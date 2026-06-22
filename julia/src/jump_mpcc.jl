using JuMP


"""
    nlp_con_row_map(model) -> Dict{MOI.ConstraintIndex, Int}

Maps each non-variable-bound constraint in `model` to its row index in the
NLPModel produced by `MathOptNLPModel(model)`. Iterates constraint types in the
same order MOI/NLPModelsJuMP does, skipping `VariableIndex` constraints (those
become variable bounds, not NLP rows).
"""
function nlp_con_row_map(model::JuMP.Model)
    moi_model = JuMP.backend(model)
    row = 0
    map = Dict{MOI.ConstraintIndex, Int}()
    for (F, S) in MOI.get(moi_model, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        for ci in MOI.get(moi_model, MOI.ListOfConstraintIndices{F,S}())
            row += 1
            map[ci] = row
        end
    end
    return map
end

"""
    nlp_var_col_map(model) -> Dict{MOI.VariableIndex, Int}

Maps each variable to its column index in the NLPModel produced by
`MathOptNLPModel(model)`.
"""
function nlp_var_col_map(model::JuMP.Model)
    vars = MOI.get(JuMP.backend(model), MOI.ListOfVariableIndices())
    return Dict(vi => i for (i, vi) in enumerate(vars))
end

"""
    var_con_complementarities(model, vars, cons) -> Vector{Tuple{Int,Int}}

Given parallel arrays `vars` (VariableRef) and `cons` (ConstraintRef) with the
same shape, return `(col_idx, row_idx)` pairs suitable as complementarity indices
in an NLPModel.
"""
function var_con_complementarities(model::JuMP.Model, vars::AbstractArray, cons::AbstractArray)
    @assert size(vars) == size(cons) "vars and cons must have the same shape"
    col_map = nlp_var_col_map(model)
    row_map = nlp_con_row_map(model)
    return [(col_map[JuMP.index(v)], row_map[JuMP.index(c)]) for (v, c) in zip(vec(vars), vec(cons))]
end

"""
    con_con_complementarities(model, cons1, cons2) -> Vector{Tuple{Int,Int}}

Given parallel arrays `cons1` and `cons2` (ConstraintRef) with the same shape,
return `(row_idx1, row_idx2)` pairs suitable as complementarity indices in an
NLPModel.
"""
function con_con_complementarities(model::JuMP.Model, cons1::AbstractArray, cons2::AbstractArray)
    @assert size(cons1) == size(cons2) "cons1 and cons2 must have the same shape"
    row_map = nlp_con_row_map(model)
    return [(row_map[JuMP.index(c1)], row_map[JuMP.index(c2)]) for (c1, c2) in zip(vec(cons1), vec(cons2))]
end

"""
    var_var_complementarities(model, vars1, vars2) -> Vector{Tuple{Int,Int}}

Given parallel arrays `vars1` and `vars2` (VariableRef) with the same shape,
return `(col_idx1, col_idx2)` pairs suitable as complementarity indices in an
NLPModel. Both variables must be bounded below by zero.
"""
function var_var_complementarities(model::JuMP.Model, vars1::AbstractArray, vars2::AbstractArray)
    @assert size(vars1) == size(vars2) "vars1 and vars2 must have the same shape"
    col_map = nlp_var_col_map(model)
    return [(col_map[JuMP.index(v1)], col_map[JuMP.index(v2)]) for (v1, v2) in zip(vec(vars1), vec(vars2))]
end

"""
    var_inds(model) -> Dict{Symbol, Array{Int}}

Returns a mapping from each named variable array in `model` (registered via
`@variable`) to an array of the corresponding MOI column indices (1-based).
Only entries whose value is an `AbstractArray{VariableRef}` are included;
scalar variables and non-variable objects are skipped.

Useful for building index lookups before constructing an NLPModel, without
having to call `nlp_var_col_map` and then scatter indices by hand.
"""
var_inds(model::JuMP.Model) = Dict(
    k => map(v -> JuMP.index(v).value, v)
    for (k, v) in object_dictionary(model)
    if v isa AbstractArray{VariableRef}
)

function _reformulation_helpers(model)
    inv_var_map = Dict(col => vi for (vi, col) in nlp_var_col_map(model))
    inv_con_map = Dict(row => ci for (ci, row) in nlp_con_row_map(model))

    get_reference(i, t) = if t == :var
        JuMP.VariableRef(model, inv_var_map[i])
    elseif t == :con
        JuMP.ConstraintRef(model, inv_con_map[i], JuMP.ScalarShape())
    else
        error("Unsupported complementarity type: $t")
    end

    get_cons_expr(con::JuMP.ConstraintRef) = begin
        obj = constraint_object(con)
        expr = obj.func - MOI.constant(obj.set)
        return expr
    end

    # Maps constraint row index -> slack variable, so repeated occurrences of
    # the same constraint reuse the existing slack (the original constraint is
    # deleted on first encounter and can't be referenced again).
    con_slack_cache = Dict{Int, JuMP.VariableRef}()

    get_or_create_con_slack!(row_idx) = get!(con_slack_cache, row_idx) do
        con = get_reference(row_idx, :con)
        slack = @variable(model, lower_bound=0)
        @constraint(model, slack == get_cons_expr(con))
        delete(model, con)
        slack
    end

    return get_reference, get_or_create_con_slack!, con_slack_cache
end

"""
    reformulate_sos1(model, ind_cc1, ind_cc2, types)::JuMP.Model

Reformulate complementarity constraints as SOS1 sets in-place.

Each triplet `(i1, i2, (t1, t2))` drawn from `zip(ind_cc1, ind_cc2, types)`
encodes one complementarity condition `x1 ⟂ x2`, where:

- `t1`, `t2` ∈ `{:var, :con}` indicate whether the index refers to a variable
  column or a constraint row (as returned by `nlp_var_col_map` /
  `nlp_con_row_map`).
- `i1`, `i2` are the corresponding 1-based column/row indices.

Behaviour by case:

| `(t1, t2)`    | Action |
|---------------|--------|
| `(:var, :var)` | Add `[x1, x2] ∈ SOS1` directly. |
| `(:var, :con)` | Introduce a non-negative slack `s = expr(con)`, delete the original constraint, add `[x1, s] ∈ SOS1`. |
| `(:con, :var)` | Symmetric to `(:var, :con)`. |
| `(:con, :con)` | Introduce two non-negative slacks, delete both original constraints, add `[s1, s2] ∈ SOS1`. |

The constraint expression used for `:con` entries is `func - constant(set)`,
i.e. the left-hand side when the constraint is written as `expr ∈ set`.
"""
function reformulate_sos1(model::JuMP.Model, ind_cc1, ind_cc2, types)::JuMP.Model
    new_model = copy(model) # Work on a copy to avoid modifying the original model's indices
    get_reference, get_or_create_con_slack!, _ = _reformulation_helpers(new_model)

    add_cc_pair!(i1, i2, t1, t2) = begin
        if (t1, t2) == (:var, :var)
            ref1 = get_reference(i1, t1)
            ref2 = get_reference(i2, t2)
            @constraint(new_model, [ref1, ref2] in JuMP.SOS1())
        elseif (t1, t2) == (:var, :con)
            ref1 = get_reference(i1, :var)
            slack = get_or_create_con_slack!(i2)
            @constraint(new_model, [ref1, slack] in JuMP.SOS1())
        elseif (t1, t2) == (:con, :var)
            add_cc_pair!(i2, i1, :var, :con)
        elseif (t1, t2) == (:con, :con)
            slack1 = get_or_create_con_slack!(i1)
            slack2 = get_or_create_con_slack!(i2)
            @constraint(new_model, [slack1, slack2] in JuMP.SOS1())
        else
            error("Unsupported complementarity type: ($t1, $t2)")
        end
    end

    foreach(((i1, i2, (t1, t2)),) -> add_cc_pair!(i1, i2, t1, t2), zip(ind_cc1, ind_cc2, types))

    return new_model
end

function reformulate_lie(model::JuMP.Model, ind_cc1, ind_cc2, types; κ, form=:eq)::JuMP.Model
    new_model = copy(model)
    get_reference, _, _ = _reformulation_helpers(new_model)

    ncc = length(ind_cc1)
    bκ(x) = 1/2 * (x + sqrt(x^2 + 4))
    con_expr(ref)  = (o = constraint_object(ref); o.func - MOI.constant(o.set))

    @variable(new_model, s1[1:ncc])
    if form == :ineq
        @variable(new_model, s2[1:ncc])
        @constraint(new_model, s1 + s2 .<= 0)
    end

    for (j, (i1, i2, (t1, t2))) in enumerate(zip(ind_cc1, ind_cc2, types))
        cc1 = get_reference(i1, t1)
        cc2 = get_reference(i2, t2)

        expr1 = t1 == :var ? cc1 : con_expr(cc1)
        expr2 = t2 == :var ? cc2 : con_expr(cc2)

        t1 == :var ? delete_lower_bound(cc1) : delete(new_model, cc1)
        t2 == :var ? delete_lower_bound(cc2) : delete(new_model, cc2)

        @constraint(new_model, expr1 == √κ * bκ(s1[j]))
        @constraint(new_model, expr2 == √κ * bκ(form == :eq ? -s1[j] : s2[j]))
    end

    return new_model
end
