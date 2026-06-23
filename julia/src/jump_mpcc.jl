using JuMP


"""
Return the 1-based NLP row index for each scalar, non-variable-bound constraint
in `model`.

This follows the constraint order used by `MathOptNLPModel(model)` and skips
`MOI.VariableIndex` constraints, which become variable bounds instead of NLP
rows.
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
Return the 1-based NLP column index for each variable in `model`, using the same
variable order as `MathOptNLPModel(model)`.
"""
function nlp_var_col_map(model::JuMP.Model)
    vars = MOI.get(JuMP.backend(model), MOI.ListOfVariableIndices())
    return Dict(vi => i for (i, vi) in enumerate(vars))
end

"""
Mutable builder for Marble complementarity pairs.

The builder stores JuMP references and resolves NLP indices when
`complementarity_indices(builder)` is called. This lets you add more variables or
constraints to the model between calls to `add_complementarities!`.
"""
mutable struct ComplementarityIndexBuilder
    model::JuMP.Model
    blocks::Vector{Tuple{Any, Any}}
end

function ComplementarityIndexBuilder(model::JuMP.Model)
    return ComplementarityIndexBuilder(model, Tuple{Any, Any}[])
end

as_vec(x::AbstractArray) = vec(x)
as_vec(x) = [x]

same_length(a, b) =
    @assert length(as_vec(a)) == length(as_vec(b)) "Complementarity arguments must have the same length"

"""
Infer the complementarity kind (`:var` or `:con`) from a JuMP reference or an
array of references.
"""
complementarity_kind(::JuMP.VariableRef) = :var
complementarity_kind(::JuMP.ConstraintRef) = :con
function complementarity_kind(x::AbstractArray)
    eltype(x) <: JuMP.VariableRef && return :var
    eltype(x) <: JuMP.ConstraintRef && return :con
    return complementarity_kind(first(x))
end
complementarity_kind(x) = error(
    "Complementarity terms must be JuMP variable or constraint references, got $(typeof(x))",
)

var_indices(col_map, vars) = [col_map[JuMP.index(v)] for v in as_vec(vars)]
con_indices(row_map, cons) = [row_map[JuMP.index(c)] for c in as_vec(cons)]
term_indices(col_map, row_map, x) =
    complementarity_kind(x) === :var ? var_indices(col_map, x) : con_indices(row_map, x)

function append_complementarity_block!(builder::ComplementarityIndexBuilder, lhs, rhs)
    same_length(lhs, rhs)
    push!(builder.blocks, (lhs, rhs))
    return builder
end

"""
Append complementarity pairs from `lhs` and `rhs`, inferring each side's kind
(`:var`/`:con`) from the type of the JuMP references passed.

`lhs` and `rhs` may each be a single JuMP variable/constraint reference or an
array of references; the two sides must have the same number of elements.
"""
function add_complementarities!(builder::ComplementarityIndexBuilder, lhs, rhs)
    return append_complementarity_block!(builder, lhs, rhs)
end

"""
Append variable-variable complementarity pairs. Each side may be a single
variable reference or an equally sized array of them.
"""
add_var_var_complementarities!(builder::ComplementarityIndexBuilder, vars1, vars2) =
    add_complementarities!(builder, vars1, vars2)

"""
Append variable-constraint complementarity pairs. Each side may be a single
reference or an equally sized array of them.
"""
add_var_con_complementarities!(builder::ComplementarityIndexBuilder, vars, cons) =
    add_complementarities!(builder, vars, cons)

"""
Append constraint-variable complementarity pairs. Each side may be a single
reference or an equally sized array of them.
"""
add_con_var_complementarities!(builder::ComplementarityIndexBuilder, cons, vars) =
    add_complementarities!(builder, cons, vars)

"""
Append constraint-constraint complementarity pairs. Each side may be a single
constraint reference or an equally sized array of them.
"""
add_con_con_complementarities!(builder::ComplementarityIndexBuilder, cons1, cons2) =
    add_complementarities!(builder, cons1, cons2)

"""
Return `(ind_cc1, ind_cc2, cc_types)` for the pairs accumulated in `builder`.
"""
function complementarity_indices(builder::ComplementarityIndexBuilder)
    col_map = nlp_var_col_map(builder.model)
    row_map = nlp_con_row_map(builder.model)
    ind_cc1 = Int[]
    ind_cc2 = Int[]
    cc_types = Tuple{Symbol, Symbol}[]

    for (lhs, rhs) in builder.blocks
        inds1 = term_indices(col_map, row_map, lhs)
        inds2 = term_indices(col_map, row_map, rhs)
        cc_type = (complementarity_kind(lhs), complementarity_kind(rhs))
        append!(ind_cc1, inds1)
        append!(ind_cc2, inds2)
        append!(cc_types, fill(cc_type, length(inds1)))
    end

    return ind_cc1, ind_cc2, cc_types
end

"""
Build `(ind_cc1, ind_cc2, cc_types)` for one or more complementarity blocks.

Each block is `(lhs, rhs)`. Each side may be a single JuMP variable/constraint
reference or an equally sized array of them; the two sides must have the same
number of elements. The kind of each side (`:var`/`:con`) is inferred from the
type of the references passed.

Example:

```julia
ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
    (model[:lambda], model[:gap_nonnegative]),  # con/con (or var/con, etc.)
    (model[:a], model[:b]),                # single references are fine
)
```
"""
function complementarity_indices(model::JuMP.Model, blocks...)
    builder = ComplementarityIndexBuilder(model)
    for block in blocks
        @assert length(block) == 2 "Each complementarity block must be `(lhs, rhs)`"
        lhs, rhs = block
        add_complementarities!(builder, lhs, rhs)
    end
    return complementarity_indices(builder)
end

"""
Return `(variable_column, constraint_row)` pairs for equally shaped variable and
constraint arrays.
"""
function var_con_complementarities(model::JuMP.Model, vars::AbstractArray, cons::AbstractArray)
    same_length(vars, cons)
    return collect(zip(
        var_indices(nlp_var_col_map(model), vars),
        con_indices(nlp_con_row_map(model), cons),
    ))
end

function var_con_complementarities(builder::ComplementarityIndexBuilder, vars::AbstractArray, cons::AbstractArray)
    return var_con_complementarities(builder.model, vars, cons)
end

"""
Return `(constraint_row1, constraint_row2)` pairs for equally shaped constraint
arrays.
"""
function con_con_complementarities(model::JuMP.Model, cons1::AbstractArray, cons2::AbstractArray)
    same_length(cons1, cons2)
    row_map = nlp_con_row_map(model)
    return collect(zip(con_indices(row_map, cons1), con_indices(row_map, cons2)))
end

function con_con_complementarities(builder::ComplementarityIndexBuilder, cons1::AbstractArray, cons2::AbstractArray)
    return con_con_complementarities(builder.model, cons1, cons2)
end

"""
Return `(variable_column1, variable_column2)` pairs for equally shaped variable
arrays.
"""
function var_var_complementarities(model::JuMP.Model, vars1::AbstractArray, vars2::AbstractArray)
    same_length(vars1, vars2)
    col_map = nlp_var_col_map(model)
    return collect(zip(var_indices(col_map, vars1), var_indices(col_map, vars2)))
end

function var_var_complementarities(builder::ComplementarityIndexBuilder, vars1::AbstractArray, vars2::AbstractArray)
    return var_var_complementarities(builder.model, vars1, vars2)
end


"""
Return MOI variable-index values for named variable arrays in `model`.

Only `AbstractArray{VariableRef}` entries in `object_dictionary(model)` are
included; scalar variables and non-variable objects are skipped.
"""
var_inds(model::JuMP.Model) = Dict(
    k => map(v -> JuMP.index(v).value, v)
    for (k, v) in object_dictionary(model)
    if v isa AbstractArray{VariableRef}
)

const SUPPORTED_COMPLEMENTARITY_TYPES = (:var, :con)

struct ComplementarityTerm
    index::Int
    kind::Symbol
end

struct ReformulationContext
    model::JuMP.Model
    var_by_col::Dict{Int,MOI.VariableIndex}
    con_by_row::Dict{Int,Any}
    con_slack_by_row::Dict{Int,JuMP.VariableRef}
end

"""State shared by reformulation passes on a copied JuMP model."""
function ReformulationContext(model::JuMP.Model)
    var_by_col = Dict{Int,MOI.VariableIndex}()
    for (var_index, col_idx) in nlp_var_col_map(model)
        var_by_col[col_idx] = var_index
    end

    con_by_row = Dict{Int,Any}()
    for (con_index, row_idx) in nlp_con_row_map(model)
        con_by_row[row_idx] = con_index
    end

    return ReformulationContext(model, var_by_col, con_by_row, Dict{Int,JuMP.VariableRef}())
end

function validate_complementarity_kind(kind::Symbol)
    kind in SUPPORTED_COMPLEMENTARITY_TYPES && return nothing
    error("Unsupported complementarity type: $kind")
end

function complementarity_term(index, kind::Symbol)
    validate_complementarity_kind(kind)
    return ComplementarityTerm(Int(index), kind)
end

function complementarity_pairs(ind_cc1, ind_cc2, types)
    n_pairs = length(ind_cc1)
    length(ind_cc2) == n_pairs || throw(DimensionMismatch(
        "ind_cc1 and ind_cc2 must have the same length",
    ))
    length(types) == n_pairs || throw(DimensionMismatch(
        "types must have the same length as the complementarity indices",
    ))

    pairs = Vector{Tuple{ComplementarityTerm,ComplementarityTerm}}(undef, n_pairs)
    for j in 1:n_pairs
        kind1, kind2 = types[j]
        pairs[j] = (
            complementarity_term(ind_cc1[j], kind1),
            complementarity_term(ind_cc2[j], kind2),
        )
    end
    return pairs
end

function variable_ref(ctx::ReformulationContext, col_idx::Int)
    haskey(ctx.var_by_col, col_idx) || error("No variable found at column index $col_idx")
    return JuMP.VariableRef(ctx.model, ctx.var_by_col[col_idx])
end

function constraint_ref(ctx::ReformulationContext, row_idx::Int)
    haskey(ctx.con_by_row, row_idx) || error("No constraint found at row index $row_idx")
    return JuMP.ConstraintRef(ctx.model, ctx.con_by_row[row_idx], JuMP.ScalarShape())
end

function constraint_expr(con::JuMP.ConstraintRef)
    obj = JuMP.constraint_object(con)
    return obj.func - MOI.constant(obj.set)
end

"""Return the reusable nonnegative slack that represents constraint row `row_idx`."""
function constraint_slack!(ctx::ReformulationContext, row_idx::Int)
    return get!(ctx.con_slack_by_row, row_idx) do
        con = constraint_ref(ctx, row_idx)
        slack = @variable(ctx.model, lower_bound = 0)
        @constraint(ctx.model, slack == constraint_expr(con))
        delete(ctx.model, con)
        slack
    end
end

function sos1_member!(ctx::ReformulationContext, term::ComplementarityTerm)
    if term.kind == :var
        return variable_ref(ctx, term.index)
    elseif term.kind == :con
        return constraint_slack!(ctx, term.index)
    end
    error("Unsupported complementarity type: $(term.kind)")
end

"""
Return a copy of `model` with complementarity pairs reformulated as SOS1 sets.

Each `(ind_cc1[j], ind_cc2[j], types[j])` defines one pair. Types are
`(:var, :var)`, `(:var, :con)`, `(:con, :var)`, or `(:con, :con)`, where
`:var` indices are NLP variable columns and `:con` indices are NLP constraint
rows.

Constraint terms are replaced with nonnegative slack variables and the original
constraints are deleted in the returned model.
"""
function reformulate_sos1(model::JuMP.Model, ind_cc1, ind_cc2, types)::JuMP.Model
    new_model = copy(model)
    ctx = ReformulationContext(new_model)

    for (term1, term2) in complementarity_pairs(ind_cc1, ind_cc2, types)
        member1 = sos1_member!(ctx, term1)
        member2 = sos1_member!(ctx, term2)
        @constraint(new_model, [member1, member2] in JuMP.SOS1())
    end
    return new_model
end
