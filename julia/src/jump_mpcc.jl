using JuMP


"""
    nlp_con_row_map(model::JuMP.Model) -> Dict{MOI.ConstraintIndex,Int}

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
    nlp_var_col_map(model::JuMP.Model) -> Dict{MOI.VariableIndex,Int}

Return the 1-based NLP column index for each variable in `model`, using the same
variable order as `MathOptNLPModel(model)`.
"""
function nlp_var_col_map(model::JuMP.Model)
    vars = MOI.get(JuMP.backend(model), MOI.ListOfVariableIndices())
    return Dict(vi => i for (i, vi) in enumerate(vars))
end

"""
    ComplementarityIndexBuilder(model::JuMP.Model)

Mutable builder for Marble complementarity index arrays.

The builder caches NLP variable-column and constraint-row maps, so construct it
after adding the variables and constraints that will appear in complementarity
pairs.
"""
mutable struct ComplementarityIndexBuilder
    model::JuMP.Model
    col_map::Dict{MOI.VariableIndex, Int}
    row_map::Dict{MOI.ConstraintIndex, Int}
    ind_cc1::Vector{Int}
    ind_cc2::Vector{Int}
    cc_types::Vector{Tuple{Symbol, Symbol}}
end

function ComplementarityIndexBuilder(model::JuMP.Model)
    return ComplementarityIndexBuilder(
        model,
        nlp_var_col_map(model),
        nlp_con_row_map(model),
        Int[],
        Int[],
        Tuple{Symbol, Symbol}[],
    )
end

# A complementarity side is either a single JuMP variable/constraint reference or
# an array of them. `_as_vec` normalizes both into a flat vector so the rest of
# the machinery never has to distinguish the two cases.
_as_vec(x::AbstractArray) = vec(x)
_as_vec(x) = [x]

_assert_same_length(a, b) = @assert length(_as_vec(a)) == length(_as_vec(b)) "Complementarity arguments must have the same length"

"""
    _complementarity_kind(x) -> Symbol

Infer the complementarity kind (`:var` or `:con`) from a JuMP reference or an
array of references.
"""
_complementarity_kind(::JuMP.VariableRef) = :var
_complementarity_kind(::JuMP.ConstraintRef) = :con
function _complementarity_kind(x::AbstractArray)
    eltype(x) <: JuMP.VariableRef && return :var
    eltype(x) <: JuMP.ConstraintRef && return :con
    return _complementarity_kind(first(x))
end
_complementarity_kind(x) = error(
    "Complementarity terms must be JuMP variable or constraint references, got $(typeof(x))",
)

_var_indices(builder::ComplementarityIndexBuilder, vars) =
    [builder.col_map[JuMP.index(v)] for v in _as_vec(vars)]

_con_indices(builder::ComplementarityIndexBuilder, cons) =
    [builder.row_map[JuMP.index(c)] for c in _as_vec(cons)]

_term_indices(builder::ComplementarityIndexBuilder, x) =
    _complementarity_kind(x) === :var ? _var_indices(builder, x) : _con_indices(builder, x)

function _append_complementarity_block!(
        builder::ComplementarityIndexBuilder,
        inds1,
        inds2,
        cc_type::Tuple{Symbol, Symbol},
    )
    append!(builder.ind_cc1, inds1)
    append!(builder.ind_cc2, inds2)
    append!(builder.cc_types, fill(cc_type, length(inds1)))
    return builder
end

"""
    add_complementarities!(builder, lhs, rhs) -> builder

Append complementarity pairs from `lhs` and `rhs`, inferring each side's kind
(`:var`/`:con`) from the type of the JuMP references passed.

`lhs` and `rhs` may each be a single JuMP variable/constraint reference or an
array of references; the two sides must have the same number of elements.
"""
function add_complementarities!(builder::ComplementarityIndexBuilder, lhs, rhs)
    _assert_same_length(lhs, rhs)
    cc_type = (_complementarity_kind(lhs), _complementarity_kind(rhs))
    return _append_complementarity_block!(
        builder,
        _term_indices(builder, lhs),
        _term_indices(builder, rhs),
        cc_type,
    )
end

"""
    add_var_var_complementarities!(builder, vars1, vars2) -> builder

Append variable-variable complementarity pairs. Each side may be a single
variable reference or an equally sized array of them.
"""
add_var_var_complementarities!(builder::ComplementarityIndexBuilder, vars1, vars2) =
    add_complementarities!(builder, vars1, vars2)

"""
    add_var_con_complementarities!(builder, vars, cons) -> builder

Append variable-constraint complementarity pairs. Each side may be a single
reference or an equally sized array of them.
"""
add_var_con_complementarities!(builder::ComplementarityIndexBuilder, vars, cons) =
    add_complementarities!(builder, vars, cons)

"""
    add_con_var_complementarities!(builder, cons, vars) -> builder

Append constraint-variable complementarity pairs. Each side may be a single
reference or an equally sized array of them.
"""
add_con_var_complementarities!(builder::ComplementarityIndexBuilder, cons, vars) =
    add_complementarities!(builder, cons, vars)

"""
    add_con_con_complementarities!(builder, cons1, cons2) -> builder

Append constraint-constraint complementarity pairs. Each side may be a single
constraint reference or an equally sized array of them.
"""
add_con_con_complementarities!(builder::ComplementarityIndexBuilder, cons1, cons2) =
    add_complementarities!(builder, cons1, cons2)

"""
    complementarity_indices(builder::ComplementarityIndexBuilder)

Return `(ind_cc1, ind_cc2, cc_types)` accumulated in `builder`.
"""
complementarity_indices(builder::ComplementarityIndexBuilder) =
    builder.ind_cc1, builder.ind_cc2, builder.cc_types

"""
    complementarity_indices(model::JuMP.Model, blocks...)

Build `(ind_cc1, ind_cc2, cc_types)` for one or more complementarity blocks.

Each block is `(lhs, rhs)`. Each side may be a single JuMP variable/constraint
reference or an equally sized array of them; the two sides must have the same
number of elements. The kind of each side (`:var`/`:con`) is inferred from the
type of the references passed.

Example:

```julia
ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
    (model[:λ], model[:gap_nonnegative]),  # con/con (or var/con, etc.)
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
    var_con_complementarities(model, vars, cons)
    var_con_complementarities(builder, vars, cons)

Return `(variable_column, constraint_row)` pairs for equally shaped variable and
constraint arrays.
"""
function var_con_complementarities(model::JuMP.Model, vars::AbstractArray, cons::AbstractArray)
    return var_con_complementarities(ComplementarityIndexBuilder(model), vars, cons)
end

function var_con_complementarities(builder::ComplementarityIndexBuilder, vars::AbstractArray, cons::AbstractArray)
    _assert_same_length(vars, cons)
    return collect(zip(_var_indices(builder, vars), _con_indices(builder, cons)))
end

"""
    con_con_complementarities(model, cons1, cons2)
    con_con_complementarities(builder, cons1, cons2)

Return `(constraint_row1, constraint_row2)` pairs for equally shaped constraint
arrays.
"""
function con_con_complementarities(model::JuMP.Model, cons1::AbstractArray, cons2::AbstractArray)
    return con_con_complementarities(ComplementarityIndexBuilder(model), cons1, cons2)
end

function con_con_complementarities(builder::ComplementarityIndexBuilder, cons1::AbstractArray, cons2::AbstractArray)
    _assert_same_length(cons1, cons2)
    return collect(zip(_con_indices(builder, cons1), _con_indices(builder, cons2)))
end

"""
    var_var_complementarities(model, vars1, vars2)
    var_var_complementarities(builder, vars1, vars2)

Return `(variable_column1, variable_column2)` pairs for equally shaped variable
arrays.
"""
function var_var_complementarities(model::JuMP.Model, vars1::AbstractArray, vars2::AbstractArray)
    return var_var_complementarities(ComplementarityIndexBuilder(model), vars1, vars2)
end

function var_var_complementarities(builder::ComplementarityIndexBuilder, vars1::AbstractArray, vars2::AbstractArray)
    _assert_same_length(vars1, vars2)
    return collect(zip(_var_indices(builder, vars1), _var_indices(builder, vars2)))
end


"""
    var_inds(model::JuMP.Model) -> Dict{Symbol,<:AbstractArray{Int}}

Return MOI variable-index values for named variable arrays in `model`.

Only `AbstractArray{VariableRef}` entries in `object_dictionary(model)` are
included; scalar variables and non-variable objects are skipped.
"""
var_inds(model::JuMP.Model) = Dict(
    k => map(v -> JuMP.index(v).value, v)
    for (k, v) in object_dictionary(model)
    if v isa AbstractArray{VariableRef}
)

const _SUPPORTED_COMPLEMENTARITY_TYPES = (:var, :con)

struct _ComplementarityTerm
    index::Int
    kind::Symbol
end

struct _ReformulationContext
    model::JuMP.Model
    var_by_col::Dict{Int,MOI.VariableIndex}
    con_by_row::Dict{Int,Any}
    con_slack_by_row::Dict{Int,JuMP.VariableRef}
end

"""State shared by reformulation passes on a copied JuMP model."""
function _ReformulationContext(model::JuMP.Model)
    var_by_col = Dict{Int,MOI.VariableIndex}()
    for (var_index, col_idx) in nlp_var_col_map(model)
        var_by_col[col_idx] = var_index
    end

    con_by_row = Dict{Int,Any}()
    for (con_index, row_idx) in nlp_con_row_map(model)
        con_by_row[row_idx] = con_index
    end

    return _ReformulationContext(model, var_by_col, con_by_row, Dict{Int,JuMP.VariableRef}())
end

function _validate_complementarity_kind(kind::Symbol)
    kind in _SUPPORTED_COMPLEMENTARITY_TYPES && return nothing
    error("Unsupported complementarity type: $kind")
end

function _complementarity_term(index, kind::Symbol)
    _validate_complementarity_kind(kind)
    return _ComplementarityTerm(Int(index), kind)
end

function _complementarity_pairs(ind_cc1, ind_cc2, types)
    n_pairs = length(ind_cc1)
    length(ind_cc2) == n_pairs || throw(DimensionMismatch(
        "ind_cc1 and ind_cc2 must have the same length",
    ))
    length(types) == n_pairs || throw(DimensionMismatch(
        "types must have the same length as the complementarity indices",
    ))

    pairs = Vector{Tuple{_ComplementarityTerm,_ComplementarityTerm}}(undef, n_pairs)
    for j in 1:n_pairs
        kind1, kind2 = types[j]
        pairs[j] = (
            _complementarity_term(ind_cc1[j], kind1),
            _complementarity_term(ind_cc2[j], kind2),
        )
    end
    return pairs
end

function _variable_ref(ctx::_ReformulationContext, col_idx::Int)
    haskey(ctx.var_by_col, col_idx) || error("No variable found at column index $col_idx")
    return JuMP.VariableRef(ctx.model, ctx.var_by_col[col_idx])
end

function _constraint_ref(ctx::_ReformulationContext, row_idx::Int)
    haskey(ctx.con_by_row, row_idx) || error("No constraint found at row index $row_idx")
    return JuMP.ConstraintRef(ctx.model, ctx.con_by_row[row_idx], JuMP.ScalarShape())
end

function _reference(ctx::_ReformulationContext, term::_ComplementarityTerm)
    if term.kind == :var
        return _variable_ref(ctx, term.index)
    elseif term.kind == :con
        return _constraint_ref(ctx, term.index)
    end
    error("Unsupported complementarity type: $(term.kind)")
end

function _constraint_expr(con::JuMP.ConstraintRef)
    obj = JuMP.constraint_object(con)
    return obj.func - MOI.constant(obj.set)
end

"""Return the reusable nonnegative slack that represents constraint row `row_idx`."""
function _constraint_slack!(ctx::_ReformulationContext, row_idx::Int)
    return get!(ctx.con_slack_by_row, row_idx) do
        con = _constraint_ref(ctx, row_idx)
        slack = @variable(ctx.model, lower_bound = 0)
        @constraint(ctx.model, slack == _constraint_expr(con))
        delete(ctx.model, con)
        slack
    end
end

function _sos1_member!(ctx::_ReformulationContext, term::_ComplementarityTerm)
    if term.kind == :var
        return _variable_ref(ctx, term.index)
    elseif term.kind == :con
        return _constraint_slack!(ctx, term.index)
    end
    error("Unsupported complementarity type: $(term.kind)")
end

"""
    reformulate_sos1(model::JuMP.Model, ind_cc1, ind_cc2, types) -> JuMP.Model

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
    ctx = _ReformulationContext(new_model)

    for (term1, term2) in _complementarity_pairs(ind_cc1, ind_cc2, types)
        member1 = _sos1_member!(ctx, term1)
        member2 = _sos1_member!(ctx, term2)
        @constraint(new_model, [member1, member2] in JuMP.SOS1())
    end
    return new_model
end
