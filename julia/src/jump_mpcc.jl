using JuMP

# NLP index maps following the ordering of `MathOptNLPModel(model)`, so an NLP
# column/row index can be recovered from a JuMP reference.

"""1-based NLP column index for each variable in `model`."""
nlp_var_col_map(model::JuMP.Model) =
    Dict(vi => i for (i, vi) in enumerate(MOI.get(JuMP.backend(model), MOI.ListOfVariableIndices())))

"""
1-based NLP row index for each scalar constraint in `model`, skipping
`MOI.VariableIndex` constraints (which become variable bounds, not NLP rows).
"""
function nlp_con_row_map(model::JuMP.Model)
    moi = JuMP.backend(model)
    rows = Dict{MOI.ConstraintIndex, Int}()
    for (F, S) in MOI.get(moi, MOI.ListOfConstraintTypesPresent())
        F == MOI.VariableIndex && continue
        for ci in MOI.get(moi, MOI.ListOfConstraintIndices{F,S}())
            rows[ci] = length(rows) + 1
        end
    end
    return rows
end

# Complementarity indexing

_as_vec(x::AbstractArray) = vec(x)
_as_vec(x) = [x]

# Infer the side kind (`:var`/`:con`) from a JuMP reference or array of them.
_cc_kind(::JuMP.VariableRef) = :var
_cc_kind(::JuMP.ConstraintRef) = :con
_cc_kind(x::AbstractArray) = _cc_kind(first(x))
_cc_kind(x) = error("Complementarity terms must be JuMP variable/constraint references, got $(typeof(x))")

"""
    complementarity_indices(model, blocks...) -> (ind_cc1, ind_cc2, cc_types)

Resolve one or more complementarity blocks into NLP indices. Each block is a
`(lhs, rhs)` pair, where each side is a single JuMP variable/constraint reference
or an equally sized array of them; the two sides must have the same length. Each
side's kind (`:var`/`:con`) is inferred from the references' type.

```julia
ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
    (model[:μ], model[:gate_violation]),   # var ⊥ con
    (model[:a], model[:b]),                # single references are fine
)
```
"""
function complementarity_indices(model::JuMP.Model, blocks...)
    col, row = nlp_var_col_map(model), nlp_con_row_map(model)
    resolve(x) = _cc_kind(x) === :var ? [col[JuMP.index(v)] for v in _as_vec(x)] :
                                        [row[JuMP.index(c)] for c in _as_vec(x)]

    ind_cc1, ind_cc2 = Int[], Int[]
    cc_types = Tuple{Symbol,Symbol}[]
    for (lhs, rhs) in blocks
        i1, i2 = resolve(lhs), resolve(rhs)
        @assert length(i1) == length(i2) "Complementarity sides must have equal length"
        append!(ind_cc1, i1)
        append!(ind_cc2, i2)
        append!(cc_types, fill((_cc_kind(lhs), _cc_kind(rhs)), length(i1)))
    end
    return ind_cc1, ind_cc2, cc_types
end

"""
Return MOI variable-index values for named variable arrays in `model`.

Only `AbstractArray{VariableRef}` entries of `object_dictionary(model)` are
included; scalar variables and non-variable objects are skipped.
"""
var_inds(model::JuMP.Model) = Dict(
    k => map(v -> JuMP.index(v).value, v)
    for (k, v) in object_dictionary(model)
    if v isa AbstractArray{VariableRef}
)

# MPCC — an NLP bundled with its resolved complementarity data

"""
    MPCC(model::JuMP.Model, blocks...)
    MPCC(nlp::AbstractNLPModel, ind_cc1, ind_cc2, cc_types; inds=Dict())

A mathematical program with complementarity constraints: an `nlp` paired with the
complementarity data Marble consumes. `ind_cc1`/`ind_cc2` index the two endpoints
of each pair as NLP variable/constraint rows, `cc_types` records each endpoint's
kind (`:var`/`:con`), and `inds` is an optional name → column map (see
[`var_inds`](@ref)). Because the indices address `nlp` rows/columns, they only
mean anything alongside the `nlp` they were resolved against.

The first form builds the `nlp` as `MathOptNLPModel(model)` and resolves each
`block` via [`complementarity_indices`](@ref); the second takes pre-resolved
indices directly.

```julia
MPCC(model, (model[:μ], model[:gate_violation]))
MPCC(MathOptNLPModel(model), ind_cc1, ind_cc2, cc_types)
```
"""
struct MPCC
    nlp::AbstractNLPModel
    ind_cc1::Vector{Int}
    ind_cc2::Vector{Int}
    cc_types::Vector{Tuple{Symbol,Symbol}}
    inds::Dict
end

function MPCC(nlp::AbstractNLPModel, ind_cc1, ind_cc2, cc_types; inds::Dict=Dict())
    return MPCC(nlp, collect(Int, ind_cc1), collect(Int, ind_cc2),
                collect(Tuple{Symbol,Symbol}, cc_types), inds)
end

function MPCC(model::JuMP.Model, blocks...)
    ind_cc1, ind_cc2, cc_types = complementarity_indices(model, blocks...)
    return MPCC(MathOptNLPModel(model), ind_cc1, ind_cc2, cc_types; inds=var_inds(model))
end

# Conversion to Marble problem data

"""
    jump_to_marble(mpcc::MPCC) -> NamedTuple

Convert an [`MPCC`](@ref) — an NLP with linear constraints and resolved
complementarity pairs — into the matrix/vector data Marble consumes.

`mpcc.ind_cc1[j]` and `mpcc.ind_cc2[j]` are the two endpoints of complementarity
pair `j`, each addressed as a variable (`:var`) or constraint (`:con`) row of
`mpcc.nlp` via `mpcc.cc_types`.

Returns a `NamedTuple` describing

    minimize    1/2 xᵀ Q x + qᵀ x + c0
    subject to  J_eq   x + b_eq   == 0
                J_ineq x + b_ineq >= 0
                0 <= (L x + l) ⊥ (R x + r) >= 0

where the last line is the elementwise complementarity condition
`(L x + l)_j ≥ 0`, `(R x + r)_j ≥ 0`, `(L x + l)_j (R x + r)_j = 0` for `j = 1..ncc`.
"""
function jump_to_marble(mpcc::MPCC)
    nlp = mpcc.nlp

    ind_cc1 = Int.(collect(mpcc.ind_cc1))
    ind_cc2 = Int.(collect(mpcc.ind_cc2))
    cc_type = mpcc.cc_types

    ncc = length(ind_cc1)

    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon

    @assert nlp.meta.nnln == 0 "Expected only linear constraints in `mpcc.nlp`"
    @assert length(ind_cc2) == ncc "Expected equal number of complementarity endpoints"
    @assert length(cc_type) == ncc "Expected one complementarity type per pair"
    @assert all(t -> all(in((:var, :con)), t), cc_type) """
    Expected cc_type entries of form (:var, :var), (:var, :con), (:con, :var), or (:con, :con)
    """

    kind1 = first.(cc_type)
    kind2 = last.(cc_type)

    valid(k, i) = 1 <= i <= (k === :var ? nvar : ncon)
    row(k, i) = k === :var ? ncon + i : i

    @assert all(valid.(kind1, ind_cc1)) "Invalid index in ind_cc1"
    @assert all(valid.(kind2, ind_cc2)) "Invalid index in ind_cc2"

    rows1 = row.(kind1, ind_cc1)
    rows2 = row.(kind2, ind_cc2)

    x0 = zeros(nvar)

    obj_sign = nlp.meta.minimize ? 1.0 : -1.0

    J0 = Matrix(NLPModels.jac(nlp, x0))
    H0 = Matrix(obj_sign * NLPModels.hess(nlp, x0))
    q  = collect(obj_sign * NLPModels.grad(nlp, x0))
    c0 = obj_sign * NLPModels.obj(nlp, x0)
    b0 = collect(NLPModels.cons(nlp, x0))

    T = promote_type(eltype(J0), eltype(H0), eltype(q), eltype(b0))

    J0 = T.(J0)
    Q  = T.(H0)
    q  = T.(q)
    b0 = T.(b0)
    c0 = T(c0)

    # Stack the constraint rows on top of an identity block so a single row index
    # addresses either a constraint (rows 1:ncon) or a variable (rows ncon+1:end)
    J_all = vcat(J0, Matrix{T}(I, nvar, nvar))
    b_all = vcat(b0, zeros(T, nvar))

    l_all = T.(vcat(collect(nlp.meta.lcon), collect(nlp.meta.lvar)))
    u_all = T.(vcat(collect(nlp.meta.ucon), collect(nlp.meta.uvar)))

    has_lb = isfinite.(l_all)
    has_ub = isfinite.(u_all)

    mask_cc = falses(ncon + nvar)
    mask_cc[unique(vcat(rows1, rows2))] .= true

    # Each endpoint becomes a slack 0 <= A x + b, so it must be bounded on at
    # least one side, report the offending pair so the input is easy to fix
    for j in 1:ncc
        @assert has_lb[rows1[j]] || has_ub[rows1[j]] "Left endpoint of complementarity pair $j (ind_cc1[$j]=$(ind_cc1[j]), $(kind1[j])) has no finite bound"
        @assert has_lb[rows2[j]] || has_ub[rows2[j]] "Right endpoint of complementarity pair $j (ind_cc2[$j]=$(ind_cc2[j]), $(kind2[j])) has no finite bound"
    end

    rowscale(A, s) = A .* reshape(s, :, 1)

    # Map a bounded row g(x) = A x + b to its nonnegative complementarity slack,
    # preferring the lower bound (slack g - lb) and flipping sign for an
    # upper-bound-only row (slack ub - g)
    function residual(rows)
        use_lb = has_lb[rows]
        sgn = ifelse.(use_lb, one(T), -one(T))

        A = rowscale(J_all[rows, :], sgn)
        b = ifelse.(
            use_lb,
            b_all[rows] .- l_all[rows],
            u_all[rows] .- b_all[rows],
        )

        return A, b
    end

    L, l = residual(rows1)
    R, r = residual(rows2)

    # A two-sided bound with lb == ub is an equality, everything else bounded and
    # not part of a complementarity pair becomes a one-sided inequality
    is_eq = has_lb .& has_ub .& (l_all .== u_all) .& .!mask_cc

    ineq_lb = has_lb .& .!is_eq .& .!mask_cc
    ineq_ub = has_ub .& .!is_eq .& .!mask_cc

    # A two-sided complementarity endpoint has its lower bound consumed by the
    # slack above, so emit the leftover upper bound ub - g >= 0 as an inequality
    cc_extra_ub = mask_cc .& has_lb .& has_ub

    J_eq = J_all[is_eq, :]
    b_eq = b_all[is_eq] .- l_all[is_eq]

    J_ineq = [
        J_all[ineq_lb, :];
        -J_all[ineq_ub, :];
        -J_all[cc_extra_ub, :];
    ]

    b_ineq = [
        b_all[ineq_lb] .- l_all[ineq_lb];
        u_all[ineq_ub] .- b_all[ineq_ub];
        u_all[cc_extra_ub] .- b_all[cc_extra_ub];
    ]

    return (
        Q=Q, q=q, c0=c0,
        J_eq=J_eq, b_eq=b_eq,
        J_ineq=J_ineq, b_ineq=b_ineq,
        L=L, l=l,
        R=R, r=r,
    )
end

# SOS1 reformulation (for handing the MPCC to an MILP solver such as Gurobi)

"""
    reformulate_sos1(model, ind_cc1, ind_cc2, cc_types) -> JuMP.Model

Return a copy of `model` with each complementarity pair reformulated as an SOS1
set. A `:var` term uses its variable directly; a `:con` term is replaced by a
nonnegative slack equal to the constraint body, and the original constraint is
deleted in the returned copy.
"""
function reformulate_sos1(model::JuMP.Model, ind_cc1, ind_cc2, cc_types)::JuMP.Model
    m = copy(model)
    var_by_col = Dict(col => vi for (vi, col) in nlp_var_col_map(m))
    con_by_row = Dict(row => ci for (ci, row) in nlp_con_row_map(m))
    slack = Dict{Int, JuMP.VariableRef}()

    # SOS1 member for term `idx` of kind `kind` in the copied model `m`.
    member(::Val{:var}, idx) = JuMP.VariableRef(m, var_by_col[idx])
    member(::Val{:con}, idx) = get!(slack, idx) do
        con = JuMP.ConstraintRef(m, con_by_row[idx], JuMP.ScalarShape())
        obj = JuMP.constraint_object(con)
        s = @variable(m, lower_bound = 0)
        @constraint(m, s == obj.func - MOI.constant(obj.set))
        delete(m, con)
        s
    end

    for j in eachindex(ind_cc1)
        k1, k2 = cc_types[j]
        a = member(Val(k1), ind_cc1[j])
        b = member(Val(k2), ind_cc2[j])
        @constraint(m, [a, b] in JuMP.SOS1())
    end
    return m
end
