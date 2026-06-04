# function to_jump(marble_data::MarbleData) 
#     model = Model()

#     (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = marble_data

#     nvar = length(q)
#     ncc = length(l)

#     @variable(model, x[1:nvar])
#     @variable(model, sL[1:ncc] >= 0)
#     @variable(model, sR[1:ncc] >= 0)

#     # Quadratic objective
#     @objective(model, Min, 1/2 * dot(x, Q, x) + q' * x + c0)

#     # Eq/Ineq constraints
#     @constraints(model, begin
#         J_eq * x + b_eq .== 0
#         J_ineq * x + b_ineq .>= 0
#     end)

#     # Complementarity slacks
#     @constraints(model, begin
#         L * x .+ l .== sL
#         R * x .+ r .== sR
#     end)

#     @constraint(model, [j = 1:ncc], [sL[j], sR[j]] in SOS1())

#     return model
# end

# Convert an NLPModels problem with linear constraints and complementarity pairs
# into the matrix/vector data Marble consumes
#
# `ind_cc1[j]` and `ind_cc2[j]` are the two endpoints of complementarity pair j,
# each addressed as a variable (`:var`) or constraint (`:con`) row via `cc_type`
#
# Returns a NamedTuple describing
#
#     minimize    1/2 xᵀ Q x + qᵀ x + c0
#     subject to  J_eq   x + b_eq   == 0
#                 J_ineq x + b_ineq >= 0
#                 0 <= (L x + l) ⊥ (R x + r) >= 0
#
# where the last line is the elementwise complementarity condition
# (L x + l)_j ≥ 0, (R x + r)_j ≥ 0, (L x + l)_j (R x + r)_j = 0 for j = 1..ncc
function jump_to_marble(
    nlp::AbstractNLPModel,
    ind_cc1,
    ind_cc2,
    cc_type::Union{
        Tuple{Symbol,Symbol},
        AbstractVector{<:Tuple{Symbol,Symbol}},
    },
)
    ind_cc1 = Int.(collect(ind_cc1))
    ind_cc2 = Int.(collect(ind_cc2))

    ncc = length(ind_cc1)

    cc_type = cc_type isa Tuple ? fill(cc_type, ncc) : collect(cc_type)

    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon

    @assert nlp.meta.nnln == 0 "Expected no nonlinear constraints in `nlp`"
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