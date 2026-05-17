function to_jump(marble_data::MarbleData) 
    model = Model()

    (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r) = marble_data

    nvar = length(q)
    ncc = length(l)

    @variable(model, x[1:nvar])
    @variable(model, sL[1:ncc] >= 0)
    @variable(model, sR[1:ncc] >= 0)

    # Quadratic objective
    @objective(model, Min, 1/2 * dot(x, Q, x) + q' * x + c0)

    # Eq/Ineq constraints
    @constraints(model, begin
        J_eq * x + b_eq .== 0
        J_ineq * x + b_ineq .>= 0
    end)

    # Complementarity slacks
    @constraints(model, begin
        L * x .+ l .== sL
        R * x .+ r .== sR
    end)

    @constraint(model, [j = 1:ncc], [sL[j], sR[j]] in SOS1())

    return model
end

function MarbleData(
    nlp::AbstractNLPModel,
    ind_cc1,
    ind_cc2,
    comp_type::Union{
        Tuple{Symbol,Symbol},
        AbstractVector{<:Tuple{Symbol,Symbol}},
    },
)::MarbleData
    ind_cc1 = Int.(collect(ind_cc1))
    ind_cc2 = Int.(collect(ind_cc2))

    ncc = length(ind_cc1)

    comp_type = comp_type isa Tuple ? fill(comp_type, ncc) : collect(comp_type)

    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon

    @assert nlp.meta.nnln == 0 "Expected no nonlinear constraints in `nlp`"
    @assert length(ind_cc2) == ncc "Expected equal number of complementarity endpoints"
    @assert length(comp_type) == ncc "Expected one complementarity type per pair"
    @assert all(t -> all(in((:var, :con)), t), comp_type) """
    Expected comp_type entries of form (:var, :var), (:var, :con), (:con, :var), or (:con, :con)
    """

    kind1 = first.(comp_type)
    kind2 = last.(comp_type)

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

    J_all = vcat(J0, Matrix{T}(I, nvar, nvar))
    b_all = vcat(b0, zeros(T, nvar))

    l_all = T.(vcat(collect(nlp.meta.lcon), collect(nlp.meta.lvar)))
    u_all = T.(vcat(collect(nlp.meta.ucon), collect(nlp.meta.uvar)))

    has_lb = isfinite.(l_all)
    has_ub = isfinite.(u_all)

    mask_cc = falses(ncon + nvar)
    mask_cc[unique(vcat(rows1, rows2))] .= true

    @assert all(has_lb[mask_cc] .| has_ub[mask_cc]) """
    Expected all complementarity endpoints to have at least one finite bound
    """

    rowscale(A, s) = A .* reshape(s, :, 1)

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

    is_eq = has_lb .& has_ub .& (l_all .== u_all) .& .!mask_cc

    ineq_lb = has_lb .& .!is_eq .& .!mask_cc
    ineq_ub = has_ub .& .!is_eq .& .!mask_cc

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

    return MarbleData(
        Q, q, c0;
        J_eq=J_eq, b_eq=b_eq,
        J_ineq=J_ineq, b_ineq=b_ineq,
        L=L, l=l,
        R=R, r=r,
    )
end