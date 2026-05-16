const VerticalMPCCModel = Tuple{AbstractNLPModel, Vector{Int}, Vector{Int}}

function to_vertical_mpcc(model::JuMP.Model)::VerticalMPCCModel
    model_copy = copy(model)
    sos1_cons = all_constraints(model_copy, Vector{VariableRef}, MOI.SOS1{Float64})
    
    ind_vcc1 = vec(map(c -> JuMP.index(constraint_object(c).func[1]).value, sos1_cons))
    ind_vcc2 = vec(map(c -> JuMP.index(constraint_object(c).func[2]).value, sos1_cons))

    for c in sos1_cons
        delete(model_copy, c)
    end

    nlp = MathOptNLPModel(model_copy)
    @assert nlp.meta.nnln == 0 "Expected no nonlinear constraints after removing SOS1 constraints"
    @assert all(nlp.meta.lvar[ind_vcc1] .== 0) && all(nlp.meta.lvar[ind_vcc2] .== 0) "Expected variables in SOS1 constraints to be of the form x[j] >= 0"

    nlp, ind_vcc1, ind_vcc2
end

function to_vertical_mpcc(ampl::AmplModel)::VerticalMPCCModel
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]

    return ampl, ind_vcc1, ind_ccc2
end

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

# function MarbleData(nlp::AbstractNLPModel, ind_vcc1, ind_vcc2)::Marble.MarbleData
#     ind_vcc1 = collect(ind_vcc1)
#     ind_vcc2 = collect(ind_vcc2)

#     nvar = nlp.meta.nvar

#     @assert nlp.meta.nnln == 0 "Expected no nonlinear constraints in `nlp`"

#     @assert length(ind_vcc1) == length(ind_vcc2) "Expected equal number of variables in each VCC pair"
#     @assert all(1 .<= ind_vcc1 .<= nvar) "Invalid index in ind_vcc1"
#     @assert all(1 .<= ind_vcc2 .<= nvar) "Invalid index in ind_vcc2"

#     comp_var_inds = sort(unique(vcat(ind_vcc1, ind_vcc2)))

#     lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
#     lcon, ucon = nlp.meta.lcon, nlp.meta.ucon

#     @assert all(isfinite.(lvar[ind_vcc1])) "Expected finite lower bounds for ind_vcc1"
#     @assert all(isfinite.(lvar[ind_vcc2])) "Expected finite lower bounds for ind_vcc2"
#     @assert all(lvar[ind_vcc1] .== 0.0) "Expected variables in ind_vcc1 to have lower bound 0"
#     @assert all(lvar[ind_vcc2] .== 0.0) "Expected variables in ind_vcc2 to have lower bound 0"

#     # Optional but useful consistency check with NLPModels metadata.
#     lower_bounded_vars = union(nlp.meta.ilow, nlp.meta.irng)
#     @assert comp_var_inds ⊆ lower_bounded_vars "VCC variables should have lower bounds"

#     ncc = length(ind_vcc1)

#     x0     = zeros(nvar)
#     J0     = Matrix(NLPModels.jac(nlp, x0))
#     H0     = Matrix(NLPModels.hess(nlp, x0))
#     b_full = NLPModels.cons(nlp, x0)

#     # CC variables are handled through L/R, so exclude them from regular
#     # variable-bound rows. Their upper bounds are added separately below.
#     reg_var_inds = setdiff(1:nvar, comp_var_inds)

#     I_reg = zeros(length(reg_var_inds), nvar)
#     for (row, j) in enumerate(reg_var_inds)
#         I_reg[row, j] = 1.0
#     end

#     # All original constraint rows are regular after to_vertical_mpcc.
#     J_all = vcat(J0, I_reg)
#     b_all = vcat(b_full, zeros(length(reg_var_inds)))
#     l_all = vcat(lcon, lvar[reg_var_inds])
#     u_all = vcat(ucon, uvar[reg_var_inds])

#     is_eq = isfinite.(l_all) .& isfinite.(u_all) .& (l_all .== u_all)

#     # Equality form:
#     #
#     #     J_eq * x + b_eq = 0
#     #
#     # For l <= c(x) <= u with l == u:
#     #
#     #     c(x) - l = 0
#     #
#     J_eq = J_all[is_eq, :]
#     b_eq = b_all[is_eq] - l_all[is_eq]

#     ineq_lb = isfinite.(l_all) .& .!is_eq
#     ineq_ub = isfinite.(u_all) .& .!is_eq

#     # Inequality form:
#     #
#     #     J_ineq * x + b_ineq >= 0
#     #
#     # Lower bounds:
#     #
#     #     c(x) >= l
#     #     c(x) - l >= 0
#     #
#     # Upper bounds:
#     #
#     #     c(x) <= u
#     #     -c(x) + u >= 0
#     #

#     # Add upper bounds on complementarity variables, since their lower bounds
#     # are represented through L/R and their regular bound rows were excluded.
#     vcc_ub_inds = comp_var_inds[isfinite.(uvar[comp_var_inds])]

#     J_vcc_ub = zeros(length(vcc_ub_inds), nvar)
#     for (row, j) in enumerate(vcc_ub_inds)
#         J_vcc_ub[row, j] = 1.0
#     end

#     J_ineq = [
#         J_all[ineq_lb, :];
#         -J_all[ineq_ub, :];
#         -J_vcc_ub;
#     ]

#     b_ineq = [
#         b_all[ineq_lb] - l_all[ineq_lb];
#         u_all[ineq_ub] - b_all[ineq_ub];
#         uvar[vcc_ub_inds];
#     ]

#     L = zeros(ncc, nvar)
#     R = zeros(ncc, nvar)

#     for k in 1:ncc
#         L[k, ind_vcc1[k]] = 1.0
#         R[k, ind_vcc2[k]] = 1.0
#     end

#     l = -lvar[ind_vcc1]
#     r = -lvar[ind_vcc2]

#     obj_sign = nlp.meta.minimize ? 1.0 : -1.0

#     Q  = obj_sign * H0
#     q  = obj_sign * NLPModels.grad(nlp, x0)
#     c0 = obj_sign * NLPModels.obj(nlp, x0)

#     return MarbleData(
#         Q,
#         q,
#         c0;
#         J_eq   = J_eq,
#         b_eq   = b_eq,
#         J_ineq = J_ineq,
#         b_ineq = b_ineq,
#         L = L,
#         R = R,
#         l = l,
#         r = r,
#     )
# end