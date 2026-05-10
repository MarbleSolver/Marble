using AmplNLReader, NLPModels, ADNLPModels
using MPCCModels: AbstractMPCCModel, MPCCModel, MPCCModelMeta, CCType, VarVar, VarCon, ConVar, ConCon
using LinearAlgebra

function from_NLPModel(mpcc::MPCCModel)::MarbleData
    nlp = mpcc.nlp
    x0  = zeros(nlp.meta.nvar)

    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon
    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar
    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon

    ind_cc1  = mpcc.meta.ind_cc1
    ind_cc2  = mpcc.meta.ind_cc2
    cc_types = mpcc.meta.cc_types
    ncc      = mpcc.meta.ncc

    # Linearize all constraints at x0:  c_full(x) = J_full * x + b_full
    J0      = Matrix(NLPModels.jac(nlp, x0))
    H0      = Matrix(NLPModels.hess(nlp, x0))
    c_at_x0 = NLPModels.cons(nlp, x0)
    b_full  = c_at_x0 - J0 * x0

    @assert nlp.meta.lin == collect(1:nlp.meta.ncon) "All constraints must be linear for MarbleData conversion"
    
    # Classify which NLP constraints / variables are in complementarity
    comp_con_inds = Set{Int}()
    comp_var_inds = Set{Int}()

    for i in 1:ncc
        t = cc_types[i]
        if t == VarVar
            push!(comp_var_inds, ind_cc1[i]); push!(comp_var_inds, ind_cc2[i])
        elseif t == VarCon
            push!(comp_var_inds, ind_cc1[i]); push!(comp_con_inds, ind_cc2[i])
        elseif t == ConVar
            push!(comp_con_inds, ind_cc1[i]); push!(comp_var_inds, ind_cc2[i])
        else  # ConCon
            push!(comp_con_inds, ind_cc1[i]); push!(comp_con_inds, ind_cc2[i])
        end
    end

    # Validate that complementarity bounds are finite (required for 0 <= expr form)
    for i in 1:ncc
        t = cc_types[i]
        if t == VarVar
            isfinite(lvar[ind_cc1[i]]) || error("VarVar pair $i: lvar[$(ind_cc1[i])] must be finite")
            isfinite(lvar[ind_cc2[i]]) || error("VarVar pair $i: lvar[$(ind_cc2[i])] must be finite")
        elseif t == VarCon
            isfinite(lvar[ind_cc1[i]]) || error("VarCon pair $i: lvar[$(ind_cc1[i])] must be finite")
            isfinite(lcon[ind_cc2[i]]) || error("VarCon pair $i: lcon[$(ind_cc2[i])] must be finite")
        elseif t == ConVar
            isfinite(lcon[ind_cc1[i]]) || error("ConVar pair $i: lcon[$(ind_cc1[i])] must be finite")
            isfinite(lvar[ind_cc2[i]]) || error("ConVar pair $i: lvar[$(ind_cc2[i])] must be finite")
        else  # ConCon
            isfinite(lcon[ind_cc1[i]]) || error("ConCon pair $i: lcon[$(ind_cc1[i])] must be finite")
            isfinite(lcon[ind_cc2[i]]) || error("ConCon pair $i: lcon[$(ind_cc2[i])] must be finite")
        end
    end

    # Regular constraints: NLP constraints not in any complementarity pair
    # Variable bounds are excluded for variables whose bounds are implicitly enforced
    # by a VarVar or VarCon/ConVar complementarity.
    reg_con_inds = [i for i in 1:ncon if i ∉ comp_con_inds]
    reg_var_inds = [i for i in 1:nvar if i ∉ comp_var_inds]

    I_nvar = Matrix{Float64}(LinearAlgebra.I, nvar, nvar)
    J_all  = vcat(J0[reg_con_inds, :],       I_nvar[reg_var_inds, :])
    b_all  = vcat(b_full[reg_con_inds],       zeros(length(reg_var_inds)))
    l_all  = vcat(lcon[reg_con_inds],         lvar[reg_var_inds])
    u_all  = vcat(ucon[reg_con_inds],         uvar[reg_var_inds])

    # Equality rows: l == u (both finite)
    eq_idx = findall(i -> isfinite(l_all[i]) && l_all[i] == u_all[i], eachindex(l_all))
    J_eq   = isempty(eq_idx) ? zeros(0, nvar) : J_all[eq_idx, :]
    b_eq   = isempty(eq_idx) ? zeros(0)       : b_all[eq_idx] .- l_all[eq_idx]

    # Inequality rows: expand finite-sided bounds to >= 0 form
    ineq_idx     = findall(i -> !(isfinite(l_all[i]) && l_all[i] == u_all[i]), eachindex(l_all))
    J_ineq_parts = Matrix{Float64}[]
    b_ineq_parts = Vector{Float64}[]
    for i in ineq_idx
        Ji = J_all[i:i, :]
        bi = b_all[i]
        isfinite(l_all[i]) && (push!(J_ineq_parts, Ji);  push!(b_ineq_parts, [bi - l_all[i]]))
        isfinite(u_all[i]) && (push!(J_ineq_parts, -Ji); push!(b_ineq_parts, [u_all[i] - bi]))
    end
    J_ineq = isempty(J_ineq_parts) ? zeros(0, nvar) : vcat(J_ineq_parts...)
    b_ineq = isempty(b_ineq_parts) ? zeros(0)       : vcat(b_ineq_parts...)

    # Complementarity: 0 <= L*x + lcc ⊥  R*x + rcc >= 0
    # Each pair contributes one row to L/lcc and one row to R/rcc.
    L_parts   = Matrix{Float64}[]
    lcc_parts = Vector{Float64}[]
    R_parts   = Matrix{Float64}[]
    rcc_parts = Vector{Float64}[]

    for i in 1:ncc
        t    = cc_types[i]
        idx1 = ind_cc1[i]
        idx2 = ind_cc2[i]

        if t == VarVar
            # 0 <= x[idx1] - lvar[idx1]  ⊥  x[idx2] - lvar[idx2] >= 0
            push!(L_parts,   I_nvar[idx1:idx1, :]); push!(lcc_parts, [-lvar[idx1]])
            push!(R_parts,   I_nvar[idx2:idx2, :]); push!(rcc_parts, [-lvar[idx2]])
        elseif t == VarCon
            # 0 <= x[idx1] - lvar[idx1]  ⊥  J_{idx2}*x + b_{idx2} - lcon[idx2] >= 0
            push!(L_parts,   I_nvar[idx1:idx1, :]); push!(lcc_parts, [-lvar[idx1]])
            push!(R_parts,   J0[idx2:idx2, :]);     push!(rcc_parts, [b_full[idx2] - lcon[idx2]])
        elseif t == ConVar
            # 0 <= J_{idx1}*x + b_{idx1} - lcon[idx1]  ⊥  x[idx2] - lvar[idx2] >= 0
            push!(L_parts,   J0[idx1:idx1, :]);     push!(lcc_parts, [b_full[idx1] - lcon[idx1]])
            push!(R_parts,   I_nvar[idx2:idx2, :]); push!(rcc_parts, [-lvar[idx2]])
        else  # ConCon
            # 0 <= J_{idx1}*x + b_{idx1} - lcon[idx1]  ⊥  J_{idx2}*x + b_{idx2} - lcon[idx2] >= 0
            push!(L_parts,   J0[idx1:idx1, :]); push!(lcc_parts, [b_full[idx1] - lcon[idx1]])
            push!(R_parts,   J0[idx2:idx2, :]); push!(rcc_parts, [b_full[idx2] - lcon[idx2]])
        end
    end

    L   = isempty(L_parts)   ? zeros(0, nvar) : vcat(L_parts...)
    lcc = isempty(lcc_parts) ? zeros(0)       : vcat(lcc_parts...)
    R   = isempty(R_parts)   ? zeros(0, nvar) : vcat(R_parts...)
    rcc = isempty(rcc_parts) ? zeros(0)       : vcat(rcc_parts...)

    # Objective: f(x) = 0.5 x'Qx + q'x + c0
    obj_sign = nlp.meta.minimize ? 1.0 : -1.0
    Q        = obj_sign * H0
    g_at_x0  = obj_sign * NLPModels.grad(nlp, x0)
    f_at_x0  = obj_sign * NLPModels.obj(nlp, x0)
    q        = g_at_x0 - Q * x0
    c0       = f_at_x0 - 0.5 * dot(x0, Q, x0) - dot(q, x0)

    return MarbleData(
        Q = Q, q = q, c0 = c0,
        J_eq   = J_eq,   b_eq   = b_eq,
        J_ineq = J_ineq, b_ineq = b_ineq,
        L = L, l = lcc,
        R = R, r = rcc,
    )
end