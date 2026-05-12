using AmplNLReader, NLPModels, ADNLPModels
using MPCCModels: MPCCModel, VarVar, VarCon, ConVar, ConCon
using LinearAlgebra

# Returns (Jrow, scalar, extra) such that Jrow*x + scalar >= 0 encodes one side
# of a complementarity pair in the solver's canonical form.  When both bounds are
# finite, the lower bound is used for the CC side and the upper bound is returned
# as `extra` to be added as a standalone inequality.
function _cc_con_row(J0, b_full, lcon, ucon, idx)
    l, u = lcon[idx], ucon[idx]
    isfinite(l) || isfinite(u) || error("Complementarity constraint $idx has no finite bound")
    J, b   = J0[idx:idx, :], b_full[idx]
    J_cc   = isfinite(l) ? J : -J
    b_cc   = isfinite(l) ? b - l : u - b
    extra  = (isfinite(l) && isfinite(u)) ? (-J, u - b) : nothing
    return J_cc, b_cc, extra
end

function _cc_var_row(nvar, lvar, uvar, idx)
    l, u = lvar[idx], uvar[idx]
    isfinite(l) || isfinite(u) || error("Complementarity variable $idx has no finite bound")
    e      = zeros(1, nvar); e[1, idx] = 1.0
    e_cc   = isfinite(l) ? e : -e
    b_cc   = isfinite(l) ? -l : u
    extra  = (isfinite(l) && isfinite(u)) ? (-e, u) : nothing
    return e_cc, b_cc, extra
end

function from_mpcc(mpcc::MPCCModel)::MarbleData
    nlp  = mpcc.nlp
    x0   = zeros(nlp.meta.nvar)
    nvar = nlp.meta.nvar
    ncon = nlp.meta.ncon
    lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
    lcon, ucon = nlp.meta.lcon, nlp.meta.ucon

    ind_cc1  = mpcc.meta.ind_cc1
    ind_cc2  = mpcc.meta.ind_cc2
    cc_types = mpcc.meta.cc_types
    ncc      = mpcc.meta.ncc

    J0     = Matrix(NLPModels.jac(nlp, x0))
    H0     = Matrix(NLPModels.hess(nlp, x0))
    b_full = NLPModels.cons(nlp, x0)  # x0=0 ⟹ c(x0) - J0*x0 = c(0)

    # Classify CC indices and validate bounds in one pass
    comp_con_inds = Set{Int}()
    comp_var_inds = Set{Int}()

    for i in 1:ncc
        t = cc_types[i]
        side1_var = t == VarVar || t == VarCon
        side2_var = t == VarVar || t == ConVar
        idx1, idx2 = ind_cc1[i], ind_cc2[i]

        if side1_var
            push!(comp_var_inds, idx1)
            isfinite(lvar[idx1]) || isfinite(uvar[idx1]) ||
                error("Complementarity variable $idx1 (pair $i, side 1) has no finite bound")
        else
            push!(comp_con_inds, idx1)
            isfinite(lcon[idx1]) || isfinite(ucon[idx1]) ||
                error("Complementarity constraint $idx1 (pair $i, side 1) has no finite bound")
        end

        if side2_var
            push!(comp_var_inds, idx2)
            isfinite(lvar[idx2]) || isfinite(uvar[idx2]) ||
                error("Complementarity variable $idx2 (pair $i, side 2) has no finite bound")
        else
            push!(comp_con_inds, idx2)
            isfinite(lcon[idx2]) || isfinite(ucon[idx2]) ||
                error("Complementarity constraint $idx2 (pair $i, side 2) has no finite bound")
        end
    end

    reg_con_inds = [i for i in 1:ncon if i ∉ comp_con_inds]
    reg_var_inds = [i for i in 1:nvar if i ∉ comp_var_inds]

    # Build variable-bound rows directly — avoids allocating a full nvar×nvar identity
    n_reg_var = length(reg_var_inds)
    I_reg = zeros(n_reg_var, nvar)
    for (row, col) in enumerate(reg_var_inds); I_reg[row, col] = 1.0; end

    J_all = vcat(J0[reg_con_inds, :], I_reg)
    b_all = vcat(b_full[reg_con_inds], zeros(n_reg_var))
    l_all = vcat(lcon[reg_con_inds],   lvar[reg_var_inds])
    u_all = vcat(ucon[reg_con_inds],   uvar[reg_var_inds])

    # Equality rows (l == u); inequality rows (everything else)
    is_eq    = [isfinite(l_all[i]) && l_all[i] == u_all[i] for i in eachindex(l_all)]
    J_eq     = J_all[is_eq, :]
    b_eq     = b_all[is_eq] .- l_all[is_eq]

    J_ineq_parts = Matrix{Float64}[]
    b_ineq_parts = Vector{Float64}[]
    for i in findall(.!is_eq)
        Ji, bi = J_all[i:i, :], b_all[i]
        isfinite(l_all[i]) && (push!(J_ineq_parts, Ji);  push!(b_ineq_parts, [bi - l_all[i]]))
        isfinite(u_all[i]) && (push!(J_ineq_parts, -Ji); push!(b_ineq_parts, [u_all[i] - bi]))
    end

    # Complementarity: 0 <= L*x + l ⊥ R*x + r >= 0
    # Doubly-bounded CC sides also emit an extra inequality (upper bound).
    L_parts = Matrix{Float64}[]; lcc_parts = Vector{Float64}[]
    R_parts = Matrix{Float64}[]; rcc_parts = Vector{Float64}[]

    cc_side = (is_var, idx) -> is_var ?
        _cc_var_row(nvar, lvar, uvar, idx) :
        _cc_con_row(J0, b_full, lcon, ucon, idx)

    for i in 1:ncc
        t = cc_types[i]
        J1, b1, extra1 = cc_side(t == VarVar || t == VarCon, ind_cc1[i])
        J2, b2, extra2 = cc_side(t == VarVar || t == ConVar, ind_cc2[i])

        push!(L_parts, J1); push!(lcc_parts, [b1])
        push!(R_parts, J2); push!(rcc_parts, [b2])

        if extra1 !== nothing; push!(J_ineq_parts, extra1[1]); push!(b_ineq_parts, [extra1[2]]); end
        if extra2 !== nothing; push!(J_ineq_parts, extra2[1]); push!(b_ineq_parts, [extra2[2]]); end
    end

    vcat_mat(parts) = isempty(parts) ? zeros(0, nvar) : vcat(parts...)
    vcat_vec(parts) = isempty(parts) ? Float64[]      : vcat(parts...)

    # Objective: f(x) = ½x'Qx + q'x + c0; x0=0 ⟹ q = ∇f(0), c0 = f(0)
    obj_sign = nlp.meta.minimize ? 1.0 : -1.0
    Q  = obj_sign * H0
    q  = obj_sign * NLPModels.grad(nlp, x0)
    c0 = obj_sign * NLPModels.obj(nlp, x0)

    return MarbleData(
        Q = Q, q = q, c0 = c0,
        J_eq   = J_eq,                    b_eq   = b_eq,
        J_ineq = vcat_mat(J_ineq_parts),  b_ineq = vcat_vec(b_ineq_parts),
        L      = vcat_mat(L_parts),       l      = vcat_vec(lcc_parts),
        R      = vcat_mat(R_parts),       r      = vcat_vec(rcc_parts),
    )
end
