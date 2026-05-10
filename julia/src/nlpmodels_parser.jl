using AmplNLReader, NLPModels, ADNLPModels, MPCC
using LinearAlgebra

function from_NLPModel(nlp::AmplModel)::MarbleData
    x0   = zeros(nlp.meta.nvar)
    lcon = nlp.meta.lcon
    ucon = nlp.meta.ucon
    lvar = nlp.meta.lvar
    uvar = nlp.meta.uvar
    nvar = nlp.meta.nvar

    # Partition constraints into complementarity vs. regular (AMPL convention:
    # cvar[i] = j ≠ 0  means constraint i is complementary to variable j)
    comp_inds = findall(nlp.meta.cvar .!= 0)
    reg_inds  = findall(nlp.meta.cvar .== 0)
    comp_vars = nlp.meta.cvar[comp_inds]
    obj_sign  = nlp.meta.minimize ? 1.0 : -1.0

    # Linearize all constraints at x0:  c_full(x) = J_full * x + b_full
    J_full  = Matrix(NLPModels.jac(nlp, x0))
    c_at_x0 = NLPModels.cons(nlp, x0)
    b_full  = c_at_x0 - J_full * x0

    # Stack regular constraints with variable bounds, EXCLUDING vars whose
    # bounds are already enforced implicitly by a complementarity constraint
    free_vars = setdiff(1:nvar, comp_vars)
    I_free    = Matrix{Float64}(LinearAlgebra.I, nvar, nvar)[free_vars, :]
    J_all = vcat(J_full[reg_inds, :], I_free)
    b_all = vcat(b_full[reg_inds],    zeros(length(free_vars)))
    l_all = vcat(lcon[reg_inds],      lvar[free_vars])
    u_all = vcat(ucon[reg_inds],      uvar[free_vars])

    # Equality rows: l == u (and finite)
    eq_idx = findall(i -> isfinite(l_all[i]) && l_all[i] == u_all[i], eachindex(l_all))
    J_eq   = isempty(eq_idx) ? zeros(0, nvar) : J_all[eq_idx, :]
    b_eq   = isempty(eq_idx) ? zeros(0)       : b_all[eq_idx] .- l_all[eq_idx]

    # Inequality rows: expand finite bounds to >= 0 form
    ineq_idx     = findall(i -> !(isfinite(l_all[i]) && l_all[i] == u_all[i]), eachindex(l_all))
    J_ineq_parts = Matrix{Float64}[]
    b_ineq_parts = Vector{Float64}[]
    for i in ineq_idx
        Ji = J_all[i:i, :]
        bi = b_all[i]
        # c(x) >= l  →   Ji * x + (bi - l) >= 0
        isfinite(l_all[i]) && (push!(J_ineq_parts, Ji);  push!(b_ineq_parts, [bi - l_all[i]]))
        # c(x) <= u  →  -Ji * x + (u - bi) >= 0
        isfinite(u_all[i]) && (push!(J_ineq_parts, -Ji); push!(b_ineq_parts, [u_all[i] - bi]))
    end
    J_ineq = isempty(J_ineq_parts) ? zeros(0, nvar) : vcat(J_ineq_parts...)
    b_ineq = isempty(b_ineq_parts) ? zeros(0)       : vcat(b_ineq_parts...)

    # Complementarity: G(x) >= lccg  ⊥  H(x) = x[comp_vars] >= lcch
    lccg = lcon[comp_inds]
    lcch = lvar[comp_vars]
    JG   = J_full[comp_inds, :]
    JH   = Matrix{Float64}(LinearAlgebra.I, nvar, nvar)[comp_vars, :]   # selector matrix
    G_at_x0 = c_at_x0[comp_inds]
    H_at_x0 = x0[comp_vars]
    L = JG;  l = G_at_x0 - JG * x0 - lccg
    R = JH;  r = H_at_x0 - JH * x0 - lcch

    # Objective: f(x) = 0.5 x'Qx + q'x + c0  (with sign flip if maximizing)
    Q       = obj_sign * Matrix(NLPModels.hess(nlp, x0))   # Symmetric → full dense
    g_at_x0 = obj_sign * NLPModels.grad(nlp, x0)
    f_at_x0 = obj_sign * NLPModels.obj(nlp, x0)
    q       = g_at_x0 - Q * x0
    c0      = f_at_x0 - 0.5 * dot(x0, Q, x0) - dot(q, x0)

    return MarbleData(
        Q = Q, q = q, c0 = c0,
        J_eq   = J_eq,   b_eq   = b_eq,
        J_ineq = J_ineq, b_ineq = b_ineq,
        L = L, l = l,
        R = R, r = r,
    )
end

function from_NLPModel(mpcc::AbstractMPCCModel)::MarbleData
    x0   = zeros(mpcc.meta.nvar)
    lcon = mpcc.meta.lcon
    ucon = mpcc.meta.ucon
    lvar = mpcc.meta.lvar
    uvar = mpcc.meta.uvar
    nvar = mpcc.meta.nvar

    # Standard constraints: c(x) = J_full * x + b_c
    J_full  = Matrix(jac(mpcc, x0))
    c_at_x0 = cons(mpcc, x0)
    b_c     = c_at_x0 - J_full * x0

    # Stack general constraints with variable bounds (identity rows)
    I_n   = Matrix{Float64}(LinearAlgebra.I, nvar, nvar)
    J_all = vcat(J_full, I_n)
    b_all = vcat(b_c,    zeros(nvar))   # bounds: c_i(x) = x_i, so b = 0
    l_all = vcat(lcon,   lvar)
    u_all = vcat(ucon,   uvar)

    # Equality rows: l == u (and finite)
    eq_idx = findall(i -> isfinite(l_all[i]) && l_all[i] == u_all[i], eachindex(l_all))
    J_eq   = isempty(eq_idx) ? zeros(0, nvar) : J_all[eq_idx, :]
    b_eq   = isempty(eq_idx) ? zeros(0)       : b_all[eq_idx] .- l_all[eq_idx]

    # Inequality rows: expand finite bounds to >= 0 form
    ineq_idx     = findall(i -> !(isfinite(l_all[i]) && l_all[i] == u_all[i]), eachindex(l_all))
    J_ineq_parts = Matrix{Float64}[]
    b_ineq_parts = Vector{Float64}[]
    for i in ineq_idx
        Ji = J_all[i:i, :]
        bi = b_all[i]
        # c(x) >= l  →   Ji * x + (bi - l) >= 0
        isfinite(l_all[i]) && (push!(J_ineq_parts, Ji);  push!(b_ineq_parts, [bi - l_all[i]]))
        # c(x) <= u  →  -Ji * x + (u - bi) >= 0
        isfinite(u_all[i]) && (push!(J_ineq_parts, -Ji); push!(b_ineq_parts, [u_all[i] - bi]))
    end
    J_ineq = isempty(J_ineq_parts) ? zeros(0, nvar) : vcat(J_ineq_parts...)
    b_ineq = isempty(b_ineq_parts) ? zeros(0)       : vcat(b_ineq_parts...)

    # Complementarity 
    lccG    = mpcc.cc_meta.lccG
    lccH    = mpcc.cc_meta.lccH
    JG      = Matrix(jacG(mpcc, x0))
    JH      = Matrix(jacH(mpcc, x0))
    G_at_x0 = consG(mpcc, x0)
    H_at_x0 = consH(mpcc, x0)
    L = JG;  l = G_at_x0 - JG * x0 - lccG
    R = JH;  r = H_at_x0 - JH * x0 - lccH

    # Objective
    Q       = Matrix(hess(mpcc, x0))
    g_at_x0 = grad(mpcc, x0)
    q       = g_at_x0 - Q * x0
    c0      = obj(mpcc, x0) - 0.5 * dot(x0, Q, x0) - dot(q, x0)

    return MarbleData(
        Q = Q, q = q, c0 = c0,
        J_eq   = J_eq,   b_eq   = b_eq,
        J_ineq = J_ineq, b_ineq = b_ineq,
        L = L, l = l,
        R = R, r = r,
    )
end