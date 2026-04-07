using AmplNLReader
using NLPModels
using SparseArrays
using LinearAlgebra
using MAT

function problem_data(; nl_path::String)::NamedTuple
    @assert ispath(nl_path) "File not found: $nl_path"
    name = splitext(basename(nl_path))[1]

    nlp = AmplModel(nl_path)
    meta = nlp.meta
    nz = meta.nvar

    # Constraint index sets
    eq_inds    = meta.jfix
    comp_inds  = findall(meta.cvar .!= 0)
    ineq_inds  = setdiff(1:meta.ncon, union(eq_inds, comp_inds))
    low_inds   = intersect(ineq_inds, meta.jlow)
    upp_inds   = intersect(ineq_inds, meta.jupp)
    rng_inds   = intersect(ineq_inds, meta.jrng)

    var_eq_inds  = meta.ifix
    var_low_inds = setdiff(meta.ilow, meta.cvar)
    var_upp_inds = meta.iupp
    var_rng_inds = meta.irng

    n_comp_c = 2 * length(comp_inds)
    n_eq_var = length(var_eq_inds)

    n_var_low = length(var_low_inds); n_var_upp = length(var_upp_inds); n_var_rng = length(var_rng_inds)
    n_ineq_var = n_var_low + n_var_upp + 2 * n_var_rng


    z0 = zeros(nz)

    #  Cost: f(z) = 0.5 z'Hz + q'z 
    # H is the (constant) quadratic cost matrix; q is the linear cost vector.
    # grad(nlp, z) = Hz + q, so q = grad(nlp, 0).

    obj_sign = meta.minimize ? 1.0 : -1.0
    H_cost = obj_sign * sparse(NLPModels.hess(nlp, z0))
    q_cost = obj_sign * NLPModels.grad(nlp, z0)
    f0_cost = NLPModels.obj(nlp, z0)

    #  Evaluate the NLP constraint Jacobian once (constant for LCQP) 
    nlp_J = jac(nlp, z0)

    #  Variable equality constraints: J*z + b = 0 
    # z[var_eq_inds] - lvar[var_eq_inds] = 0
    J_eq_var = sparse(1:n_eq_var, var_eq_inds, ones(n_eq_var), n_eq_var, nz)
    b_eq_var = -meta.lvar[var_eq_inds]

    #  Variable inequality constraints: J*z + b ≥ 0 
    # [z[var_low] - lvar; -z[var_upp] + uvar; z[var_rng] - lvar; -z[var_rng] + uvar]
    ineq_var_rows = [1:n_var_low;
                     n_var_low+1:n_var_low+n_var_upp;
                     n_var_low+n_var_upp+1:n_var_low+n_var_upp+n_var_rng;
                     n_var_low+n_var_upp+n_var_rng+1:n_ineq_var]
    ineq_var_cols = [var_low_inds; var_upp_inds; var_rng_inds; var_rng_inds]
    ineq_var_vals = [ones(n_var_low); -ones(n_var_upp); ones(n_var_rng); -ones(n_var_rng)]
    J_ineq_var = sparse(ineq_var_rows, ineq_var_cols, ineq_var_vals, n_ineq_var, nz)
    b_ineq_var = [-meta.lvar[var_low_inds];
                   meta.uvar[var_upp_inds];
                  -meta.lvar[var_rng_inds];
                   meta.uvar[var_rng_inds]]

    #  NLP equality constraints: J*z + b = 0 
    # c(z)[eq_inds] - lcon[eq_inds] = 0
    J_eq_c = nlp_J[eq_inds, :]
    b_eq_c = -meta.lcon[eq_inds]

    #  NLP inequality constraints: J*z + b ≥ 0 
    # [c[low]-lcon; -c[upp]+ucon; c[rng]-lcon; -c[rng]+ucon]
    J_ineq_c = [ nlp_J[low_inds, :];
                -nlp_J[upp_inds, :];
                 nlp_J[rng_inds, :];
                -nlp_J[rng_inds, :]]
    b_ineq_c = [-meta.lcon[low_inds];
                 meta.ucon[upp_inds];
                -meta.lcon[rng_inds];
                 meta.ucon[rng_inds]]

    #  Complementarity constraints 
    # Interleaved pairs:
    #   Odd  rows (1:2:end): z[cvar[i]] - lvar[cvar[i]]    (variable component)
    #   Even rows (2:2:end): c(z)[i]    - lcon[i]          (NLP constraint component)
    n_pairs = length(comp_inds)
    cvar    = meta.cvar[comp_inds]
    J_comp_c = spzeros(n_comp_c, nz)
    J_comp_c[1:2:end, :] = sparse(1:n_pairs, cvar, ones(n_pairs), n_pairs, nz)
    J_comp_c[2:2:end, :] = nlp_J[comp_inds, :]
    b_comp_c = Vector{Float64}(undef, n_comp_c)
    b_comp_c[1:2:end] .= -meta.lvar[cvar]
    b_comp_c[2:2:end] .= -meta.lcon[comp_inds]

    return (
        name   = name,
        cost   = (H=H_cost, q=q_cost, f0=f0_cost),
        eq_c   = (J=[J_eq_var; J_eq_c],     b=[b_eq_var; b_eq_c]),
        ineq_c = (J=[J_ineq_var; J_ineq_c], b=[b_ineq_var; b_ineq_c]),
        comp_c = (J=J_comp_c,               b=b_comp_c),
    )
end

function save_data(; data::NamedTuple, savedir::String = @__DIR__)
    @assert isdir(savedir) "Directory not found: $savedir"

    # Save data as sparse .mat file
    matwrite(joinpath(savedir, "$(data.name).mat"), Dict("H" => data.cost.H, "q" => data.cost.q, "f0" => data.cost.f0,
                                      "J_eq" => data.eq_c.J, "b_eq" => data.eq_c.b,
                                      "J_ineq" => data.ineq_c.J, "b_ineq" => data.ineq_c.b,
                                      "J_comp" => data.comp_c.J, "b_comp" => data.comp_c.b))
end