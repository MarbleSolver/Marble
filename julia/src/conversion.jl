function to_vertical_mpcc(model::JuMP.Model)::Tuple{AbstractNLPModel, Vector{Int}, Vector{Int}}
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

function to_jump(ampl::AmplModel)::JuMP.Model
    @assert ampl.meta.nnln == 0 "Expected no nonlinear constraints"

    has_finite_bound(lb, ub) = isfinite(lb) || isfinite(ub)

    function nonnegative_slack!(model, expr; ub = Inf, name = "slack")
        s = @variable(model, base_name = name)
        set_lower_bound(s, 0.0)

        if isfinite(ub)
            @assert ub >= 0.0 "Slack upper bound must be nonnegative"
            set_upper_bound(s, ub)
        end

        @constraint(model, s == expr)
        return s
    end

    function bounded_member_slack!(model, expr, lb, ub; name = "slack")
        @assert has_finite_bound(lb, ub) "Expected at least one finite bound"

        if isfinite(lb)
            slack_ub = isfinite(ub) ? ub - lb : Inf
            @assert !isfinite(ub) || slack_ub >= 0.0 "Inconsistent bounds: lb > ub"

            return nonnegative_slack!(
                model,
                expr - lb;
                ub = slack_ub,
                name = name,
            )
        else
            return nonnegative_slack!(
                model,
                ub - expr;
                name = name,
            )
        end
    end

    nv   = ampl.meta.nvar
    nc   = ampl.meta.ncon
    lvar = ampl.meta.lvar
    uvar = ampl.meta.uvar
    lcon = ampl.meta.lcon
    ucon = ampl.meta.ucon
    cvar = ampl.meta.cvar

    x0 = zeros(nv)

    c0 = NLPModels.obj(ampl, x0)
    g  = NLPModels.grad(ampl, x0)
    H  = NLPModels.hess(ampl, x0)
    A  = NLPModels.jac(ampl, x0)

    b = NLPModels.cons(ampl, x0)

    cc_inds = findall(!=(0), cvar)
    cc_var_inds = sort!(collect(Set(cvar[cc_inds])))

    @assert all(has_finite_bound.(lvar[cc_var_inds], uvar[cc_var_inds])) "Expected variables in complementarity constraints to have at least one finite bound"

    reg_inds = [
        i for i in 1:nc
        if cvar[i] == 0 && has_finite_bound(lcon[i], ucon[i])
    ]

    model = Model()

    # Create variables initially without bounds.
    @variable(model, x[1:nv])

    # A complementarity variable can be used directly only when the
    # complementarity member is exactly x[j] >= 0.
    #
    # Use exact == 0.0 instead of ≈ to avoid silently changing the model.
    use_direct_var_member = falses(nv)
    for j in cc_var_inds
        use_direct_var_member[j] = isfinite(lvar[j]) && lvar[j] == 0.0
    end

    # Now impose variable bounds intelligently
    #
    # Case 1: variable is not in complementarity:
    #   impose all original bounds directly.
    #
    # Case 2: variable is in complementarity and can be reused directly:
    #   impose all original bounds directly, because x[j] itself is the
    #   nonnegative complementarity member
    #
    # Case 3: variable is in complementarity but needs a shifted/flipped slack:
    #   do not impose its bounds here. The slack equality and slack bounds
    #   will enforce them exactly

    for j in eachindex(x)
        if j ∈ cc_var_inds && !use_direct_var_member[j]
            # Bounds will be enforced by the complementarity-side slack.
            continue
        end

        isfinite(lvar[j]) && set_lower_bound(x[j], lvar[j])
        isfinite(uvar[j]) && set_upper_bound(x[j], uvar[j])
    end

    Q = H + H' - spdiagm(0 => diag(H))

    @objective(model, Min, c0 + dot(g, x) + 0.5 * x' * Q * x)

    if !isempty(reg_inds)
        @constraint(model, lcon[reg_inds] .<= A[reg_inds, :] * x + b[reg_inds] .<= ucon[reg_inds])
    end

    cc_var_slacks = Dict{Int, VariableRef}()

    for i in cc_inds
        j = cvar[i]

        row_expr = A[i, :]' * x + b[i]

        s_c = bounded_member_slack!(
            model,
            row_expr,
            lcon[i],
            ucon[i];
            name = "cc_row_slack[$i]",
        )

        s_v =
            if use_direct_var_member[j]
                x[j]
            else
                get!(cc_var_slacks, j) do
                    bounded_member_slack!(
                        model,
                        x[j],
                        lvar[j],
                        uvar[j];
                        name = "cc_var_slack[$j]",
                    )
                end
            end

        @constraint(model, [s_c, s_v] in SOS1())
    end

    return model
end

function MarbleData(nlp::AbstractNLPModel, ind_vcc1, ind_vcc2)::Marble.MarbleData
    ind_vcc1 = collect(ind_vcc1)
    ind_vcc2 = collect(ind_vcc2)

    nvar = nlp.meta.nvar

    @assert nlp.meta.nnln == 0 "Expected no nonlinear constraints after to_vertical_mpcc"

    @assert length(ind_vcc1) == length(ind_vcc2) "Expected equal number of variables in each VCC pair"
    @assert allunique(ind_vcc1) "Expected ind_vcc1 to contain unique indices"
    @assert allunique(ind_vcc2) "Expected ind_vcc2 to contain unique indices"
    @assert isempty(intersect(ind_vcc1, ind_vcc2)) "Expected disjoint VCC index sets"

    @assert all(1 .<= ind_vcc1 .<= nvar) "Invalid index in ind_vcc1"
    @assert all(1 .<= ind_vcc2 .<= nvar) "Invalid index in ind_vcc2"

    comp_var_inds = sort(unique(vcat(ind_vcc1, ind_vcc2)))

    lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
    lcon, ucon = nlp.meta.lcon, nlp.meta.ucon

    @assert all(isfinite.(lvar[ind_vcc1])) "Expected finite lower bounds for ind_vcc1"
    @assert all(isfinite.(lvar[ind_vcc2])) "Expected finite lower bounds for ind_vcc2"
    @assert all(lvar[ind_vcc1] .== 0.0) "Expected variables in ind_vcc1 to have lower bound 0"
    @assert all(lvar[ind_vcc2] .== 0.0) "Expected variables in ind_vcc2 to have lower bound 0"

    # Optional but useful consistency check with NLPModels metadata.
    lower_bounded_vars = union(nlp.meta.ilow, nlp.meta.irng)
    @assert comp_var_inds ⊆ lower_bounded_vars "VCC variables should have lower bounds"

    ncc = length(ind_vcc1)

    x0     = zeros(nvar)
    J0     = Matrix(NLPModels.jac(nlp, x0))
    H0     = Matrix(NLPModels.hess(nlp, x0))
    b_full = NLPModels.cons(nlp, x0)

    # CC variables are handled through L/R, so exclude them from regular
    # variable-bound rows. Their upper bounds are added separately below.
    reg_var_inds = setdiff(1:nvar, comp_var_inds)

    I_reg = zeros(length(reg_var_inds), nvar)
    for (row, j) in enumerate(reg_var_inds)
        I_reg[row, j] = 1.0
    end

    # All original constraint rows are regular after to_vertical_mpcc.
    J_all = vcat(J0, I_reg)
    b_all = vcat(b_full, zeros(length(reg_var_inds)))
    l_all = vcat(lcon, lvar[reg_var_inds])
    u_all = vcat(ucon, uvar[reg_var_inds])

    is_eq = isfinite.(l_all) .& isfinite.(u_all) .& (l_all .== u_all)

    # Equality form:
    #
    #     J_eq * x + b_eq = 0
    #
    # For l <= c(x) <= u with l == u:
    #
    #     c(x) - l = 0
    #
    J_eq = J_all[is_eq, :]
    b_eq = b_all[is_eq] - l_all[is_eq]

    ineq_lb = isfinite.(l_all) .& .!is_eq
    ineq_ub = isfinite.(u_all) .& .!is_eq

    # Inequality form:
    #
    #     J_ineq * x + b_ineq >= 0
    #
    # Lower bounds:
    #
    #     c(x) >= l
    #     c(x) - l >= 0
    #
    # Upper bounds:
    #
    #     c(x) <= u
    #     -c(x) + u >= 0
    #

    # Add upper bounds on complementarity variables, since their lower bounds
    # are represented through L/R and their regular bound rows were excluded.
    vcc_ub_inds = comp_var_inds[isfinite.(uvar[comp_var_inds])]

    J_vcc_ub = zeros(length(vcc_ub_inds), nvar)
    for (row, j) in enumerate(vcc_ub_inds)
        J_vcc_ub[row, j] = 1.0
    end

    J_ineq = [
        J_all[ineq_lb, :];
        -J_all[ineq_ub, :];
        -J_vcc_ub;
    ]

    b_ineq = [
        b_all[ineq_lb] - l_all[ineq_lb];
        u_all[ineq_ub] - b_all[ineq_ub];
        uvar[vcc_ub_inds];
    ]

    L = zeros(ncc, nvar)
    R = zeros(ncc, nvar)

    for k in 1:ncc
        L[k, ind_vcc1[k]] = 1.0
        R[k, ind_vcc2[k]] = 1.0
    end

    l = -lvar[ind_vcc1]
    r = -lvar[ind_vcc2]

    obj_sign = nlp.meta.minimize ? 1.0 : -1.0

    Q  = obj_sign * H0
    q  = obj_sign * NLPModels.grad(nlp, x0)
    c0 = obj_sign * NLPModels.obj(nlp, x0)

    return MarbleData(
        Q,
        q,
        c0;
        J_eq   = J_eq,
        b_eq   = b_eq,
        J_ineq = J_ineq,
        b_ineq = b_ineq,
        L = L,
        R = R,
        l = l,
        r = r,
    )
end