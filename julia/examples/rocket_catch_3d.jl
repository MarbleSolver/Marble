import ForwardDiff as FD

rocket_Rx(θ) = [1.0 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
rocket_Ry(θ) = [cos(θ) 0 sin(θ); 0 1.0 0; -sin(θ) 0 cos(θ)]
rocket_thrust_axis_body(δr, δp) = rocket_Rx(δr) * rocket_Ry(δp) * [0.0; 0.0; 1.0]

function rocket_dynamics(x, u; m, g, l, J)
    q, v, ω = x[4:7], x[8:10], x[11:13]
    thrust_axis = rocket_thrust_axis_body(u[2], u[3])
    force_body  = u[1] * thrust_axis
    torque_body = cross([0.0; 0.0; -l], force_body)

    [
        v;
        0.5 * L(q) * _H * ω;
        [0.0; 0.0; -g] + (1 / m) * qtoQ(q) * force_body;
        J \ (torque_body - cross(ω, J * ω))
    ]
end

function linearized_rocket_dynamics(x̄, ū, Δt; m, g, l, J)
    nx, nu = 13, 3
    E = quat_state_lift(x̄[4:7], 6)
    Af = FD.jacobian(x -> rocket_dynamics(x, ū; m=m, g=g, l=l, J=J), x̄)
    Bf = FD.jacobian(u -> rocket_dynamics(x̄, u; m=m, g=g, l=l, J=J), ū)
    cf = rocket_dynamics(x̄, ū; m=m, g=g, l=l, J=J)
    sysc = ss(Af, [Bf cf], I(nx), zeros(nx, nu + 1))
    sysd = c2d(sysc, Δt, :zoh)
    return E' * sysd.A * E, E' * sysd.B[:, 1:nu], E' * sysd.B[:, end]
end

function first_order_plume_axis_world(ϕ, δr, δp)
    [
        -δp - 2 * ϕ[2];
         δr + 2 * ϕ[1];
        -1.0;
    ]
end

function first_order_plume_clearance(ϕ, δr, δp; tower_normal)
    return tower_normal' * first_order_plume_axis_world(ϕ, δr, δp)
end

function rocket_catch_3d(; N=40, T=4.0)
    Δt = T / (N - 1)
    nx̃, nu = 12, 3

    m, g = 1.0, 9.81
    l = 0.7
    J = Diagonal([0.25, 0.25, 0.06])

    q0 = [1.0; 0.0; 0.0; 0.0]
    x̄ = [zeros(3); q0; zeros(6)]
    ū = [m * g; 0.0; 0.0]
    Ã, B̃, c̃ = linearized_rocket_dynamics(x̄, ū, Δt; m=m, g=g, l=l, J=J)

    catch_altitude = 2.0
    trigger_altitude = 6.0
    tower_normal = [0.0, 1.0, 0.0]
    tower_y = 0.0
    arm_half_width = 0.45
    arm_opening_angle = deg2rad(14.0)
    arm_slope = tan(arm_opening_angle)

    catch_A = [
         0.0 1.0 0.0;
         1.0 arm_slope 0.0;
        -1.0 arm_slope 0.0;
    ]
    catch_b = [tower_y; -arm_half_width; -arm_half_width]
    n_catch_halfspaces = size(catch_A, 1)
    plume_away_from_tower_index = n_catch_halfspaces + 1
    n_state_triggered_constraints = plume_away_from_tower_index
    nh = n_state_triggered_constraints

    thrust_min, thrust_max = 0.25 * m * g, 1.8 * m * g
    gimbal_max = deg2rad(15.0)
    ϕmax = deg2rad.([25.0, 25.0, 15.0])
    ωmax = [1.2, 1.2, 0.6]

    x0_full = [
        [5.0; 5.0; 22.0];
        rptoq([0.04; -0.06; 0.0]);
        [0.0; 1.0; -1.2];
        zeros(3);
    ]
    xF_full = [
        [0.0; 1.0; catch_altitude];
        q0;
        zeros(6);
    ]
    xstart = state_to_delta(x0_full, x̄)
    xfinal = state_to_delta(xF_full, x̄)

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx̃])
    @variable(model, u[1:N, 1:nu])
    @variable(model, gp[1:N] >= 0)
    @variable(model, gn[1:N] >= 0)
    @variable(model, hp[1:N, 1:nh] >= 0)
    @variable(model, hn[1:N, 1:nh] >= 0)

    height_is_below_trigger = gp
    height_is_above_trigger = gn
    state_triggered_margin = hp
    state_triggered_violation = hn

    stc_penalty = 1.0
    @objective(model, Min,
        sum(Δt / 2 * ((ū[1] + u[k, 1])^2 + 15.0 * (u[k, 2]^2 + u[k, 3]^2)) for k=1:N)
        + stc_penalty * Δt * sum(gp[k] + gn[k] + sum(hp[k, j] + hn[k, j] for j=1:nh) for k=1:N)
    )

    @constraint(model, ic, x[1, :] .== xstart)
    @constraint(model, tc, x[N, :] .== xfinal)
    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ã * x[k, :] + B̃ * u[k, :] + c̃)

    @constraint(model, thrust_lb[k=1:N], ū[1] + u[k, 1] >= thrust_min)
    @constraint(model, thrust_ub[k=1:N], ū[1] + u[k, 1] <= thrust_max)
    @constraint(model, gimbal_lb[k=1:N, j=2:3], u[k, j] >= -gimbal_max)
    @constraint(model, gimbal_ub[k=1:N, j=2:3], u[k, j] <=  gimbal_max)
    @constraint(model, rp_lb[k=1:N, j=1:3], x[k, 3+j] >= -ϕmax[j])
    @constraint(model, rp_ub[k=1:N, j=1:3], x[k, 3+j] <=  ϕmax[j])
    @constraint(model, omega_lb[k=1:N, j=1:3], x[k, 9+j] >= -ωmax[j])
    @constraint(model, omega_ub[k=1:N, j=1:3], x[k, 9+j] <=  ωmax[j])

    # State-triggered constraints
    @expression(model, rocket_CoM[k=1:N, d=1:3], x̄[d] + x[k, d])
    @constraint(model, trigger_altitude_split[k=1:N],
        trigger_altitude - rocket_CoM[k, 3] == height_is_below_trigger[k] - height_is_above_trigger[k]
    )

    # When active, the rocket CoM must lie inside the catch region:
    #   catch_A[j, :] ⋅ rocket_CoM >= catch_b[j]    for every catch half-space j
    @expression(model, catch_halfspace_margin[k=1:N, j=1:n_catch_halfspaces],
        sum(catch_A[j, d] * rocket_CoM[k, d] for d=1:3) - catch_b[j]
    )
    @constraint(model, catch_halfspace[k=1:N, j=1:n_catch_halfspaces],
        catch_halfspace_margin[k, j] == state_triggered_margin[k, j] - state_triggered_violation[k, j]
    )

    # When active, the engine plume must point away from the tower plane. The
    # plume is the negative thrust direction, and this first-order expression uses
    # the small-angle attitude state ϕ = x[k, 4:6]
    @expression(model, plume_axis_world[k=1:N], first_order_plume_axis_world(x[k, 4:6], ū[2] + u[k, 2], ū[3] + u[k, 3]))
    @expression(model, plume_away_from_tower_margin[k=1:N], tower_normal' * plume_axis_world[k])
    @constraint(model, engine_plume_away_from_tower[k=1:N],
        plume_away_from_tower_margin[k] ==
        state_triggered_margin[k, plume_away_from_tower_index] -
        state_triggered_violation[k, plume_away_from_tower_index]
    )

    @constraint(model, total_state_triggered_violation[k=1:N],
        sum(state_triggered_violation[k, j] for j=1:n_state_triggered_constraints) >= 0
    )

    ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
        (model[:gp], model[:total_state_triggered_violation]),
    )

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, Δt, x̄, ū, m, g, l, J, catch_altitude, trigger_altitude,
                       tower_normal, tower_y, arm_half_width, arm_opening_angle,
                       catch_A, catch_b, n_catch_halfspaces, plume_away_from_tower_index,
                       thrust_min, thrust_max, gimbal_max, ϕmax, ωmax))
end

problem = rocket_catch_3d(N=40, T=4.0);

##

solver = Marble.Solver()
Marble.setup!(solver, problem; verbosity = 1)
results = Marble.solve!(solver)

z = Marble.z(results)