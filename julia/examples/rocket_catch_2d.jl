using JuMP, Marble
import ForwardDiff as FD
using ControlSystems
using LinearAlgebra

using Pkg; Pkg.activate(@__DIR__)

# State  x = [px, py, vx, vy, θ, ω, α]   Control  u = [T, α̇]
function rocket_dynamics(x, u; m, g, l, J)
    R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    px, py, vx, vy, θ, ω, α = x
    T, α̇ = u
    a = (1 / m) * R(θ + α) * [0; T]
    [vx; vy; a[1]; a[2] - g; ω; -(l / J) * T * sin(α); α̇]
end

function linearized_rocket_dynamics(x̄, ū, Δt; m, g, l, J)
    nx, nu = 7, 2
    A = FD.jacobian(x -> rocket_dynamics(x, ū; m, g, l, J), x̄)
    B = FD.jacobian(u -> rocket_dynamics(x̄, u; m, g, l, J), ū)
    c = rocket_dynamics(x̄, ū; m, g, l, J)
    sysd = c2d(ss(A, [B c], I(nx), zeros(nx, nu + 1)), Δt, :zoh)
    return sysd.A, sysd.B[:, 1:nu], sysd.B[:, end]
end

function rocket_catch_2d(; N=40, T=5.0)
    m, g = 1.0, 9.81
    l    = 0.5                       # half-length of the rocket
    J    = (1 / 12) * m * (2l)^2
    Δt   = T / (N - 1)
    nx, nu = 7, 2

    x̄ = zeros(nx)
    ū = [m * g, 0.0]
    Ad, Bd, cd = linearized_rocket_dynamics(x̄, ū, Δt; m, g, l, J)

    tower_height, catch_height, catch_arm_length = 3.0, 1.5, 1.0
    θ_max, ω_max, α_max = 0.2, 0.6, 0.5
    T_max, α̇_max        = 12.0, 0.8

    xinit = [10.0, 20.0, -3.0, -8.0, 0.15, 0.2, 0.0]

    model = JuMP.Model()

    @variable(model, x[1:N, 1:nx])
    @variable(model, u[1:N-1, 1:nu])
    # State-triggered constraint: trigger g = 1.3·tower_height - py, split g = gp - gn
    @variable(model, gp[1:N] >= 0)
    @variable(model, gn[1:N] >= 0)
    # Keep-out conditions h ≥ 0, split h = hp - hn (violation lives in hn ≥ 0)
    @variable(model, hp[1:N, 1:3] >= 0)
    @variable(model, hn[1:N, 1:3] >= 0)

    w = 1.0
    @objective(model, Min,
        sum(0.5 * Δt * x[k, 7]^2 for k=1:N)
        + sum(0.5 * Δt * (ū[1] + u[k, 1])^2 for k=1:N-1)
        + w * Δt * sum(gp[k] + gn[k] + sum(hp[k, :]) + sum(hn[k, :]) for k=1:N)
    )

    # Boundary conditions: start in flight, end caught at the tower arm
    @constraint(model, ic, x[1, :] .== xinit)
    @constraint(model, tc_px_lb, x[N, 1] >= 0.0)
    @constraint(model, tc_px_ub, x[N, 1] <= catch_arm_length)
    @constraint(model, tc_py, x[N, 2] == catch_height)
    @constraint(model, tc_v, x[N, 3:4] .== 0.0)
    @constraint(model, tc_θ, x[N, 5] == 0.0)

    @constraint(model, dynamics[k=1:N-1], x[k+1, :] .== Ad * x[k, :] + Bd * u[k, :] + cd)

    # State and control bounds
    @constraint(model, θ_lb[k=1:N], x[k, 5] >= -θ_max)
    @constraint(model, θ_ub[k=1:N], x[k, 5] <= θ_max)
    @constraint(model, ω_lb[k=1:N], x[k, 6] >= -ω_max)
    @constraint(model, ω_ub[k=1:N], x[k, 6] <= ω_max)
    @constraint(model, α_lb[k=1:N], x[k, 7] >= -α_max)
    @constraint(model, α_ub[k=1:N], x[k, 7] <= α_max)
    @constraint(model, T_lb[k=1:N-1], ū[1] + u[k, 1] >= 0.0)
    @constraint(model, T_ub[k=1:N-1], ū[1] + u[k, 1] <= T_max)
    @constraint(model, α̇_lb[k=1:N-1], u[k, 2] >= -α̇_max)
    @constraint(model, α̇_ub[k=1:N-1], u[k, 2] <= α̇_max)

    # Trigger and keep-out expressions (R_wb ≈ I + θ·[0 -1; 1 0] for small θ)
    @expression(model, g_trig[k=1:N], 1.3 * tower_height - x[k, 2])
    @expression(model, h_stc[k=1:N, d=1:3],
        [x[k, 5] + x[k, 7],          # gimbal aligned:  θ + α ≥ 0
         x[k, 1] - l * x[k, 5],      # rocket top   x-coord ≥ 0
         x[k, 1] + l * x[k, 5]][d])  # rocket bottom x-coord ≥ 0

    @constraint(model, g_decomp[k=1:N], gp[k] - gn[k] == g_trig[k])
    @constraint(model, h_decomp[k=1:N, d=1:3], hp[k, d] - hn[k, d] == h_stc[k, d])
    # Triggered when below the tower: gp > 0 forces zero keep-out violation
    @constraint(model, h_violation[k=1:N], sum(hn[k, :]) >= 0)

    ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
        (model[:gp], model[:h_violation]),
    )

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, Δt, x̄, ū, m, g, l, J,
                        tower_height, catch_height, catch_arm_length,
                        θ_max, ω_max, α_max, T_max, α̇_max, xinit))
end

problem = rocket_catch_2d(N=40, T=5.0);

##

solver = Marble.Solver()
Marble.setup!(solver, problem; verbosity = 1)
results = Marble.solve!(solver)

z = Marble.z(results)
