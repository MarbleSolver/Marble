using JuMP, Marble
using Combinatorics

using Pkg; Pkg.activate(@__DIR__)

function multi_robot_coordination(; N=30, T=5.0, d_min=5.0)
    Δt = T / (N - 1)
    m = 1.0
    n_robots = 3

    x0 = [
        Float64[0, -4, 0, 0],
        Float64[4, 1, 0, 0],
        Float64[-4, 1, 0, 0]
    ]
    xF = [
        Float64[0, 4, 0, 0],
        Float64[-4, 1, 0, 0],
        Float64[4, 1, 0, 0]
    ]

    n_slacks = binomial(n_robots, 2)

    model = JuMP.Model()

    # x[k, i, :] = [px, py, vx, vy] for robot i at time node k
    @variable(model, x[1:N, 1:n_robots, 1:4])
    # u[k, i, :] = [ax, ay] control acceleration for robot i
    @variable(model, u[1:N-1, 1:n_robots, 1:2])
    # s stores |relative position| componentwise for each robot pair
    @variable(model, s[1:N, 1:n_slacks, 1:2])
    # λ_plus, λ_minus are KKT multipliers for s - d >= 0 and s + d >= 0
    @variable(model, λ_plus[1:N, 1:n_slacks, 1:2] >= 0)
    @variable(model, λ_minus[1:N, 1:n_slacks, 1:2] >= 0)

    @objective(model, Min, sum(Δt * 0.5 * (u[k, i, :]' * u[k, i, :]) for i in 1:n_robots, k in 1:N-1))

    @constraint(model, ic[i=1:n_robots], x[1, i, :] .== x0[i])
    @constraint(model, fc[i=1:n_robots], x[N, i, :] .== xF[i])

    continuous_dynamics(x, u; m) = [x[3:4]; u ./ m]
    @constraint(model, dynamics[k=1:N-1, i=1:n_robots],
        x[k+1, i, :] .== x[k, i, :] + Δt * continuous_dynamics(x[k+1, i, :], u[k, i, :]; m=m))

    pairs = collect(combinations(1:n_robots, 2))
    # d[k, i, :] is the relative position for robot pair pairs[i]
    @expression(model, d[k=1:N, i=1:n_slacks, j=1:2], begin
        p1, p2 = pairs[i]
        x[k, p1, j] - x[k, p2, j]
    end)

    # Enforce s = |d| through KKT conditions, then require L1 distance >= d_min
    @constraint(model, kkt_abs_stationarity[k=1:N, i=1:n_slacks, j=1:2],
        1 - λ_plus[k, i, j] - λ_minus[k, i, j] == 0
    )
    @constraint(model, kkt_abs_s_minus_d[k=1:N, i=1:n_slacks, j=1:2], s[k, i, j] - d[k, i, j] >= 0)
    @constraint(model, kkt_abs_s_plus_d[k=1:N, i=1:n_slacks, j=1:2], s[k, i, j] + d[k, i, j] >= 0)
    @constraint(model, [k=1:N, i=1:n_slacks], sum(s[k, i, j] for j=1:2) >= d_min)

    ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
        (model[:λ_plus], model[:kkt_abs_s_minus_d]),
        (model[:λ_minus], model[:kkt_abs_s_plus_d]),
    )

    return (; model, ind_cc1, ind_cc2, cc_types,
              inds=var_inds(model),
              params=(; N, T, d_min, n_robots, pairs=collect(combinations(1:n_robots, 2))))
end

problem = multi_robot_coordination(N=30, T=5.0, d_min=5.0);

##

solver = Marble.Solver()
Marble.setup!(solver, problem; verbosity = 1)
results = Marble.solve!(solver)

z = Marble.z(results)