using NLPModels
using Marble

function multi_robot_coordination(; N=50, T=5.0)::AbstractNLPModel
    Δt = T / (N - 1)
    m = 1.0

    inds, nvar = indexed_components((
        xA = (4, N),   # states [r, v] ∈ R^4
        uA = (2, N-1), # controls [a] ∈ R^2
        xB = (4, N),   # states [r, v] ∈ R^4
        uB = (2, N-1), # controls [a] ∈ R^2
        s  = (2, N),   # slacks for L1 norm
        λ  = (2, N)    # dual variables for L1 norm
    ))

    x0 = [zeros(2); zeros(2)]
    xf = [[10, 10]; zeros(2)]

    # Cost function
    stage_cost(k, x, u) = Δt * 1/2 * u' * u
    terminal_cost(x) = 0.0

    # Boundary conditions (= 0)
    initial_condition(x, x0) = x - x0
    terminal_condition(x, xf) = x - xf

    # Dynamics (implicit Euler)
    continuous_dynamics(x, u; m) = [x[2:3]; u / m]
    dynamics(x0, x1, u; m) = x1 - (x0 + Δt * continuous_dynamics(x1, u; m=m))

    # L1 norm constraint on the distance between the two robots

    # L1_norm(x) = ∑ |x_1| + ... + |x_n|
    # The L1 norm of vector x can be represented as a solution of the following LP:
    #  minimize   1^T s
    #  subject to s >= x
    #             s >= -x
    # With the following KKT conditions (λ⁺, λ⁻ are dual vars for s >= x, s >= -x):
    #   Stationarity:          λ⁺ + λ⁻ = 1
    #   Complementary slack:   λ⁺_i * (s_i - x_i) = 0,  λ⁻_i * (s_i + x_i) = 0
    #   Primal feasibility:    s_i >= x_i,  s_i >= -x_i
    #   Dual feasibility:      λ⁺_i >= 0,  λ⁻_i >= 0
    # At optimality s_i = |x_i|, so L1_norm(x) = 1^T s.
    # 
    # So, for each vector x that we want to represent the L1 norm as, we write the
    # following constraints:
    #   0 ≤ λ ⟂ s - x ≥ 0
    #   0 ≤ (1 - λ) ⟂ s + x ≥ 0

    L1_norm(rA, rB, s, λ) = [
        λ,                  # ≥ 0
        s - (rA - rB),      # ≥ 0
        1 .- λ,             # ≥ 0
        s + (rA - rB),      # ≥ 0
    ]

    separation(s; d_min) = sum(s) - d_min  # ≥ 0

end