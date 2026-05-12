using NLPModels, ADNLPModels
using MPCCModels
using Marble

include("nlp_utils.jl")

function multi_robot_coordination(; N=30, T=5.0)
    Δt = T / (N - 1)
    m = 1.0
    d_min = 2.5

    inds, nvar = indexed_components((
        xA = (4, N),   # states [r, v] ∈ R^4
        uA = (2, N-1), # controls [a] ∈ R^2
        xB = (4, N),   # states [r, v] ∈ R^4
        uB = (2, N-1), # controls [a] ∈ R^2
        s  = (2, N),   # slacks for L1 norm
        λ  = (2, N)    # dual variables for L1 norm
    ))

    x0A = [-5, 0, 0, 0]
    x0B = [5, 0, 0, 0]
    xfA = [5, 10, 0, 0]
    xfB = [-5, 10, 0, 0]

    # Cost function
    stage_cost(u) = Δt .* 1/2 .* (u' * u)

    # Boundary conditions (= 0)
    initial_condition(x, x0) = x - x0
    terminal_condition(x, xf) = x - xf

    # Dynamics (implicit Euler)
    continuous_dynamics(x, u; m) = [x[3:4]; u / m]
    dynamics(x0, x1, u; m) = x1 - (x0 + Δt .* continuous_dynamics(x1, u; m=m))

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

    block_cc_to_interleaved(x, y) = begin
        @assert length(x) == length(y) "Vectors must have the same length for interleaving"
        [x y]'[:]
    end

    L1_norm_KKT(rA, rB, s, λ) = [
        block_cc_to_interleaved(λ, s - (rA - rB));  # λ ⟂ s - (rA - rB), length=4
        block_cc_to_interleaved(1 .- λ, s + (rA - rB))  # (1 - λ) ⟂ s + (rA - rB), length=4
    ]

    separation(s; d_min) = sum(s) - d_min  # ≥ 0

    # Build the problem constraints
    cb = CBuilder()
    
    add_eq!(cb, :ic_A, x -> initial_condition(x[inds.xA[1]],   x0A), 4)
    add_eq!(cb, :ic_B, x -> initial_condition(x[inds.xB[1]],   x0B), 4)
    add_eq!(cb, :tc_A, x -> terminal_condition(x[inds.xA[end]], xfA), 4)
    add_eq!(cb, :tc_B, x -> terminal_condition(x[inds.xB[end]], xfB), 4)

    for k=1:N-1
        add_eq!(cb, :dynamics_A, x -> dynamics(x[inds.xA[k]], x[inds.xA[k+1]], x[inds.uA[k]]; m=m), 4)
        add_eq!(cb, :dynamics_B, x -> dynamics(x[inds.xB[k]], x[inds.xB[k+1]], x[inds.uB[k]]; m=m), 4)
    end

    for k=1:N
        add_cc!(cb, :L1_norm, x -> begin
            rA = x[inds.xA[k][1:2]]  # position of robot A at time k
            rB = x[inds.xB[k][1:2]]  # position of robot B at time k

            L1_norm_KKT(rA, rB, x[inds.s[k]], x[inds.λ[k]])
        end, 8)

        add_ineq!(cb, :separation, x -> separation(x[inds.s[k]]; d_min=d_min), 1; lb=0.0)
    end

    c, lcon, ucon, cinds, ncon, ind_ccc1, ind_ccc2 = assemble(cb)

    # Build the problem cost function
    f(x) = sum(stage_cost(x[inds.uA[k]]) + stage_cost(x[inds.uB[k]]) for k=1:N-1; init=0.0)

    nlp = ADNLPModel(f, zeros(nvar), c, lcon, ucon; name="Multi-Robot Coordination")
    MPCCModels.MPCCModelConCon(nlp, ind_ccc1, ind_ccc2), inds
end

mpcc, inds = multi_robot_coordination()
data = from_mpcc(mpcc)

# Instantiate probem
problem = Marble.Problem(data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)

# Create an instance of Marble and solve the problem
solver = Marble.Solver()
Marble.set_problem!(solver, problem)

opts = Marble.SolverOptions()
Marble.verbosity!(opts, true)

result = Marble.solve!(solver, opts)

## Visualization
using Plots, Printf; gr()

N = 30; T = 5.0; d_min = 2.5
ts = range(0.0, T, N)

z = Marble.z(result)

rA = [z[inds.xA[k][1:2]] for k = 1:N]
vA = [z[inds.xA[k][3:4]] for k = 1:N]
rB = [z[inds.xB[k][1:2]] for k = 1:N]
vB = [z[inds.xB[k][3:4]] for k = 1:N]
l1_dist = [sum(z[inds.s[k]]) for k = 1:N]

# Fixed axis limits so the frame doesn't jump during animation
all_x = [r[1] for r in [rA; rB]]
all_y = [r[2] for r in [rA; rB]]
pad = 1.5
xlims_traj = (minimum(all_x) - pad, maximum(all_x) + pad)
ylims_traj = (minimum(all_y) - pad, maximum(all_y) + pad)

arrow_scale = 0.15

# ── Static right plot (L1 norm) ───────────────────────────────────────────────
p2 = plot(
    ts, l1_dist;
    label="‖rA − rB‖₁", color=:mediumpurple, lw=2.5, fill=(0, 0.08, :mediumpurple),
)
hline!(p2, [d_min];
    label="d_min = $(d_min)", color=:black, ls=:dash, lw=1.5,
)
plot!(p2;
    xlabel="Time (s)", ylabel="L1 distance (m)",
    title="Robot Separation",
    legend=:topright,
    framestyle=:box,
    grid=true, gridalpha=0.3,
    ylims=(0, max(maximum(l1_dist) * 1.15, d_min * 2.0)),
)

# ── Animation ─────────────────────────────────────────────────────────────────
anim = @animate for k = 1:N
    # Ghost full-path trails
    p1 = plot(
        [r[1] for r in rA], [r[2] for r in rA];
        color=:royalblue, lw=1, alpha=0.15, label="",
    )
    plot!(p1,
        [r[1] for r in rB], [r[2] for r in rB];
        color=:firebrick, lw=1, alpha=0.15, label="",
    )

    # Trace so far
    if k > 1
        plot!(p1,
            [r[1] for r in rA[1:k]], [r[2] for r in rA[1:k]];
            label="Robot A", color=:royalblue, lw=2.5,
        )
        plot!(p1,
            [r[1] for r in rB[1:k]], [r[2] for r in rB[1:k]];
            label="Robot B", color=:firebrick, lw=2.5,
        )
    end

    # Start markers (always visible)
    scatter!(p1, [rA[1][1]], [rA[1][2]]; marker=:circle, ms=6, color=:royalblue, label="")
    scatter!(p1, [rB[1][1]], [rB[1][2]]; marker=:circle, ms=6, color=:firebrick,  label="")

    # Goal markers (always visible, hollow)
    scatter!(p1, [rA[end][1]], [rA[end][2]]; marker=:rect, ms=6, color=:royalblue, label="", alpha=0.35)
    scatter!(p1, [rB[end][1]], [rB[end][2]]; marker=:rect, ms=6, color=:firebrick,  label="", alpha=0.35)

    # Current position dots
    scatter!(p1, [rA[k][1]], [rA[k][2]]; marker=:circle, ms=11, color=:royalblue, label="")
    scatter!(p1, [rB[k][1]], [rB[k][2]]; marker=:circle, ms=11, color=:firebrick,  label="")

    # Velocity arrows at current position
    quiver!(p1, [rA[k][1]], [rA[k][2]];
        quiver=([vA[k][1] * arrow_scale], [vA[k][2] * arrow_scale]),
        color=:royalblue, lw=1.5,
    )
    quiver!(p1, [rB[k][1]], [rB[k][2]];
        quiver=([vB[k][1] * arrow_scale], [vB[k][2] * arrow_scale]),
        color=:firebrick, lw=1.5,
    )

    plot!(p1;
        xlabel="x (m)", ylabel="y (m)",
        title=@sprintf("Robot Trajectories  (t = %.2f s)", ts[k]),
        aspect_ratio=:equal,
        legend=:topleft,
        framestyle=:box,
        grid=true, gridalpha=0.3,
        xlims=xlims_traj, ylims=ylims_traj,
    )

    plot(p1, p2; layout=(1, 2), size=(1000, 460), margin=5Plots.mm, dpi=120)
end

gif(anim; fps=15)
