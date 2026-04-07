import rcqp
import numpy as np

s = rcqp.Solver()

# Simple 2-variable unconstrained QP:
#   min  0.5 * x' * I * x + [1, 2]' * x
# Optimal solution: x* = [-1, -2]

nz = 2
H = np.eye(nz)
g = np.array([1.0, 2.0])
J_empty = np.zeros((0, nz))
c_empty = np.zeros(0)

prob = rcqp.Problem(
    H, g, 0.0,
    J_empty, c_empty,   # equality
    J_empty, c_empty,   # inequality
    J_empty, c_empty,   # complementarity
)

print(f"Problem: nz={prob.nz}, n_eq={prob.n_eq}, n_ineq={prob.n_ineq}, n_comp={prob.n_comp}")

opts = rcqp.SolverOptions()
opts.max_iters = 500
opts.convergence_kkt_norm = 1e-6
print(f"Options: {opts}")

solver = rcqp.Solver(opts)
solver.set_problem(prob, np.ones(nz))

converged = solver.solve(opts)
print(f"Converged: {converged}")

ws = solver.get_workspace()
print(f"Solution z:  {ws.z}")
print(f"Expected:    [-1. -2.]")
print(f"KKT residual norm: {np.linalg.norm(ws.kkt_residual):.2e}")
