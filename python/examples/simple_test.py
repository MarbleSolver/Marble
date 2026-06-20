# Construct a simple test problem minimizing x'x
# where x[1] = 1
#       x[2] >= 1
#       0 <= (x[3] + 1) ⟂ (x[4] - 1) >= 0  -->  solution is x[3] = 0, x[4] = 1
import numpy as np
import marble

Q = 2.0 * np.eye(4)
q = np.zeros(4)

J_eq,   c_eq   = [[1.0, 0.0, 0.0, 0.0]], [-1.0]   # x[1] == 1
J_ineq, c_ineq = [[0.0, 1.0, 0.0, 0.0]], [-1.0]   # x[2] >= 1
L, l           = [[0.0, 0.0, 1.0, 0.0]], [1.0]    # x[3] + 1
R, r           = [[0.0, 0.0, 0.0, 1.0]], [-1.0]   # x[4] - 1

solver = marble.Solver()
solver.setup(Q, q, 0.0,
             J_eq=J_eq, c_eq=c_eq, J_ineq=J_ineq, c_ineq=c_ineq,
             L=L, l=l, R=R, r=r, verbosity=1)
results = solver.solve()

z = np.asarray(results.z)
print(z)
print(solver.problem.obj(z))
print(solver.problem.residual_eq(z))
print(solver.problem.residual_ineq(z))
print(solver.problem.residual_comp(z))
