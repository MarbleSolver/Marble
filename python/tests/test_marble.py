"""Marble Python test suite.

Mirror of the Julia suite (julia/test/runtests.jl): the same example problem
written directly as matrices (from julia/examples/simple_test.jl) and the same
eight test cases in the same order. Each test sets up and runs the solver fresh.
Run with ``python -m unittest`` or ``python tests/test_marble.py``.
"""
import unittest

import numpy as np
import scipy.sparse as sp

import marble


# Example problem (julia/examples/simple_test.jl), written directly as matrices:
#   min x'x
#   s.t. x1 = 1,  x2 >= 1,  0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
#   optimum x* = [1, 1, 0, 1]
Q = 2.0 * np.eye(4)
q = np.zeros(4)
C0 = 0.0
J_EQ,   B_EQ   = [[1.0, 0.0, 0.0, 0.0]], [-1.0]
J_INEQ, B_INEQ = [[0.0, 1.0, 0.0, 0.0]], [-1.0]
L, EL          = [[0.0, 0.0, 1.0, 0.0]], [1.0]
R, ER          = [[0.0, 0.0, 0.0, 1.0]], [-1.0]
ZSTAR = np.array([1.0, 1.0, 0.0, 1.0])


# Build the solver, set up the example problem (dense or sparse), and solve.
# Option settings are forwarded to setup as keyword arguments.
def setup_and_solve(sparse_problem=False, **options):
    conv = (lambda M: sp.csc_matrix(M)) if sparse_problem else (lambda M: M)
    m = marble.Solver()
    m.setup(conv(Q), q, C0,
            J_eq=conv(J_EQ), b_eq=B_EQ, J_ineq=conv(J_INEQ), b_ineq=B_INEQ,
            L=conv(L), l=EL, R=conv(R), r=ER, **options)
    return m, m.solve()


class TestMarble(unittest.TestCase):

    def test_problem_dimensions(self):
        m, _ = setup_and_solve()
        p = m.problem
        self.assertTrue(p.nz == 4 and p.n_eq == 1 and p.n_ineq == 1 and p.n_comp == 1)

    def test_solves_to_known_optimum(self):
        _, res = setup_and_solve()
        z = np.asarray(res.z)
        self.assertTrue(res.converged and np.allclose(z, ZSTAR, atol=1e-4))

    def test_equality_constraint_satisfied(self):
        # x1 = 1
        _, res = setup_and_solve()
        self.assertTrue(abs(np.asarray(res.z)[0] - 1.0) < 1e-4)

    def test_inequality_constraint_satisfied(self):
        # x2 >= 1
        _, res = setup_and_solve()
        self.assertTrue(np.asarray(res.z)[1] >= 1.0 - 1e-4)

    def test_complementarity_satisfied(self):
        # 0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
        _, res = setup_and_solve()
        z = np.asarray(res.z)
        a = z[2] + 1.0
        b = z[3] - 1.0
        self.assertTrue(a >= -1e-4 and b >= -1e-4 and abs(a * b) < 1e-4)

    def test_objective_value(self):
        # x'x = 3 at the optimum
        m, res = setup_and_solve()
        self.assertTrue(abs(m.problem.obj(np.asarray(res.z)) - 3.0) < 1e-3)

    def test_dense_and_sparse_solve_agree(self):
        _, rd = setup_and_solve(sparse_problem=False)
        _, rs = setup_and_solve(sparse_problem=True)
        self.assertTrue(rd.converged and rs.converged
                        and np.allclose(np.asarray(rd.z), np.asarray(rs.z), atol=1e-6))

    def test_complementarity_only_qpcc(self):
        # same cost x'x, only the complementarity 0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
        # x1, x2 are unconstrained -> 0; x3 = 0, x4 = 1  =>  x* = [0, 0, 0, 1]
        m = marble.Solver()
        m.setup(Q, q, C0, L=L, l=EL, R=R, r=ER)
        res = m.solve()
        z = np.asarray(res.z)
        self.assertTrue(res.converged and np.allclose(z, [0.0, 0.0, 0.0, 1.0], atol=1e-4))

    def test_options_are_honored(self):
        m, res = setup_and_solve(max_iters=123, convergence_kkt_norm=1e-8)
        self.assertTrue(m.options.max_iters == 123
                        and m.options.convergence_kkt_norm == 1e-8
                        and res.converged)


if __name__ == "__main__":
    unittest.main()
