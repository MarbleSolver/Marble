"""Marble Python test suite.

Mirror of the Julia suite (julia/test/runtests.jl): the same example problem
written directly as matrices (from julia/examples/simple_test.jl) and the same
16 test cases in the same order — nine solve tests followed by seven dimension
validation cases. Each test sets up and runs the solver fresh.
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


def sparse_from_tuple(t):
    rows, cols, colptr, rowval, nzval = t
    return sp.csc_matrix((np.asarray(nzval), np.asarray(rowval), np.asarray(colptr)),
                         shape=(rows, cols))


def logical_kkt_matrix(solver):
    p = solver.problem
    w = solver.get_workspace()
    relax = solver.get_relaxation_map()

    H = sparse_from_tuple(p.cost_hessian).toarray()
    J_eq = sparse_from_tuple(p.J_eq).toarray()
    J_ineq = sparse_from_tuple(p.J_ineq).toarray()
    Lmat = sparse_from_tuple(p.L).toarray()
    Rmat = sparse_from_tuple(p.R).toarray()

    nz, ne, ni, nc = p.nz, p.n_eq, p.n_ineq, p.n_comp
    nvars = nz + ni + nc + ne + ni + 2 * nc
    z = slice(0, nz)
    v = slice(nz, nz + ni)
    sigma = slice(nz + ni, nz + ni + nc)
    m_eq = slice(nz + ni + nc, nz + ni + nc + ne)
    m_ineq = slice(nz + ni + nc + ne, nz + 2 * ni + nc + ne)
    m_comp_L = slice(nz + 2 * ni + nc + ne, nz + 2 * ni + 2 * nc + ne)
    m_comp_R = slice(nz + 2 * ni + 2 * nc + ne, nvars)

    K = np.zeros((nvars, nvars))
    K[z, z] = H
    K[z, m_eq] = J_eq.T
    K[m_eq, z] = J_eq
    K[z, m_ineq] = J_ineq.T
    K[m_ineq, z] = J_ineq
    K[z, m_comp_L] = Lmat.T
    K[m_comp_L, z] = Lmat
    K[z, m_comp_R] = Rmat.T
    K[m_comp_R, z] = Rmat

    kappa = w.relax_param
    inv_penalty = 1.0 / w.penalty_param

    if ni > 0:
        s_ineq = np.asarray(w.s_ineq)
        m_ineq_val = np.asarray(w.m_ineq)
        b_neg = np.asarray(relax.b(-s_ineq, kappa))
        b_prime = np.asarray(relax.b_prime(s_ineq, kappa))
        b_neg_prime = np.asarray(relax.b_prime(-s_ineq, kappa))
        b_double_prime = np.asarray(relax.b_double_prime(s_ineq, kappa))
        diag_v = -b_double_prime * (m_ineq_val + b_neg) + b_prime * b_neg_prime
        K[v, v] = np.diag(diag_v)
        K[v, m_ineq] = np.diag(-b_prime)
        K[m_ineq, v] = np.diag(-b_prime)

    if nc > 0:
        s_comp = np.asarray(w.s_comp)
        m_comp_L_val = np.asarray(w.m_comp_L)
        m_comp_R_val = np.asarray(w.m_comp_R)
        b_prime = np.asarray(relax.b_prime(s_comp, kappa))
        b_neg_prime = np.asarray(relax.b_prime(-s_comp, kappa))
        b_double_prime = np.asarray(relax.b_double_prime(s_comp, kappa))
        diag_sigma = -b_double_prime * (m_comp_L_val + m_comp_R_val)
        K[sigma, sigma] = np.diag(diag_sigma)
        K[sigma, m_comp_L] = np.diag(-b_prime)
        K[m_comp_L, sigma] = np.diag(-b_prime)
        K[sigma, m_comp_R] = np.diag(b_neg_prime)
        K[m_comp_R, sigma] = np.diag(b_neg_prime)

    K[m_eq, m_eq] = -inv_penalty * np.eye(ne)
    K[m_ineq, m_ineq] = -inv_penalty * np.eye(ni)
    K[m_comp_L, m_comp_L] = -inv_penalty * np.eye(nc)
    K[m_comp_R, m_comp_R] = -inv_penalty * np.eye(nc)
    return K


class TestMarble(unittest.TestCase):

    def test_problem_dimensions(self):
        m, _ = setup_and_solve()
        p = m.problem
        self.assertTrue(p.nz == 4 and p.n_eq == 1 and p.n_ineq == 1 and p.n_comp == 1)

    def test_solves_to_known_optimum(self):
        _, res = setup_and_solve()
        z = np.asarray(res.z)
        self.assertTrue(res.converged and np.allclose(z, ZSTAR, atol=1e-4))
        self.assertGreaterEqual(res.factorizations, res.iters_inner)
        self.assertEqual(res.factorizations,
                         res.iters_inner + res.factorizations_ldlt
                         + res.factorizations_inertia + res.factorizations_linesearch)

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
        m, res = setup_and_solve(max_iters=123, convergence_kkt_norm=1e-8,
                                 use_relax_correction=False)
        self.assertTrue(m.options.max_iters == 123
                        and m.options.convergence_kkt_norm == 1e-8
                        and m.options.use_relax_correction is False
                        and res.converged)

    def test_zero_sized_constraint_blocks(self):
        m = marble.Solver()
        m.setup(np.eye(2), np.zeros(2))
        res = m.solve()
        self.assertTrue(res.converged and np.allclose(np.asarray(res.z), np.zeros(2), atol=1e-8))

    def test_left_right_complementarity_accessors(self):
        m, res = setup_and_solve()
        z = np.asarray(res.z)
        np.testing.assert_allclose(m.problem.residual_comp_L(z), np.asarray(L) @ z + np.asarray(EL), atol=1e-8)
        np.testing.assert_allclose(m.problem.residual_comp_R(z), np.asarray(R) @ z + np.asarray(ER), atol=1e-8)
        self.assertEqual(np.asarray(res.m_comp_L).shape, (1,))
        self.assertEqual(np.asarray(res.m_comp_R).shape, (1,))
        np.testing.assert_allclose(np.asarray(res.m_comp), np.r_[res.m_comp_L, res.m_comp_R])

    def test_kkt_and_ldlt_accessors(self):
        m, _ = setup_and_solve()
        kkt = m.get_kkt_system()
        ldlt = kkt.ldlt

        self.assertEqual(kkt.n_primals, 6)
        self.assertEqual(kkt.n_duals, 4)
        self.assertEqual(kkt.n_vars, 10)
        self.assertEqual(len(np.asarray(kkt.residual)), kkt.n_vars)

        rows, cols, colptr, rowval, nzval = kkt.matrix
        self.assertEqual(rows, kkt.n_vars)
        self.assertEqual(cols, kkt.n_vars)
        self.assertEqual(len(np.asarray(colptr)), cols + 1)
        self.assertEqual(len(np.asarray(rowval)), len(np.asarray(nzval)))

        self.assertTrue(ldlt.built)
        np.testing.assert_allclose(kkt.scaling, ldlt.scaling)
        np.testing.assert_array_equal(np.sort(ldlt.perm), np.arange(kkt.n_vars))
        np.testing.assert_array_equal(np.sort(ldlt.iperm), np.arange(kkt.n_vars))

    def test_ldlt_stored_matrix_matches_scaled_permuted_kkt(self):
        m, _ = setup_and_solve()
        m.update_relaxed_slack_values()
        kkt = m.get_kkt_system()
        kkt.update_kkt_system()
        ldlt = kkt.ldlt

        K = logical_kkt_matrix(m)
        n = kkt.n_vars
        scaling = np.asarray(ldlt.scaling)
        perm = np.asarray(ldlt.perm)
        S = np.diag(scaling)
        P = np.zeros((n, n))
        P[perm, np.arange(n)] = 1.0

        stored = sparse_from_tuple(ldlt.matrix).toarray()
        expected = np.triu(P.T @ S @ K @ S @ P)
        np.testing.assert_allclose(stored, expected, atol=1e-9, rtol=1e-9)

        x_logical = np.arange(1.0, n + 1.0)
        np.testing.assert_allclose(ldlt.logical_to_stored(x_logical),
                                   P.T @ S @ x_logical, atol=1e-12, rtol=1e-12)

        x_stored = np.arange(1.0, n + 1.0)
        np.testing.assert_allclose(ldlt.stored_to_logical(x_stored),
                                   S @ P @ x_stored, atol=1e-12, rtol=1e-12)


class TestDimensionValidation(unittest.TestCase):
    """Mismatched / missing blocks must raise ValueError, never segfault.

    Mirror of the Julia suite's "dimension validation" testset: the same seven
    cases in the same order.
    """

    def test_missing_l_raises(self):
        # L given (1 row) but l omitted
        m = marble.Solver()
        with self.assertRaises(ValueError):
            m.setup(Q=np.eye(2), q=np.zeros(2),
                    L=np.zeros((1, 2)), R=np.zeros((1, 2)), r=np.zeros(1))

    def test_missing_r_raises(self):
        # L and l given but R, r omitted
        m = marble.Solver()
        with self.assertRaises(ValueError):
            m.setup(Q=np.eye(2), q=np.zeros(2), L=np.zeros((1, 2)), l=np.zeros(1))

    def test_non_square_Q_raises(self):
        # Non-square Q
        with self.assertRaises(ValueError):
            marble.Solver().setup(Q=np.zeros((2, 3)), q=np.zeros(2))

    def test_Q_q_size_mismatch_raises(self):
        # Q size disagrees with len(q)
        with self.assertRaises(ValueError):
            marble.Solver().setup(Q=np.eye(3), q=np.zeros(2))

    def test_eq_row_mismatch_raises(self):
        # J_eq rows disagree with b_eq length
        with self.assertRaises(ValueError):
            marble.Solver().setup(Q=np.eye(2), q=np.zeros(2),
                                  J_eq=np.zeros((2, 2)), b_eq=np.zeros(1))

    def test_ineq_col_mismatch_raises(self):
        # J_ineq columns disagree with len(q)
        with self.assertRaises(ValueError):
            marble.Solver().setup(Q=np.eye(2), q=np.zeros(2),
                                  J_ineq=np.zeros((1, 3)), b_ineq=np.zeros(1))

    def test_sparse_mismatch_raises(self):
        # Sparse blocks: L given but l omitted
        with self.assertRaises(ValueError):
            marble.Solver().setup(Q=sp.csc_matrix(np.eye(2)), q=np.zeros(2),
                                  L=sp.csc_matrix(np.zeros((1, 2))),
                                  R=sp.csc_matrix(np.zeros((1, 2))), r=np.zeros(1))


if __name__ == "__main__":
    unittest.main()
