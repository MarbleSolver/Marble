# Marble test suite
#
# Mirror of the Python suite (python/tests/test_marble.py): the same example
# problem written directly as matrices (from julia/examples/simple_test.jl) and
# the same 16 test cases in the same order — nine solve tests followed by seven
# dimension validation cases.

using Test
using LinearAlgebra
using SparseArrays
using Marble

# Example problem (julia/examples/simple_test.jl), written directly as matrices:
#   min x'x
#   s.t. x1 = 1,  x2 >= 1,  0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
#   optimum x* = [1, 1, 0, 1]
const Q = 2.0 * Matrix(1.0I, 4, 4)
const q = zeros(4)
const c0 = 0.0
const J_eq,   b_eq   = [1.0 0.0 0.0 0.0], [-1.0]
const J_ineq, b_ineq = [0.0 1.0 0.0 0.0], [-1.0]
const L, l          = [0.0 0.0 1.0 0.0], [1.0]
const R, r          = [0.0 0.0 0.0 1.0], [-1.0]
const z_opt = [1.0, 1.0, 0.0, 1.0]

# Build the solver, set up the example problem (dense or sparse), and solve.
# Option settings are forwarded to setup! as keyword arguments.
function setup_and_solve(; sparse_problem = false, opts...)
    conv = sparse_problem ? sparse : identity
    solver = Marble.Solver()
    Marble.setup!(solver, conv(Q), q, c0;
        J_eq = conv(J_eq), b_eq = b_eq, J_ineq = conv(J_ineq), b_ineq = b_ineq,
        L = conv(L), l = l, R = conv(R), r = r, opts...)
    return solver, Marble.solve!(solver)
end

function sparse_from_tuple(t)
    rows, cols, colptr, rowval, nzval = t
    return SparseMatrixCSC(rows, cols, collect(colptr), collect(rowval), collect(nzval))
end

function logical_kkt_matrix(solver)
    p = solver.problem
    w = Marble.get_workspace(solver)
    relax = Marble.get_relaxation_map(solver)

    H = Matrix(sparse_from_tuple(Marble.cost_hessian(p)))
    J_eq = Matrix(sparse_from_tuple(Marble.J_eq(p)))
    J_ineq = Matrix(sparse_from_tuple(Marble.J_ineq(p)))
    Lmat = Matrix(sparse_from_tuple(Marble.L(p)))
    Rmat = Matrix(sparse_from_tuple(Marble.R(p)))

    nz = Marble.nz(p)
    ne = Marble.n_eq(p)
    ni = Marble.n_ineq(p)
    nc = Marble.n_comp(p)
    nvars = nz + ni + nc + ne + ni + 2nc

    z = 1:nz
    v = (last(z) + 1):(last(z) + ni)
    sigma = (last(v) + 1):(last(v) + nc)
    m_eq = (last(sigma) + 1):(last(sigma) + ne)
    m_ineq = (last(m_eq) + 1):(last(m_eq) + ni)
    m_comp_L = (last(m_ineq) + 1):(last(m_ineq) + nc)
    m_comp_R = (last(m_comp_L) + 1):(last(m_comp_L) + nc)

    K = zeros(nvars, nvars)
    K[z, z] .= H
    K[z, m_eq] .= J_eq'
    K[m_eq, z] .= J_eq
    K[z, m_ineq] .= J_ineq'
    K[m_ineq, z] .= J_ineq
    K[z, m_comp_L] .= Lmat'
    K[m_comp_L, z] .= Lmat
    K[z, m_comp_R] .= Rmat'
    K[m_comp_R, z] .= Rmat

    kappa = Marble.relax_param(w)
    inv_penalty = 1.0 / Marble.penalty_param(w)

    if ni > 0
        s_ineq = collect(Marble.s_ineq(w))
        m_ineq_val = collect(Marble.m_ineq(w))
        b_neg = collect(Marble.b(relax, -s_ineq, kappa))
        b_prime = collect(Marble.b_prime(relax, s_ineq, kappa))
        b_neg_prime = collect(Marble.b_prime(relax, -s_ineq, kappa))
        b_double_prime = collect(Marble.b_double_prime(relax, s_ineq, kappa))
        diag_v = -b_double_prime .* (m_ineq_val .+ b_neg) .+
                 b_prime .* b_neg_prime
        K[v, v] .= Diagonal(diag_v)
        K[v, m_ineq] .= Diagonal(-b_prime)
        K[m_ineq, v] .= Diagonal(-b_prime)
    end

    if nc > 0
        s_comp = collect(Marble.s_comp(w))
        m_comp_L_val = collect(Marble.m_comp_L(w))
        m_comp_R_val = collect(Marble.m_comp_R(w))
        b_prime = collect(Marble.b_prime(relax, s_comp, kappa))
        b_neg_prime = collect(Marble.b_prime(relax, -s_comp, kappa))
        b_double_prime = collect(Marble.b_double_prime(relax, s_comp, kappa))
        diag_sigma = -b_double_prime .* (m_comp_L_val .+ m_comp_R_val)
        K[sigma, sigma] .= Diagonal(diag_sigma)
        K[sigma, m_comp_L] .= Diagonal(-b_prime)
        K[m_comp_L, sigma] .= Diagonal(-b_prime)
        K[sigma, m_comp_R] .= Diagonal(b_neg_prime)
        K[m_comp_R, sigma] .= Diagonal(b_neg_prime)
    end

    K[m_eq, m_eq] .= -inv_penalty .* I(ne)
    K[m_ineq, m_ineq] .= -inv_penalty .* I(ni)
    K[m_comp_L, m_comp_L] .= -inv_penalty .* I(nc)
    K[m_comp_R, m_comp_R] .= -inv_penalty .* I(nc)
    return K
end

@testset "Marble" begin

    @testset "problem dimensions" begin
        solver, _ = setup_and_solve()
        p = solver.problem
        @test Marble.nz(p) == 4 && Marble.n_eq(p) == 1 && Marble.n_ineq(p) == 1 && Marble.n_comp(p) == 1
    end

    @testset "solves to known optimum" begin
        _, res = setup_and_solve()
        z = collect(Marble.z(res))
        @test Marble.converged(res) && isapprox(z, z_opt, atol = 1e-4)
        @test Marble.factorizations(res) >= Marble.iters_inner(res)
        @test Marble.factorizations(res) ==
              Marble.iters_inner(res) +
              Marble.factorizations_ldlt(res) +
              Marble.factorizations_inertia(res) +
              Marble.factorizations_linesearch(res)
    end

    @testset "equality constraint satisfied" begin
        # x1 = 1
        _, res = setup_and_solve()
        @test abs(collect(Marble.z(res))[1] - 1.0) < 1e-4
    end

    @testset "inequality constraint satisfied" begin
        # x2 >= 1
        _, res = setup_and_solve()
        @test collect(Marble.z(res))[2] >= 1.0 - 1e-4
    end

    @testset "complementarity satisfied" begin
        # 0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
        _, res = setup_and_solve()
        z = collect(Marble.z(res))
        a = z[3] + 1.0
        b = z[4] - 1.0
        @test a >= -1e-4 && b >= -1e-4 && abs(a * b) < 1e-4
    end

    @testset "objective value" begin
        # x'x = 3 at the optimum
        solver, res = setup_and_solve()
        @test abs(Marble.obj(solver, collect(Marble.z(res))) - 3.0) < 1e-3
    end

    @testset "dense and sparse solve agree" begin
        _, rd = setup_and_solve(sparse_problem = false)
        _, rs = setup_and_solve(sparse_problem = true)
        @test Marble.converged(rd) && Marble.converged(rs) &&
              isapprox(collect(Marble.z(rd)), collect(Marble.z(rs)), atol = 1e-6)
    end

    @testset "complementarity-only QPCC" begin
        # same cost x'x, only the complementarity 0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
        # x1, x2 are unconstrained -> 0; x3 = 0, x4 = 1  =>  x* = [0, 0, 0, 1]
        solver = Marble.Solver()
        Marble.setup!(solver, Q, q, c0; L = L, l = l, R = R, r = r)
        res = Marble.solve!(solver)
        z = collect(Marble.z(res))
        @test Marble.converged(res) && isapprox(z, [0.0, 0.0, 0.0, 1.0], atol = 1e-4)
    end

    @testset "options are honored" begin
        s2, r2 = setup_and_solve(max_iters = 123, convergence_kkt_norm = 1e-8,
                                 use_relax_correction = false)
        @test Marble.max_iters(s2.options) == 123 &&
              isapprox(Marble.convergence_kkt_norm(s2.options), 1e-8) &&
              Marble.use_relax_correction(s2.options) == false &&
              Marble.converged(r2)
    end

    @testset "zero-sized constraint blocks" begin
        solver = Marble.Solver()
        Marble.setup!(solver, Matrix(1.0I, 2, 2), zeros(2))
        res = Marble.solve!(solver)
        @test Marble.converged(res) && isapprox(collect(Marble.z(res)), zeros(2), atol = 1e-8)
    end

    @testset "left/right complementarity accessors" begin
        solver, res = setup_and_solve()
        z = collect(Marble.z(res))
        @test isapprox(collect(Marble.residual_comp_L(solver.problem, z)), vec(L * z .+ l), atol = 1e-8)
        @test isapprox(collect(Marble.residual_comp_R(solver.problem, z)), vec(R * z .+ r), atol = 1e-8)
        @test length(collect(Marble.m_comp_L(res))) == 1
        @test length(collect(Marble.m_comp_R(res))) == 1
        @test collect(Marble.m_comp(res)) == vcat(collect(Marble.m_comp_L(res)), collect(Marble.m_comp_R(res)))
    end

    @testset "uninitialized solver errors" begin
        solver = Marble.Solver()
        @test_throws ErrorException Marble.solve!(solver)
        @test_throws ErrorException Marble.convergence(solver)
        @test_throws ErrorException Marble.get_problem(solver)
        @test_throws ErrorException Marble.get_workspace(solver)
        @test_throws ErrorException Marble.get_kkt_system(solver)
        @test_throws ErrorException Marble.update_primal_residuals!(solver)
        @test_throws ErrorException Marble.update_relaxed_slack_values!(solver)
        @test_throws ErrorException Marble.filter_linesearch!(solver)
        @test_throws ErrorException Marble.apply_newton_step!(solver, 1.0)
        @test_throws ErrorException Marble.entry_from_solution(solver)
    end

    @testset "KKT and LDLT accessors" begin
        solver, _ = setup_and_solve()
        kkt = Marble.get_kkt_system(solver)
        ldlt = Marble.ldlt(kkt)

        @test Marble.n_primals(kkt) == 6
        @test Marble.n_duals(kkt) == 4
        @test Marble.n_vars(kkt) == 10
        @test length(collect(Marble.residual(kkt))) == Marble.n_vars(kkt)

        rows, cols, colptr, rowval, nzval = Marble.matrix(kkt)
        @test rows == Marble.n_vars(kkt)
        @test cols == Marble.n_vars(kkt)
        @test length(collect(colptr)) == cols + 1
        @test length(collect(rowval)) == length(collect(nzval))

        @test Marble.built(ldlt)
        @test collect(Marble.scaling(kkt)) == collect(Marble.scaling(ldlt))
        @test sort(collect(Marble.perm(kkt))) == collect(0:Marble.n_vars(kkt)-1)
        @test sort(collect(Marble.iperm(kkt))) == collect(0:Marble.n_vars(kkt)-1)
    end

    @testset "LDLT internal matrix equals P' S K S P" begin
        solver, _ = setup_and_solve()
        Marble.update_relaxed_slack_values!(solver)
        kkt = Marble.get_kkt_system(solver)
        Marble.update_kkt_system!(kkt)
        ldlt = Marble.ldlt(kkt)

        K = logical_kkt_matrix(solver)
        n = Marble.n_vars(kkt)
        scaling = collect(Marble.scaling(ldlt))
        perm = collect(Marble.perm(ldlt)) .+ 1
        S = Diagonal(scaling)
        P = sparse(perm, 1:n, ones(n), n, n)

        internal = Matrix(sparse_from_tuple(Marble.matrix(ldlt)))
        expected = Matrix(triu(sparse(P' * S * K * S * P)))
        @test isapprox(internal, expected, atol = 1e-9, rtol = 1e-9)

        x_problem = collect(1.0:n)
        @test isapprox(collect(Marble.problem_to_internal(ldlt, x_problem)),
                       P' * S * x_problem, atol = 1e-12, rtol = 1e-12)

        x_internal = collect(1.0:n)
        @test isapprox(collect(Marble.internal_to_problem(ldlt, x_internal)),
                       S * P * x_internal, atol = 1e-12, rtol = 1e-12)
    end

    @testset "dimension validation" begin
        # Mismatched / missing blocks must error cleanly, never segfault. These
        # mirror the seven cases in the Python suite (TestDimensionValidation)
        Q2 = Matrix(1.0I, 2, 2)
        # L given (1 row) but l omitted
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Q2, zeros(2);
            L = zeros(1, 2), R = zeros(1, 2), r = zeros(1))
        # L and l given but R, r omitted
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Q2, zeros(2);
            L = zeros(1, 2), l = zeros(1))
        # Non-square Q
        @test_throws ErrorException Marble.setup!(Marble.Solver(), zeros(2, 3), zeros(2))
        # Q size disagrees with length(q)
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Matrix(1.0I, 3, 3), zeros(2))
        # J_eq rows disagree with b_eq length
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Q2, zeros(2);
            J_eq = zeros(2, 2), b_eq = zeros(1))
        # J_ineq columns disagree with length(q)
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Q2, zeros(2);
            J_ineq = zeros(1, 3), b_ineq = zeros(1))
        # Sparse blocks: L given but l omitted
        @test_throws ErrorException Marble.setup!(Marble.Solver(), sparse(Q2), zeros(2);
            L = sparse(zeros(1, 2)), R = sparse(zeros(1, 2)), r = zeros(1))
    end
end
