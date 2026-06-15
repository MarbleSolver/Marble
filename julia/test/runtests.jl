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
const C0 = 0.0
const J_EQ,   c_eq   = [1.0 0.0 0.0 0.0], [-1.0]
const J_INEQ, c_ineq = [0.0 1.0 0.0 0.0], [-1.0]
const L, EL          = [0.0 0.0 1.0 0.0], [1.0]
const R, ER          = [0.0 0.0 0.0 1.0], [-1.0]
const ZSTAR = [1.0, 1.0, 0.0, 1.0]

# Build the solver, set up the example problem (dense or sparse), and solve.
# Option settings are forwarded to setup! as keyword arguments.
function setup_and_solve(; sparse_problem = false, opts...)
    conv = sparse_problem ? sparse : identity
    solver = Marble.Solver()
    Marble.setup!(solver, conv(Q), q, C0;
        J_eq = conv(J_EQ), c_eq = c_eq, J_ineq = conv(J_INEQ), c_ineq = c_ineq,
        L = conv(L), l = EL, R = conv(R), r = ER, opts...)
    return solver, Marble.solve!(solver)
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
        @test Marble.converged(res) && isapprox(z, ZSTAR, atol = 1e-4)
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
        Marble.setup!(solver, Q, q, C0; L = L, l = EL, R = R, r = ER)
        res = Marble.solve!(solver)
        z = collect(Marble.z(res))
        @test Marble.converged(res) && isapprox(z, [0.0, 0.0, 0.0, 1.0], atol = 1e-4)
    end

    @testset "options are honored" begin
        s2, r2 = setup_and_solve(max_iters = 123, convergence_kkt_norm = 1e-8)
        @test Marble.max_iters(s2.options) == 123 &&
              isapprox(Marble.convergence_kkt_norm(s2.options), 1e-8) &&
              Marble.converged(r2)
    end

    @testset "dimension validation" begin
        # Mismatched / missing blocks must error cleanly, never segfault. Validation
        # happens in Julia (see _validate_problem_dims) because CxxWrap cannot
        # translate the core's C++ exception into a catchable Julia error. These
        # mirror the seven cases in the Python suite (TestDimensionValidation)
        Q2 = Matrix(1.0I, 2, 2)
        # L given (1 row) but l omitted
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), Q2, zeros(2);
            L = zeros(1, 2), R = zeros(1, 2), r = zeros(1))
        # L and l given but R, r omitted
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), Q2, zeros(2);
            L = zeros(1, 2), l = zeros(1))
        # Non-square Q.
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), zeros(2, 3), zeros(2))
        # Q size disagrees with length(q)
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), Matrix(1.0I, 3, 3), zeros(2))
        # J_eq rows disagree with c_eq length
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), Q2, zeros(2);
            J_eq = zeros(2, 2), c_eq = zeros(1))
        # J_ineq columns disagree with length(q)
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), Q2, zeros(2);
            J_ineq = zeros(1, 3), c_ineq = zeros(1))
        # Sparse blocks: L given but l omitted
        @test_throws DimensionMismatch Marble.setup!(Marble.Solver(), sparse(Q2), zeros(2);
            L = sparse(zeros(1, 2)), R = sparse(zeros(1, 2)), r = zeros(1))
    end
end
