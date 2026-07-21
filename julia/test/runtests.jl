# Marble test suite
using Test
using LinearAlgebra
using SparseArrays
using Marble
import ForwardDiff as FD

# Example problem (julia/examples/simple_test.jl), written directly as matrices:
#   min x'x
#   s.t. x1 = 1,  x2 >= 1,  0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
#   optimum x* = [1, 1, 0, 1]
const Q = 2.0 * Matrix(1.0I, 4, 4)
const q = zeros(4)
const C0 = 0.0
const J_EQ,   B_EQ   = [1.0 0.0 0.0 0.0], [-1.0]
const J_INEQ, B_INEQ = [0.0 1.0 0.0 0.0], [-1.0]
const L, EL          = [0.0 0.0 1.0 0.0], [1.0]
const R, ER          = [0.0 0.0 0.0 1.0], [-1.0]
const ZSTAR = [1.0, 1.0, 0.0, 1.0]

# Build the solver, set up the example problem (dense or sparse), and solve.
# Option settings are forwarded to setup! as keyword arguments.
function setup_and_solve(; sparse_problem = false, opts...)
    conv = sparse_problem ? sparse : identity
    solver = Marble.Solver()
    Marble.setup!(solver, conv(Q), q, C0;
        J_eq = conv(J_EQ), b_eq = B_EQ, J_ineq = conv(J_INEQ), b_ineq = B_INEQ,
        L = conv(L), l = EL, R = conv(R), r = ER, opts...)
    return solver, Marble.solve!(solver)
end

retract(::Val{:Softplus}, x, κ) = 0.5*(x + sqrt.(x.^2 .+ 4*κ))
retract(::Val{:Exp}, x, κ) = sqrt(κ)*exp.(x)
retract(::Val{:ScaledExp}, x, κ) = sqrt(κ)*exp.(x./sqrt(κ))

@testset "Marble" begin
    @testset "smoke tests" begin
        solver, _ = setup_and_solve()
        for name in propertynames(solver)
            @test getproperty(solver, name) isa Any
        end
        problem, workspace = solver.problem, solver.workspace
        for name in propertynames(problem)
            @test getproperty(problem, name) isa Any
        end
        for name in propertynames(workspace)
            @test getproperty(workspace, name) isa Any
        end
    end
    @testset "retraction maps" begin
        solver, _ = setup_and_solve()
        x = [0; 0.5; -0.5; -10; 10; randn(4)]
        κ = 1e-1

        tol = 1e-10
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            Marble.update_settings!(solver, retraction_type = i-1)
            @test isapprox(Marble.retract(solver, x, κ), retract(retract_type, x, κ), atol = tol)
            @test isapprox(Marble.retract_deriv(solver, x, κ), diag(FD.jacobian(_x -> retract(retract_type, _x, κ), x)), atol=tol)
            @test isapprox(Marble.retract_second_deriv(solver, x, κ), 
                    diag(FD.jacobian(_x -> diag(FD.jacobian(_x -> retract(retract_type, _x, κ), _x)), x)), atol=tol)
            @test isapprox(Marble.retract_drelax(solver, x, κ), FD.derivative(_κ -> retract(retract_type, x, _κ), κ), atol=tol)
            @test isapprox(Marble.retract_deriv_drelax(solver, x, κ), FD.derivative(_κ -> diag(FD.jacobian(_x -> retract(retract_type, _x, _κ), x)), κ), atol=tol)
        end
    end

    @testset "problem dimensions" begin
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            solver, _ = setup_and_solve(retraction_type = i-1)
            p = solver.problem
            @test p.nz == 4 && p.n_eq == 1 && p.n_ineq == 1 && p.n_comp == 1
        end
    end

    @testset "solves to known optimum" begin
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            solver, res = setup_and_solve(retraction_type = i-1)
            @test res.converged && isapprox(res.z, ZSTAR, atol = 1e-4)
        end
    end

    @testset "equality constraint satisfied" begin
        # x1 = 1
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            solver, res = setup_and_solve(retraction_type = i-1)
            @test abs(res.z[1] - 1.0) < 1e-4
        end
    end

    @testset "inequality constraint satisfied" begin
        # x2 >= 1
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            solver, res = setup_and_solve(retraction_type = i-1)
            @test res.z[2] >= 1.0 - 1e-4
        end
    end

    @testset "complementarity satisfied" begin
        # 0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            solver, res = setup_and_solve(retraction_type = i-1)
            z = res.z
            a = z[3] + 1.0
            b = z[4] - 1.0
            @test a >= -1e-4 && b >= -1e-4 && abs(a * b) < 1e-4
        end
    end

    @testset "objective value" begin
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            solver, res = setup_and_solve(retraction_type = i-1)
            @test abs(Marble.obj(solver, res.z)) - 3.0 < 1e-3
        end
    end

    @testset "dense and sparse solve agree" begin
        _, rd = setup_and_solve(sparse_problem = false)
        _, rs = setup_and_solve(sparse_problem = true)
        @test rd.converged && rs.converged &&
              isapprox(rd.z, rs.z, atol = 1e-6)
    end

    @testset "complementarity-only QPCC" begin
        # same cost x'x, only the complementarity 0 <= (x3 + 1) ⟂ (x4 - 1) >= 0
        # x1, x2 are unconstrained -> 0; x3 = 0, x4 = 1  =>  x* = [0, 0, 0, 1]
        solver = Marble.Solver()
        Marble.setup!(solver, Q, q, C0; L = L, l = EL, R = R, r = ER)
        res = Marble.solve!(solver)
        @test res.converged && isapprox(res.z, [0.0, 0.0, 0.0, 1.0], atol = 1e-4)
    end

    @testset "options are honored" begin
        s2, r2 = setup_and_solve(max_iters = 123, convergence_kkt_norm = 1e-8)
        @test Marble.max_iters(s2.options) == 123 &&
              isapprox(Marble.convergence_kkt_norm(s2.options), 1e-8) &&
              r2.converged
    end

    @testset "dimension validation" begin
        # Mismatched / missing blocks must error cleanly, never segfault. Validation
        # happens in Julia (see _validate_problem_dims) because CxxWrap cannot
        # translate the core's C++ exception into a catchable Julia error. These
        # mirror the seven cases in the Python suite (TestDimensionValidation)
        Q2 = Matrix(1.0I, 2, 2)
        # L given (1 row) but l omitted
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Q2, zeros(2);
            L = zeros(1, 2), R = zeros(1, 2), r = zeros(1))
        # L and l given but R, r omitted
        @test_throws ErrorException Marble.setup!(Marble.Solver(), Q2, zeros(2);
            L = zeros(1, 2), l = zeros(1))
        # Non-square Q.
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

    function kkt_residual(retract_type, solver, solution, κ, ρ; symmetric = true)
        # Extract solution, multipliers, and problem data
        prob, workspace = solver.problem, solver.workspace
        z, s_ineq, s_comp = solution[solver.z_inds], solution[solver.s_ineq_inds], solution[solver.s_comp_inds]
        m_eq, m_ineq, m_comp_L, m_comp_R =
            solution[solver.m_eq_inds], solution[solver.m_ineq_inds], solution[solver.m_comp_L_inds], solution[solver.m_comp_R_inds]
        H, g, J_eq, c_eq, J_ineq, c_ineq, L_comp, l_comp, R_comp, r_comp = prob.cost_hessian, prob.cost_gradient, prob.J_eq, prob.c_eq, prob.J_ineq, prob.c_ineq, prob.L_comp, prob.l_comp, prob.R_comp, prob.r_comp
        m_eq_est, m_ineq_est, m_comp_L_est, m_comp_R_est = workspace.m_eq_est, workspace.m_ineq_est, workspace.m_comp_L_est, workspace.m_comp_R_est

        # Construct KKT residual
        res = zeros(promote_type(eltype(solution), typeof(κ)), length(solution))
        # z stationarity
        res[solver.z_inds] = H*z + g + J_eq'*m_eq + J_ineq'*m_ineq + L_comp'*m_comp_L + R_comp'*m_comp_R

        # Inequality slack stationarity
        p_neg_ineq = retract(retract_type, -s_ineq, κ)
        dp_ineq = FD.jacobian(_x -> retract(retract_type, _x, κ), s_ineq)
        if symmetric
            res[solver.s_ineq_inds] = dp_ineq * (-m_ineq - p_neg_ineq)
        else
            res[solver.s_ineq_inds] = -m_ineq - p_neg_ineq
        end

        # Complementarity slack stationarity
        dp_comp = FD.jacobian(_x -> retract(retract_type, _x, κ), s_comp)
        dp_neg_comp = FD.jacobian(_x -> retract(retract_type, _x, κ), -s_comp)
        res[solver.s_comp_inds] = -dp_comp * m_comp_L + dp_neg_comp * m_comp_R

        # Equality primal feasibility
        res[solver.m_eq_inds] = J_eq*z + c_eq - 1/ρ*(m_eq - m_eq_est)

        # Inequality primal feasibility
        p_ineq = retract(retract_type, s_ineq, κ)
        res[solver.m_ineq_inds] = J_ineq*z + c_ineq - p_ineq - 1/ρ*(m_ineq - m_ineq_est)

        # Complementarity primal feasibility
        p_comp_L = retract(retract_type, s_comp, κ)
        p_comp_R = retract(retract_type, -s_comp, κ)
        res[solver.m_comp_L_inds] = L_comp*z + l_comp - p_comp_L - 1/ρ*(m_comp_L - m_comp_L_est)
        res[solver.m_comp_R_inds] = R_comp*z + r_comp - p_comp_R - 1/ρ*(m_comp_R - m_comp_R_est)
        return res
    end

    @testset "KKT residual" begin
        solver, _ = setup_and_solve()
        prob, workspace = solver.problem, solver.workspace
        res = workspace.kkt_residual

        workspace.solution .= randn(length(workspace.solution))
        ρ, κ = 1e1, 1e-1
        Marble.update_residuals!(solver)

        tol = 1e-10
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            Marble.update_settings!(solver, retraction_type = i-1)
            Marble.update_KKT_residual!(solver, κ, ρ)
            res = kkt_residual(retract_type, solver, workspace.solution, κ, ρ)
            @test isapprox(workspace.kkt_residual, res, atol=tol)
        end
    end

    @testset "KKT system" begin
        solver, _ = setup_and_solve()
        workspace = solver.workspace

        workspace.solution .= randn(length(workspace.solution))
        ρ, κ = 1e1, 1e-1
        
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            Marble.update_settings!(solver, retraction_type = i-1)
            Marble.update_KKT_system!(solver, κ, ρ)

            # Get kkt matrix from solver
            kkt_mat = workspace.kkt_system # upper triangular
            kkt_mat = kkt_mat + kkt_mat' - Diagonal(diag(kkt_mat)) # symmetric
            scale_mat = spdiagm(1 ./workspace.scaling)
            kkt_mat = scale_mat*kkt_mat[workspace.amd_iperm_vec, workspace.amd_iperm_vec]*scale_mat

            # Compare to forward diff with symmetrizing term
            test_mat = FD.jacobian(_s -> kkt_residual(retract_type, solver, _s, κ, ρ; symmetric=false), workspace.solution)
            dp_ineq = FD.jacobian(_x -> retract(retract_type, _x, κ), workspace.s_ineq)
            test_mat[solver.s_ineq_inds, :] = dp_ineq * test_mat[solver.s_ineq_inds, :]

            @test isapprox(kkt_mat, test_mat, atol = 1e-10)
        end
    end

    @testset "KKT relaxation derivative" begin
        solver, _  = setup_and_solve()
        workspace = solver.workspace
        workspace.solution .= randn(length(workspace.solution))
        κ = 1e-1
        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            Marble.update_settings!(solver, retraction_type = i-1)
            Marble.update_dKKT_residual_drelax!(solver, κ)
            @test isapprox(workspace.dkkt_residual_drelax, FD.derivative(_κ -> kkt_residual(retract_type, solver, workspace.solution, _κ, 1e1), κ), atol=1e-10)
        end
    end

    @testset "filter entry" begin
        solver, _  = setup_and_solve()
        prob, workspace = solver.problem, solver.workspace
        workspace.solution .= randn(length(workspace.solution))

        for (i, retract_type) in enumerate([Val(:Softplus), Val(:Exp), Val(:ScaledExp)])
            ρ, κ = 1e1, 1e-1
            Marble.update_settings!(solver, retraction_type = i-1)
            Marble.update_residuals!(solver)

            res = kkt_residual(retract_type, solver, workspace.solution, κ, ρ, symmetric = false)
            viols = [res[solver.m_eq_inds]; res[solver.m_ineq_inds]; res[solver.m_comp_L_inds]; res[solver.m_comp_R_inds]]
            viol = norm(viols, 1)/length(viols)
            obj = 0.5*workspace.z'*prob.cost_hessian*workspace.z + prob.cost_gradient'*workspace.z + prob.cost_const - 
                 κ*sum(log.(retract(retract_type, workspace.s_ineq, κ))) + 
                 0.5*1/ρ*(workspace.m_eq'*workspace.m_eq + workspace.m_ineq'*workspace.m_ineq + workspace.m_comp_L'*workspace.m_comp_L + workspace.m_comp_R'*workspace.m_comp_R)

            isapprox(collect(Marble.entry_from_solution(solver, κ, ρ)), [viol; obj], atol = 1e-10)
        end
    end
end
