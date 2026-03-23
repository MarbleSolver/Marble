using Pkg;
Pkg.activate(joinpath(@__DIR__))
using CxxWrap
using Revise
using CILogDomain
using SparseArrays
using LinearAlgebra

module RCQP
  using CxxWrap
  @wrapmodule(() -> joinpath(@__DIR__, "submodules/RCQP/build/librcqp_wrapper.so"))

  function __init__()
    @initcxx
  end
end

name = :simple_test

problem = create_problem(name);
solver_options = SolverOptions()
solver_options.verbose = true
solver_options.max_iters = 10_000
z_ours, solver = solve(problem; options=solver_options);

# Check eigen to julia and back
println(RCQP.test_array([1.0, 2.0, 3.0]))

arr = [1.0 2.0; 3.0 4.0]
println(RCQP.test_matrix(arr, 2, 2))
println(arr)

# Set up problem and check sizes
H, g = Matrix(solver.cost_hess), solver.cost_grad
J_eq, J_ineq, J_comp = Matrix(solver.conjac[solver.eq_inds, :]), Matrix(solver.conjac[solver.ineq_inds, :]), Matrix(solver.conjac[solver.comp_inds, :])
c_eq, c_ineq, c_comp = solver.conrhs[solver.eq_inds], solver.conrhs[solver.ineq_inds], solver.conrhs[solver.comp_inds]

prob = RCQP.Problem(H, g, J_eq, c_eq, J_ineq, c_ineq, 
            J_comp, c_comp)
@assert RCQP.nz(prob) == solver.nz
@assert RCQP.n_eq(prob) == solver.n_eq
@assert RCQP.n_ineq(prob) == solver.n_ineq
@assert RCQP.n_comp(prob) == solver.n_comp

# Set up solver, test setting and getting problem
rcqp = RCQP.Solver()

# Check retraction and retraction derivative
s = randn(10)
@assert norm(RCQP.retract(rcqp, s, 1e-2) - solver.r(s, 1e-2)) < 1e-10
@assert norm(RCQP.retract_deriv(rcqp, s, 1e-2) - solver.d_r(s, 1e-2)) < 1e-10
@assert norm(RCQP.retract_second_deriv(rcqp, s, 1e-2) - solver.dd_r(s, 1e-2)) < 1e-10

# Get workspace
workspace = RCQP.get_workspace(rcqp)

# Set problem
RCQP.set_problem(rcqp, prob)
prob = RCQP.get_problem(rcqp)

@assert RCQP.nz(prob) == solver.nz
@assert RCQP.n_eq(prob) == solver.n_eq
@assert RCQP.n_ineq(prob) == solver.n_ineq
@assert RCQP.n_comp(prob) == solver.n_comp

# Test indices
RCQP.z_inds(rcqp) == solver.kkt_inds.z .- 1
RCQP.s_ineq_inds(rcqp) == solver.kkt_inds.v .- 1
RCQP.s_comp_inds(rcqp) == solver.kkt_inds.σ .- 1
RCQP.m_eq_inds(rcqp) == solver.kkt_inds.λ .- 1
RCQP.m_ineq_inds(rcqp) == solver.kkt_inds.μ .- 1
RCQP.m_comp_inds(rcqp) == solver.kkt_inds.τ .- 1

# Update ineq, comp and penalty terms
iter = solver.iters[2]
RCQP.update_KKT_ineq(rcqp, iter.v, sqrt(iter.κ))
RCQP.update_KKT_comp(rcqp, iter.σ, iter.τ, sqrt(iter.κ))
RCQP.update_KKT_penalty(rcqp, 1/iter.ρ)

# Check KKT structure
function get_kkt(rcqp)
    nr, nc, colptr, rowval, nzval = RCQP.kkt_system(RCQP.get_workspace(rcqp))
    return SparseMatrixCSC(nr, nc, colptr.+1, rowval.+1, nzval)
end
kkt = get_kkt(rcqp)
hess = lagrangian_hessian(solver, iter)
hess_sparse = copy(hess) # can't compare now that kkt has all diag elements in structure
kkt_sparse = copy(kkt)
hess_sparse.nzval .= 1
kkt_sparse.nzval .= 1
# @assert norm((triu(hess_sparse) .!= 0) - (kkt_sparse .!= 0), Inf) == 0

# Check indices into nzval (can't compare against solver one because we only use upper triangular form)
kkt_sparse.nzval[RCQP.z_z_inds(rcqp) .+ 1] = solver.kkt_inds.z
@assert norm(diag(kkt_sparse[solver.kkt_inds.z, solver.kkt_inds.z]) - solver.kkt_inds.z, Inf) == 0
kkt_sparse.nzval[RCQP.s_ineq_s_ineq_inds(rcqp) .+ 1] = solver.kkt_inds.v
@assert norm(kkt_sparse[solver.kkt_inds.v, solver.kkt_inds.v] - diagm(solver.kkt_inds.v), Inf) == 0
kkt_sparse.nzval[RCQP.s_ineq_m_ineq_inds(rcqp) .+ 1] = solver.kkt_inds.v*2
@assert norm(kkt_sparse[solver.kkt_inds.v, solver.kkt_inds.μ] - 2*diagm(solver.kkt_inds.v), Inf) == 0
kkt_sparse.nzval[RCQP.s_comp_s_comp_inds(rcqp) .+ 1] = solver.kkt_inds.σ
@assert norm(kkt_sparse[solver.kkt_inds.σ, solver.kkt_inds.σ] - diagm(solver.kkt_inds.σ), Inf) == 0
kkt_sparse.nzval[RCQP.s_comp_m_comp_inds(rcqp) .+ 1] =  vcat([[a; b] for (a, b) in zip(solver.kkt_inds.σ, -solver.kkt_inds.σ)]...)
@assert norm(kkt_sparse[solver.kkt_inds.σ, solver.kkt_inds.τ] - kron(diagm(solver.kkt_inds.σ), [1 -1]), Inf) == 0

# Check stationarity rows
@assert norm(triu(hess)[solver.kkt_inds.z, :] - kkt[solver.kkt_inds.z, :], Inf) < 1e-10 
@assert norm(triu(hess)[solver.kkt_inds.v, :] - kkt[solver.kkt_inds.v, :], Inf) < 1e-10 
@assert norm(triu(hess)[solver.kkt_inds.σ, :] - kkt[solver.kkt_inds.σ, :], Inf) < 1e-10 

# Check entire matrix
@assert norm(triu(hess) - kkt, Inf) < 1e-10

# Check across all iterates
for iter in solver.iters
    RCQP.update_KKT_ineq(rcqp, iter.v, sqrt(iter.κ))
    RCQP.update_KKT_comp(rcqp, iter.σ, iter.τ, sqrt(iter.κ))
    RCQP.update_KKT_penalty(rcqp, 1/iter.ρ)
    hess = lagrangian_hessian(solver, iter)
    kkt = get_kkt(rcqp)
    @assert norm(triu(hess) - kkt, Inf) < 1e-10
end

# Check writing to workspace
function set_from_iter(rcqp, iter)
    workspace = RCQP.get_workspace(rcqp)
    RCQP.z(workspace) .= iter.z
    RCQP.s_ineq(workspace) .= iter.v
    RCQP.s_comp(workspace) .= iter.σ
    RCQP.m_eq(workspace) .= iter.λ
    RCQP.m_ineq(workspace) .= iter.μ
    RCQP.m_comp(workspace) .= iter.τ
    RCQP.m_eq_est(workspace) .= iter.α
    RCQP.m_ineq_est(workspace) .= iter.β
    RCQP.m_comp_est(workspace) .= iter.γ
    RCQP.update_KKT_residual(rcqp, sqrt(iter.κ), 1/iter.ρ)
    RCQP.update_KKT_system(rcqp, sqrt(iter.κ), 1/iter.ρ)
end
set_from_iter(rcqp, solver.iters[2])
residual = RCQP.kkt_residual(workspace)
kkt = get_kkt(rcqp)
@assert norm(lagrangian_gradient(solver, iter) - residual, Inf) < 1e-10
@assert norm(triu(lagrangian_hessian(solver, iter)) - kkt, Inf) < 1e-10

# Check across all iteraters
for iter in solver.iters
    # Write current solution guess
    set_from_iter(rcqp, iter)

    # Check against iterate
    residual = RCQP.kkt_residual(workspace)
    kkt = get_kkt(rcqp)
    @assert norm(lagrangian_gradient(solver, iter) - residual, Inf) < 1e-10
    @assert norm(triu(lagrangian_hessian(solver, iter)) - kkt, Inf) < 1e-10
end

# Test factorization and solve
iter = solver.iters[1]
set_from_iter(rcqp, iter)
hess = lagrangian_hessian(solver, iter)
res = RCQP.kkt_residual(RCQP.get_workspace(rcqp))

RCQP.analytical_factorization(rcqp)
RCQP.numerical_factorization(rcqp)
RCQP.backsolve(rcqp)
qdldl_soln = RCQP.newton_step(RCQP.get_workspace(rcqp))

@assert norm(lagrangian_gradient(solver, iter) - res, Inf) < 1e-10
@assert norm(triu(lagrangian_hessian(solver, iter)) - kkt, Inf) < 1e-10