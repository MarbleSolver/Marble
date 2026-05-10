using Pkg;
Pkg.activate(joinpath(@__DIR__))
using CxxWrap
using Revise
using CILogDomain
using SparseArrays
using LinearAlgebra

module Marble
  using CxxWrap
  @wrapmodule(() -> joinpath(@__DIR__, "submodules/Marble/build/libmarble_wrapper"))

  function __init__()
    @initcxx
  end
end

name = :hopper
##

problem = create_problem(name)#, eq = 1, ineq = 1, comp = 1);
solver_options = SolverOptions()
solver_options.verbose = true
solver_options.max_iters = 10_000
z_ours, solver = solve(problem; options=solver_options);

# Check eigen to julia and back
println(Marble.test_array([1.0, 2.0, 3.0]))

arr = [1.0 2.0; 3.0 4.0]
println(Marble.test_matrix(arr, 2, 2))
println(arr)

# Set up problem and check sizes
H, g, f0 = Matrix(solver.cost_hess), solver.cost_grad, solver.cost_const
J_eq, J_ineq, J_comp = Matrix(solver.conjac[solver.eq_inds, :]), Matrix(solver.conjac[solver.ineq_inds, :]), Matrix(solver.conjac[solver.comp_inds, :])
c_eq, c_ineq, c_comp = solver.conrhs[solver.eq_inds], solver.conrhs[solver.ineq_inds], solver.conrhs[solver.comp_inds]

prob = Marble.Problem(H, g, f0, J_eq, c_eq, J_ineq, c_ineq, 
            J_comp, c_comp)
@assert Marble.nz(prob) == solver.nz
@assert Marble.n_eq(prob) == solver.n_eq
@assert Marble.n_ineq(prob) == solver.n_ineq
@assert Marble.n_comp(prob) == solver.n_comp

# Set up solver, test setting and getting problem
marble_solver = Marble.Solver()

# Check retraction and retraction derivative
s = randn(10)
@assert norm(Marble.retract(marble_solver, s, 1e-2) - solver.r(s, 1e-2)) < 1e-10
@assert norm(Marble.retract_deriv(marble_solver, s, 1e-2) - solver.d_r(s, 1e-2)) < 1e-10
@assert norm(Marble.retract_second_deriv(marble_solver, s, 1e-2) - solver.dd_r(s, 1e-2)) < 1e-10

# Set problem
Marble.set_problem(marble_solver, prob)
prob = Marble.get_problem(marble_solver)
@assert Marble.nz(prob) == solver.nz
@assert Marble.n_eq(prob) == solver.n_eq
@assert Marble.n_ineq(prob) == solver.n_ineq
@assert Marble.n_comp(prob) == solver.n_comp

# Get workspace
workspace = Marble.get_workspace(marble_solver)

# Ruiz scaling check
@assert norm(Marble.scaling(workspace) - diag(solver.ruiz_mat), Inf) < 1e-10

# Test indices
Marble.z_inds(marble_solver) == solver.kkt_inds.z .- 1
Marble.s_ineq_inds(marble_solver) == solver.kkt_inds.v .- 1
Marble.s_comp_inds(marble_solver) == solver.kkt_inds.σ .- 1
Marble.m_eq_inds(marble_solver) == solver.kkt_inds.λ .- 1
Marble.m_ineq_inds(marble_solver) == solver.kkt_inds.μ .- 1
Marble.m_comp_inds(marble_solver) == solver.kkt_inds.τ .- 1

# Check KKT structure
function get_kkt(marble_solver)
    nr, nc, colptr, rowval, nzval = Marble.kkt_system(Marble.get_workspace(marble_solver))
    return copy(SparseMatrixCSC(nr, nc, colptr.+1, rowval.+1, nzval))
end
kkt_sparse = get_kkt(marble_solver)
perm = Marble.amd_perm_vec(workspace) .+ 1
iperm = Marble.amd_iperm_vec(workspace) .+ 1

# Check indices into nzval (can't compare against solver one because we only use upper triangular form)
# Also the matrix is permuted, so we have to check against permuted indices
kkt_sparse.nzval[:] = 1:length(kkt_sparse.nzval)
kkt_full = (kkt_sparse + kkt_sparse' - spdiagm(diag(kkt_sparse)))[iperm, iperm] # Unpermuted version

norm(diagm(kkt_sparse.nzval[Marble.z_z_inds(marble_solver) .+ 1]) - kkt_full[solver.kkt_inds.z, solver.kkt_inds.z], Inf)
norm(diagm(kkt_sparse.nzval[Marble.s_ineq_s_ineq_inds(marble_solver) .+ 1]) - kkt_full[solver.kkt_inds.v, solver.kkt_inds.v], Inf)
norm(diagm(kkt_sparse.nzval[Marble.s_ineq_m_ineq_inds(marble_solver) .+ 1]) - kkt_full[solver.kkt_inds.v, solver.kkt_inds.μ], Inf)
norm(diagm(kkt_sparse.nzval[Marble.s_comp_s_comp_inds(marble_solver) .+ 1]) - kkt_full[solver.kkt_inds.σ, solver.kkt_inds.σ], Inf)
norm(kkt_sparse.nzval[Marble.s_comp_m_comp_inds(marble_solver) .+ 1]  - kkt_full[solver.kkt_inds.σ, solver.kkt_inds.τ].nzval, Inf)

# Update ineq, comp and penalty terms
iter = solver.iters[2]
Marble.update_KKT_ineq(marble_solver, iter.v, sqrt(iter.κ))
Marble.update_KKT_comp(marble_solver, iter.σ, iter.τ, sqrt(iter.κ))
Marble.update_KKT_penalty(marble_solver, 1/iter.ρ)
kkt = get_kkt(marble_solver)[iperm, iperm] # Unpermuted
kkt = (kkt + kkt' - spdiagm(diag(kkt)))

# Check stationarity rows
scaling = solver.ruiz_mat
hess = scaling*lagrangian_hessian(solver, iter)*scaling
@assert norm(hess[solver.kkt_inds.z, :] - kkt[solver.kkt_inds.z, :], Inf) < 1e-10 
@assert norm(hess[solver.kkt_inds.v, :] - kkt[solver.kkt_inds.v, :], Inf) < 1e-10 
@assert norm(hess[solver.kkt_inds.σ, :] - kkt[solver.kkt_inds.σ, :], Inf) < 1e-10 

# Check entire matrix
@assert norm(hess - kkt, Inf) < 1e-10

# Check across all iterates
for iter in solver.iters
    Marble.update_KKT_ineq(marble_solver, iter.v, sqrt(iter.κ))
    Marble.update_KKT_comp(marble_solver, iter.σ, iter.τ, sqrt(iter.κ))
    Marble.update_KKT_penalty(marble_solver, 1/iter.ρ)
    hess = (scaling*lagrangian_hessian(solver, iter)*scaling)[perm, perm]
    kkt = get_kkt(marble_solver)
    @assert norm(triu(hess) - kkt, Inf) < 1e-10
end

# Check writing to workspace
function set_from_iter(marble_solver, iter)
    workspace = Marble.get_workspace(marble_solver)
    Marble.z(workspace) .= iter.z
    Marble.s_ineq(workspace) .= iter.v
    Marble.s_comp(workspace) .= iter.σ
    Marble.m_eq(workspace) .= iter.λ
    Marble.m_ineq(workspace) .= iter.μ
    Marble.m_comp(workspace) .= iter.τ
    Marble.m_eq_est(workspace) .= iter.α
    Marble.m_ineq_est(workspace) .= iter.β
    Marble.m_comp_est(workspace) .= iter.γ
    Marble.update_KKT_residual(marble_solver, sqrt(iter.κ), 1/iter.ρ)
    Marble.update_KKT_system(marble_solver, sqrt(iter.κ), 1/iter.ρ)
    Marble.update_KKT_primal_regularizer(marble_solver, iter.reg)
end
set_from_iter(marble_solver, solver.iters[2])
residual = Marble.kkt_residual(workspace)
kkt = get_kkt(marble_solver)
@assert norm(lagrangian_gradient(solver, iter) - residual, Inf) < 1e-10
@assert norm(triu((scaling*(lagrangian_hessian(solver, iter) + iter.reg*solver.reg_mat)*scaling)[perm, perm]) - kkt, Inf) < 1e-10

# Check across all iteraters
for iter in solver.iters
    # Write current solution guess
    set_from_iter(marble_solver, iter)

    # Check against iterate
    residual = Marble.kkt_residual(workspace)
    kkt = get_kkt(marble_solver)
    @assert norm(lagrangian_gradient(solver, iter) - residual, Inf) < 1e-10
    @assert norm(triu((scaling*(lagrangian_hessian(solver, iter) + iter.reg*solver.reg_mat)*scaling)[perm, perm]) - kkt, Inf) < 1e-10
end

# Check regularizer updating
Marble.update_KKT_primal_regularizer(marble_solver, 0.0)
kkt = get_kkt(marble_solver)
Marble.update_KKT_primal_regularizer(marble_solver, 1e-4)
kkt_reg = get_kkt(marble_solver)
@assert norm((kkt_reg - kkt) - 1e-4*(scaling*solver.reg_mat*scaling)[perm, perm], Inf) < 1e-12
Marble.update_KKT_primal_regularizer(marble_solver, 1.23e-7)
kkt_reg = copy(get_kkt(marble_solver))
@assert norm((kkt_reg - kkt) - 1.23e-7*(scaling*solver.reg_mat*scaling)[perm, perm], Inf) < 1e-12
Marble.update_KKT_primal_regularizer(marble_solver, 0.0)
kkt_reg = copy(get_kkt(marble_solver))
@assert norm(kkt_reg - kkt, Inf) < 1e-14

# Test factorization and solve
iter = solver.iters[5]
set_from_iter(marble_solver, iter)
res = copy(Marble.kkt_residual(Marble.get_workspace(marble_solver)))
reg = 1e-7
Marble.update_KKT_primal_regularizer(marble_solver, reg)
kkt = get_kkt(marble_solver)[iperm, iperm] # Unpermuted
kkt = (kkt + kkt' - spdiagm(diag(kkt)))

@assert Marble.analytical_factorization(marble_solver)
@assert Marble.numerical_factorization(marble_solver)
Marble.backsolve(marble_solver)
qdldl_soln = Marble.newton_step(Marble.get_workspace(marble_solver))
@assert(norm(kkt*(scaling \ qdldl_soln) + (scaling*res), Inf) < 1e-9)

# Test for each iter
using QDLDL
for iter in solver.iters
    if iter.type != :Newton
        continue
    end
    set_from_iter(marble_solver, iter)

    # Get C++ terms
    res = copy(Marble.kkt_residual(Marble.get_workspace(marble_solver)))
    kkt = get_kkt(marble_solver)[iperm, iperm] # Unpermuted
    kkt = (kkt + kkt' - spdiagm(diag(kkt)))

    # Solve C++ system
    @assert Marble.analytical_factorization(marble_solver)
    @assert Marble.numerical_factorization(marble_solver)
    @assert Marble.check_inertia(marble_solver)
    Marble.backsolve(marble_solver)
    qdldl_soln = Marble.newton_step(Marble.get_workspace(marble_solver))

    # Check against Julia
    hess = lagrangian_hessian(solver, iter) + iter.reg*solver.reg_mat
    grad = lagrangian_gradient(solver, iter)
    F = QDLDL.qdldl(hess)
    qdldl_soln_julia = QDLDL.solve(F, -grad)
    @assert norm(hess*qdldl_soln + grad, Inf) < 1e-5

    # Check iterate passes filter
    marble_filter_viol, marble_filter_obj = Marble.entry_from_solution(marble_solver, sqrt(iter.κ), 1/iter.ρ)
    iter_filter_obj, iter_filter_viol = filter_criteria(solver, iter)

    @assert abs(marble_filter_viol - iter_filter_viol) < 1e-10
    @assert abs(marble_filter_obj - iter_filter_obj) < 1e-10
end

println("All tests passed!")
