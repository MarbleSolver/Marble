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