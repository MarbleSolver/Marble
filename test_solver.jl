using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using CxxWrap
using Revise
using CILogDomain
using SparseArrays
using LinearAlgebra

module RCQP
  using CxxWrap
  @wrapmodule(() -> joinpath(@__DIR__, "build/librcqp_wrapper"))

  function __init__()
    @initcxx
  end
end

name = :simple_test
problem = create_problem(name)#, eq = 1, ineq = 1, comp = 1);
solver = Solver(SolverOptions())
set_problem!(solver, problem)
print_solver_details(solver)

# Set up problem
H, g, f0 = Matrix(solver.cost_hess), solver.cost_grad, solver.cost_const
J_eq, J_ineq, J_comp = Matrix(solver.conjac[solver.eq_inds, :]), Matrix(solver.conjac[solver.ineq_inds, :]), Matrix(solver.conjac[solver.comp_inds, :])
c_eq, c_ineq, c_comp = solver.conrhs[solver.eq_inds], solver.conrhs[solver.ineq_inds], solver.conrhs[solver.comp_inds]
prob = RCQP.Problem(H, g, f0, J_eq, c_eq, J_ineq, c_ineq,  J_comp, c_comp)

opts = RCQP.SolverOptions()
RCQP.verbosity!(opts, 10)
rcqp = RCQP.Solver(opts)
RCQP.set_problem(rcqp, prob)
RCQP.solve(rcqp, opts)