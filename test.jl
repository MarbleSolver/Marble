using Pkg;
Pkg.activate(joinpath(@__DIR__))
using CxxWrap
using Revise
using CILogDomain

name = :simple_test

problem = create_problem(name);
solver_options = SolverOptions()
solver_options.verbose = true
solver_options.max_iters = 10_000
z_ours, solver = solve(problem; options=solver_options);

module RCQP
  using CxxWrap
  @wrapmodule(() -> joinpath(@__DIR__, "build", "librcqp_wrapper"))

  function __init__()
    @initcxx
  end
end

println(RCQP.test_array([1.0, 2.0, 3.0]))

arr = [1.0 2.0; 3.0 4.0]
println(RCQP.test_matrix(arr, 2, 2))
println(arr)

H, g = Matrix(solver.cost_hess), solver.cost_grad
J_eq, J_ineq, J_comp = Matrix(solver.conjac[solver.eq_inds, :]), Matrix(solver.conjac[solver.ineq_inds, :]), Matrix(solver.conjac[solver.comp_inds, :])
J_comp_l, J_comp_r = J_comp[1:2:end, :], J_comp[2:2:end, :]
c_eq, c_ineq, c_comp = solver.conrhs[solver.eq_inds], solver.conrhs[solver.ineq_inds], solver.conrhs[solver.comp_inds]
c_comp_l, c_comp_r = c_comp[1:2:end], c_comp[2:2:end]

test = RCQP.Problem(H, g, J_eq, c_eq, J_ineq, c_ineq, 
            J_comp_l, c_comp_l, J_comp_r, c_comp_r)
@assert RCQP.nz(test) == solver.nz
@assert RCQP.n_eq(test) == solver.n_eq
@assert RCQP.n_ineq(test) == solver.n_ineq
@assert RCQP.n_comp(test) == solver.n_comp