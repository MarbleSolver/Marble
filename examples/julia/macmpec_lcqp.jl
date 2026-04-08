# Run all of the MacMPEC LCQP problems with Marble and extract the results

using HDF5
using JSON

module Marble
    using CxxWrap

    libpath = joinpath(@__DIR__, "..", "..", "build", "lib", "librcqp_julia")
    @wrapmodule(() -> libpath)

    function __init__()
        @initcxx
    end
end

## Function to solve an individual problem

function solve_macmpec_lcqp(name)
    # Load data from .h5 file
    data = h5open(joinpath(@__DIR__, "..", "data", "macmpec", "h5", name * ".h5"), "r") do f
        Dict(k => read(f, k) for k in keys(f))
    end

    # Cost terms
    f0 = data["f0"]
    q  = data["q"]
    H  = data["H"]

    obj(z) = 0.5 * z' * H * z + q' * z + f0

    # Constraint Jacobians/affine terms
    J_eq   = data["J_eq"]
    b_eq   = data["b_eq"]
    J_ineq = data["J_ineq"]
    b_ineq = data["b_ineq"]
    J_comp = data["J_comp"]
    b_comp = data["b_comp"]

    # Create an instance of Marble and solve the problem
    solver = Marble.Solver()

    # Instantiate probem with sparse matrices
    problem = Marble.Problem(H, q, f0, J_eq, b_eq, J_ineq, b_ineq, J_comp, b_comp)
    
    @info "$(Marble.n_eq(problem)) equality constraints, $(Marble.n_ineq(problem)) inequality constraints, $(Marble.n_comp(problem)) complementarity constraints"

    Marble.set_problem!(solver, problem)

    @assert Marble.solve!(solver, Marble.SolverOptions()) "Solver failed to converge on problem $name"
    @info "\tObjective value: $(obj(Marble.z(Marble.get_workspace(solver))))"
end

## Run the function for MacMPEC LCQP problems

json_path = joinpath(@__DIR__, "../data/macmpec/macmpec_lcqp.json")
json_data = JSON.parsefile(json_path)

for key in sort(collect(keys(json_data)))
    @info "Solving MacMPEC problem: $key"
    solve_macmpec_lcqp(key)
end