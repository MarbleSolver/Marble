# Run all of the MacMPEC LCQP problems with Marble and extract the results

using MAT, SparseArrays
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
    # Load data from .mat file
    data = matread(joinpath(@__DIR__, "..", "data", "macmpec", "mat", name * ".mat"))

    atleast1d(x) = ndims(x) == 0 ? [x] : x

    # Cost terms
    f0 = data["f0"]
    q = atleast1d(data["q"])
    H = data["H"]

    obj(z) = 0.5 * z' * H * z + q' * z + f0

    # Constraint Jacobians/affine terms
    J_eq = data["J_eq"]
    b_eq = atleast1d(data["b_eq"])
    J_ineq = data["J_ineq"]
    b_ineq = atleast1d(data["b_ineq"])
    J_comp = data["J_comp"]
    b_comp = atleast1d(data["b_comp"])

    # Create an instance of Marble and solve the problem
    solver = Marble.Solver()

    # Instantiate probem with sparse matrices
    problem = Marble.Problem(
        size(q,      1), H.colptr, H.rowval, H.nzval, q, f0,
        size(J_eq,   1), J_eq.colptr,   J_eq.rowval,   J_eq.nzval,   b_eq,
        size(J_ineq, 1), J_ineq.colptr, J_ineq.rowval, J_ineq.nzval, b_ineq,
        size(J_comp, 1), J_comp.colptr, J_comp.rowval, J_comp.nzval, b_comp)
        
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