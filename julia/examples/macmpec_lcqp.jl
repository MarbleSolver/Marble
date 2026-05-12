# Run all of the MacMPEC QPCC problems with Marble and extract the results

using HDF5
using JSON
using Marble
using AmplNLReader
using MPCCModels

## Function to solve an individual problem
function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MPCCModels.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

function solve_macmpec_lcqp(name)
    nl_path = joinpath(@__DIR__, "..", "..", "examples", "data", "macmpec", "ampl_nl", name * ".nl")
    ampl_model = AmplModel(nl_path)
    mpcc_model = mpcc_from_ampl(ampl_model)
    data = from_mpcc(mpcc_model)

    # Instantiate probem
    problem = Marble.Problem(data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)

    # Create an instance of Marble and solve the problem
    solver = Marble.Solver()
    Marble.set_problem!(solver, problem)

    opts = Marble.SolverOptions()
    Marble.verbosity!(opts, true)

    result = Marble.solve!(solver, opts)
    @assert Marble.converged(result) "Solver failed to converge on problem $name"

    # @info("Solving MacMPEC problem: $name",
    #     n_eq = Marble.n_eq(problem),
    #     n_ineq = Marble.n_ineq(problem),
    #     n_comp = Marble.n_comp(problem),
    #     iterations = Marble.iterations(result),
    #     objective = Marble.obj(data, Marble.z(result)),
    # )
end

## Run the function for MacMPEC QPCC problems

json_path = joinpath(@__DIR__, "../../examples/data/macmpec/macmpec_lcqp.json")
json_data = JSON.parsefile(json_path)

for key in sort(collect(keys(json_data)))[end:end]
    solve_macmpec_lcqp(key)
end

@info("All MacMPEC QPCC problems solved successfully")