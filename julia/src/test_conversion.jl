using AmplNLReader
using JuMP, AmplNLWriter, Gurobi
using Marble



nlp = AmplModel(joinpath(@__DIR__, "..", "..", "examples", "data", "macmpec", "ampl_nl", "bard1.nl"))
data = from_NLPModel(nlp)
problem = Marble.Problem(data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)

# model = _marbledata_to_jump(data; comp=COMP_SOS1)

# const GUROBI_AMPL = "/Applications/AMPL/gurobi"
# set_optimizer(model, () -> Gurobi.Optimizer())
# # set_optimizer_attribute(model, "OutputFlag", 1)
# optimize!(model)

# status = termination_status(model)
# value.(model[:x])
# primal  = primal_status(model)