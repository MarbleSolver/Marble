using Marble
using AmplNLReader
using MPCCModels

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MPCCModels.MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

name = "bard1"
nl_path = joinpath(@__DIR__, "..", "..", "examples", "data", "macmpec", "ampl_nl", name * ".nl")
ampl = AmplModel(nl_path)
mpcc = mpcc_from_ampl(ampl)
marbledata = from_mpcc(mpcc)

##
using JuMP, Gurobi

jump_model = convert(JuMP.Model, marbledata)
set_optimizer(jump_model, Gurobi.Optimizer)
optimize!(jump_model)

# nlp  = mpcc.nlp
# nvar = nlp.meta.nvar
# ncon = nlp.meta.ncon
# lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
# lcon, ucon = nlp.meta.lcon, nlp.meta.ucon

# ind_cc1  = mpcc.meta.ind_cc1
# ind_cc2  = mpcc.meta.ind_cc2
# cc_types = mpcc.meta.cc_types
# ncc      = mpcc.meta.ncc

# model = JuMP.Model()

# @variable(model, lvar[i] <= x[i=1:nvar] <= uvar[i])

# @objective(model, nlp.meta.minimize ? Min : Max, NLPModels.obj(nlp, x))