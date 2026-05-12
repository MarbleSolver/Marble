using Marble
using MPCCModels: MPCCModelVarCon
using AmplNLReader

function mpcc_from_ampl(ampl::AmplNLReader.AmplModel)
    # First we find the nonzero elements in the cvar vector:
    # cvar = 0 → normal nonlinear constraint
    # cvar ≠ 0 → complementarity with the cvar-th variable
    ind_ccc2 = findall(!iszero, ampl.meta.cvar)
    # Then we store the corresponding variables
    ind_vcc1 = ampl.meta.cvar[ind_ccc2]
    return MPCCModelVarCon(ampl, ind_vcc1, ind_ccc2)
end

name = "bard1"
nl_path = joinpath(@__DIR__, "..", "..", "examples", "data", "macmpec", "ampl_nl", name * ".nl")
ampl_model = AmplModel(nl_path)
mpcc_model = mpcc_from_ampl(ampl_model)
marble_data = from_mpcc(mpcc_model)
marble_data2 = from_mpcc(ampl_model)