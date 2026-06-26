# Construct a simple test problem using JuMP. The problem is written as:
# min   x'*x
# s.t.  x[1] = 1
#       x[2] ≥ 1
#       0 ≤ (x[3] + 1) ⟂ (x[4] - 1) ≥ 0
#       w ≥ 10

using Pkg; Pkg.activate(@__DIR__)
using Revise
using JuMP, Marble, NLPModelsJuMP

# Construct problem using JuMP
model = JuMP.Model()

@variable(model, x[1:4])
@variable(model, w)
@objective(model, Min, x'*x)
@constraint(model, x[1] == 1)
@constraint(model, x[2] >= 1)
@constraint(model, w >= 10)

# For complementarities, we define each inequality and then specify indices
# for the complementarity pairs
@constraint(model, x3_comp, x[3] + 1 >= 0)
@constraint(model, x4_comp, x[4] - 1 >= 0)

ind_cc1, ind_cc2, cc_types = complementarity_indices(model,
    (model[:x3_comp], model[:x4_comp])
)

solver = Marble.Solver()
Marble.setup!(solver, model, ind_cc1, ind_cc2, cc_types; verbosity = 1)
results = Marble.solve!(solver)
z = Marble.z(results)

println(z)
println(Marble.obj(solver, z))
println(Marble.residual_eq(solver, z))
println(Marble.residual_ineq(solver, z))
println(Marble.residual_comp(solver, z))