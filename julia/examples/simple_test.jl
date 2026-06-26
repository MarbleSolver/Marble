# Construct a simple test problem using JuMP. The problem is written as:
# min   x'*x
# s.t.  x[1] = 1
#       x[2] ≥ 1
#       0 ≤ (x[3] + 1) ⟂ (x[4] - 1) ≥ 0
# The solution is x = [1, 1, 0, 1] with an objective of 3
using Pkg; Pkg.activate(@__DIR__)
using Revise
using JuMP, Marble, NLPModelsJuMP

# Construct problem using JuMP
model = JuMP.Model()

@variable(model, x[1:4])
@objective(model, Min, x'*x)
@constraint(model, x[1] == 1)
@constraint(model, x[2] >= 1)

# For complementarities, we define each inequality and then specify indices
# for the complementarity pairs
@constraint(model, x3_comp, x[3] + 1 >= 0)
@constraint(model, x4_comp, x[4] - 1 >= 0)

mpcc = MPCC(model,
    (model[:x3_comp], model[:x4_comp])
)

solver = Marble.Solver()
Marble.setup!(solver, mpcc; verbosity = 1)
results = Marble.solve!(solver)
z = Marble.z(results)

println(z)
println(Marble.obj(solver, z))
println(Marble.residual_eq(solver, z))
println(Marble.residual_ineq(solver, z))
println(Marble.residual_comp(solver, z))