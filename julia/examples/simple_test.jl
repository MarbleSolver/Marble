# Construct a simple test problem using JuMP minimizing x'x
# where x[1] = 1
#       x[2] ≥ 1
#       0 ≤ (x[3] + 1) ⟂ (x[4] - 1) ≥ 0 --> solution is x[3] = 0, x[4] = 1
using Pkg; Pkg.activate(@__DIR__)
using JuMP, Marble, NLPModelsJuMP

# Construct problem using JuMP
model = JuMP.Model()

@variable(model, x[1:4])
@objective(model, Min, x'*x)
@constraint(model, x[1] == 1)
@constraint(model, x[2] >= 1)

# For complementarities, we define each inequality and then specify indices
# for the complementarity pairs
@constraint(model, x3_comp, x[3] + 1 >= 0) # x3_comp a constraint variable
@constraint(model, x4_comp, x[4] - 1 >= 0)
comps = con_con_complementarities(model, [x3_comp,], [x4_comp,])

# Construct a marble problem struct (completely unclear why we do this in Julia)
marble_data = MarbleData(MathOptNLPModel(model), first.(comps), last.(comps), [(:con, :con),])

# Solve the problem (again unclear why we have a wrapper function for this)
opts = Marble.SolverOptions()
Marble.verbosity!(opts, 1)
result = Marble.solve(marble_data; opts = opts)
println(Marble.z(result))
