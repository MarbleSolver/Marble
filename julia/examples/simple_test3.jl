using Pkg; Pkg.activate(@__DIR__)
using Revise
using JuMP, Marble, NLPModelsJuMP

# Construct problem using JuMP
model = JuMP.Model()

@variable(model, x[1:2])
@objective(model, Min, 1/2 * x'*x)
@constraint(model, cc1_lhs, x[1] >= 0)
@constraint(model, cc1_rhs, x[2] >= 0)
@constraint(model, cc2_lhs, 1 - x[1] >= 0)
@constraint(model, cc2_rhs, 1 - x[2] >= 0)

mpcc = MPCC(model,
    (model[:cc1_lhs], model[:cc1_rhs]),
    (model[:cc2_lhs], model[:cc2_rhs])
)

solver = Marble.Solver()
Marble.setup!(solver, mpcc; verbosity = 1, relaxation_initial=0.1)
results = Marble.solve!(solver)
z = Marble.z(results)

println(z)
println(Marble.obj(solver, z))
println(Marble.residual_eq(solver, z))
println(Marble.residual_ineq(solver, z))
println(Marble.residual_comp(solver, z))