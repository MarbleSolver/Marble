# using JuMP, AmplNLWriter

# const GUROBI_AMPL = "/Applications/AMPL/gurobi"

# model = Model(() -> AmplNLWriter.Optimizer(GUROBI_AMPL))

# @variable(model, x[1:3])

# @objective(model, Min, 1/2 * x' * x)
# @constraint(model, sum(x) == 1)
# @constraint(model, 2*x[1]+1 ⟂ x[2])
# @constraint(model, [x[1], x[2]] .>= 0)
# # @complementarity(model, x[1], x[2])
# # @objective(model, Min, sum((x[i] - i)^2 for i in 1:3))
# # @constraint(model, sum(x) == 1)

# optimize!(model)
# @show value.(x)

# comps = all_constraints(model, Vector{AffExpr}, MOI.Complements)

# for c in comps
#     obj = constraint_object(c)
#     n = obj.set.dimension  # number of complementarity pairs
#     lhs = obj.func[1:n-1]
#     rhs = obj.func[n:end]
#     println("$lhs ⟂ $rhs")
# end

using JuMP, Complementarity
m = Model()
@variable(m, x>=0)
@objective(m, Min, x^2)
@complements(m, 0 <= x+2,   x >= 0, simple)
# solve(m)
# @show getvalue(x)