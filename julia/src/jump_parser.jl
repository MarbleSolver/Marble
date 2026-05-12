using JuMP, NLPModels
using LinearAlgebra

@enum CompFormulation COMP_PERP COMP_SOS1

# Convert a MarbleData (canonical QPCC form) to a JuMP model.
# Introduces explicit slack variables s1 = L*x+l and s2 = R*x+r for the two
# complementarity sides so the same comp formulation options apply.
function Base.convert(::Type{JuMP.Model}, data::MarbleData)::JuMP.Model
    nvar = length(data.q)
    ncc  = isnothing(data.l) ? 0 : length(data.l)

    model = JuMP.Model()
    @variable(model, x[1:nvar])

    @objective(model, Min, data.c0 + dot(data.q, x) + 0.5 * dot(x, data.Q * x))

    if !isnothing(data.J_eq) && !isempty(data.J_eq)
        @constraint(model, data.J_eq * x + data.b_eq .== 0)
    end
    if !isnothing(data.J_ineq) && !isempty(data.J_ineq)
        @constraint(model, data.J_ineq * x + data.b_ineq .>= 0)
    end

    if ncc > 0
        @variable(model, s1[1:ncc] >= 0)
        @variable(model, s2[1:ncc] >= 0)
        @constraint(model, s1 .== data.L * x + data.l)
        @constraint(model, s2 .== data.R * x + data.r)

        for k in 1:ncc
            @constraint(model, [s1[k], s2[k]] in SOS1())
        end
    end

    return model
end