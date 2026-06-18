
using Pkg; Pkg.activate(@__DIR__)
using Revise
using JuMP, Marble, NLPModelsJuMP
using SparseArrays

# Construct problem using JuMP
model = JuMP.Model()

@variable(model, x[1:4])
@variable(model, b1 in Parameter(1.0))    # value x[1] must equal
@variable(model, b2 in Parameter(2.0))    # lower bound on x[2]
@variable(model, b3 in Parameter(3.0))

@objective(model, Min, x'*x)
@constraint(model, x1_eq, 2 * b3 * x[1] == b1)
@constraint(model, x2_ineq, x[2] >= b2)

p' * H * x 

"""
y = [x1 x2 x3 x4 b1 b2 b3]

y' * H * y = 

H_11 * x1^2 + H_22 * x2^2 + H_33 * x3^2 + H_44 * x4^2 + H_55 * b1^2 + H_66 * b2^2 + H_77 * b3^2 + 
2 * H_12 * x1 * x2 + 2 * H_13 * x1 * x3 + 2 * H_14 * x1 * x4 + 2 * H_15 * x1 * b1 + 2 * H_16 * x1 * b2 + 2 * H_17 * x1 * b3 +
2 * H_23 * x2 * x3 + 2 * H_24 * x2 * x4 + 2 * H_25 * x2 * b1 + 2 * H_26 * x2 * b2 + 2 * H_27 * x2 * b3 +
2 * H_34 * x3 * x4 + 2 * H_35 * x3 * b1 + 2 * H_36 * x3 * b2 + 2 * H_37 * x3 * b3 +
2 * H_45 * x4 * b1 + 2 * H_46 * x4 * b2 + 2 * H_47 * x4 * b3 +
2 * H_56 * b1 * b2 + 2 * H_57 * b1 * b3 +
2 * H_67 * b2 * b3

"""

# For complementarities, we define each inequality and then specify indices
# for the complementarity pairs
@constraint(model, x3_comp, x[3] + 1 >= 0)
@constraint(model, x4_comp, x[4] - 1 >= 0)

function build_evaluator(model::Model)
    rows = ConstraintRef[]
    nlp = MOI.Nonlinear.Model()
    for (F, S) in list_of_constraint_types(model)
        F == VariableRef && continue          # skip variable bounds
        for ci in all_constraints(model, F, S)
            push!(rows, ci)
            obj = constraint_object(ci)
            MOI.Nonlinear.add_constraint(nlp, obj.func, obj.set)
        end
    end
    MOI.Nonlinear.set_objective(nlp, objective_function(model))
    x = all_variables(model)
    evaluator = MOI.Nonlinear.Evaluator(
        nlp,
        MOI.Nonlinear.SparseReverseMode(),
        index.(x),                            # defines column ordering
    )
    MOI.initialize(evaluator, [:Grad, :Jac, :Hess])
    return evaluator, x, rows
end

evaluator, x, rows = build_evaluator(model)
constraint_row = Dict(ci => i for (i, ci) in enumerate(rows))

xv = ones(num_variables(model))
n, m = length(x), length(rows)

# Objective gradient ∇f
g = zeros(n)
MOI.eval_objective_gradient(evaluator, g, xv)

# Constraint Jacobian J_g  (m × n)
Js = MOI.jacobian_structure(evaluator)        # Vector{Tuple{Int,Int}} (row, col)
Jv = zeros(length(Js))
MOI.eval_constraint_jacobian(evaluator, Jv, xv)
J = SparseArrays.sparse(first.(Js), last.(Js), Jv, m, n)

param_cols = findall(is_parameter, x)
decision_cols = findall(!is_parameter, x)

param_values_dict = Dict(v => parameter_value(v) for v in all_variables(model) if is_parameter(v))
param_values = get.(Ref(param_values_dict), x[param_cols], 0.0)


# Hessian of the cost fn
Hs = MOI.hessian_objective_structure(evaluator)
Hv = zeros(length(Hs))
MOI.eval_hessian_objective(evaluator, Hv, xv)
H = SparseArrays.sparse(first.(Hs), last.(Hs), Hv, n, n)