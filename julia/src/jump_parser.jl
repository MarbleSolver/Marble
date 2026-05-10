using JuMP, NLPModels
using LinearAlgebra

@enum CompFormulation COMP_PERP COMP_SOS1

function _marbledata_to_jump(data::MarbleData;
                             comp::CompFormulation=COMP_PERP)::JuMP.Model
    model = JuMP.Model()
    nvar  = length(data.q)
    ncc   = length(data.l)

    @variable(model, x[1:nvar])

    # Objective: Min c0 + q'x + 0.5 x'Qx
    @objective(model, Min, data.c0 + data.q' * x + 0.5 * x' * data.Q * x)

    # Equality: J_eq * x + b_eq = 0
    if !isempty(data.J_eq)
        @constraint(model, data.J_eq * x + data.b_eq .== 0)
    end

    # Inequality: J_ineq * x + b_ineq >= 0
    if !isempty(data.J_ineq)
        @constraint(model, data.J_ineq * x + data.b_ineq .>= 0)
    end

    if ncc > 0
        # Auxiliary variable s = R*x + r so both comp sides are named variables
        @variable(model, s[1:ncc])
        @constraint(model, s .== data.R * x + data.r)
        @constraint(model, data.L * x + data.l .>= 0)
        @constraint(model, s .>= 0)

        if comp == COMP_PERP
            # (L*x + l) ⟂ s
            @constraint(model, (data.L * x + data.l) ⟂ s)

        else  # COMP_SOS1
            # Each pair (g_k, s_k) is an SOS1 set: at most one can be nonzero
            g = data.L * x + data.l
            for k in 1:ncc
                @constraint(model, [g[k], s[k]] in SOS1())
            end
        end
    end

    return model
end

to_JuMP(nlp::NLPModels.AbstractNLPModel)::JuMP.Model = _marbledata_to_jump(from_NLPModel(nlp))

# # function parse_model(model::JuMP.Model)

# const GUROBI_AMPL = "/Applications/AMPL/gurobi"

# model = Model()

# @variable(model, x[1:4])

# @objective(model, Min, 1/2 * x' * x)
# @constraint(model, sum(x) == 1)
# @constraint(model, x[1:2] ⟂ x[3:4])
# @constraint(model, x .>= 0.0)
# # @constraint(model, x in Nonnegatives())
# # @constraint(model, x[2]^2 <= 1)

# # # Inner model that will store the scalarized reformulation
# # inner = MOIU.Model{Float64}()

# # # Force ScalarizeBridge
# # bridged = MOIB.Constraint.SingleBridgeOptimizer{
# #     MOIB.Constraint.ScalarizeBridge{Float64}
# # }(inner)

# # # Copy the JuMP model into the bridged optimizer
# # MOI.copy_to(bridged, JuMP.backend(model))

# # Verify that the model is in a form that we can handle

# @assert objective_function_type(model) ∈ [AffExpr, QuadExpr] "Objective function is not affine or quadratic"
# @assert objective_sense(model) == MOI.MIN_SENSE "Objective must be a minimization problem"

# const GEQ_SETS  = [MOI.GreaterThan, MOI.Nonnegatives]
# const LEQ_SETS  = [MOI.LessThan, MOI.Nonpositives]
# const EQ_SETS   = [MOI.EqualTo, MOI.Zeros]
# const COMP_SETS = [MOI.Complements]
# const VALID_SETS = [GEQ_SETS; LEQ_SETS; EQ_SETS; COMP_SETS]

# cons = all_constraints(model; include_variable_in_set_constraints=true)
# for c in cons
#     obj = constraint_object(c)
#     etype = typeof(obj.func)
#     ctype = typeof(obj.set)

#     @assert any(ctype <: T for T in VALID_SETS) "Constraint $c has unsupported set type $ctype"
#     @assert etype <: Union{AffExpr, Vector{AffExpr}, VariableRef, Vector{VariableRef}} "Constraint $c has unsupported expression type $etype"
# end

# vars = all_variables(model)
# n_vars = length(vars)
# var_idx = Dict(v => i for (i, v) in enumerate(vars))

# function extract_row(expr, n_vars, var_idx)
#     row = zeros(n_vars)
#     if expr isa AffExpr
#         for (var, coeff) in expr.terms
#             row[var_idx[var]] = coeff
#         end
#     elseif expr isa VariableRef
#         row[var_idx[expr]] = 1.0
#     end
#     return row
# end

# expr_constant(expr) = expr isa AffExpr ? expr.constant : 0.0

# function to_variable_ref(expr)
#     expr isa VariableRef && return expr
#     if expr isa AffExpr && length(expr.terms) == 1
#         var, coeff = first(expr.terms)
#         coeff == 1.0 && iszero(expr.constant) && return var
#     end
#     error("Both sides of a complementarity constraint must be variable references, got: $(expr)")
# end

# # First pass: collect all variables appearing in complementarity constraints,
# # asserting both sides are pure variable references
# comp_vars = Set{VariableRef}()
# for c in cons
#     obj = constraint_object(c)
#     obj.set isa MOI.Complements || continue
#     n_pairs = div(obj.set.dimension, 2)
#     for i in 1:n_pairs
#         push!(comp_vars, to_variable_ref(obj.func[i]))
#         push!(comp_vars, to_variable_ref(obj.func[n_pairs + i]))
#     end
# end

# # Returns true if expr is a single comp variable with no offset (x >= 0 bound)
# function is_comp_var_nonneg(expr, comp_vars)
#     expr isa VariableRef && return expr ∈ comp_vars
#     expr isa AffExpr || return false
#     length(expr.terms) == 1 || return false
#     var, coeff = first(expr.terms)
#     return coeff == 1.0 && iszero(expr.constant) && var ∈ comp_vars
# end

# eq_rows,    eq_rhs   = Vector{Float64}[], Float64[]
# ineq_rows,  ineq_rhs = Vector{Float64}[], Float64[]
# comp_rows_f, m_comp  = Vector{Float64}[], Float64[]
# comp_rows_g, n_comp  = Vector{Float64}[], Float64[]

# # Second pass: build Jacobians
# # Convention: A_eq*x + b_eq = 0, A_ineq*x + b_ineq >= 0
# for c in cons
#     obj = constraint_object(c)
#     set, func = obj.set, obj.func

#     if set isa MOI.EqualTo
#         push!(eq_rows, extract_row(func, n_vars, var_idx))
#         push!(eq_rhs, expr_constant(func) - set.value)

#     elseif set isa MOI.Zeros
#         for f in func
#             push!(eq_rows, extract_row(f, n_vars, var_idx))
#             push!(eq_rhs, expr_constant(f))
#         end

#     elseif set isa MOI.GreaterThan
#         # Drop 0 <= x_i for any variable that appears in a complementarity constraint
#         iszero(set.lower) && is_comp_var_nonneg(func, comp_vars) && continue
#         push!(ineq_rows, extract_row(func, n_vars, var_idx))
#         push!(ineq_rhs, expr_constant(func) - set.lower)

#     elseif set isa MOI.Nonnegatives
#         for f in func
#             is_comp_var_nonneg(f, comp_vars) && continue
#             push!(ineq_rows, extract_row(f, n_vars, var_idx))
#             push!(ineq_rhs, expr_constant(f))
#         end

#     elseif set isa MOI.LessThan
#         # Negate to convert a'x + c <= u  →  -a'x + (u - c) >= 0
#         push!(ineq_rows, -extract_row(func, n_vars, var_idx))
#         push!(ineq_rhs, set.upper - expr_constant(func))

#     elseif set isa MOI.Nonpositives
#         for f in func
#             push!(ineq_rows, -extract_row(f, n_vars, var_idx))
#             push!(ineq_rhs, -expr_constant(f))
#         end

#     elseif set isa MOI.Complements
#         n_pairs = div(set.dimension, 2)
#         for i in 1:n_pairs
#             push!(comp_rows_f, extract_row(obj.func[i], n_vars, var_idx))
#             push!(m_comp, expr_constant(obj.func[i]))
#             push!(comp_rows_g, extract_row(obj.func[n_pairs + i], n_vars, var_idx))
#             push!(n_comp, expr_constant(obj.func[n_pairs + i]))
#         end
#     end
# end

# to_matrix(rows) = isempty(rows) ? zeros(0, n_vars) : mapreduce(transpose, vcat, rows)

# A_eq   = to_matrix(eq_rows);   b_eq   = eq_rhs
# A_ineq = to_matrix(ineq_rows); b_ineq = ineq_rhs
# M_comp = to_matrix(comp_rows_f)
# N_comp = to_matrix(comp_rows_g)