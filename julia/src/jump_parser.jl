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

# # Partition MPCCModel indices into CC-involved vs regular sets.
# function _cc_index_sets(mpcc::MPCCModel)
#     nlp = mpcc.nlp
#     nvar, ncon = nlp.meta.nvar, nlp.meta.ncon
#     ind_cc1, ind_cc2, cc_types, ncc =
#         mpcc.meta.ind_cc1, mpcc.meta.ind_cc2, mpcc.meta.cc_types, mpcc.meta.ncc

#     comp_con = Set{Int}(); comp_var = Set{Int}()
#     for i in 1:ncc
#         t = cc_types[i]
#         (t == VarVar || t == VarCon) ? push!(comp_var, ind_cc1[i]) : push!(comp_con, ind_cc1[i])
#         (t == VarVar || t == ConVar) ? push!(comp_var, ind_cc2[i]) : push!(comp_con, ind_cc2[i])
#     end
#     reg_con = [i for i in 1:ncon if i ∉ comp_con]
#     reg_var = [i for i in 1:nvar if i ∉ comp_var]
#     return reg_con, reg_var
# end

# # Algebraic (QPCC) formulation: linearizes constraints and uses Hessian at x₀=0 for
# # the objective. Produces a JuMP model with only affine/quadratic expressions, which
# # QP-based solvers (Gurobi, CPLEX, Mosek) can accept. Only exact for LP/QPCCs.
# function _to_JuMP_algebraic(mpcc::MPCCModel, comp::CompFormulation)::JuMP.Model
#     nlp  = mpcc.nlp
#     x0   = zeros(nlp.meta.nvar)
#     nvar = nlp.meta.nvar
#     ncon = nlp.meta.ncon
#     lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
#     lcon, ucon = nlp.meta.lcon, nlp.meta.ucon
#     ind_cc1, ind_cc2, cc_types, ncc =
#         mpcc.meta.ind_cc1, mpcc.meta.ind_cc2, mpcc.meta.cc_types, mpcc.meta.ncc

#     J0     = Matrix(NLPModels.jac(nlp, x0))
#     H0     = Matrix(NLPModels.hess(nlp, x0))
#     b_full = NLPModels.cons(nlp, x0)

#     reg_con, reg_var = _cc_index_sets(mpcc)

#     model = JuMP.Model()
#     @variable(model, x[1:nvar])

#     obj_sign = nlp.meta.minimize ? 1.0 : -1.0
#     Q  = obj_sign * H0
#     q  = obj_sign * NLPModels.grad(nlp, x0)
#     c0 = obj_sign * NLPModels.obj(nlp, x0)
#     @objective(model, Min, c0 + dot(q, x) + 0.5 * dot(x, Q * x))

#     for i in reg_var
#         isfinite(lvar[i]) && set_lower_bound(x[i], lvar[i])
#         isfinite(uvar[i]) && set_upper_bound(x[i], uvar[i])
#     end

#     for i in reg_con
#         expr = dot(J0[i, :], x) + b_full[i]
#         l, u = lcon[i], ucon[i]
#         if isfinite(l) && l == u
#             @constraint(model, expr == l)
#         else
#             isfinite(l) && @constraint(model, expr >= l)
#             isfinite(u) && @constraint(model, expr <= u)
#         end
#     end

#     ncc == 0 && return model

#     @variable(model, s1[1:ncc] >= 0)
#     @variable(model, s2[1:ncc] >= 0)

#     function add_slack!(s_var, is_var::Bool, idx::Int)
#         if is_var
#             l, u = lvar[idx], uvar[idx]
#             isfinite(l) || isfinite(u) || error("CC variable $idx has no finite bound")
#             if isfinite(l)
#                 @constraint(model, s_var == x[idx] - l)
#                 isfinite(u) && @constraint(model, x[idx] <= u)
#             else
#                 @constraint(model, s_var == u - x[idx])
#             end
#         else
#             expr = dot(J0[idx, :], x) + b_full[idx]
#             l, u = lcon[idx], ucon[idx]
#             isfinite(l) || isfinite(u) || error("CC constraint $idx has no finite bound")
#             if isfinite(l)
#                 @constraint(model, s_var == expr - l)
#                 isfinite(u) && @constraint(model, expr <= u)
#             else
#                 @constraint(model, s_var == u - expr)
#             end
#         end
#     end

#     for i in 1:ncc
#         t = cc_types[i]
#         add_slack!(s1[i], t == VarVar || t == VarCon, ind_cc1[i])
#         add_slack!(s2[i], t == VarVar || t == ConVar, ind_cc2[i])
#     end

#     if comp == COMP_PERP
#         @constraint(model, s1 ⟂ s2)
#     else
#         for k in 1:ncc; @constraint(model, [s1[k], s2[k]] in SOS1()); end
#     end

#     return model
# end

# # Nonlinear formulation: wraps NLPModels functions as JuMP nonlinear operators so the
# # full nonlinear MPCC is preserved. Requires a solver that supports UserDefinedFunction
# # (e.g., Ipopt, KNITRO) — QP solvers like Gurobi will reject this form.
# function _to_JuMP_nonlinear(mpcc::MPCCModel, comp::CompFormulation)::JuMP.Model
#     nlp  = mpcc.nlp
#     nvar = nlp.meta.nvar
#     ncon = nlp.meta.ncon
#     lvar, uvar = nlp.meta.lvar, nlp.meta.uvar
#     lcon, ucon = nlp.meta.lcon, nlp.meta.ucon
#     ind_cc1, ind_cc2, cc_types, ncc =
#         mpcc.meta.ind_cc1, mpcc.meta.ind_cc2, mpcc.meta.cc_types, mpcc.meta.ncc

#     reg_con, reg_var = _cc_index_sets(mpcc)

#     model = JuMP.Model()
#     @variable(model, x[1:nvar])

#     op_obj = add_nonlinear_operator(
#         model, nvar,
#         (xv...) -> NLPModels.obj(nlp, collect(Float64, xv)),
#         (g, xv...) -> (NLPModels.grad!(nlp, collect(Float64, xv), g); nothing);
#         name = :nlp_obj,
#     )
#     @objective(model, nlp.meta.minimize ? Min : Max, op_obj(x...))

#     # Lazily register c_i(x) as a JuMP operator; gradient via jtprod! with eᵢ.
#     con_ops = Dict{Int, Any}()
#     function get_con_op(i::Int)
#         haskey(con_ops, i) && return con_ops[i]
#         ei   = (e = zeros(ncon); e[i] = 1.0; e)
#         f_i  = let i = i;  (xv...) -> NLPModels.cons(nlp, collect(Float64, xv))[i]; end
#         g!_i = let ei = ei; (g, xv...) -> (NLPModels.jtprod!(nlp, collect(Float64, xv), ei, g); nothing); end
#         return con_ops[i] = add_nonlinear_operator(model, nvar, f_i, g!_i; name = Symbol("nlp_con_", i))
#     end

#     for i in reg_var
#         isfinite(lvar[i]) && set_lower_bound(x[i], lvar[i])
#         isfinite(uvar[i]) && set_upper_bound(x[i], uvar[i])
#     end

#     for i in reg_con
#         op_i = get_con_op(i)
#         l, u = lcon[i], ucon[i]
#         if isfinite(l) && l == u
#             @constraint(model, op_i(x...) == l)
#         else
#             isfinite(l) && @constraint(model, op_i(x...) >= l)
#             isfinite(u) && @constraint(model, op_i(x...) <= u)
#         end
#     end

#     ncc == 0 && return model

#     @variable(model, s1[1:ncc] >= 0)
#     @variable(model, s2[1:ncc] >= 0)

#     function add_slack!(s_var, is_var::Bool, idx::Int)
#         if is_var
#             l, u = lvar[idx], uvar[idx]
#             isfinite(l) || isfinite(u) || error("CC variable $idx has no finite bound")
#             if isfinite(l)
#                 @constraint(model, s_var == x[idx] - l)
#                 isfinite(u) && @constraint(model, x[idx] <= u)
#             else
#                 @constraint(model, s_var == u - x[idx])
#             end
#         else
#             op_i = get_con_op(idx)
#             l, u = lcon[idx], ucon[idx]
#             isfinite(l) || isfinite(u) || error("CC constraint $idx has no finite bound")
#             if isfinite(l)
#                 @constraint(model, s_var == op_i(x...) - l)
#                 isfinite(u) && @constraint(model, op_i(x...) <= u)
#             else
#                 @constraint(model, s_var == u - op_i(x...))
#             end
#         end
#     end

#     for i in 1:ncc
#         t = cc_types[i]
#         add_slack!(s1[i], t == VarVar || t == VarCon, ind_cc1[i])
#         add_slack!(s2[i], t == VarVar || t == ConVar, ind_cc2[i])
#     end

#     if comp == COMP_PERP
#         @constraint(model, s1 ⟂ s2)
#     else
#         for k in 1:ncc; @constraint(model, [s1[k], s2[k]] in SOS1()); end
#     end

#     return model
# end

# # algebraic=true  → affine/quadratic expressions only; compatible with Gurobi/CPLEX/Mosek.
# #                   Only exact when the underlying NLP is an LP or QP.
# # algebraic=false → full nonlinear operators; requires Ipopt, KNITRO, or similar.
# function to_JuMP(mpcc::MPCCModel;
#                  comp::CompFormulation = COMP_PERP,
#                  algebraic::Bool = false)::JuMP.Model
#     algebraic ? _to_JuMP_algebraic(mpcc, comp) : _to_JuMP_nonlinear(mpcc, comp)
# end


# ------ Old MarbleData → JuMP reference (QPCC only) ------
# function _marbledata_to_jump(data::MarbleData;
#                              comp::CompFormulation=COMP_PERP)::JuMP.Model
#     model = JuMP.Model()
#     nvar  = length(data.q)
#     ncc   = length(data.l)
#     @variable(model, x[1:nvar])
#     @objective(model, Min, data.c0 + data.q' * x + 0.5 * x' * data.Q * x)
#     if !isempty(data.J_eq);   @constraint(model, data.J_eq * x + data.b_eq .== 0); end
#     if !isempty(data.J_ineq); @constraint(model, data.J_ineq * x + data.b_ineq .>= 0); end
#     if ncc > 0
#         @variable(model, s[1:ncc])
#         @constraint(model, s .== data.R * x + data.r)
#         @constraint(model, data.L * x + data.l .>= 0)
#         @constraint(model, s .>= 0)
#         if comp == COMP_PERP
#             @constraint(model, (data.L * x + data.l) ⟂ s)
#         else
#             g = data.L * x + data.l
#             for k in 1:ncc; @constraint(model, [g[k], s[k]] in SOS1()); end
#         end
#     end
#     return model
# end
# to_JuMP(nlp::NLPModels.AbstractNLPModel) = _marbledata_to_jump(from_mpcc(nlp))
