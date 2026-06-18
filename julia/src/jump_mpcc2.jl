struct _JumpMPCCRow
    A::Vector{Float64}
    b::Float64
    lb::Float64
    ub::Float64
end

struct _JumpMPCCEndpoint
    row::Int
    side::Symbol
    label::String
end

struct JumpMPCCModel
    model::JuMP.Model
    vars::Vector{JuMP.VariableRef}
    col_map::Dict{MOI.VariableIndex,Int}
    constraints::Vector{Any}
    row_by_constraint::Dict{Any,Int}
    endpoints1::Vector{_JumpMPCCEndpoint}
    endpoints2::Vector{_JumpMPCCEndpoint}
end

function _jump_var_col_map(model::JuMP.Model)
    vars = [v for v in JuMP.all_variables(model) if !JuMP.is_parameter(v)]
    return vars, Dict(JuMP.index(v) => i for (i, v) in enumerate(vars))
end

_set_bounds(set::MOI.GreaterThan) = (Float64(set.lower), Inf)
_set_bounds(set::MOI.LessThan) = (-Inf, Float64(set.upper))
_set_bounds(set::MOI.EqualTo) = (Float64(set.value), Float64(set.value))
_set_bounds(set::MOI.Interval) = (Float64(set.lower), Float64(set.upper))
_set_bounds(set) = error("Unsupported JuMP constraint set $(typeof(set)); expected ==, >=, <=, or interval bounds")

struct _JumpPolynomial
    c::Float64
    q::Vector{Float64}
    Q::Matrix{Float64}
end

_zero_poly(nvar::Int) = _JumpPolynomial(0.0, zeros(Float64, nvar), zeros(Float64, nvar, nvar))
_const_poly(c::Real, nvar::Int) = _JumpPolynomial(Float64(c), zeros(Float64, nvar), zeros(Float64, nvar, nvar))

function _degree(p::_JumpPolynomial)
    any(!iszero, p.Q) && return 2
    any(!iszero, p.q) && return 1
    return 0
end

function _scale_poly(p::_JumpPolynomial, α::Real)
    α = Float64(α)
    return _JumpPolynomial(α * p.c, α .* p.q, α .* p.Q)
end

function _add_poly(a::_JumpPolynomial, b::_JumpPolynomial)
    return _JumpPolynomial(a.c + b.c, a.q .+ b.q, a.Q .+ b.Q)
end

_sub_poly(a::_JumpPolynomial, b::_JumpPolynomial) = _add_poly(a, _scale_poly(b, -1.0))

function _mul_poly(a::_JumpPolynomial, b::_JumpPolynomial)
    nvar = length(a.q)
    _degree(a) + _degree(b) <= 2 || error("Expression is higher than quadratic in JuMP decision variables")

    c = a.c * b.c
    q = a.c .* b.q .+ b.c .* a.q
    Q = a.c .* b.Q .+ b.c .* a.Q .+ a.q * transpose(b.q) .+ b.q * transpose(a.q)
    return _JumpPolynomial(c, q, Q)
end

function _pow_poly(base::_JumpPolynomial, exponent::_JumpPolynomial)
    _degree(exponent) == 0 || error("Polynomial exponents cannot depend on JuMP decision variables")
    p = exponent.c
    isinteger(p) || error("Only integer powers are supported in JuMP expressions")
    n = Int(p)
    0 <= n <= 2 || error("Only powers 0, 1, and 2 are supported in JuMP expressions")
    n == 0 && return _const_poly(1.0, length(base.q))
    n == 1 && return base
    return _mul_poly(base, base)
end

function _parameter_value_or_error(v::JuMP.VariableRef)
    JuMP.is_parameter(v) && return JuMP.parameter_value(v)
    error("Expression contains decision variable $v inside a nonlinear coefficient")
end

function _parameter_only_value(expr)
    return Float64(JuMP.value(_parameter_value_or_error, expr))
end

function _poly_from_variable(var::JuMP.VariableRef, col_map, nvar::Int)
    if JuMP.is_parameter(var)
        return _const_poly(JuMP.parameter_value(var), nvar)
    end
    i = get(col_map, JuMP.index(var), nothing)
    isnothing(i) && error("Variable $var is not part of the Marble decision variable map")
    q = zeros(Float64, nvar)
    q[i] = 1.0
    return _JumpPolynomial(0.0, q, zeros(Float64, nvar, nvar))
end

function _poly_from_expr(expr::Real, col_map, nvar::Int)
    return _const_poly(expr, nvar)
end

function _poly_from_expr(expr::JuMP.VariableRef, col_map, nvar::Int)
    return _poly_from_variable(expr, col_map, nvar)
end

function _poly_from_expr(expr::JuMP.AffExpr, col_map, nvar::Int)
    p = _const_poly(JuMP.constant(expr), nvar)
    for (coef, var) in JuMP.linear_terms(expr)
        p = _add_poly(p, _scale_poly(_poly_from_variable(var, col_map, nvar), coef))
    end
    return p
end

function _poly_from_expr(expr::JuMP.QuadExpr, col_map, nvar::Int)
    p = _poly_from_expr(expr.aff, col_map, nvar)
    for (coef, var1, var2) in JuMP.quad_terms(expr)
        term = _mul_poly(
            _poly_from_variable(var1, col_map, nvar),
            _poly_from_variable(var2, col_map, nvar),
        )
        p = _add_poly(p, _scale_poly(term, coef))
    end
    return p
end

function _poly_from_expr(expr::JuMP.NonlinearExpr, col_map, nvar::Int)
    args = getfield(expr, :args)
    head = getfield(expr, :head)

    if head == :+
        p = _zero_poly(nvar)
        for arg in args
            p = _add_poly(p, _poly_from_expr(arg, col_map, nvar))
        end
        return p
    elseif head == :-
        if length(args) == 1
            return _scale_poly(_poly_from_expr(args[1], col_map, nvar), -1.0)
        elseif length(args) == 2
            return _sub_poly(
                _poly_from_expr(args[1], col_map, nvar),
                _poly_from_expr(args[2], col_map, nvar),
            )
        end
    elseif head == :*
        p = _const_poly(1.0, nvar)
        for arg in args
            p = _mul_poly(p, _poly_from_expr(arg, col_map, nvar))
        end
        return p
    elseif head == :/
        length(args) == 2 || error("Unsupported division expression $expr")
        numerator = _poly_from_expr(args[1], col_map, nvar)
        denominator = _poly_from_expr(args[2], col_map, nvar)
        _degree(denominator) == 0 || error("Division by expressions containing decision variables is unsupported")
        return _scale_poly(numerator, inv(denominator.c))
    elseif head == :^
        length(args) == 2 || error("Unsupported power expression $expr")
        return _pow_poly(
            _poly_from_expr(args[1], col_map, nvar),
            _poly_from_expr(args[2], col_map, nvar),
        )
    end

    try
        return _const_poly(_parameter_only_value(expr), nvar)
    catch err
        err isa ErrorException || rethrow()
        error("Unsupported nonlinear JuMP expression $expr; expression must be quadratic in decision variables after parameters are fixed")
    end
end

function _poly_from_expr(expr, ::Any, ::Int)
    error("Unsupported JuMP expression $(typeof(expr)); expected scalar affine, quadratic, or supported nonlinear expression")
end

function _affine_data(func, col_map, nvar::Int)
    p = _poly_from_expr(func, col_map, nvar)
    any(!iszero, p.Q) && error("JuMP constraints must be affine in decision variables after parameters are fixed")
    return p.q, p.c
end

function _constraint_row(con::JuMP.ConstraintRef, col_map, nvar::Int)
    obj = JuMP.constraint_object(con)
    A, b = _affine_data(obj.func, col_map, nvar)
    lb, ub = _set_bounds(obj.set)
    return _JumpMPCCRow(A, b, lb, ub)
end

function _variable_row(var::JuMP.VariableRef, col_map, nvar::Int)
    A = zeros(Float64, nvar)
    A[col_map[JuMP.index(var)]] = 1.0

    if JuMP.is_fixed(var)
        value = Float64(JuMP.fix_value(var))
        return _JumpMPCCRow(A, 0.0, value, value)
    end

    lb = JuMP.has_lower_bound(var) ? Float64(JuMP.lower_bound(var)) : -Inf
    ub = JuMP.has_upper_bound(var) ? Float64(JuMP.upper_bound(var)) : Inf
    return _JumpMPCCRow(A, 0.0, lb, ub)
end

function _constraint_side(con::JuMP.ConstraintRef)
    set = JuMP.constraint_object(con).set
    set isa MOI.GreaterThan && return :lower
    set isa MOI.LessThan && return :upper
    set isa MOI.Interval && return :auto
    set isa MOI.EqualTo && error("Complementarity endpoint $con is an equality; endpoints must be inequalities")
    error("Unsupported complementarity endpoint set $(typeof(set)) for $con")
end

function _validate_endpoint_row(row::_JumpMPCCRow, label::String)
    has_lb = isfinite(row.lb)
    has_ub = isfinite(row.ub)
    (has_lb || has_ub) || error("Complementarity endpoint $label has no finite lower or upper bound")
    !(has_lb && has_ub && row.lb == row.ub) || error(
        "Complementarity endpoint $label is an equality; endpoints must be inequalities",
    )
    return nothing
end

function _jump_endpoint(
        model::JuMP.Model,
        endpoint::JuMP.VariableRef,
        row_by_constraint,
        col_map,
        variable_row_offset::Int,
    )
    JuMP.owner_model(endpoint) === model || error("Variable endpoint $endpoint belongs to a different JuMP model")
    row = variable_row_offset + col_map[JuMP.index(endpoint)]
    return _JumpMPCCEndpoint(row, :auto, string(endpoint))
end

function _jump_endpoint(
        model::JuMP.Model,
        endpoint::JuMP.ConstraintRef,
        row_by_constraint,
        col_map,
        variable_row_offset::Int,
    )
    JuMP.owner_model(endpoint) === model || error("Constraint endpoint $endpoint belongs to a different JuMP model")
    obj = JuMP.constraint_object(endpoint)
    side = _constraint_side(endpoint)

    if obj.func isa JuMP.VariableRef
        JuMP.is_parameter(obj.func) && error("Parameter bound $endpoint cannot be a Marble complementarity endpoint")
        row = variable_row_offset + col_map[JuMP.index(obj.func)]
    else
        key = JuMP.index(endpoint)
        haskey(row_by_constraint, key) || error("Constraint endpoint $endpoint was not found in the scalar constraint rows")
        row = row_by_constraint[key]
    end

    return _JumpMPCCEndpoint(row, side, string(endpoint))
end

function _jump_endpoint(::JuMP.Model, endpoint, ::Any, ::Any, ::Int)
    error("Complementarity endpoints must be JuMP VariableRef or ConstraintRef objects, got $(typeof(endpoint))")
end

function _objective_data(model::JuMP.Model, col_map, nvar::Int)
    sense = JuMP.objective_sense(model)
    sense == MOI.FEASIBILITY_SENSE && return zeros(Float64, nvar, nvar), zeros(Float64, nvar), 0.0
    obj_sign = sense == MOI.MIN_SENSE ? 1.0 : sense == MOI.MAX_SENSE ? -1.0 :
        error("Unsupported JuMP objective sense $sense")

    poly = _poly_from_expr(JuMP.objective_function(model), col_map, nvar)
    return obj_sign .* poly.Q, obj_sign .* poly.q, obj_sign * poly.c
end

function _jump_rows(cache::JumpMPCCModel)
    rows = _JumpMPCCRow[]
    nvar = length(cache.vars)
    for con in cache.constraints
        push!(rows, _constraint_row(con, cache.col_map, nvar))
    end
    for var in cache.vars
        push!(rows, _variable_row(var, cache.col_map, nvar))
    end
    return rows
end

function _row_matrix(row_indices, rows::Vector{_JumpMPCCRow}, nvar::Int, scale = i -> 1.0)
    M = zeros(Float64, length(row_indices), nvar)
    b = zeros(Float64, length(row_indices))
    for (out_row, row_idx) in enumerate(row_indices)
        s = scale(row_idx)
        row = rows[row_idx]
        M[out_row, :] .= s .* row.A
        b[out_row] = s == 1.0 ? row.b - row.lb : row.ub - row.b
    end
    return M, b
end

function _endpoint_residual(
        endpoint::_JumpMPCCEndpoint,
        rows::Vector{_JumpMPCCRow},
        consumed_lb::AbstractVector{Bool},
        consumed_ub::AbstractVector{Bool},
    )
    row = rows[endpoint.row]
    has_lb = isfinite(row.lb)
    has_ub = isfinite(row.ub)
    side = endpoint.side == :auto ? (has_lb ? :lower : :upper) : endpoint.side

    if side == :lower
        has_lb || error("Complementarity endpoint $(endpoint.label) has no finite lower bound")
        consumed_lb[endpoint.row] = true
        return row.A, row.b - row.lb
    elseif side == :upper
        has_ub || error("Complementarity endpoint $(endpoint.label) has no finite upper bound")
        consumed_ub[endpoint.row] = true
        return -row.A, row.ub - row.b
    end

    error("Unsupported complementarity endpoint side $side")
end

function _push_residual!(Mrows, brows, A, b)
    push!(Mrows, collect(Float64, A))
    push!(brows, Float64(b))
    return nothing
end

function _rows_to_matrix(Mrows::Vector{Vector{Float64}}, brows::Vector{Float64}, nvar::Int)
    M = zeros(Float64, length(Mrows), nvar)
    for (i, row) in enumerate(Mrows)
        M[i, :] .= row
    end
    return M, copy(brows)
end

function _expand_jump_complementarity_blocks(model, blocks, row_by_constraint, col_map, variable_row_offset, rows)
    endpoints1 = _JumpMPCCEndpoint[]
    endpoints2 = _JumpMPCCEndpoint[]

    for block in blocks
        length(block) == 2 || error("Each complementarity block must be `(cc1, cc2)`")
        cc1, cc2 = block
        side1 = _as_vec(cc1)
        side2 = _as_vec(cc2)
        length(side1) == length(side2) || throw(DimensionMismatch(
            "Complementarity block sides must have the same length; got $(length(side1)) and $(length(side2))",
        ))

        for (raw1, raw2) in zip(side1, side2)
            ep1 = _jump_endpoint(model, raw1, row_by_constraint, col_map, variable_row_offset)
            ep2 = _jump_endpoint(model, raw2, row_by_constraint, col_map, variable_row_offset)
            _validate_endpoint_row(rows[ep1.row], ep1.label)
            _validate_endpoint_row(rows[ep2.row], ep2.label)
            push!(endpoints1, ep1)
            push!(endpoints2, ep2)
        end
    end

    return endpoints1, endpoints2
end

"""
    prepare_jump_to_marble(model::JuMP.Model, complementarity_blocks) -> JumpMPCCModel

Prepare a reusable direct JuMP-to-Marble converter.

The returned cache stores the decision-variable order, constraint row order, and
complementarity endpoint routing. Call `jump_to_marble(cache)` after changing
JuMP parameter values to re-query the current objective Hessian/gradient,
constraint Jacobians, bounds, and complementarity residuals.
"""
function prepare_jump_to_marble(model::JuMP.Model, complementarity_blocks)
    vars, col_map = _jump_var_col_map(model)
    nvar = length(vars)

    constraints = JuMP.all_constraints(model; include_variable_in_set_constraints = false)
    row_by_constraint = Dict{Any,Int}()
    rows = _JumpMPCCRow[]

    for con in constraints
        push!(rows, _constraint_row(con, col_map, nvar))
        row_by_constraint[JuMP.index(con)] = length(rows)
    end

    variable_row_offset = length(rows)
    for var in vars
        push!(rows, _variable_row(var, col_map, nvar))
    end

    endpoints1, endpoints2 = _expand_jump_complementarity_blocks(
        model,
        complementarity_blocks,
        row_by_constraint,
        col_map,
        variable_row_offset,
        rows,
    )

    return JumpMPCCModel(model, vars, col_map, constraints, row_by_constraint, endpoints1, endpoints2)
end

"""
    jump_to_marble(cache::JumpMPCCModel) -> NamedTuple

Refresh the numeric Marble data from a prepared JuMP MPCC converter.

This re-queries JuMP expressions and parameter values, so it is the efficient
path to use after `JuMP.set_parameter_value`.
"""
function jump_to_marble(cache::JumpMPCCModel)
    nvar = length(cache.vars)
    rows = _jump_rows(cache)

    for ep in Iterators.flatten((cache.endpoints1, cache.endpoints2))
        _validate_endpoint_row(rows[ep.row], ep.label)
    end

    nrows = length(rows)
    consumed_lb = falses(nrows)
    consumed_ub = falses(nrows)

    Lrows = Vector{Float64}[]
    lvals = Float64[]
    Rrows = Vector{Float64}[]
    rvals = Float64[]

    for (ep1, ep2) in zip(cache.endpoints1, cache.endpoints2)
        A, b = _endpoint_residual(ep1, rows, consumed_lb, consumed_ub)
        _push_residual!(Lrows, lvals, A, b)
        A, b = _endpoint_residual(ep2, rows, consumed_lb, consumed_ub)
        _push_residual!(Rrows, rvals, A, b)
    end

    eq_rows = Int[]
    ineq_lb_rows = Int[]
    ineq_ub_rows = Int[]

    for (i, row) in enumerate(rows)
        has_lb = isfinite(row.lb)
        has_ub = isfinite(row.ub)

        if has_lb && has_ub && row.lb == row.ub
            (consumed_lb[i] || consumed_ub[i]) || push!(eq_rows, i)
            continue
        end

        has_lb && !consumed_lb[i] && push!(ineq_lb_rows, i)
        has_ub && !consumed_ub[i] && push!(ineq_ub_rows, i)
    end

    J_eq, b_eq = _row_matrix(eq_rows, rows, nvar)
    J_ineq_lb, b_ineq_lb = _row_matrix(ineq_lb_rows, rows, nvar)
    J_ineq_ub, b_ineq_ub = _row_matrix(ineq_ub_rows, rows, nvar, i -> -1.0)
    J_ineq = [J_ineq_lb; J_ineq_ub]
    b_ineq = [b_ineq_lb; b_ineq_ub]
    L, l = _rows_to_matrix(Lrows, lvals, nvar)
    R, r = _rows_to_matrix(Rrows, rvals, nvar)
    Q, q, c0 = _objective_data(cache.model, cache.col_map, nvar)

    return (
        Q = Q, q = q, c0 = c0,
        J_eq = J_eq, b_eq = b_eq,
        J_ineq = J_ineq, b_ineq = b_ineq,
        L = L, l = l,
        R = R, r = r,
    )
end

"""
    jump_to_marble(model::JuMP.Model, complementarity_blocks) -> NamedTuple

Convert a JuMP quadratic model and a list of complementarity endpoint blocks
directly into the matrix/vector data Marble consumes, without first building an
`NLPModelsJuMP.MathOptNLPModel`.

Each entry of `complementarity_blocks` must be `(cc1, cc2)`. Each side may be a
single `JuMP.VariableRef`/`JuMP.ConstraintRef` or an array of references; array
sides are paired elementwise and must have equal length. Complementarity
endpoints must be one-sided or two-sided inequalities.

Returns a `NamedTuple` describing

    minimize    1/2 xᵀ Q x + qᵀ x + c0
    subject to  J_eq   x + b_eq   == 0
                J_ineq x + b_ineq >= 0
                0 <= (L x + l) ⊥ (R x + r) >= 0
"""
function jump_to_marble(model::JuMP.Model, complementarity_blocks)
    return jump_to_marble(prepare_jump_to_marble(model, complementarity_blocks))
end
