module Marble
    using CxxWrap
    using SparseArrays
    using NLPModels, JuMP, NLPModelsJuMP
    using LinearAlgebra
    using AmplNLReader
    using Preferences

    const _libmarble_julia = @load_preference("libmarble_julia_path",
        joinpath(@__DIR__, "..", "..", "build", "lib", "libmarble_julia"))

    @wrapmodule(() -> _libmarble_julia)

    function __init__()
        @initcxx
    end

    function setup!(solver::Marble.Solver,model::Model, ind_cc1, ind_cc2, cc_type; kwargs...)
        Marble.update_settings!(solver; kwargs...)
        setup!(solver, MathOptNLPModel(model), ind_cc1, ind_cc2, cc_type; kwargs...)
        return nothing
    end

    function setup!(solver::Marble.Solver,nlp::AbstractNLPModel, ind_cc1, ind_cc2, cc_type; kwargs...)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)
        data = jump_to_marble(nlp, ind_cc1, ind_cc2, cc_type)
        _set_problem!(solver, opts, data.Q, data.q, data.c0,
                      data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)
        return nothing
    end

    function setup!(solver::Marble.Solver, Q::AbstractMatrix, q::AbstractVector, c0::Real=0.0;
                    J_eq=nothing, b_eq=nothing, J_ineq=nothing, b_ineq=nothing,
                    L=nothing, l=nothing, R=nothing, r=nothing, kwargs...)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)

        n = length(q)
        J_eq   = isnothing(J_eq)   ? zeros(0, n) : J_eq
        b_eq   = isnothing(b_eq)   ? zeros(0)    : b_eq
        J_ineq = isnothing(J_ineq) ? zeros(0, n) : J_ineq
        b_ineq = isnothing(b_ineq) ? zeros(0)    : b_ineq
        L = isnothing(L) ? zeros(0, n) : L
        l = isnothing(l) ? zeros(0)    : l
        R = isnothing(R) ? zeros(0, n) : R
        r = isnothing(r) ? zeros(0)    : r

        _set_problem!(solver, opts, Q, q, Float64(c0),
                      J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        return nothing
    end

    # Check that every block has consistent dimensions, throwing DimensionMismatch
    # on any mismatch. The compiled core's Problem constructor performs the same
    # check, but CxxWrap does not translate C++ exceptions into Julia exceptions
    # (they call std::terminate and abort the process), so we must validate here :(
    function _validate_problem_dims(Q, q, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        n = length(q)
        size(Q) == (n, n) || throw(DimensionMismatch(
            "Q must have size ($n, $n) to match length(q)=$n, got $(size(Q))"))

        for (Jname, J, cname, c) in (("J_eq", J_eq, "b_eq", b_eq),
                                     ("J_ineq", J_ineq, "b_ineq", b_ineq))
            size(J, 1) == length(c) || throw(DimensionMismatch(
                "$Jname has $(size(J, 1)) rows but $cname has length $(length(c)); they must match"))
            (size(J, 1) == 0 || size(J, 2) == n) || throw(DimensionMismatch(
                "$Jname has $(size(J, 2)) columns but expected $n (= length(q))"))
        end

        nL, nR, nl, nr = size(L, 1), size(R, 1), length(l), length(r)
        (nL == nR == nl == nr) || throw(DimensionMismatch(
            "complementarity blocks L, l, R, r must share the same number of rows; " *
            "got L rows=$nL, l length=$nl, R rows=$nR, r length=$nr " *
            "(every complementarity pair needs a row in each of L, l, R and r)"))
        if nL > 0
            size(L, 2) == n || throw(DimensionMismatch(
                "L has $(size(L, 2)) columns but expected $n (= length(q))"))
            size(R, 2) == n || throw(DimensionMismatch(
                "R has $(size(R, 2)) columns but expected $n (= length(q))"))
        end
        return nothing
    end

    # Build a Marble.Problem (dense or sparse, based on storage) and hand it to the
    # solver. When any block is sparse every block is converted to a SparseMatrixCSC
    # so the solver sees consistent compressed-sparse data
    function _set_problem!(solver::Marble.Solver, opts, Q, q, c0,
                           J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        _validate_problem_dims(Q, q, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        blocks = (Q, J_eq, J_ineq, L, R)
        fvec(v) = collect(Float64, v)
        if any(b -> b isa AbstractSparseMatrix, blocks)
            Qs, Es, Is, Ls, Rs = sparse(Q), sparse(J_eq), sparse(J_ineq), sparse(L), sparse(R)
            prob = Marble.Problem(size(Qs, 2),
                Qs.colptr, Qs.rowval, Qs.nzval, fvec(q), c0,
                size(Es, 1), Es.colptr, Es.rowval, Es.nzval, fvec(b_eq),
                size(Is, 1), Is.colptr, Is.rowval, Is.nzval, fvec(b_ineq),
                size(Ls, 1), Ls.colptr, Ls.rowval, Ls.nzval, fvec(l),
                             Rs.colptr, Rs.rowval, Rs.nzval, fvec(r))
        else
            fmat(M) = Matrix{Float64}(M)
            prob = Marble.Problem(
                fmat(Q), fvec(q), c0,
                fmat(J_eq), fvec(b_eq), fmat(J_ineq), fvec(b_ineq),
                fmat(L), fvec(l), fmat(R), fvec(r))
        end
        Marble.set_problem!(solver, prob, opts)
        return nothing
    end

    const OPTIONS = [
        :convergence_kkt_norm, :convergence_eq_violation, :convergence_ineq_violation,
        :convergence_comp_violation, :outer_step_kkt_norm,
        :penalty_initial, :penalty_max, :penalty_scaling,
        :relaxation_initial, :relaxation_min, :relaxation_scaling,
        :max_iters, :max_iters_linesearch,
        :gamma_objective, :gamma_constraint, :ruiz_iterations,
        :output_dir, :verbosity, :print_every, :debug,
        :debug_output_path, :debug_log_every
    ]

    function update_settings!(solver::Marble.Solver; kwargs...)
        opts = Marble.options(solver)
        for (opt_name, opt_val) in kwargs
            try
                opt_setter = getfield(Marble, Symbol(opt_name, :!))
                opt_setter(opts, opt_val)
            catch err
                if err isa UndefVarError || err isa MethodError
                    @warn "Option $(opt_name) not found in Marble.SolverOptions. Skipping."
                else
                    rethrow(err)
                end
            end
        end
    end

    function Base.show(io::IO, opts::Marble.SolverOptions)
        title = " Marble Options "
        
        max_key_len = maximum(length, string.(Marble.OPTIONS))
        total_width = max(max_key_len + 25, length(title) + 10)
        pad_left = (total_width - length(title)) ÷ 2
        
        printstyled(io, "─"^total_width * "\n", color=:light_black)
        printstyled(io, " "^pad_left * title * "\n", bold=true, color=:cyan)
        printstyled(io, "─"^total_width * "\n", color=:light_black)
        for opt in Marble.OPTIONS
            opt_val = getfield(Marble, opt)(opts)
            key_str = rpad(string(opt), max_key_len)
            print(io, "  ", key_str, " ")
            printstyled(io, "│ ", color=:light_black)
            println(io, opt_val)
        end
        printstyled(io, "─"^total_width * "\n", color=:light_black)
    end

    function convert_Eigen_sparse(nr, nc, colptr, rowval, nzval)
        return SparseMatrixCSC(nr, nc, colptr .+ 1, rowval .+ 1, nzval)
    end

    Base.getproperty(solver::Marble.Solver, sym::Symbol) = solver_property(solver, Val(sym))
    solver_property(solver::Marble.Solver, ::Val{:options}) = CxxWrap.dereference_argument(Marble.options(solver))
    solver_property(solver::Marble.Solver, ::Val{:problem}) = CxxWrap.dereference_argument(Marble.get_problem(solver))
    solver_property(solver::Marble.Solver, ::Val{sym}) where sym = getfield(solver, sym)

    Base.setproperty!(solver::Marble.Solver, sym::Symbol, val) = @warn "Setting properties of Marble.Solver is not supported"

    Base.propertynames(solver::Marble.Solver, private::Bool=false) = (:options, :problem, fieldnames(typeof(solver))...)

    Base.getproperty(problem::Marble.Problem, sym::Symbol) = problem_property(problem, Val(sym))
    problem_property(problem::Marble.Problem, ::Val{:cost_hessian}) = convert_Eigen_sparse(Marble.cost_hessian(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:J_eq}) = convert_Eigen_sparse(Marble.J_eq(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:J_ineq}) = convert_Eigen_sparse(Marble.J_ineq(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:J_comp}) = convert_Eigen_sparse(Marble.J_comp(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:c_eq}) = Marble.c_eq(problem)
    problem_property(problem::Marble.Problem, ::Val{:c_ineq}) = Marble.c_ineq(problem)
    problem_property(problem::Marble.Problem, ::Val{:c_comp}) = Marble.c_comp(problem)
    problem_property(problem::Marble.Problem, ::Val{:cost_gradient}) = Marble.cost_gradient(problem)
    problem_property(problem::Marble.Problem, ::Val{sym}) where sym = getfield(problem, sym)

    Base.setproperty!(problem::Marble.Problem, sym::Symbol, val) = @warn "Setting properties of Marble.Solver is not supported"

    Base.propertynames(problem::Marble.Problem, private::Bool=false) = 
        (:cost_hessian, :J_eq, :J_ineq, :J_comp, :c_eq, :c_ineq, :c_comp, :cost_gradient, fieldnames(typeof(problem))...)

    obj(solver::Marble.Solver, z::Vector{Float64}) = Marble.obj(Marble.get_problem(solver), z)
    residual_eq(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_eq(Marble.get_problem(solver), z)
    residual_ineq(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_ineq(Marble.get_problem(solver), z)
    residual_comp(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_comp(Marble.get_problem(solver), z)

    # Utility functions for problem construction with JuMP
    include("jump_mpcc.jl")
    export nlp_con_row_map, nlp_var_col_map, var_var_complementarities, var_con_complementarities, con_con_complementarities
    export var_inds, reformulate_sos1, reformulate_lie

    # Functions to convert stuff to MarbleData
    include("conversion.jl")
    export jump_to_marble

    # Debug log loader
    include("debug.jl")
    export SolverDebugLog, load_debug_log, get_field, get_iterates
end
