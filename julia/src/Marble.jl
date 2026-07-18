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

        prob = Problem(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        Marble.set_problem!(solver, prob, opts)
        return nothing
    end

    # Extra constructor for a Marble.Problem that handles type conversion, either to sparse
    # matrices based by their data or to dense FLoat64 matrices depending on the input. In
    # the sparse case every block is converted to a SparseMatrixCSC
    function Problem(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        blocks = (Q, J_eq, J_ineq, L, R)
        fvec(v) = collect(Float64, v)
        if any(b -> b isa AbstractSparseMatrix, blocks)
            Qs, Es, Is, Ls, Rs = sparse(Q), sparse(J_eq), sparse(J_ineq), sparse(L), sparse(R)
            return Marble.Problem(size(Qs, 2),
                Qs.colptr, Qs.rowval, Qs.nzval, fvec(q), Float64(c0),
                size(Es, 1), Es.colptr, Es.rowval, Es.nzval, fvec(b_eq),
                size(Is, 1), Is.colptr, Is.rowval, Is.nzval, fvec(b_ineq),
                size(Ls, 1), Ls.colptr, Ls.rowval, Ls.nzval, fvec(l),
                             Rs.colptr, Rs.rowval, Rs.nzval, fvec(r))
        else
            fmat(M) = Matrix{Float64}(M)
            return Marble.Problem(
                fmat(Q), fvec(q), Float64(c0),
                fmat(J_eq), fvec(b_eq), fmat(J_ineq), fvec(b_ineq),
                fmat(L), fvec(l), fmat(R), fvec(r))
        end
    end

    const OPTIONS = [
        :convergence_kkt_norm, :convergence_eq_violation, :convergence_ineq_violation,
        :convergence_comp_violation, :outer_step_kkt_norm,
        :penalty_initial, :penalty_max, :penalty_scaling,
        :relaxation_initial, :relaxation_min, :relaxation_scaling,
        :max_iters, :max_iters_linesearch,
        :gamma_objective, :gamma_constraint, :ruiz_iterations,
        :inertia_warmstart,
        :verbosity, :retraction_type
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

    Base.getproperty(solver::Marble.Solver, sym::Symbol) = solver_property(solver, Val(sym))
    solver_property(solver::Marble.Solver, ::Val{:options}) = CxxWrap.dereference_argument(Marble.options(solver))
    solver_property(solver::Marble.Solver, ::Val{:problem}) = CxxWrap.dereference_argument(Marble.get_problem(solver))
    solver_property(solver::Marble.Solver, ::Val{sym}) where sym = getfield(solver, sym)

    Base.propertynames(solver::Marble.Solver, private::Bool=false) = (:options, :problem,fieldnames(typeof(solver))...)

    obj(solver::Marble.Solver, z::Vector{Float64}) = Marble.obj(Marble.get_problem(solver), z)
    obj(data::NamedTuple, z::Vector{Float64}) = Marble.obj(Problem(data...), z)

    residual_eq(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_eq(Marble.get_problem(solver), z)
    residual_eq(data::NamedTuple, z::Vector{Float64}) = Marble.residual_eq(Problem(data...), z)

    residual_ineq(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_ineq(Marble.get_problem(solver), z)
    residual_ineq(data::NamedTuple, z::Vector{Float64}) = Marble.residual_ineq(Problem(data...), z)

    residual_comp(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_comp(Marble.get_problem(solver), z)
    residual_comp(data::NamedTuple, z::Vector{Float64}) = Marble.residual_comp(Problem(data...), z)

    # Problem construction with JuMP: MPCC, complementarity indexing, and
    # the conversion of an MPCC into the matrix/vector data Marble consumes.
    include("jump_mpcc.jl")

    export MPCC, complementarity_indices
    export reformulate_sos1, var_inds
    export mpcc_to_marble

    # Build a Marble problem from an MPCC (defined here since it depends on
    # both MPCC/mpcc_to_marble above and the build_problem helper).
    function setup!(solver::Marble.Solver, mpcc::MPCC; kwargs...)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)
        data = mpcc_to_marble(mpcc)
        prob = Problem(data.Q, data.q, data.c0,
                       data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)
        Marble.set_problem!(solver, prob, opts)
        return nothing
    end
end
