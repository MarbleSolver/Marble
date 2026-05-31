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

    function setup!(solver::Marble.Solver,model::Model, ind_cc1, ind_cc2, comp_type; kwargs...)
        Marble.update_settings!(solver; kwargs...)
        setup!(solver, MathOptNLPModel(model), ind_cc1, ind_cc2, comp_type; kwargs...)
        return nothing
    end

    function setup!(solver::Marble.Solver,nlp::AbstractNLPModel, ind_cc1, ind_cc2, comp_type; kwargs...)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)
        data = jump_to_marble(nlp, ind_cc1, ind_cc2, comp_type)
        Marble.set_problem!(solver, data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r, opts)
        return nothing
    end

    function setup!(solver::Marble.Solver, Q, q, 
                    J_eq=zeros(0, size(Q, 1)), b_eq=zeros(0),
                    J_ineq=zeros(0, size(Q, 1)), b_ineq=zeros(0),
                    L=zeros(0, size(Q, 1)), l=zeros(0),
                    R=zeros(0, size(Q, 1)), r=zeros(0); kwargs...)
                    println(J_eq)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)
        Marble.set_problem!(solver, Q, q, 0.0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r, opts)
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

    Base.getproperty(solver::Marble.Solver, sym::Symbol) = solver_property(solver, Val(sym))
    solver_property(solver::Marble.Solver, ::Val{:options}) = CxxWrap.dereference_argument(Marble.options(solver))
    solver_property(solver::Marble.Solver, ::Val{:problem}) = CxxWrap.dereference_argument(Marble.get_problem(solver))
    solver_property(solver::Marble.Solver, ::Val{sym}) where sym = getfield(solver, sym)

    Base.propertynames(solver::Marble.Solver, private::Bool=false) = (:options, :problem,fieldnames(typeof(solver))...)

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
