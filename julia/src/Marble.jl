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

    # Solver data struct
    export MarbleData, estimate_initial_point
    export obj, residual_eq, residual_ineq, residual_comp

    @Base.kwdef mutable struct MarbleData
        Q::AbstractMatrix; q::AbstractVector; c0::Number
        J_eq::AbstractMatrix; b_eq::AbstractVector
        J_ineq::AbstractMatrix; b_ineq::AbstractVector
        L::AbstractMatrix; l::AbstractVector
        R::AbstractMatrix; r::AbstractVector
    end

    function MarbleData(Q::AbstractMatrix, q::AbstractVector, c0::Number=0.0;
                        J_eq=nothing, b_eq=nothing,
                        J_ineq=nothing, b_ineq=nothing,
                        L=nothing, l=nothing, R=nothing, r=nothing)
        n = size(Q, 2)
        T = eltype(Q)
        return MarbleData(
            Q=Q, q=q, c0=c0,
            J_eq  = isnothing(J_eq)   ? zeros(T, 0, n) : J_eq,
            b_eq  = isnothing(b_eq)   ? zeros(T, 0)    : b_eq,
            J_ineq = isnothing(J_ineq) ? zeros(T, 0, n) : J_ineq,
            b_ineq = isnothing(b_ineq) ? zeros(T, 0)    : b_ineq,
            L = isnothing(L) ? zeros(T, 0, n) : L,
            l = isnothing(l) ? zeros(T, 0)    : l,
            R = isnothing(R) ? zeros(T, 0, n) : R,
            r = isnothing(r) ? zeros(T, 0)    : r,
        )
    end

    function estimate_initial_point(data::MarbleData, z0; opts, κ=nothing, eps=1e-12, multiplier_rtol=1e-8)
        z = convert(Vector{Float64}, z0)
        κ = isnothing(κ) ? Float64(Marble.relaxation_initial(opts)) : Float64(κ)

        g = convert(Vector{Float64}, data.J_ineq * z + data.b_ineq)
        a = convert(Vector{Float64}, data.L * z + data.l)
        b = convert(Vector{Float64}, data.R * z + data.r)

        g_safe = max.(g, eps)
        a_safe = max.(a, eps)
        b_safe = max.(b, eps)

        s_ineq = g_safe .- κ ./ g_safe
        s_comp = a_safe .- b_safe

        n_eq = length(data.b_eq)
        n_ineq = length(data.b_ineq)
        n_comp = length(data.l)

        # m_ineq = -κ ./ g_safe
        # grad = convert(Vector{Float64}, data.Q * z + data.q)
        # rhs = -(grad + data.J_ineq' * m_ineq)

        # A = Matrix([data.J_eq' data.L' data.R'])
        # m_free = size(A, 2) == 0 ? Float64[] : pinv(A; rtol=multiplier_rtol) * rhs

        # m_eq = n_eq == 0 ? Float64[] : m_free[1:n_eq]
        # m_comp_lr = n_comp == 0 ? Float64[] : m_free[(n_eq + 1):end]
        # m_comp = zeros(2 * n_comp)
        # for i in 1:n_comp
        #     m_comp[2i - 1] = m_comp_lr[i]
        #     m_comp[2i] = m_comp_lr[n_comp + i]
        # end

        m_ineq = zeros(n_ineq)
        m_eq = zeros(n_eq)
        m_comp = zeros(2 * n_comp)

        return (;
            z,
            s_ineq,
            s_comp,
            m_eq,
            m_ineq,
            m_comp,
            m_eq_est=copy(m_eq),
            m_ineq_est=copy(m_ineq),
            m_comp_est=copy(m_comp),
        )
    end

    function _initialize_point(data::MarbleData; opts, z0)
        init = estimate_initial_point(data, z0; opts=opts)
        ip = Marble.InitialPoint()
        Marble.z!(ip, init.z)
        Marble.s_ineq!(ip, init.s_ineq)
        Marble.s_comp!(ip, init.s_comp)
        Marble.m_eq!(ip, init.m_eq)
        Marble.m_ineq!(ip, init.m_ineq)
        Marble.m_comp!(ip, init.m_comp)
        Marble.m_eq_est!(ip, init.m_eq_est)
        Marble.m_ineq_est!(ip, init.m_ineq_est)
        Marble.m_comp_est!(ip, init.m_comp_est)
        return ip
    end

    function solve(data::MarbleData; opts, z0=nothing)
        problem = Marble.Problem(data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)
        solver = Marble.Solver()
        Marble.set_problem!(solver, problem)

        if isnothing(z0)
            return Marble.solve!(solver, opts)
        else
            ip = _initialize_point(data; opts=opts, z0=z0)
            return Marble.solve!(solver, opts, ip)
        end
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

    function update_settings!(opts::Marble.SolverOptions; kwargs...)
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

    # Utility functions for problem construction with JuMP
    include("jump_mpcc.jl")
    export nlp_con_row_map, nlp_var_col_map, var_var_complementarities, var_con_complementarities, con_con_complementarities
    export var_inds, reformulate_sos1, reformulate_lie

    # Functions to convert stuff to MarbleData
    include("conversion.jl")
    export to_jump

    # Debug log loader
    include("debug.jl")
    export SolverDebugLog, load_debug_log, get_field, get_iterates
end
