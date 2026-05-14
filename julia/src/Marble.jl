module Marble
    using CxxWrap
    using SparseArrays
    using NLPModels, ADNLPModels, JuMP, NLPModelsJuMP
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
    export MarbleData
    export obj, residual_eq, residual_ineq, residual_comp

    @Base.kwdef mutable struct MarbleData
        Q::AbstractMatrix; q::AbstractVector; c0::Number
        J_eq::AbstractMatrix; b_eq::AbstractVector
        J_ineq::AbstractMatrix; b_ineq::AbstractVector
        L::AbstractMatrix; l::AbstractVector
        R::AbstractMatrix; r::AbstractVector
    end

    function MarbleData(Q::AbstractMatrix, q::AbstractVector, c0::Number;
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

    function obj(data::MarbleData, x)
        0.5 * x' * data.Q * x + data.q' * x + data.c0
    end

    function residual_eq(data::MarbleData, x)
        data.J_eq * x + data.b_eq
    end

    function residual_ineq(data::MarbleData, x)
        min.(data.J_ineq * x + data.b_ineq, 0.0)
    end

    function residual_comp(data::MarbleData, x)
        (data.L * x + data.l) .* (data.R * x + data.r)
    end

    function solve(data::MarbleData; opts)
        problem = Marble.Problem(data.Q, data.q, data.c0, data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)
        solver = Marble.Solver()
        Marble.set_problem!(solver, problem)
        Marble.verbosity!(opts, true)

        return Marble.solve!(solver, opts)
    end

    # Functions to convert stuff to MarbleData
    include("conversion.jl")
    export to_vertical_mpcc, to_jump
end
