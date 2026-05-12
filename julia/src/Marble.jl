module Marble
    using CxxWrap
    using SparseArrays
    using NLPModels, ADNLPModels
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
        Q::Union{AbstractMatrix, Nothing}; q::Union{AbstractVector, Nothing}; c0::Union{Number, Nothing}
        J_eq::Union{AbstractMatrix, Nothing}; b_eq::Union{AbstractVector, Nothing}
        J_ineq::Union{AbstractMatrix, Nothing}; b_ineq::Union{AbstractVector, Nothing}
        L::Union{AbstractMatrix, Nothing}; l::Union{AbstractVector, Nothing}
        R::Union{AbstractMatrix, Nothing}; r::Union{AbstractVector, Nothing}
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

    # Functions to convert stuff to MarbleData
    include("nlpmodels_parser.jl")
    include("jump_parser.jl")
    export from_mpcc, to_JuMP, _marbledata_to_jump, CompFormulation, COMP_PERP, COMP_SOS1

    include("indexed_components.jl")
    export indexed_components
end
