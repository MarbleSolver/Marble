using JSON

struct SolverDebugLog
    header::Dict{String,Any}
    iterates::Vector{Dict{String,Any}}
    footer::Dict{String,Any}
end

"""
    load_debug_log(path) -> SolverDebugLog

Read a JSONL debug log written by the C++ solver when `debug=true`.
Returns a `SolverDebugLog` with fields:
- `header`   – problem dimensions and solver parameters
- `iterates` – one Dict per logged iteration (z, multipliers, violations, etc.)
- `footer`   – final iterate + convergence summary
"""
function load_debug_log(path::AbstractString)
    header   = nothing
    iterates = Dict{String,Any}[]
    footer   = nothing

    for line in eachline(path)
        isempty(strip(line)) && continue
        rec = JSON.parse(line)
        t = rec["type"]
        if t == "header"
            header = rec
        elseif t == "iterate"
            push!(iterates, rec)
        elseif t == "footer"
            footer = rec
        end
    end

    header === nothing && error("Debug log missing header: $path")
    footer === nothing && error("Debug log missing footer: $path")

    return SolverDebugLog(header, iterates, footer)
end

"""
    get_field(log, field) -> Vector

Extract a scalar field (e.g. "kkt", "penalty_param") from every iterate as a Vector.
"""
function get_field(log::SolverDebugLog, field::AbstractString)
    return [it[field] for it in log.iterates]
end

"""
    get_iterates(log, field) -> Matrix  (n_iters × n_vars)

Extract a vector field (e.g. "z", "m_eq") from every iterate.
Each row is one iteration.
"""
function get_iterates(log::SolverDebugLog, field::AbstractString)
    vecs = [Float64.(it[field]) for it in log.iterates]
    isempty(vecs) && return Matrix{Float64}(undef, 0, 0)
    return reduce(hcat, vecs)'   # (n_iters, n_vars)
end
