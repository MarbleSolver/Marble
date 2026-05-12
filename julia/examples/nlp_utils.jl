struct _Block
    name::Symbol
    eval::Function          # x -> AbstractVector
    n::Int
    lo::Vector{Float64}
    hi::Vector{Float64}
end

mutable struct CBuilder
    blocks::Vector{_Block}
    ind_ccc1::Vector{Int}  # For complementarity constraints: index of first variable/con constraint
    ind_ccc2::Vector{Int}  # For complementarity constraints: index of second variable/con constraint
end
CBuilder() = CBuilder(_Block[], Int[], Int[])

function add!(b::CBuilder, name::Symbol, eval, n::Int; lo=-Inf, hi=Inf)
    lo_v = lo isa AbstractVector ? collect(Float64, lo) : fill(Float64(lo), n)
    hi_v = hi isa AbstractVector ? collect(Float64, hi) : fill(Float64(hi), n)
    push!(b.blocks, _Block(name, eval, n, lo_v, hi_v))
    return b
end

# Convenience aliases
add_eq!(b, name, eval, neq; val=0.0)    = add!(b, name, eval, neq; lo=val,  hi=val)
add_ineq!(b, name, eval, nineq; lb=0.0, ub=Inf) = begin
    @assert isfinite(lb) || isfinite(ub) "At least one of lb or ub must be finite for an inequality constraint"
    @assert !(isfinite(lb) && isfinite(ub) && lb > ub) "Inconsistent bounds: lb=$lb > ub=$ub"

    add!(b, name, eval, nineq; lo=lb, hi=ub)
end
add_cc!(b, name, eval, ncc; lb=0.0, ub=Inf) = begin
    @assert isfinite(lb) || isfinite(ub) "At least one of lb or ub must be finite for a complementarity constraint"
    @assert iseven(ncc) "Number of variables n must be even for a complementarity constraint"

    ncon = sum(blk -> blk.n, b.blocks, init=0)
    add_ineq!(b, name, eval, ncc; lb=lb, ub=ub)
    append!(b.ind_ccc1, ncon .+ (1:2:ncc))
    append!(b.ind_ccc2, ncon .+ (2:2:ncc))
end

function assemble(b::CBuilder)
    # Group ranges by name; same-name pushes accumulate into a vector
    inds_dict = Dict{Symbol, Vector{UnitRange{Int}}}()
    los, his = Vector{Float64}[], Vector{Float64}[]
    off = 0
    for blk in b.blocks
        rng = (off+1):(off+blk.n)
        push!(get!(inds_dict, blk.name, UnitRange{Int}[]), rng)
        push!(los, blk.lo); push!(his, blk.hi)
        off += blk.n
    end
    # Single-element name → bare range; multi-element name → vector of ranges
    inds = (; (k => (length(v)==1 ? v[1] : v) for (k,v) in pairs(inds_dict))...)
    lcon = reduce(vcat, los)
    ucon = reduce(vcat, his)
    fns  = [blk.eval for blk in b.blocks]
    c(x) = reduce(vcat, f(x) for f in fns)
    return (; c, lcon, ucon, inds, ncon=off, ind_ccc1=b.ind_ccc1, ind_ccc2=b.ind_ccc2)
end