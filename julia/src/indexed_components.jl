function _build_index!(dims::Tuple, offset::Ref{Int})
    if length(dims) == 1
        # Leaf: allocate a contiguous block of dims[1] elements
        m = dims[1]
        idx = (offset[] + 1):(offset[] + m)
        offset[] += m
        return idx
    else
        # Outer dimension is last; recurse for each index along it
        outer = dims[end]
        inner_dims = dims[1:end-1]
        return [_build_index!(inner_dims, offset) for _ in 1:outer]
    end
end

function indexed_components(components::NamedTuple)
    # Generalized indexed components for N-dimensional specs.
    #
    # Each component spec is either:
    #   - A single integer m:            one flat block of length m → UnitRange
    #   - A tuple (d1, d2, ..., dn):     d1 is the leaf block size; d2..dn are
    #                                    outer indexing dimensions (dn is outermost)
    #
    # Indexing convention:
    #   spec = m              →  inds               is UnitRange
    #   spec = (m, T)         →  inds[t]            is UnitRange   (t in 1:T)
    #   spec = (m, r, T)      →  inds[t][i]         is UnitRange   (t in 1:T, i in 1:r)
    #   spec = (m, d2, d3, …) →  inds[…][j][i]     is UnitRange

    comp_keys = keys(components)
    offset = Ref(0)

    vals = map(comp_keys) do key
        spec = components[key]
        dims = spec isa Tuple ? spec : (spec,)
        _build_index!(dims, offset)
    end

    return NamedTuple{comp_keys}(vals), offset[]
end
