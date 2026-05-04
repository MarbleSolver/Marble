module Marble
    using CxxWrap

    @wrapmodule(() -> joinpath(@__DIR__, "..", "..", "build", "lib", "librcqp_julia"))

    function __init__()
        @initcxx
    end
end
