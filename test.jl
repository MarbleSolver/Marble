using Pkg;
Pkg.activate(joinpath(@__DIR__, "../.."))
using CxxWrap
using Revise

module RCQP
  using CxxWrap
  @wrapmodule(() -> joinpath(@__DIR__, "build", "librcqp_wrapper"))

  function __init__()
    @initcxx
  end
end

println(RCQP.test_array([1.0, 2.0, 3.0]))

arr = [1.0 2.0; 3.0 4.0]
println(RCQP.test_matrix(arr, 2, 2))
println(arr)