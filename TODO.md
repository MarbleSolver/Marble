- Add tests for Julia
- Add tests for Python
- Let user specify Julia install (for CxxWrap.jl)? Optionally wrap C++ install process into Julia, either through binary builder or `build.jl`
- Add/clean up examples
    - Simple test
    - Push box
    - Hopper
    - Rocket
    - Quadrotor

## Perf

- Benchmark line-search residual delta caching instead of recomputing every constraint product from `z`.
- Avoid duplicate filter sufficient-progress checks when accepting and pruning entries.
- Evaluate warm-starting regularizer search from the previous accepted inertia/line-search regularizer.
