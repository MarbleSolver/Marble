# RCQP
A C++ solver for quadratic programs with linear complementarity constraints, with Python and Julia bindings.

## Prerequisites

| Dependency | Required for | Install |
|---|---|---|
| CMake ≥ 3.28 | all | [cmake.org](https://cmake.org/download/) or `brew install cmake` |
| Eigen3 | all | `brew install eigen` / `apt install libeigen3-dev` |
| nlohmann-json | all | `brew install nlohmann-json` / `apt install nlohmann-json3-dev` |
| Python ≥ 3.8 + dev headers | Python bindings | [python.org](https://www.python.org/) |
| pybind11-stubgen | `.pyi` autocomplete stubs | `pip install pybind11-stubgen` |
| Julia ≥ 1.9 + CxxWrap.jl | Julia bindings | see below |

**Julia setup** (only needed for Julia bindings):
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
```

## Building

The project uses CMake presets defined in `CMakePresets.json`.

### Python bindings (default)
```bash
cmake --preset python
cmake --build --preset python
```
Output: `build/python/rcqp.cpython-<version>-<platform>.so`

### Julia bindings
```bash
cmake --preset julia
cmake --build --preset julia
```
CMake auto-detects JlCxx by calling `julia -e 'using CxxWrap; print(CxxWrap.prefix_path())'` — no manual paths required.

Output: `build/lib/librcqp_julia.dylib` (macOS) / `build/lib/librcqp_julia.so` (Linux)

### Both
```bash
cmake --preset all
cmake --build --preset all
```

<!-- ### Build specific targets
```bash
cmake --build --preset python --target rcqp          # core static lib only
cmake --build --preset python --target rcqp_python   # Python .so + regenerate .pyi
cmake --build --preset julia  --target rcqp_julia    # Julia shared lib
```

### Build output layout
```
build/
  lib/
    librcqp.a          # core library (static)
    libqdldl.a         # QDLDL (fetched + bundled automatically)
  python/
    rcqp.<ext>.so      # Python extension module
typings/
  rcqp.pyi             # auto-generated type stubs (for IDE autocomplete)
``` -->

## Python — pip install (editable)

For a proper install into your virtual environment (no `sys.path` hacks):
```bash
pip install scikit-build-core pybind11-stubgen
pip install -e . --no-build-isolation
```

After this, `import rcqp` works from anywhere in that environment.

## Python — usage without installing

Add `build/python/` to your path at runtime:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "build" / "python"))

import rcqp
import numpy as np

opts = rcqp.SolverOptions()
opts.max_iters = 500

solver = rcqp.Solver(opts)
solver.set_problem(prob, scaling)
converged = solver.solve(opts)

ws = solver.get_workspace()
print(ws.z)   # primal solution
```

## VSCode — Python autocomplete

Autocomplete and type-checking are driven by the `.pyi` stubs in `typings/`, which are regenerated automatically each time you build `rcqp_python` (requires `pybind11-stubgen`).

Two config files in this repo wire everything up for VSCode automatically:

### `.vscode/settings.json`
```json
{
  "python.analysis.extraPaths": ["${workspaceFolder}/build/python"],
  "python.analysis.stubPath": "${workspaceFolder}/typings",
  "python.analysis.useLibraryCodeForTypes": true
}
```

- `extraPaths` — tells Pylance where the compiled `.so` lives so it can be imported
- `stubPath` — tells Pylance where to find the `.pyi` stub file for type info

### Install the Pylance extension
Install the **Pylance** extension in VSCode (`ms-python.vscode-pylance`). It picks up the above settings automatically.

If stubs are stale or missing, rebuild:
```bash
cmake --build --preset python --target rcqp_python
```

If `pybind11-stubgen` is not installed, CMake will warn but the build still succeeds — you just won't get updated stubs.
```bash
pip install pybind11-stubgen
```

## Julia — usage

```julia
using CxxWrap

module RCQP
    using CxxWrap
    @wrapmodule(() -> joinpath(@__DIR__, "build/lib/librcqp_julia"))
    function __init__()
        @initcxx
    end
end

prob = RCQP.Problem(H, g, 0.0, J_eq, c_eq, J_ineq, c_ineq, J_comp, c_comp)

opts = RCQP.SolverOptions()
RCQP.max_iters!(opts, 500)

solver = RCQP.Solver(opts)
RCQP.set_problem(solver, prob, ones(nz))
converged = RCQP.solve(solver, opts)

ws = RCQP.get_workspace(solver)
z  = copy(RCQP.z(ws))
```

See `tester.jl` for a complete working example.
