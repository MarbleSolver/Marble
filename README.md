# Marble
[![CMake Build](https://github.com/MarbleSolver/RCQP/actions/workflows/cmake-build.yml/badge.svg)](https://github.com/MarbleSolver/RCQP/actions/workflows/cmake-build.yml)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://roboticexplorationlab.org/Marble/)

A C++ solver for quadratic programs with linear complementarity constraints, with Python and Julia bindings.

## Prerequisites

| Dependency | Required for |
|---|---|
| CMake ≥ 3.28 | all |
| Eigen3 | all |
| nlohmann-json | all |
| Python ≥ 3.8 + dev headers | Python bindings |
| pybind11-stubgen | `.pyi` autocomplete stubs |
| Julia ≥ 1.9 + CxxWrap.jl | Julia bindings |

**Julia setup** (only needed for Julia bindings):
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
```

## Building

The project uses CMake presets defined in `CMakePresets.json`. Execute the commands below from the root of the repository.

### Python bindings (default)
```bash
cmake --preset python
cmake --build --preset python
```
Output: `build/python/marble/_core.cpython-<version>-<platform>.so` (the compiled core inside the `marble` package)

### Julia bindings
```bash
cmake --preset julia
cmake --build --preset julia
```
CMake auto-detects JlCxx by calling `julia -e 'using CxxWrap; print(CxxWrap.prefix_path())'`

Output: `build/lib/libmarble_julia.dylib` (macOS) / `build/lib/libmarble_julia.so` (Linux)

### Both
```bash
cmake --preset all
cmake --build --preset all
```

## Docker

The included `Dockerfile` builds both Python and Julia bindings inside a self-contained image.

```bash
docker build -t marble .
docker run -it marble
```

Inside the container:
- `python` uses the venv at `/opt/venv`; `import marble` works immediately
- Julia shared library is at `build/lib/libmarble_julia.so`


## Julia: use from your own environment

The Julia bindings in `julia/` form a package named `Marble`. It is not registered, so
add it to your environment with `Pkg.develop`.

**1. Build the Julia shared library** (this needs `CxxWrap` available to the Julia that
CMake uses, see [Julia bindings](#julia-bindings) above):
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
cmake --preset julia
cmake --build --preset julia
```
This writes `build/lib/libmarble_julia.{dylib,so}`, which `Marble.jl` loads on import.
By default it looks in this repo's `build/lib/`, so an in-repo build needs no extra
configuration.

**2. Add the package to your environment and install its dependencies:**
```julia
using Pkg
Pkg.develop(path="/path/to/RCQP/julia")
Pkg.instantiate()      # pulls in CxxWrap and the other dependencies
```

**3. Use it:** build a solver, set up the problem matrices, then solve. Matrices may be
dense or sparse, and solver options are passed to `setup!` as keyword arguments.
```julia
using Marble
using LinearAlgebra

# min 1/2 x'x  s.t.  x1 + x2 = 2  ->  x* = [1, 1]
solver = Marble.Solver()
Marble.setup!(solver, Matrix(1.0I, 2, 2), [0.0, 0.0]; J_eq = [1.0 1.0], b_eq = [-2.0])
res = Marble.solve!(solver)
println(Marble.converged(res), " ", collect(Marble.z(res)))   # true  ~[1.0, 1.0]
```

If you build or move the shared library elsewhere, point the package at it with a
Preference, then restart Julia so the new path takes effect:
```julia
using Preferences
set_preferences!("Marble", "libmarble_julia_path" => "/abs/path/to/libmarble_julia")
```
The path omits the file extension; `Marble.jl` appends the platform's `.dylib` / `.so`.


## Python: use from your own virtual environment

marble is not on PyPI, so install it from a local checkout of this repository. This
works with any virtual environment, a fresh one or an existing project's.

**1. Install the system build dependencies** (CMake, Eigen3, and nlohmann-json are
needed at build time):
```bash
# macOS
brew install cmake eigen nlohmann-json

# Ubuntu/Debian
sudo apt install cmake libeigen3-dev nlohmann-json3-dev
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv                 # or activate an existing environment
source .venv/bin/activate            # Windows: .venv\Scripts\activate
```

**3. Install marble from the repo's `python/` directory:**
```bash
pip install /path/to/RCQP/python
```
pip builds the C++ extension in an isolated environment, fetching `scikit-build-core`
and `pybind11` automatically, and pulls in `numpy` and `scipy`. Nothing else needs to
be installed first.

After this, `import marble` works wherever that environment is active:
```python
import numpy as np
import marble

# min 1/2 x'x + q'x  ->  x* = -q
solver = marble.Solver()
solver.setup(np.eye(2), np.array([1.0, 2.0]))
print(solver.solve().z)              # [-1. -2.]
```

**Developing marble itself:** install the build tools into your environment and use an
editable install so source edits are picked up without reinstalling:
```bash
pip install scikit-build-core pybind11
pip install -e /path/to/RCQP/python --no-build-isolation
```

## Python: usage without installing

After building (`cmake --build --preset python`), the `marble` package is staged at
`build/python/marble/`. Add `build/python/` to your path at runtime, then import it:

```python
import sys

rcqp_root = "/path/to/RCQP"            # your checkout of this repo
sys.path.insert(0, rcqp_root + "/build/python")

import marble
```

## VSCode: Python autocomplete

The high-level API in `python/marble/__init__.py` is type-annotated, and the
compiled core ships a `python/marble/_core.pyi` stub that is regenerated each time
you build `marble_python` (requires `pybind11-stubgen`). Pylance reads both from the
`marble` package, so a single setting wires everything up:

### `.vscode/settings.json`
```json
{
  "python.analysis.extraPaths": ["${workspaceFolder}/build/python"],
  "python.analysis.useLibraryCodeForTypes": true
}
```

- `extraPaths`:  tells Pylance where the staged `marble` package (with `_core.so` and `_core.pyi`) lives so it can be imported

### Install the Pylance extension
Install the **Pylance** extension in VSCode (`ms-python.vscode-pylance`). It picks up the above setting automatically.

If stubs are stale or missing, rebuild:
```bash
cmake --build --preset python --target marble_python
```

If `pybind11-stubgen` is not installed, CMake will warn but the build still succeeds. Install it with:
```bash
pip install pybind11-stubgen
```
