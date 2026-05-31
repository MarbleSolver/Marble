# Marble

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/marble-solver-logos-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/marble-solver-logos-light.svg">
    <img alt="Marble" src="assets/marble-solver-logos-light.svg" width="360">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/Marble/Marble/actions/workflows/cmake-build.yml"><img alt="CMake Build" src="https://github.com/MarbleSolver/Marble/actions/workflows/cmake-build.yml/badge.svg"></a>
  <a href="https://roboticexplorationlab.org/Marble/"><img alt="Docs" src="https://img.shields.io/badge/docs-dev-blue.svg"></a>
</p>

A C++ solver for quadratic programs with linear complementarity constraints, with Python and Julia bindings.

## Prerequisites

Install these yourself:

| Dependency | Version | Needed for |
|---|---|---|
| CMake | ≥ 3.28 | everything |
| A C++17 compiler | — | everything |
| Eigen3 | ≥ 3.3 | everything |
| nlohmann-json | — | everything |
| Python + dev headers | ≥ 3.8 | Python bindings |
| Julia + CxxWrap.jl | ≥ 1.10 | Julia bindings |
| pybind11-stubgen | — | optional: `.pyi` autocomplete stubs |

CMake fetches the rest automatically (from source if they aren't already installed), so you don't need to install them yourself:

| Dependency | Version | Needed for |
|---|---|---|
| QDLDL | — | everything |
| pybind11 | ≥ 2.13 | Python bindings |

**Julia setup** (only for the Julia bindings): make `CxxWrap.jl` available to the Julia that CMake will use.
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
```

## Building from source

Install the system build dependencies:

```bash
# macOS
brew install cmake eigen nlohmann-json

# Ubuntu/Debian
sudo apt install cmake libeigen3-dev nlohmann-json3-dev
```

The project ships CMake presets in `CMakePresets.json`. Run the commands below from the repository root.

### Python bindings (default)
```bash
cmake --preset python
cmake --build --preset python
```
Output: `build/python/marble/_core.cpython-<version>-<platform>.so` — the compiled core staged inside the `marble` package.

### Julia bindings
```bash
cmake --preset julia
cmake --build --preset julia
```
CMake auto-detects JlCxx by running `julia -e 'using CxxWrap; print(CxxWrap.prefix_path())'`; override with `-DJlCxx_DIR=<path>` if needed.

Output: `build/lib/libmarble_julia.dylib` (macOS) / `build/lib/libmarble_julia.so` (Linux).

### Both
```bash
cmake --preset all
cmake --build --preset all
```

<!-- ## Docker

The included `Dockerfile` builds both Python and Julia bindings inside a self-contained image.

```bash
docker build -t marble .
docker run -it marble
```

Inside the container:
- `python` uses the venv at `/opt/venv`; `import marble` works immediately
- Julia shared library is at `build/lib/libmarble_julia.so` -->

## Using Marble from Python

marble is not on PyPI, so build the bindings from this repository and use them from your own virtual environment (a fresh one or an existing project's).

**1. Build the Python bindings** (needs the dependencies from [Building from source](#building-from-source)):
```bash
cmake --preset python
cmake --build --preset python
```
This stages the `marble` package at `build/python/marble/`.

**2. Make it importable.** Either install it into your virtual environment from the repo's `python/` directory (pip recompiles the extension and pulls in `numpy` and `scipy`):
```bash
python -m venv .venv                 # or activate an existing environment
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install /path/to/Marble/python
```
or, without installing, add the staged package to your path at runtime:
```python
import sys
sys.path.insert(0, "/path/to/Marble/build/python")
```

**3. Use it:**
```python
import numpy as np
import marble

# min 1/2 x'x  s.t.  x1 + x2 = 2  ->  x* = [1, 1]
solver = marble.Solver()
solver.setup(np.eye(2), np.zeros(2), J_eq=np.array([[1.0, 1.0]]), b_eq=np.array([-2.0]))
res = solver.solve()
print(res.converged, res.z)          # True  [1. 1.]
```

## Using Marble from Julia

The Julia bindings in `julia/` form a package named `Marble`. It is not registered, so add it to your environment with `Pkg.develop`.

**1. Build the Julia shared library** (needs `CxxWrap` available to the Julia that CMake uses, see [Julia bindings](#julia-bindings)):
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
cmake --preset julia
cmake --build --preset julia
```
This writes `build/lib/libmarble_julia.{dylib,so}`, which `Marble.jl` loads on import. By default it looks in this repo's `build/lib/`, so an in-repo build needs no extra configuration.

**2. Add the package to your environment and install its dependencies:**
```julia
using Pkg
Pkg.develop(path="/path/to/Marble/julia")
Pkg.instantiate()      # pulls in CxxWrap and the other dependencies
```

**3. Use it:**
```julia
using Marble
using LinearAlgebra

# min 1/2 x'x  s.t.  x1 + x2 = 2  ->  x* = [1, 1]
solver = Marble.Solver()
Marble.setup!(solver, Matrix(1.0I, 2, 2), [0.0, 0.0]; J_eq = [1.0 1.0], b_eq = [-2.0])
res = Marble.solve!(solver)
println(Marble.converged(res), " ", collect(Marble.z(res)))   # true  ~[1.0, 1.0]
```

If you build or move the shared library elsewhere, point the package at it with a Preference, then restart Julia so the new path takes effect:
```julia
using Preferences
set_preferences!("Marble", "libmarble_julia_path" => "/abs/path/to/libmarble_julia")
```
The path omits the file extension; `Marble.jl` appends the platform's `.dylib` / `.so`.

## Developing Marble

### Editable Python install
Install the build tools into your environment and use an editable install so source edits are picked up without reinstalling:
```bash
pip install scikit-build-core pybind11
pip install -e /path/to/Marble/python --no-build-isolation
```

### VSCode: Python autocomplete

The high-level API in `python/marble/__init__.py` is type-annotated, and the compiled core ships a `python/marble/_core.pyi` stub that is regenerated each time you build `marble_python` (requires `pybind11-stubgen`). Pylance reads both from the `marble` package, so a single setting wires everything up.

Add to `.vscode/settings.json`:
```json
{
  "python.analysis.extraPaths": ["${workspaceFolder}/build/python"],
  "python.analysis.useLibraryCodeForTypes": true
}
```
`extraPaths` tells Pylance where the staged `marble` package (with `_core.so` and `_core.pyi`) lives so it can be imported.

Then install the **Pylance** extension (`ms-python.vscode-pylance`); it picks up the setting automatically.

If stubs are stale or missing, rebuild:
```bash
cmake --build --preset python --target marble_python
```
If `pybind11-stubgen` is not installed, CMake warns but the build still succeeds. Install it with:
```bash
pip install pybind11-stubgen
```
