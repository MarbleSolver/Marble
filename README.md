# Marble
[![CMake Build](https://github.com/MarbleSolver/RCQP/actions/workflows/cmake-build.yml/badge.svg)](https://github.com/MarbleSolver/RCQP/actions/workflows/cmake-build.yml)

A C++ solver for quadratic programs with linear complementarity constraints, with Python and Julia bindings.

This repository is currently a work-in-progress and will receive some significant clean up. If you want to get started we recommend checking out the Getting Started with [Julia](https://roboticexplorationlab.org/Marble/getting_started/julia/) or [Python](https://roboticexplorationlab.org/Marble/getting_started/python/) pages. We will try to make sure that the interfaces (Julia and Python) do not change, just the internals.

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
Output: `build/python/marble.cpython-<version>-<platform>.so`

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

## Python:  pip install (editable)

For a proper install into your virtual environment:

**1. Install system dependencies** (CMake, Eigen3, and nlohmann-json are still required at build time):
```bash
# macOS
brew install cmake eigen nlohmann-json

# Ubuntu/Debian
sudo apt install cmake libeigen3-dev nlohmann-json3-dev
```

**2. Install Python build dependencies into your venv:**
```bash
pip install scikit-build-core pybind11
```

**3. Install marble:**
```bash
pip install -e . --no-build-isolation
```

After this, `import marble` works from anywhere in that environment.

## Python:  usage without installing

Add `build/python/` to your path at runtime:
```python
import sys
from pathlib import Path

# Insert the path to the `build` directory in PYTHONPATH
path_to_marble = ...
sys.path.insert(0, str(path_to_marble / "build" / "python"))

import marble
```

## VSCode: Python autocomplete

Autocomplete and type-checking are driven by the `.pyi` stubs in `typings/`, which are regenerated automatically each time you build `marble_python` (requires `pybind11-stubgen`).

Two config files in this repo wire everything up for VSCode automatically:

### `.vscode/settings.json`
```json
{
  "python.analysis.extraPaths": ["${workspaceFolder}/build/python"],
  "python.analysis.stubPath": "${workspaceFolder}/typings",
  "python.analysis.useLibraryCodeForTypes": true
}
```

- `extraPaths`:  tells Pylance where the compiled `.so` lives so it can be imported
- `stubPath`:  tells Pylance where to find the `.pyi` stub file for type info

### Install the Pylance extension
Install the **Pylance** extension in VSCode (`ms-python.vscode-pylance`). It picks up the above settings automatically.

If stubs are stale or missing, rebuild:
```bash
cmake --build --preset python --target marble_python
```

If `pybind11-stubgen` is not installed, CMake will warn but the build still succeeds. Ensure `pybind11-stubgen` is installed with:
```bash
pip install pybind11-stubgen
```
