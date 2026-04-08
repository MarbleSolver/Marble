# Marble
[![CMake Build](https://github.com/MarbleSolver/RCQP/actions/workflows/cmake-build.yml/badge.svg)](https://github.com/MarbleSolver/RCQP/actions/workflows/cmake-build.yml)

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
CMake auto-detects JlCxx by calling `julia -e 'using CxxWrap; print(CxxWrap.prefix_path())'`

Output: `build/lib/librcqp_julia.dylib` (macOS) / `build/lib/librcqp_julia.so` (Linux)

### Both
```bash
cmake --preset all
cmake --build --preset all
```

## Docker

The included `Dockerfile` builds both Python and Julia bindings inside a self-contained image.

```bash
docker build -t rcqp .
docker run -it rcqp
```

Inside the container:
- `python` uses the venv at `/opt/venv`; `import rcqp` works immediately
- Julia shared library is at `build/lib/librcqp_julia.so`

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

**3. Install rcqp:**
```bash
pip install -e . --no-build-isolation
```

After this, `import rcqp` works from anywhere in that environment.

## Python:  usage without installing

Add `build/python/` to your path at runtime:
```python
import sys
from pathlib import Path

# Insert the path to the `build` directory in PYTHONPATH
path_to_rcqp = ...
sys.path.insert(0, str(path_to_rcqp / "build" / "python"))

import rcqp
```

## VSCode: Python autocomplete

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

- `extraPaths`:  tells Pylance where the compiled `.so` lives so it can be imported
- `stubPath`:  tells Pylance where to find the `.pyi` stub file for type info

### Install the Pylance extension
Install the **Pylance** extension in VSCode (`ms-python.vscode-pylance`). It picks up the above settings automatically.

If stubs are stale or missing, rebuild:
```bash
cmake --build --preset python --target rcqp_python
```

If `pybind11-stubgen` is not installed, CMake will warn but the build still succeeds. Ensure `pybind11-stubgen` is installed with:
```bash
pip install pybind11-stubgen
```
