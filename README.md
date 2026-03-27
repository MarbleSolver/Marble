# RCQP
A C++ solver for quadratic programs with linear complementarity constraints.

## Installation & Compilation

### MacOS and Ubuntu
First install the CxxWrap Julia package:
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
```
We require at least v0.17.5, which can be checked after installation using `Pkg.status("CxxWrap")`

Inside the repo root, run the following commands:
```bash
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(julia -e 'using CxxWrap; print(CxxWrap.prefix_path())')" .. && cmake --build .
```
