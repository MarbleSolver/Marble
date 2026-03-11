# RCQP
A C++ solver for quadratic programs with linear complementarity constraints.

## Installation & Compilation

### MacOS
First install the CxxWrap Julia package:
```bash
julia -e 'using Pkg; Pkg.add("CxxWrap")'
```

Inside the repo root, run the following commands:
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH="$(julia -e 'using CxxWrap; print(CxxWrap.prefix_path())')" ..
cmake --build .
```
