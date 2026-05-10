FROM julia:1.11

ENV DEBIAN_FRONTEND=noninteractive

# Compiler toolchain + C++ headers that the Marble build needs at compile time:
#   git               — required by CMake FetchContent to clone qdldl at configure time
#   libeigen3-dev     — linear algebra (matrix/vector types)
#   nlohmann-json3-dev — JSON (used for solver config/output)
#   python3-venv      — needed to create an isolated Python environment below
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3 \
    python3-venv \
    python3-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

# CxxWrap.jl is the Julia package that lets Julia call into a compiled C++ library.
# It provides the JlCxx CMake target that julia_bindings.cpp links against.
RUN julia -e 'using Pkg; Pkg.add("CxxWrap")'

# Create an isolated Python virtual environment at /opt/venv.
# This sidesteps the "externally-managed-environment" restriction on the
# system Python, which blocks pip from installing packages system-wide.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV

# Prepend the venv to PATH so every subsequent `python` / `pip` / `cmake`
# command in this file (and in the running container) uses the venv.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install cmake and pybind11 into the venv.
#   cmake   — apt only provides ~3.25; this project requires 3.28+, so we get
#             a newer version from PyPI (pip's cmake package bundles the binary)
#   pybind11 — the C++/Python bridge; CMake needs its cmake config files to
#              find and link pybind11 when building the Python extension module
#   numpy   — runtime dependency for working with the solver in Python
RUN pip install cmake pybind11 numpy

WORKDIR /marble
COPY . .

# Build both the Python and Julia bindings in one pass using the "all" preset
# defined in CMakePresets.json.
#
#   cmake --preset all
#     "configure" step: reads CMakeLists.txt, finds all dependencies, and
#     generates the actual build files (Makefiles / Ninja files) in build/.
#     Equivalent to:
#       cmake -B build \
#             -DCMAKE_BUILD_TYPE=Release \
#             -DMARBLE_BUILD_PYTHON=ON \
#             -DMARBLE_BUILD_JULIA=ON \
#             -DMARBLE_GENERATE_PYI=ON
#
#   cmake --build --preset all
#     "build" step: runs the compiler using the files generated above.
#     Produces:
#       build/python/marble.<cpython-tag>.so  — Python extension module
#       build/lib/libmarble_julia.so          — Julia shared library
#
RUN cmake --preset all && cmake --build --preset all

# Add the directory containing the compiled Python extension to PYTHONPATH
# so that `import marble` works inside the venv without a separate `pip install`.
ENV PYTHONPATH="/marble/build/python"

CMD ["/bin/bash"]
