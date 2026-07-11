cmake_minimum_required(VERSION 3.28)  # 3.28 needed for EXCLUDE_FROM_ALL in FetchContent_Declare

project(marble LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Build options
# ---------------------------------------------------------------------------
option(MARBLE_BUILD_PYTHON  "Build Python bindings"                   ON)
option(MARBLE_BUILD_JULIA   "Build Julia bindings"                    ON)
option(MARBLE_GENERATE_PYI  "Generate .pyi stubs via pybind11-stubgen" ON)

# ---------------------------------------------------------------------------
# Output directories  (keeps build/ tidy)
# ---------------------------------------------------------------------------
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
include(FetchContent)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json CONFIG REQUIRED)

# QDLDL – prefer a system install, otherwise fetch as static-only.
# EXCLUDE_FROM_ALL suppresses QDLDL's cmake --install rules so they don't
# interfere with pip install / scikit-build-core (requires CMake 3.28).
find_package(qdldl CONFIG QUIET)
if(NOT qdldl_FOUND)
    set(QDLDL_BUILD_STATIC_LIB ON  CACHE BOOL "" FORCE)
    set(QDLDL_BUILD_SHARED_LIB OFF CACHE BOOL "" FORCE)
    set(QDLDL_BUILD_DEMO_EXE   OFF CACHE BOOL "" FORCE)
    set(QDLDL_UNITTESTS        OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        qdldl
        GIT_REPOSITORY https://github.com/osqp/qdldl.git
        GIT_TAG        master
        EXCLUDE_FROM_ALL
    )
    FetchContent_MakeAvailable(qdldl)
endif()

# Normalise QDLDL to a single interface target regardless of how it was found
add_library(marble_qdldl INTERFACE)
if(TARGET qdldl::qdldlstatic)
    target_link_libraries(marble_qdldl INTERFACE qdldl::qdldlstatic)
elseif(TARGET qdldl::qdldl)
    target_link_libraries(marble_qdldl INTERFACE qdldl::qdldl)
elseif(TARGET qdldlstatic)
    target_link_libraries(marble_qdldl INTERFACE qdldlstatic)
elseif(TARGET qdldl)
    target_link_libraries(marble_qdldl INTERFACE qdldl)
else()
    message(FATAL_ERROR "Could not find a usable QDLDL target")
endif()

# ---------------------------------------------------------------------------
# Core library  (static → gets bundled into each wrapper, no rpath issues)
# ---------------------------------------------------------------------------
add_library(marble STATIC
    src/solver.cpp
    src/problem.cpp
    src/workspace.cpp
    src/filter.cpp
)
target_include_directories(marble PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(marble PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json marble_qdldl)

# ---------------------------------------------------------------------------
# Python bindings  (cmake -DMARBLE_BUILD_PYTHON=ON  — default ON)
# ---------------------------------------------------------------------------
if(MARBLE_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.13.6
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    # The compiled extension is the private submodule `marble._core`; the pure
    # Python `marble` package (python/marble/) wraps it with the high-level API.
    pybind11_add_module(marble_python MODULE bindings/python_bindings.cpp)
    target_link_libraries(marble_python PRIVATE marble)
    set_target_properties(marble_python PROPERTIES
        OUTPUT_NAME            _core
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/marble
    )

    # Stage the pure Python package next to the freshly built extension so that
    # adding build/python to sys.path is enough to `import marble` from the tree.
    add_custom_command(TARGET marble_python POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/python/marble ${CMAKE_BINARY_DIR}/python/marble
        COMMENT "Staging marble package into build/python/marble"
        VERBATIM
    )

    # .pyi stub generation for the compiled core (VSCode / Pylance autocomplete)
    if(MARBLE_GENERATE_PYI)
        find_program(PYBIND11_STUBGEN pybind11-stubgen)
        if(PYBIND11_STUBGEN)
            add_custom_command(TARGET marble_python POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E env
                        PYTHONPATH=${CMAKE_BINARY_DIR}/python
                        ${PYBIND11_STUBGEN} marble._core
                        --output-dir ${CMAKE_SOURCE_DIR}/python
                        --numpy-array-remove-parameters
                        --ignore-unresolved-names "^(m|n)$"
                COMMAND ${CMAKE_COMMAND} -E copy
                        ${CMAKE_SOURCE_DIR}/python/marble/_core.pyi
                        ${CMAKE_BINARY_DIR}/python/marble/_core.pyi
                COMMENT "Generating marble/_core.pyi stubs"
                VERBATIM
            )
        else()
            message(WARNING
                "pybind11-stubgen not found — .pyi stubs won't be regenerated. "
                "Run: pip install pybind11-stubgen")
        endif()
    endif()

    # Wheel layout: site-packages/marble/{__init__.py, _core.so}
    install(TARGETS marble_python LIBRARY DESTINATION marble)
endif()

# ---------------------------------------------------------------------------
# Julia bindings  (cmake -DMARBLE_BUILD_JULIA=ON)
# ---------------------------------------------------------------------------
if(MARBLE_BUILD_JULIA)
    # Auto-detect JlCxx prefix by asking Julia where CxxWrap installed it.
    # This works on any machine that has Julia + CxxWrap.jl without any
    # hardcoded paths. Users can still override with -DJlCxx_DIR=<path>.
    if(NOT DEFINED JlCxx_DIR)
        find_program(JULIA_EXECUTABLE julia)
        if(JULIA_EXECUTABLE)
            execute_process(
                COMMAND ${JULIA_EXECUTABLE} --startup-file=no -e
                        "using CxxWrap; print(CxxWrap.prefix_path())"
                OUTPUT_VARIABLE _jlcxx_prefix
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(_jlcxx_prefix)
                set(JlCxx_DIR "${_jlcxx_prefix}/lib/cmake/JlCxx"
                    CACHE PATH "Path to JlCxxConfig.cmake (auto-detected from Julia)")
                message(STATUS "Auto-detected JlCxx_DIR: ${JlCxx_DIR}")
            else()
                message(FATAL_ERROR
                    "Could not detect JlCxx prefix from Julia.\n"
                    "Make sure CxxWrap.jl is installed: julia -e 'using Pkg; Pkg.add(\"CxxWrap\")'\n"
                    "Or set JlCxx_DIR manually: cmake -DJlCxx_DIR=<path/to/cmake/JlCxx>")
            endif()
        else()
            message(FATAL_ERROR
                "julia executable not found.\n"
                "Install Julia from https://julialang.org/downloads or set JlCxx_DIR manually.")
        endif()
    endif()

    find_package(JlCxx REQUIRED)
    add_library(marble_julia SHARED bindings/julia_bindings.cpp)
    target_link_libraries(marble_julia PRIVATE marble JlCxx::cxxwrap_julia)
    set_target_properties(marble_julia PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()
