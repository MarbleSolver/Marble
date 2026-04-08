cmake_minimum_required(VERSION 3.10.2)
project(rcqp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(Eigen3 3.3 REQUIRED NO_MODULE)   # provides Eigen3::Eigen
find_package(QDLDL)
find_package(nlohmann_json CONFIG REQUIRED)

# Fetch QDLDL if not found
if(NOT QDLDL_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      qdldl
      GIT_REPOSITORY https://github.com/oxfordcontrol/QDLDL.git
      GIT_TAG master
    )
    FetchContent_MakeAvailable(qdldl)
    # If the fetched project provides a target, use it; otherwise adjust below.
endif()

# Create alias if needed for consistent naming
if(NOT TARGET qdldl::qdldl)
    if(TARGET qdldl)
        add_library(qdldl::qdldl ALIAS qdldl)
    endif()
endif()

# Add JlCxx for Julia bindings
find_package(JlCxx)
get_target_property(JlCxx_location JlCxx::cxxwrap_julia LOCATION)
get_filename_component(JlCxx_location ${JlCxx_location} DIRECTORY)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${JlCxx_location}")

message(STATUS "Found JlCxx at ${JlCxx_location}")

# Fetch a specific version of spdlog to avoid fmt/spdlog compatibility issues
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.12.0 # Or whatever version you prefer
)

# Make sure spdlog builds its bundled fmt instead of looking for the system one
set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "Use bundled fmt" FORCE)

FetchContent_MakeAvailable(spdlog)

add_definitions(-DSPDLOG_COMPILED_LIB)

# Setup library
add_library(rcqp SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/src/solver.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/problem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/workspace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/filter.cpp
)

target_include_directories(rcqp
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(rcqp PUBLIC cxx_std_17)

# Wrapper library for Julia
add_library(rcqp_wrapper SHARED wrapper.cpp)
target_link_libraries(rcqp_wrapper 
    PRIVATE 
    rcqp
    JlCxx::cxxwrap_julia)

# Link dependencies 
target_link_libraries(rcqp
  PUBLIC
    Eigen3::Eigen
    qdldl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)