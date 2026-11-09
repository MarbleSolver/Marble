#pragma once
#include <stdexcept>

/**
 * Base class of every error Marble throws (Python: marble.MarbleError, a RuntimeError).
 *
 * Marble deliberately never throws std::invalid_argument, std::out_of_range and the like:
 * inside a Julia process those types resolve to the RTTI of Julia's bundled libstdc++, so
 * they can't be caught as std::exception and CxxWrap aborts the process instead of raising
 * a Julia error. std::runtime_error and types derived from it are unaffected.
 */
struct MarbleError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * Invalid problem data, option values or method arguments
 * (Python: marble.InvalidInputError, which is also a ValueError).
 */
struct InvalidInputError : MarbleError {
    using MarbleError::MarbleError;
};
