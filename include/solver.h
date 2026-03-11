#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"

class Solver {
public:
    std::shared_ptr<Problem> prob;

    /**
     * Construct a solver instance
     * TODO: take in options
     */
    Solver(); 

    /**
     * Sets the problem for the solver, populates the KKT system, computes sparsity indexing
     */
    void set_problem(Problem& prob);

    /**
     * Returns the problem currently set for the solver
     */
    Problem& get_problem() {
        return *prob;
    }
};