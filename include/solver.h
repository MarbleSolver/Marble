#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"

class Solver {
public:
    // Problem instance
    std::shared_ptr<Problem> prob;

    // Workspace for solve
    std::shared_ptr<Workspace> workspace;

    // Indices that define the structure of the KKT system (order of rows and columns)
    Eigen::VectorXi z_inds; // Original primal variables
    Eigen::VectorXi s_ineq_inds; // Inequality slacks
    Eigen::VectorXi s_comp_inds; // Complementarity slacks
    Eigen::VectorXi m_eq_inds; // Equality multipliers
    Eigen::VectorXi m_ineq_inds; // Inequality multipliers
    Eigen::VectorXi m_comp_inds; // Complementarity multipliers

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

    /**
     * Returns the workspace used by the solver
     */
    Workspace& get_workspace() {
        return *workspace;
    }
};