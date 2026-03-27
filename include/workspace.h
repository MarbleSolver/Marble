#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include <cstring>

class Workspace {
public:
    // Stacked solution vector [z; s_ineq; s_comp; m_eq; m_ineq; m_comp]
    Vec solution;

    // Views into the solution vector
    Vec z;
    Vec s_ineq;
    Vec s_comp;
    Vec m_eq;
    Vec m_ineq;
    Vec m_comp;

    Vec newton_step;

    // Current multiplier estimates for AL
    Vec m_eq_est;
    Vec m_ineq_est;
    Vec m_comp_est;
    
    // KKT residual, driven to 0 in each subproblem solve
    Vec kkt_residual;

    // Constraint evaluations
    Vec residual_eq;
    Vec residual_ineq;
    Vec residual_comp;

    /**
     * KKT system matrix, stored in sparse format with the structure intended to be fixed
     * after it is initialize by Solver::set_problem.
     */
    SMat kkt_system;

    // Empty constructor
    Workspace() = default;

    /**
     * Inserts a sparse block matrix into a sparse matrix represented as a 
     * triplet list given the top right corner of the block and the block matrix]
     * while preserving the sparsity pattern of the original block matrix.
     * 
     * @warning This function does not check whether the elements being inserted 
     * already exist in the triplet list.
     * 
     * @param triplets the triplet list representing the sparse matrix to be modified
     * @param block the block matrix to be inserted
     * @param row_start the starting row index of the block in the sparse matrix
     * @param col_start the starting column index of the block in the sparse matrix
     */
    void appendBlockTriplets(std::vector<Eigen::Triplet<double>>& triplets, const SMat& block, int row_start, int col_start);

    /**
     * Given a row and column index into kkt_system, find the data index in the underlying
     * kkt_system.valuePtr() array, returning -1 if it doesn't exist.
     * 
     * @warning this function assumes that the KKT system is in the compressed format
     * 
     * @param row row index
     * @param col col_index
     */
    int findValuePtrIndex(int row, int col);
};