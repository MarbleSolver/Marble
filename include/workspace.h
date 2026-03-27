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
    Eigen::Map<Vec> z;
    Eigen::Map<Vec> s_ineq;
    Eigen::Map<Vec> s_comp;
    Eigen::Map<Vec> m_eq;
    Eigen::Map<Vec> m_ineq;
    Eigen::Map<Vec> m_comp;

    // Current multiplier estimates for AL
    Vec m_eq_est;
    Vec m_ineq_est;
    Vec m_comp_est;
    
    // KKT residual, driven to 0 in each subproblem solve
    Vec kkt_residual;

    // Diagonal terms for s_ineq and s_comp stationarity, stored
    // to allow for updating primal regularizer cheaply
    Vec s_ineq_stationarity;
    Vec s_comp_stationarity;

    // Constraint evaluations
    Vec residual_eq;
    Vec residual_ineq;
    Vec residual_comp;

    /**
     * KKT system matrix, stored in sparse format with the structure intended to be fixed
     * after it is initialize by Solver::set_problem.
     */
    SMat kkt_system;

    // The KKT system is permuted and scaled for conditioning using the following vectors
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, QDLDL_int> amd_perm; // AMD permutation for KKT system, reduces in-fill
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> amd_perm_vec; // AMD permutation for KKT system, reduces in-fill
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> amd_iperm_vec; // Inverse AMD permutation for KKT system
    Vec scaling;

    // KKT step
    Vec newton_step;

    // Work arrays for QDLDL
    // Workspace arrays required by QDLDL
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> etree;
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> Lnz;
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> iwork;
    std::vector<QDLDL_bool> bwork;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> fwork;
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> Lp;
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> Li;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> Lx;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> D;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> Dinv;
    QDLDL_int sum_Lnz;

    // Empty constructor
    Workspace(const Problem& prob);

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