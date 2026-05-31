#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include <cstring>

/**
 * Contains all elements computed during each solver iteration, including current solution estimates, KKT residual terms and hessians, and Newton step
 * 
 * The workspace is initialized in Solver::set_problem() after the problem dimensions are known, and is updated during each iteration of the solver, minimizing allocations,
 * though more work needs to be done to fully leverage this.
 */
class Workspace {
public:
    Vec solution;               /** Stacked solution vector [z; s_ineq; s_comp; m_eq; m_ineq; m_comp] */

    // Views into the solution vector
    Eigen::Map<Vec> z;          /** Primal variables */
    Eigen::Map<Vec> s_ineq;     /** Inequality slacks (in the retraction domain) */
    Eigen::Map<Vec> s_comp;     /** Complementarity slacks (in the retraction domain) */
    Eigen::Map<Vec> m_eq;       /** Equality multipliers */
    Eigen::Map<Vec> m_ineq;     /** Inequality multipliers */
    Eigen::Map<Vec> m_comp;     /** Complementarity multipliers */

    // Current multiplier estimates for AL
    Vec m_eq_est;               /** Equality multiplier estimates for AL */
    Vec m_ineq_est;             /** Inequality multiplier estimates for AL */
    Vec m_comp_est;             /** Complementarity multiplier estimates for AL */

    
    Vec kkt_residual; /** KKT residual, driven to 0 in each subproblem solve */

    // Diagonal terms for s_ineq and s_comp stationarity, stored
    // to allow for updating primal regularizer cheaply
    Vec s_ineq_stationarity; /** Inequality stationarity terms */
    Vec s_comp_stationarity; /** Complementarity stationarity terms */

    // Constraint evaluations
    Vec residual_eq;        /** Equality constraint residuals */
    Vec residual_ineq;      /** Inequality constraint residuals */
    Vec residual_comp;      /** Complementarity constraint residuals */

    // Relaxation and penalty parameters
    double relax_param;    /** Relaxation parameter for the retraction map */
    double penalty_param;  /** Penalty parameter for the augmented Lagrangian */

    /**
     * KKT system matrix, stored in sparse format with the structure intended to be fixed
     * after it is initialize by Solver::set_problem.
     */
    SMat kkt_system;

    // The KKT system is permuted and scaled for conditioning using the following vectors
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, QDLDL_int> amd_perm; /** AMD permutation for KKT system, reduces in-fill */
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> amd_perm_vec; /** AMD permutation for KKT system, reduces in-fill */
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> amd_iperm_vec; /** Inverse AMD permutation for KKT system */
    Vec scaling; /** Diagonal scaling for KKT system, improves conditioning (ruiz equilibration) */

    // KKT step
    Vec newton_step; /** Newton step for the KKT system (possibly computed with regularization) */

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
     * @warning 
     * This function does not check whether the elements being inserted 
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
     * @warning 
     * This function assumes that the KKT system is in the compressed format
     * 
     * @param row row index
     * @param col col_index
     */
    int findValuePtrIndex(int row, int col);
};