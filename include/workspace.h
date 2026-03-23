#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "qdldl.h"

class Workspace {
public:
    // Current solution vectors
    Vec z;
    Vec s_ineq;
    Vec s_comp;
    Vec m_eq;
    Vec m_ineq;
    Vec m_comp;

    // Current multiplier estimates for AL
    Vec m_eq_est;
    Vec m_ineq_est;
    Vec m_comp_est;
    
    // KKT residual, driven to 0 in each subproblem solve
    Vec kkt_residual;

    /**
     * KKT system matrix, stored in sparse format with the structure intended to be fixed
     * after it is initialize by Solver::set_problem.
     */
    SMat kkt_system;

    // KKT step
    Vec newton_step;

    // Work arrays for QDLDL
    // Workspace arrays required by QDLDL
    std::vector<QDLDL_int> etree;
    std::vector<QDLDL_int> Lnz;
    std::vector<QDLDL_int> iwork;
    std::vector<QDLDL_bool> bwork;
    std::vector<QDLDL_float> fwork;
    std::vector<QDLDL_int> Lp;
    std::vector<QDLDL_int> Li;
    std::vector<QDLDL_float> Lx;
    std::vector<QDLDL_float> D;
    std::vector<QDLDL_float> Dinv;
    QDLDL_int sum_Lnz;

    // Empty constructor
    Workspace();

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