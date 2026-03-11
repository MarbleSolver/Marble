#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"

class Workspace {
public:
    /**
     * KKT system matrix, stored in sparse format with the structure intended to be fixed
     * after it is initialize by Solver::set_problem.
     */
    SMat kkt_system;

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
};