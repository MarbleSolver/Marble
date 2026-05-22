#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

#include "qdldl.h"

using SMat = Eigen::SparseMatrix<double, Eigen::ColMajor, QDLDL_int>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

/**
 * A brief description of your class.
 * 
 * A more detailed description of what the class does.
 */
class Problem {
public:
    // Problem dimensions
    const int nz; // Number of decision variables in the original problem
    const int n_eq; // Number of equality constraints
    const int n_ineq; // Number of inequality constraints
    const int n_comp; // Number of complementarity constraints

    // Quadratic cost definition
    SMat cost_hessian;
    Vec cost_hessian_diag; // Needed for regularizer updating
    Vec cost_gradient;
    double cost_const{0.0}; // Constant term in the cost, not needed for optimization but useful for comparisons

    // Constraint definitions, all of the form Jx + c = 0
    SMat J_eq;
    Vec c_eq;
    SMat J_ineq;
    Vec c_ineq;
    SMat J_comp;
    Vec c_comp;

    Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
            SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
            SMat L, Vec l, SMat R, Vec r);

    Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
            Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
            Mat L, Vec l, Mat R, Vec r);
};