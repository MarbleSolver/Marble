#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

using SMat = Eigen::SparseMatrix<double>;
using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

class Problem {
public:
    // Problem dimensions
    const int nz; // Number of decision variables in the original problem
    const int n_eq; // Number of equality constraints
    const int n_ineq; // Number of inequality constraints
    const int n_comp; // Number of complementarity constraints

    // Quadratic cost definition
    SMat cost_hessian;
    Vec cost_gradient;
    double cost_const{0.0}; // Constant term in the cost, not needed for optimization but useful for comparisons

    // Constraint definitions, all of the form Jx + c = 0
    SMat J_eq;
    Vec c_eq;
    SMat J_ineq;
    Vec c_ineq;
    SMat J_comp;
    Vec c_comp;

    /**
     * Construct a new Problem given problem data
     * TODO: support empty variables
     */
    Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
            SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
            SMat J_comp, Vec c_comp);

    /**
     * Construct a new Problem given problem data
     * TODO: support empty variables
     */
    Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
            Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
            Mat J_comp, Vec c_comp);
};