#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>

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

    // Constraint definitions, all of the form Jx + c = 0
    SMat J_eq;
    Vec c_eq;
    SMat J_ineq;
    Vec c_ineq;
    SMat J_comp_l;
    Vec c_comp_l;
    SMat J_comp_r;
    Vec c_comp_r;

    /**
     * Construct a new Problem given problem data
     * TODO: support empty variables
     */
    Problem(SMat cost_hessian, Vec cost_gradient,
            SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
            SMat J_comp_l, Vec c_comp_l, SMat J_comp_r, Vec c_comp_r);

    /**
     * Construct a new Problem given problem data
     * TODO: support empty variables
     */
    Problem(Mat cost_hessian, Vec cost_gradient,
            Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
            Mat J_comp_l, Vec c_comp_l, Mat J_comp_r, Vec c_comp_r);
};