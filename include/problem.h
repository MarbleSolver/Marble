#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

#include "utils.h"   // Mat, Vec, SMat

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
    Vec cost_gradient;
    double cost_const{0.0}; // Constant term in the cost

    // Constraint definitions:
    //   J_eq z + c_eq = 0
    //   J_ineq z + c_ineq >= 0
    //   0 <= L z + l  ⟂  R z + r >= 0
    SMat J_eq;
    Vec c_eq;
    SMat J_ineq;
    Vec c_ineq;
    SMat L;
    Vec l;
    SMat R;
    Vec r;

    //Constructors
    Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
            SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
            SMat L, Vec l, SMat R, Vec r);

    Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
            Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
            Mat L, Vec l, Mat R, Vec r);

    /* Returns the objective given primal variables z*/
    double obj(const Vec& z) const;

    /* Returns the residual of the equality constraints */
    Vec residual_eq(const Vec& z) const;

    /* Returns the residual of the inequality constraints */
    Vec residual_ineq(const Vec& z) const;

    /* Returns L*z + l */
    Vec residual_comp_L(const Vec& z) const;

    /* Returns R*z + r */
    Vec residual_comp_R(const Vec& z) const;

    /* Returns (L*z + l) .* (R*z + r) */
    Vec residual_comp(const Vec& z) const;

private:
    // Verify all stored blocks have mutually consistent dimensions. Called from
    // the constructors; throws std::invalid_argument on any mismatch
    void validate_dims() const;
};
