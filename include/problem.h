#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <iostream>

#include "utils.h"

/**
 * Quadratic program data in Marble's native constraint form
 */
class Problem {
public:
    const int nz;     // Number of decision variables
    const int n_eq;   // Number of equality constraints
    const int n_ineq; // Number of inequality constraints
    const int n_comp; // Number of complementarity constraints

    SMat cost_hessian;
    Vec cost_gradient;
    double cost_const{0.0};

    SMat J_eq;
    Vec b_eq;
    SMat J_ineq;
    Vec b_ineq;
    SMat L;
    Vec l;
    SMat R;
    Vec r;

    /**
     * Construct a problem from sparse matrices
     *
     * @param cost_hessian Quadratic objective Hessian
     * @param cost_gradient Linear objective coefficient
     * @param cost_const Constant objective offset
     * @param J_eq Equality constraint matrix
     * @param b_eq Equality constraint offset
     * @param J_ineq Inequality constraint matrix
     * @param b_ineq Inequality constraint offset
     * @param L Left complementarity matrix
     * @param l Left complementarity offset
     * @param R Right complementarity matrix
     * @param r Right complementarity offset
     */
    Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
            SMat J_eq, Vec b_eq, SMat J_ineq, Vec b_ineq,
            SMat L, Vec l, SMat R, Vec r);

    /**
     * Construct a problem from dense matrices
     *
     * @param cost_hessian Quadratic objective Hessian
     * @param cost_gradient Linear objective coefficient
     * @param cost_const Constant objective offset
     * @param J_eq Equality constraint matrix
     * @param b_eq Equality constraint offset
     * @param J_ineq Inequality constraint matrix
     * @param b_ineq Inequality constraint offset
     * @param L Left complementarity matrix
     * @param l Left complementarity offset
     * @param R Right complementarity matrix
     * @param r Right complementarity offset
     */
    Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
            Mat J_eq, Vec b_eq, Mat J_ineq, Vec b_ineq,
            Mat L, Vec l, Mat R, Vec r);

    /**
     * Evaluate the quadratic objective
     *
     * @param z Primal decision vector
     * @return Objective value at z
     */
    double obj(const Vec& z) const;

    /**
     * Evaluate equality residuals
     *
     * @param z Primal decision vector
     * @return J_eq * z + b_eq
     */
    Vec residual_eq(const Vec& z) const;

    /**
     * Evaluate inequality residuals
     *
     * @param z Primal decision vector
     * @return J_ineq * z + b_ineq
     */
    Vec residual_ineq(const Vec& z) const;

    /**
     * Evaluate left complementarity residuals
     *
     * @param z Primal decision vector
     * @return L * z + l
     */
    Vec residual_comp_L(const Vec& z) const;

    /**
     * Evaluate right complementarity residuals
     *
     * @param z Primal decision vector
     * @return R * z + r
     */
    Vec residual_comp_R(const Vec& z) const;

    /**
     * Evaluate elementwise complementarity products
     *
     * @param z Primal decision vector
     * @return Elementwise product of left and right complementarity residuals
     */
    Vec residual_comp(const Vec& z) const;

private:
    /**
     * Check that all matrices and vectors have compatible dimensions
     *
     * @throws std::invalid_argument if any dimension is inconsistent
     */
    void validate_dims() const;
};
