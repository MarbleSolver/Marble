#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"

class Solver {
public:
    // Problem instance
    std::shared_ptr<Problem> prob;

    // Workspace for solve
    std::shared_ptr<Workspace> workspace;

    //Dimensions for the subproblems solved during the solver
    int n_primals;
    int n_duals;
    int n_vars;

    // Indices that define the structure of the KKT system (order of rows and columns)
    Eigen::VectorXi z_inds; // Original primal variables
    Eigen::VectorXi s_ineq_inds; // Inequality slacks
    Eigen::VectorXi s_comp_inds; // Complementarity slacks
    Eigen::VectorXi m_eq_inds; // Equality multipliers
    Eigen::VectorXi m_ineq_inds; // Inequality multipliers
    Eigen::VectorXi m_comp_inds; // Complementarity multipliers

    // Indices into the sparse KKT matrix valuePtr for jacobians of the KKT residual that are updated
    // based on the nonlinear terms (s_ineq, s_comp, relaxation_param, and penalty_param)
    Eigen::VectorXi s_ineq_s_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_ineq_m_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_s_comp_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_m_comp_inds;  
    Eigen::VectorXi penalty_inds;

    /**
     * Construct a solver instance
     * TODO: take in options
     */
    Solver(); 

    
    /**
     * Retraction map (elementwise)
     */
    Vec retract(const Vec& s, double sqrt_relax_param);

    /**
     * Retraction map derivative (elementwise)
     */
    Vec retract_deriv(const Vec& s, double sqrt_relax_param);

    /**
     * Retraction map second derivative (elementwise)
     */
    Vec retract_second_deriv(const Vec& s, double sqrt_relax_param);

    /**
     * Sets the problem for the solver, populates the KKT system, computes sparsity indexing
     */
    void set_problem(Problem& prob);

    /**
     * Returns the problem currently set for the solver
     */
    Problem& get_problem() {
        return *prob;
    }

    /**
     * Construct and initialize KKT sparsity
     */
    void initialize_kkt_sparsity();

    /**
     * Compute KKT residual given the current guess stored in the workspace
     */
    void update_KKT_residual(double sqrt_relax_param, double inv_penalty_param);

    /**
     * Update KKT system given the current guess stored in the workspace
     */
    void update_KKT_system(double sqrt_relax_param, double inv_penalty_param);

    /**
     * Update the KKT terms associated with s_ineq (no dependence on m_ineq)
     */
    void update_KKT_ineq(const Vec& s_ineq, double sqrt_relax_param);

    /**
     * Update the KKT terms associated with s_comp and m_comp
     */
    void update_KKT_comp(const Vec& s_comp, Vec m_comp, double sqrt_relax_param);

    /**
     * Update the KKT penalty diagonal
     */
    void update_KKT_penalty(const double inv_penalty_param);

    /**
     * Perform an analytical factorization of the KKT system using QDLDL
     */
    void analytical_factorization();

    /**
     * Returns the workspace used by the solver
     */
    Workspace& get_workspace() {
        return *workspace;
    }
};