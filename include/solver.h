#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"
#include "filter.h"

class Solver {
public:
    struct Options {
        // KKT Inf norm must be less than this value for convergence
        double convergence_kkt_norm{1e-4};
        // Equality constraint violation Inf norm must be less than this value for convergence
        double convergence_eq_violation{1e-4};
        // Inequality constraint violation Inf norm must be less than this value for convergence
        double convergence_ineq_violation{1e-4};
        // Complementarity constraint violation Inf norm must be less than this value for convergence
        double convergence_comp_violation{1e-4};
        // KKT Inf norm must be less than this value to take an outer step in the algorithm
        double outer_step_kkt_norm{1e-6};
        // Initial AL penalty parameter
        double penalty_initial{1.0};
        // Maximum AL penalty parameter
        double penalty_max{1e6};
        // AL penalty parameter scaling factor, multiplies current penalty parameter
        double penalty_scaling{10.0};
        // Initial relaxation parameter for complementarity and inequality constraints
        double relaxation_initial{1e-1};
        // Minimum relaxation parameter for complementarity and inequality constraints
        double relaxation_min{1e-7};
        // Relaxation parameter scaling factor, multiplies current relaxation parameter
        double relaxation_scaling{0.5};
        // Maximum number of iterations for the solver, iterations refers to outer + inner iterations
        int max_iters{1000};
        // Maximum number of iterations for the filter linesearch
        int max_iters_linesearch{10};
        // (filter) Sufficient progress parameter for objective value decrease
        double gamma_objective{1e-5};
        // (filter) Sufficient progress parameter for constraint violation decrease
        double gamma_constraint{1e-5};

        Options() = default;
    };

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

    // Indices into the complementarity residual for extracting the comp residual violation for filter evaluation
    Eigen::VectorXi comp_L_inds;
    Eigen::VectorXi comp_R_inds;

    // Indices into the sparse KKT matrix valuePtr for jacobians of the KKT residual that are updated
    // based on the nonlinear terms (s_ineq, s_comp, relaxation_param, and penalty_param)
    Eigen::VectorXi s_ineq_s_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_ineq_m_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_s_comp_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_m_comp_inds;  
    Eigen::VectorXi penalty_inds;
    Eigen::VectorXi regularizer_inds;
    
    // Filter for linesearch
    // TODO: initialize filter with options, should be private member of solver
    std::shared_ptr<Filter> filter;

    Solver() : Solver(Options()) {}

    Solver(const Options& options)
        : options(options),
          filter(std::make_shared<Filter>(options.gamma_objective, options.gamma_constraint)),
          workspace(std::make_shared<Workspace>()) {}
    
    /**
     * Retraction map (elementwise)
     */
    Vec retract(const Vec& s, double sqrt_relax_param) const;

    /**
     * Retraction map derivative (elementwise)
     */
    Vec retract_deriv(const Vec& s, double sqrt_relax_param) const;

    /**
     * Retraction map second derivative (elementwise)
     */
    Vec retract_second_deriv(const Vec& s, double sqrt_relax_param) const;

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
     * @brief Update the KKT regularization terms on the first `n_vars` diagonal entries
     * 
     * @param regularizer Positive regularizer
     */
    void update_KKT_regularizer(const double regularizer);

    /**
     * Returns the workspace used by the solver
     */
    Workspace& get_workspace() {
        return *workspace;
    }

    /**
     * @brief Get the filter object
     * 
     * @return Filter& Filter object used for linesearch in the solver
     */
    Filter& get_filter() {
        return *filter;
    }

    Filter::Entry entry_from_solution(double sqrt_relax_param, double inv_penalty_param) const;

    /**
     * @brief Perform backtracking filter linesearch given a step direction
     * 
     * @param newton_step Search direction from Newton solve
     * @param sqrt_relax_param Square root of the complementarity and inequality relaxation parameter 
     * @param inv_penalty_param Inverse of the AL penalty parameter
     * @param max_iters Maximum number of iterations for the linesearch
     * @warning This function modifies the workspace solution to store the candidate solution, and updates the constraint residuals based on the candidate solution, which are used to evaluate the filter conditions. If the linesearch fails, the workspace solution is left at the last candidate solution evaluated.
     * @return true Linesearch succeeded, new iterate is stored in workspace x_candidate
     * @return false Linesearch failed
     */
    bool filter_linesearch(const Vec &newton_step, const double sqrt_relax_param, const double inv_penalty_param, int max_iters);

    /**
     * Solve the current problem instance 
     */
    bool solve(const Options &options);

private:
    // Solver options
    const Options options;

    // KKT system regularizers to try in Newton step
    const std::vector<double> kkt_system_regularizers = {
        0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
    };

    /**
     * Determine if the solver has converged based on KKT residual norm, constraint satisfaction
     */
    bool convergence(const Options &options);

    /**
     * Compute a search direction
     */
    Vec compute_newton_step(double kkt_system_regularizer);
};