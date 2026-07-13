#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"
#include "filter.h"
#include <chrono>
#include <utility>

/**
 * A brief description of your class.
 * 
 * A more detailed description of what the class does.
 */
struct SolveResult {
    bool converged;
    int iterations;
    int iterations_outer;
    int iterations_inner;
    int factorizations;
    double setup_time_s;
    double solve_time_s;
    Vec z;        // Primal solution
    Vec s_ineq;   // Inequality slack solution
    Vec s_comp;   // Complementarity slack solution
    Vec m_eq;     // Equality multiplier solution
    Vec m_ineq;   // Inequality multiplier solution
    Vec m_comp_L; // Complementarity multiplier solution (left / L_comp side)
    Vec m_comp_R; // Complementarity multiplier solution (right / R_comp side)
};

class Solver {
public:
    /**
     * Solver options
     */
    struct Options {
        /// KKT Inf norm must be less than this value for convergence
        double convergence_kkt_norm{1e-4};
        /// Equality constraint violation Inf norm must be less than this value for convergence
        double convergence_eq_violation{1e-4};
        /// Inequality constraint violation Inf norm must be less than this value for convergence
        double convergence_ineq_violation{1e-4};
        /// Complementarity constraint violation Inf norm must be less than this value for convergence
        double convergence_comp_violation{1e-5};
        /// KKT Inf norm must be less than this value to take an outer step in the algorithm
        double outer_step_kkt_norm{1e-6};
        /// Initial AL penalty parameter
        double penalty_initial{10.0};
        /// Maximum AL penalty parameter
        double penalty_max{1e6};
        /// AL penalty parameter scaling factor, multiplies current penalty parameter
        double penalty_scaling{10.0};
        /// Initial relaxation parameter for complementarity and inequality constraints
        double relaxation_initial{1e-1};
        /// Minimum relaxation parameter for complementarity and inequality constraints
        double relaxation_min{1e-7};
        /// Relaxation parameter scaling factor, multiplies current relaxation parameter
        double relaxation_scaling{0.5};
        /// Maximum number of iterations for the solver, iterations refers to outer + inner iterations
        int max_iters{1000};
        /// Maximum number of iterations for the filter linesearch
        int max_iters_linesearch{10};
        /// (filter) Sufficient progress parameter for objective value decrease
        double gamma_objective{1e-5};
        /// (filter) Sufficient progress parameter for constraint violation decrease
        double gamma_constraint{1e-5};
        /// Number of ruiz iterations for scaling
        int ruiz_iterations{10};
        /// Warm-start each inner Newton step's inertia regularizer from the last successful value
        /// (false restarts the regularizer from 0 at every inner step)
        bool inertia_warmstart{true};
        /// Initialize s_comp with small random noise (true) or zeros (false)
        bool comp_init_random{true};
        /// Seed for the small random noise used to initialize s_comp
        /// (negative leaves the RNG unseeded, i.e. non-deterministic across runs)
        int comp_init_seed{-1};
        /// Verbosity level, maps to spdlog thresholds:
        ///   0 = silent
        ///   1 = errors only (reason for non-convergence)
        ///   2 = + warnings (regularization bumps, penalty capped, ...)
        ///   3 = + solver header and per-iteration table
        int verbosity{0};
        /// Print a per-iteration row every N iterations (only used at verbosity >= 3)
        int print_every{1};

        Options() = default;
    };

    // Problem instance
    std::shared_ptr<const Problem> prob;

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
    Eigen::VectorXi m_comp_L_inds; // Complementarity multipliers (left / L_comp side)
    Eigen::VectorXi m_comp_R_inds; // Complementarity multipliers (right / R_comp side)

    // Indices into the sparse KKT matrix valuePtr for jacobians of the KKT residual that are updated
    // based on the nonlinear terms (s_ineq, s_comp, relaxation_param, and penalty_param)
    Eigen::VectorXi z_z_inds; // Diagonal matrix only for regularizer updates
    Eigen::VectorXi s_ineq_s_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_ineq_m_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_s_comp_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_m_comp_L_inds; // Diagonal matrix (s_comp row, left multiplier)
    Eigen::VectorXi s_comp_m_comp_R_inds; // Diagonal matrix (s_comp row, right multiplier)
    Eigen::VectorXi penalty_inds;
    Eigen::VectorXi regularizer_inds;
    
    // Filter for linesearch
    // TODO: initialize filter with options, should be private member of solver
    std::shared_ptr<Filter> filter;

    Solver() : filter(std::make_shared<Filter>(options.gamma_objective, options.gamma_constraint)) {}
    
    /**
     * Retraction map (elementwise)
     */
    Vec retract(const Eigen::Ref<const Vec>& s, double relax_param) const;

    /**
     * Retraction map derivative (elementwise)
     */
    Vec retract_deriv(const Eigen::Ref<const Vec>& s, double relax_param) const;

    /**
     * Retraction map second derivative (elementwise)
     */
    Vec retract_second_deriv(const Eigen::Ref<const Vec>& s, double relax_param) const;

    /**
     * Derivative of the retraction map wrt the relaxation parameter kappa (elementwise)
     */
    Vec retract_drelax(const Eigen::Ref<const Vec>& s, double relax_param) const;

    /**
     * Mixed derivative of the retraction map wrt s and the relaxation parameter kappa (elementwise)
     */
    Vec retract_deriv_drelax(const Eigen::Ref<const Vec>& s, double relax_param) const;

    /**
     * Ruiz equilibration for current problem data using copies of H and J_*.
     * Writes concatenated scaling vector [d; e_eq; e_ineq; e_comp] into workspace->scaling.
     */
    void ruiz_equilibration(int niter = 10);

    /**
     * Sets the problem for the solver, computes sparsity indexing. The Problem
     * is validated on construction (see Problem's constructors), so callers reach
     * this with consistent dimensions.
     */
    void set_problem(Problem problem, Solver::Options& options);

    /**
     * Returns the problem currently set for the solver
     */
    const Problem& get_problem() {
        return *prob;
    }

    /**
     * Construct and initialize KKT sparsity
     */
    void initialize_kkt_sparsity();

    /**
     * Compute KKT residual given the current guess stored in the workspace
     */
    void update_KKT_residual(double relax_param, double inv_penalty_param);

    void update_dKKT_residual_drelax(double relax_param);

    /**
     * Update KKT system given the current guess stored in the workspace
     */
    void update_KKT_system(double relax_param, double inv_penalty_param);

    /**
     * Update the KKT terms associated with s_ineq (no dependence on m_ineq)
     */
    void update_KKT_ineq(const Eigen::Ref<const Vec>& s_ineq, double relax_param);

    /**
     * Update the KKT terms associated with s_comp and m_comp
     */
    void update_KKT_comp(const Eigen::Ref<const Vec>& s_comp, const Eigen::Ref<const Vec>& m_comp_L,
                         const Eigen::Ref<const Vec>& m_comp_R, double relax_param);

    /**
     * Update the KKT penalty diagonal
     */
    void update_KKT_penalty(const double inv_penalty_param);

    /**
     * Update the KKT regularizer
     */
    void update_KKT_primal_regularizer(const double reg);

    /**
     * Perform an analytical factorization of the KKT system using QDLDL
     */
    bool analytical_factorization();

    /**
     * Perform an numerical factorization of the KKT system using QDLDL
     */
    bool numerical_factorization();

    /**
     * The KKT system should define a saddle point, with n_primal positive and n_dual negative eigenvalues
     * We can check this (called the inertia) after numerical_factorization() by checking the number of
     * positive and negative elements of D because LDLt factorizations preserve inertia
     */
    bool check_inertia();

    /**
     * Solve K x = b using the currently factorized KKT matrix, where K is the
     * unscaled system (the Ruiz scaling stored in workspace->scaling is applied
     * internally). Populates x with the solution.
     *
     * @param b right-hand side vector (length n_vars)
     * @param x solution vector to populate (length n_vars); must not alias b
     */
    void backsolve(const Eigen::Ref<const Vec>& b, Vec& x);

    /**
     * Compute AMD ordering
     */
    void compute_amd_ordering();

    /**
     * Returns the workspace used by the solver
     */
    Workspace& get_workspace() {
        return *workspace;
    }

    /**
     * Get the filter object
     * 
     * @return Filter& Filter object used for linesearch in the solver
     */
    Filter& get_filter() {
        return *filter;
    }

    Filter::Entry entry_from_solution(double relax_param, double inv_penalty_param) const;

    /**
     * Perform backtracking filter linesearch given a step direction
     * 
     * @param relax_param Complementarity and inequality relaxation parameter (kappa)
     * @param inv_penalty_param Inverse of the AL penalty parameter
     * @param max_iters Maximum number of iterations for the linesearch
     * @warning This function modifies the workspace solution to store the candidate solution, and updates the constraint residuals based on the candidate solution, which are used to evaluate the filter conditions. If the linesearch fails, the workspace solution is restored to its original value before returning.
     * @return true Linesearch succeeded, new iterate is stored in workspace x_candidate
     * @return false Linesearch failed
     */
    bool filter_linesearch(const double relax_param, const double inv_penalty_param, int max_iters);

    /**
     * Solve the current problem instance.
     */
    SolveResult solve();

    /**
     * Determine if the solver has converged based on KKT residual norm, constraint satisfaction
     */
    bool convergence() const;

    // Solver options
    Options options;

private:
    // Counts numerical_factorization() calls; reset at the start of each solve()
    int n_factorizations{0};

    // Timing for set_problem and solve, in seconds
    double setup_time_s{0.0};
    double solve_time_s{0.0};

    // KKT system regularizers to try in Newton step
    const std::vector<double> kkt_system_regularizers = {
        0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
    };
};
