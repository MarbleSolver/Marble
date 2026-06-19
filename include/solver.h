#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"
#include "filter.h"
#include <chrono>
#include <filesystem>
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
    int factorizations_ldlt;
    int factorizations_inertia;
    int factorizations_linesearch;
    double setup_time_s;
    double solve_time_s;
    Vec z;        // Primal solution
    Vec s_ineq;   // Inequality slack solution
    Vec s_comp;   // Complementarity slack solution
    Vec m_eq;     // Equality multiplier solution
    Vec m_ineq;   // Inequality multiplier solution
    Vec m_comp;   // Complementarity multiplier solution
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
        /// Output directory for solution and solve information
        std::filesystem::path output_dir{"/dev/null"};
        /// Verbosity level: 0=silent, 1=per-iteration table + footer
        int verbosity{0};
        /// Print a row every N iterations (only used when verbosity >= 1)
        int print_every{1};
        /// Write a JSONL debug log with iterates and solver state (one JSON object per line)
        bool debug{false};
        /// Path to the debug log file (used when debug=true)
        std::string debug_output_path{"solver_debug.jsonl"};
        /// Log every N iterations (1 = every iteration)
        int debug_log_every{1};

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
    // TODO: currently a duplicate of the one in Problems, remove
    Eigen::VectorXi comp_L_inds;
    Eigen::VectorXi comp_R_inds;

    // Indices into the sparse KKT matrix valuePtr for jacobians of the KKT residual that are updated
    // based on the nonlinear terms (s_ineq, s_comp, relaxation_param, and penalty_param)
    Eigen::VectorXi z_z_inds; // Diagonal matrix only for regularizer updates
    Eigen::VectorXi s_ineq_s_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_ineq_m_ineq_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_s_comp_inds; // Diagonal matrix
    Eigen::VectorXi s_comp_m_comp_inds;  
    Eigen::VectorXi penalty_inds;
    Eigen::VectorXi regularizer_inds;
    
    // Filter for linesearch
    // TODO: initialize filter with options, should be private member of solver
    std::shared_ptr<Filter> filter;

    Solver() : filter(std::make_shared<Filter>(options.gamma_objective, options.gamma_constraint)) {}
    
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

    Vec d_retract_d_relax_param(const Vec& s, double relax_param) const;
    Vec d_dretract_d_relax_param(const Vec& s, double relax_param) const;


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
     * Populates the KKT system, computes sparsity indexing
     */
    void set_problem(const Solver::Options& options);

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

    void update_residual_relax_grad(double relax_param);

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
    void update_KKT_comp(const Vec& s_comp, const Vec& m_comp, double sqrt_relax_param);

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

    // // Route 2: condensed inertia test. Returns true iff the FULL KKT system at this
    // // `reg` would have inertia (n_primals, n_duals, 0). Does NOT need a factorization.
    // // Equivalent to: update_KKT_primal_regularizer(reg); numerical_factorization(); check_inertia();
    // bool check_inertia_condensed(double reg, double sigma_tol = 1e-10) const;

    // // Smallest reg in kkt_system_regularizers that passes the condensed test
    // // (so the inner loop does at most one full factorization for the step).
    // double route2_min_regularizer() const;

    // double min_kkt_regularizer() const;

    // double comp_block_reg_min(double atol) const;

    double comp_block_reg_min(double sigma_tol) const;

    Mat condensed_reduced_hessian(double reg) const;

    double reduced_hessian_min_eig() const;

    double primal_KKT_posdef_reg(double atol) const;

    void relaxation_tangent_correction(double relax_param_new, double relax_param_old);

    /**
     * The KKT system should define a saddle point, with n_primal positive and n_dual negative eigenvalues
     * We can check this (called the inertia) after numerical_factorization() by checking the number of
     * positive and negative elements of D because LDLt factorizations preserve inertia
     */
    bool check_inertia();

    /**
     * Solve the KKT system using the factorized matrix, populating the solution
     * in workspace->newton_step.
     */
    void backsolve(Vec &step, const Vec& rhs);

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

    Filter::Entry entry_from_solution(double sqrt_relax_param, double inv_penalty_param) const;

    /**
     * Perform backtracking filter linesearch given a step direction
     * 
     * @param sqrt_relax_param Square root of the complementarity and inequality relaxation parameter 
     * @param inv_penalty_param Inverse of the AL penalty parameter
     * @param max_iters Maximum number of iterations for the linesearch
     * @warning This function modifies the workspace solution to store the candidate solution, and updates the constraint residuals based on the candidate solution, which are used to evaluate the filter conditions. If the linesearch fails, the workspace solution is restored to its original value before returning.
     * @return true Linesearch succeeded, new iterate is stored in workspace x_candidate
     * @return false Linesearch failed
     */
    bool filter_linesearch(const double sqrt_relax_param, const double inv_penalty_param, int max_iters);

    /**
     * Solve the current problem instance.
     */
    SolveResult solve();

    /**
     * Determine if the solver has converged based on KKT residual norm, constraint satisfaction
     */
    bool convergence(const Options &options);

    // Solver options
    Options options;

private:
    // Counts numerical_factorization() calls; reset at the start of each solve()
    int n_factorizations{0};

    // Counts the # of factorizations due to LDLT numerical failure
    int n_factorizations_ldlt{0};

    // Counts the # of factorizations due to incorrect inertia
    int n_factorizations_inertia{0};

    // Counts the # of factorizations due to linesearch failure
    int n_factorizations_linesearch{0};

    // Timing for set_problem and solve, in seconds
    double setup_time_s{0.0};
    double solve_time_s{0.0};

    // KKT system regularizers to try in Newton step
    const std::vector<double> kkt_system_regularizers = {
        0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
    };
};
