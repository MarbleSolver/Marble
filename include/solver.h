#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"
#include "filter.h"
#include <filesystem>
#include <utility>
#include "spdlog/spdlog.h"
#include "fmt/format.h"

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
        double convergence_comp_violation{1e-5};
        // KKT Inf norm must be less than this value to take an outer step in the algorithm
        double outer_step_kkt_norm{1e-6};
        // Initial AL penalty parameter
        double penalty_initial{10.0};
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
        // Number of ruiz iterations for scaling
        double ruiz_iterations{10};
        // Output directory for solution and solve information
        std::filesystem::path output_dir{"/dev/null"};
        // Verbosity level
        int verbosity{0};

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

    // Track iterations
    int total_iters = 0;
    int outer_iters = 0;
    int inner_iters = 0;
    int ls_iters = 0;

    // Track regularizer for logging
    double regularizer = 0.0;

    Solver() : Solver(Options()) {}

    Solver(const Options& options)
        : options(options),
          filter(std::make_shared<Filter>(options.gamma_objective, options.gamma_constraint)) {}
    
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
     * Ruiz equilibration for current problem data using copies of H and J_*.
     * Writes concatenated scaling vector [d; e_eq; e_ineq; e_comp] into workspace->scaling.
     */
    void ruiz_equilibration(int niter = 10);

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
     * Solve the KKT system using the factorized matrix, populating the solution
     * in workspace->newton_step.
     */
    void backsolve();

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
     * @param sqrt_relax_param Square root of the complementarity and inequality relaxation parameter 
     * @param inv_penalty_param Inverse of the AL penalty parameter
     * @param max_iters Maximum number of iterations for the linesearch
     * @warning This function modifies the workspace solution to store the candidate solution, and updates the constraint residuals based on the candidate solution, which are used to evaluate the filter conditions. If the linesearch fails, the workspace solution is left at the last candidate solution evaluated.
     * @return true Linesearch succeeded, new iterate is stored in workspace x_candidate
     * @return false Linesearch failed
     */
    bool filter_linesearch(const double sqrt_relax_param, const double inv_penalty_param, int max_iters);

    /**
     * Solve the current problem instance 
     */
    bool solve(const Options &options);

    /**
     * Determine if the solver has converged based on KKT residual norm, constraint satisfaction
     */
    bool convergence(const Options &options);

    /**
     * Print iteration log
     */
    void print_iteration_log(bool outer) const;

    /**
     * Print solver details after initialization
     */
    void print_solver_details() const;

private:
    // Solver options
    const Options options;

    // KKT system regularizers to try in Newton step
    const std::vector<double> kkt_system_regularizers = {
        0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
    };
};