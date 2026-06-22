#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "workspace.h"
#include "relaxation_map.h"
#include "filter.h"
#include "kkt_system.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <utility>

class Solver {
public:
    struct Result {
        bool converged{false};
        int iters{0};
        int iters_outer{0};
        int iters_inner{0};
        int factorizations{0};
        int factorizations_ldlt{0};
        int factorizations_inertia{0};
        int factorizations_linesearch{0};
        double setup_time_s{0.0};
        double solve_time_s{0.0};
        Vec z;        // Primal solution
        Vec s_ineq;   // Inequality slack solution
        Vec s_comp;   // Complementarity slack solution
        Vec m_eq;     // Equality multiplier solution
        Vec m_ineq;   // Inequality multiplier solution
        Vec m_comp_L; // Left complementarity multiplier solution
        Vec m_comp_R; // Right complementarity multiplier solution
    };

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
        double relax_initial{1e-1};
        /// Minimum relaxation parameter for complementarity and inequality constraints
        double relaxation_min{1e-7};
        /// Relaxation parameter scaling factor, multiplies current relaxation parameter
        double relaxation_scaling{0.5};
        /// Apply first-order iterate correction when decreasing relaxation parameter
        bool use_relax_correction{true};
        /// Maximum number of iters for the solver, iters refers to outer + inner iters
        int max_iters{1000};
        /// Maximum number of iters for the filter linesearch
        int max_iters_linesearch{10};
        /// (filter) Sufficient progress parameter for objective value decrease
        double gamma_objective{1e-5};
        /// (filter) Sufficient progress parameter for constraint violation decrease
        double gamma_constraint{1e-5};
        /// Number of ruiz iters for scaling
        int ruiz_iters{10};
        // /// Output directory for solution and solve information
        // std::filesystem::path output_dir{"/dev/null"};
        /// Verbosity level: 0=silent, 1=per-iteration table + footer
        int verbosity{0};
        /// Print a row every N iters (only used when verbosity >= 1)
        int print_every{1};
        // /// Write a JSONL debug log with iterates and solver state (one JSON object per line)
        // bool debug{false};
        // /// Path to the debug log file (used when debug=true)
        // std::string debug_output_path{"solver_debug.jsonl"};
        // /// Log every N iters (1 = every iteration)
        // int debug_log_every{1};

        Options() = default;
    };

    Options options;

    std::shared_ptr<Problem> prob{nullptr};
    std::shared_ptr<Workspace> workspace{nullptr};
    Filter filter;

    // Constructed in set_problem() once the problem dimensions are known
    // (MarbleKKTSystem is non-default-constructible and pinned).
    std::unique_ptr<MarbleKKTSystem> kkt_system;

    Solver() = default;

    // Set the problem for the solver and perform necessary precomputation for solving
    void set_problem(Problem problem, Solver::Options& options);

    bool has_problem() const { return prob && workspace && kkt_system; }
    void require_problem_set(const char* caller) const;

    Problem& get_problem() const { require_problem_set("get_problem"); return *prob; }
    Workspace& get_workspace() const { require_problem_set("get_workspace"); return *workspace; }
    Filter& get_filter() { return filter; }
    const RelaxationMap& get_relaxation_map() const { return relaxation_map; }

    // Update the cached relaxation map values stored in the solver's workspace
    void update_relaxed_slack_values();
    void update_primal_residuals();

    Filter::Entry entry_from_solution() const;

    void apply_newton_step(double step_size);

    /**
     * Perform backtracking filter linesearch given a step direction
     * 
     * @param relax_param Complementarity and inequality relaxation parameter kappa
     * @param inv_penalty_param Inverse of the AL penalty parameter
     * @param max_iters Maximum number of iters for the linesearch
     * @warning This function modifies the workspace solution to store the candidate solution, and updates the constraint residuals based on the candidate solution, which are used to evaluate the filter conditions. If the linesearch fails, the workspace solution is restored to its original value before returning.
     * @return true Linesearch succeeded, new iterate is stored in workspace x_candidate
     * @return false Linesearch failed
     */
    bool filter_linesearch();

    Result solve();

    bool convergence() const;

    // SolveParameters current_parameters() const;
    // void initialize_solve_state();
    // SolveMetrics compute_metrics() const;
    // bool should_take_outer_step(double outer_step_kkt_norm_adjustment) const;
    // void take_outer_step(int iter, int& last_outer_step_iter, double& outer_step_kkt_norm_adjustment);
    // StepStats take_inner_step(const SolveParameters& params);
    // double factor_with_inertia_correction(double initial_regularizer);
    // double solve_with_filter_linesearch(double initial_regularizer, const SolveParameters& params);
    // SolveResult finalize_result(bool converged, int n_iter_outer, int n_iter_inner,
    //                             const std::chrono::steady_clock::time_point& t0);

private:
    // // Counts numerical_factorization() calls; reset at the start of each solve()
    // int n_factorizations{0};

    // // Counts the # of factorizations due to LDLT numerical failure
    // int n_factorizations_ldlt{0};

    // // Counts the # of factorizations due to incorrect inertia
    // int n_factorizations_inertia{0};

    // // Counts the # of factorizations due to linesearch failure
    // int n_factorizations_linesearch{0};

    // // Timing for set_problem and solve, in seconds
    // double setup_time_s{0.0};
    // double solve_time_s{0.0};

    // Relaxation map and derivatives
    RelaxationMap relaxation_map;

    // KKT system regularizers to try in Newton step
    const std::vector<double> kkt_system_regularizers = {
        0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
    };
};
