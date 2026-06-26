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
    /**
     * Solve output and iteration diagnostics
     */
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

    /**
     * User-tunable solver parameters
     */
    struct Options {
        // KKT infinity norm threshold for convergence
        double convergence_kkt_norm{1e-4};
        // Equality violation threshold for convergence
        double convergence_eq_violation{1e-4};
        // Inequality violation threshold for convergence
        double convergence_ineq_violation{1e-4};
        // Complementarity violation threshold for convergence
        double convergence_comp_violation{1e-5};
        // KKT infinity norm threshold for an outer step
        double outer_step_kkt_norm{1e-6};
        // Initial augmented Lagrangian penalty parameter
        double penalty_initial{10.0};
        // Maximum augmented Lagrangian penalty parameter
        double penalty_max{1e6};
        // Penalty scaling factor
        double penalty_scaling{10.0};
        // Initial relaxation parameter
        double relax_initial{1e-1};
        // Minimum relaxation parameter
        double relax_min{1e-7};
        // Relaxation scaling factor
        double relax_scaling{0.5};
        // Apply first-order correction when decreasing relaxation
        bool use_relax_correction{true};
        // Maximum total inner plus outer iterations
        int max_iters{1000};
        // Maximum backtracking steps per filter line search
        int max_iters_linesearch{10};
        // Filter merit progress parameter
        double gamma_objective{1e-5};
        // Filter feasibility progress parameter
        double gamma_constraint{1e-5};
        // Number of Ruiz equilibration iterations
        int ruiz_iters{10};
        // Verbosity level
        int verbosity{0};

        /**
         * Construct options with default values
         */
        Options() = default;
    };

    Options options;

    std::shared_ptr<Problem> prob{nullptr};
    std::shared_ptr<Workspace> workspace{nullptr};
    Filter filter;

    std::unique_ptr<KKTSystem> kkt_system;

    /**
     * Construct an empty solver
     */
    Solver() = default;

    /**
     * Set the problem and build the fixed KKT structure
     *
     * @param problem Problem data to solve
     * @param options Solver options to copy into the solver
     */
    void set_problem(Problem problem, Solver::Options& options);

    /**
     * Check whether the solver has been initialized with a problem
     *
     * @return True when problem, workspace, and KKT system are present
     */
    bool has_problem() const { return prob && workspace && kkt_system; }

    /**
     * Throw unless a problem has been set
     *
     * @param caller Name of the caller used in the error message
     */
    void require_problem_set(const char* caller) const;

    /**
     * Access the active problem
     *
     * @return Mutable problem reference
     */
    Problem& get_problem() const { require_problem_set("get_problem"); return *prob; }

    /**
     * Access the active workspace
     *
     * @return Mutable workspace reference
     */
    Workspace& get_workspace() const { require_problem_set("get_workspace"); return *workspace; }

    /**
     * Access the line-search filter
     *
     * @return Mutable filter reference
     */
    Filter& get_filter() { return filter; }

    /**
     * Access the relaxation map
     *
     * @return Relaxation map reference
     */
    const RelaxationMap& get_relaxation_map() const { return relaxation_map; }

    /**
     * Refresh cached relaxation values in the workspace
     */
    void update_relaxed_slack_values();

    /**
     * Refresh primal constraint residuals in the workspace
     */
    void update_primal_residuals();

    /**
     * Build a filter entry from the current workspace state
     *
     * @return Filter entry containing feasibility and merit values
     */
    Filter::Entry entry_from_solution() const;

    /**
     * Apply a scaled Newton step relative to the currently applied step size
     *
     * @param step_size New line-search step size
     */
    void apply_newton_step(double step_size);

    /**
     * Run backtracking filter line search along the current Newton direction
     *
     * @return True when an acceptable candidate is accepted
     */
    bool filter_linesearch();

    /**
     * Run the solver on the active problem
     *
     * @return Solver result and diagnostics
     */
    Result solve();

    /**
     * Check current convergence criteria
     *
     * @return True when all solver convergence tolerances are met
     */
    bool convergence() const;

private:
    RelaxationMap relaxation_map;

    const std::vector<double> kkt_system_regularizers = {
        0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0,
        1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8
    };
};
