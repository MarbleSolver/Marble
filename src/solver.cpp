#include "solver.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <string>

void Solver::require_problem_set(const char* caller) const {
    if (!has_problem()) {
        throw std::runtime_error(std::string("Solver::") + caller +
                                 " called before set_problem/setup!");
    }
}

void Solver::set_problem(Problem problem, Solver::Options& options) {
    this->prob = std::make_shared<Problem>(std::move(problem));
    this->workspace = std::make_shared<Workspace>(*this->prob);
    this->filter = Filter(options.gamma_objective, options.gamma_constraint);

    this->options = options;

    // Initialize KKT system with fixed sparsity pattern
    this->kkt_system = std::make_unique<KKTSystem>(this->prob, this->workspace);

    // Build the KKT system with Ruiz scaling from problem data
    const Vec s = this->kkt_system->ruiz_equilibration(options.ruiz_iters);
    this->kkt_system->build(s);
}

bool Solver::convergence() const {
    require_problem_set("convergence");

    // KKT residual norm
    double kkt_residual_norm = kkt_system->residual.lpNorm<Eigen::Infinity>();

    // Primal feasibility
    double eq_violation = workspace->residual_eq.lpNorm<Eigen::Infinity>();
    double ineq_violation = workspace->residual_ineq.cwiseMin(0).lpNorm<Eigen::Infinity>();

    double comp_violation = workspace->residual_comp_L.cwiseProduct(workspace->residual_comp_R).lpNorm<Eigen::Infinity>();

    return kkt_residual_norm < options.convergence_kkt_norm &&
           eq_violation < options.convergence_eq_violation &&
           ineq_violation < options.convergence_ineq_violation &&
           comp_violation < options.convergence_comp_violation;
}

void Solver::update_relaxed_slack_values() {
    require_problem_set("update_relaxed_slack_values");

    relaxation_map.evaluate(workspace->s_ineq, workspace->relax_param, workspace->ineq_retract);
    relaxation_map.evaluate(workspace->s_comp, workspace->relax_param, workspace->comp_retract);
}

void Solver::update_primal_residuals() {
    require_problem_set("update_primal_residuals");

    workspace->residual_eq = prob->residual_eq(workspace->z);
    workspace->residual_ineq = prob->residual_ineq(workspace->z);
    workspace->residual_comp_L = prob->residual_comp_L(workspace->z);
    workspace->residual_comp_R = prob->residual_comp_R(workspace->z);
}

void Solver::apply_newton_step(double step_size) {
    require_problem_set("apply_newton_step");

    workspace->solution += (step_size - workspace->newton_step_size) * workspace->newton_step;
    workspace->newton_step_size = step_size;
}

// TODO: is maybe weird that this is part of Solver and not Filter... but need access to the workspace and problem
// to compute these quantities
Filter::Entry Solver::entry_from_solution() const {
    require_problem_set("entry_from_solution");

    double inv_penalty_param = 1.0 / workspace->penalty_param;
    const RelaxedSlackCache ineq = workspace->ineq_retract;
    const RelaxedSlackCache comp = workspace->comp_retract;

    Vec m_eq_primal_feas = workspace->residual_eq - inv_penalty_param * (workspace->m_eq - workspace->m_eq_est);

    Vec m_ineq_primal_feas = workspace->residual_ineq - ineq.b -
        inv_penalty_param * (workspace->m_ineq - workspace->m_ineq_est);

    Vec m_comp_L_primal_feas = workspace->residual_comp_L - comp.b -
        inv_penalty_param * (workspace->m_comp_L - workspace->m_comp_L_est);
    Vec m_comp_R_primal_feas = workspace->residual_comp_R - comp.b_neg -
        inv_penalty_param * (workspace->m_comp_R - workspace->m_comp_R_est);

    double candidate_constraint_violation = kkt_system->n_duals == 0 ? 0.0 :
        (m_eq_primal_feas.lpNorm<1>() + m_ineq_primal_feas.lpNorm<1>() +
         m_comp_L_primal_feas.lpNorm<1>() + m_comp_R_primal_feas.lpNorm<1>()) / kkt_system->n_duals;

    double candidate_objective =
        0.5 * workspace->z.transpose() * prob->cost_hessian * workspace->z +
        prob->cost_gradient.dot(workspace->z) + prob->cost_const -
        workspace->relax_param * ineq.b.array().log().sum() +
        0.5 * inv_penalty_param *
            (workspace->m_eq.squaredNorm() + workspace->m_ineq.squaredNorm() +
             workspace->m_comp_L.squaredNorm() + workspace->m_comp_R.squaredNorm());

    return {
        candidate_constraint_violation,
        candidate_objective
    };
}

bool Solver::filter_linesearch() {
    require_problem_set("filter_linesearch");

    double step_size = 1.0;

    // TODO: this can be done much more efficiently due to the linearity
    // we should compute a delta for each constraint and the cost from the newton_step
    // and use those scaled vectors to assembly the feasibility candidates
    // We should also re-use the terms in the bottom half of the KKT residual

    for (int i = 0; i < options.max_iters_linesearch; ++i) {
        // Update constraint residuals from candidate solution
        apply_newton_step(step_size);
        update_relaxed_slack_values();
        update_primal_residuals();

        const Filter::Entry candidate = entry_from_solution();

        if (filter.acceptable(candidate)) {
            filter.update(candidate);
            return true;
        }

        // If not acceptable, shrink step size and try again
        step_size *= 0.5;
    }

    // Restore original solution before returning failure
    apply_newton_step(0.0);
    update_relaxed_slack_values();
    update_primal_residuals();

    return false;
}

Solver::Result Solver::solve() {
    require_problem_set("solve");

    // Start solve timer
    const auto t0 = std::chrono::steady_clock::now();

    Result result;

    // Initialize penalty and relax param
    workspace->penalty_param = options.penalty_initial;
    workspace->relax_param = options.relax_initial;
    filter.clear();

    int iters_outer = 0, iters_inner = 0;
    int factorizations = 0, factorizations_ldlt = 0, factorizations_inertia = 0, factorizations_linesearch = 0;

    double beta = 1.0;

    int last_outer_step_iter = -1;

    for (int iter = 0; iter < options.max_iters; ++iter) {
        // Keep cached residuals consistent after line search and kappa correction steps.
        update_primal_residuals();

        // Update cached relaxation map values for current slacks/relaxation params
        update_relaxed_slack_values();

        // Compute updated KKT residual
        kkt_system->update_residual();
        double residual_norm = kkt_system->residual.lpNorm<Eigen::Infinity>();

        // Check for convergence, exit if converged
        if (convergence()) {
            result.converged = true;
            break;
        }

        // If the residual norm is small, take an outer step
        if (residual_norm < beta * options.outer_step_kkt_norm) {
            iters_outer++;

            if (iter > 0 && last_outer_step_iter == iter - 1 && workspace->relax_param <= options.relaxation_min) {
                beta *= 0.1;
            }

            // Update Lagrange multiplier estimates
            workspace->m_eq_est = workspace->m_eq;
            workspace->m_ineq_est = workspace->m_ineq;
            workspace->m_comp_L_est = workspace->m_comp_L;
            workspace->m_comp_R_est = workspace->m_comp_R;

            // Update penalty parameter if not already at max
            if (workspace->penalty_param < options.penalty_max) {
                workspace->penalty_param = std::min(workspace->penalty_param * options.penalty_scaling, options.penalty_max);
            }
            // Update relaxation parameter and compute first-order correction to the iterate
            else {
                kkt_system->update_residual_relax_grad();
                // kkt_system->update_kkt_system();

                // if (!kkt_system->numerical_factorization()) {
                //     throw std::runtime_error("Kappa correction factorization failed at iter " +
                //                              std::to_string(iter));
                // }
                // ++factorizations;
                // ++factorizations_ldlt;

                const double relax_param_new = std::max(workspace->relax_param * options.relaxation_scaling, options.relaxation_min);
                const double delta = relax_param_new - workspace->relax_param;

                if (options.use_relax_correction && delta != 0.0) {
                    kkt_system->compute_step(workspace->relax_correction_step,
                                             -delta * kkt_system->grad_residual_relax_param);
                    workspace->solution += workspace->relax_correction_step;
                } else {
                    workspace->relax_correction_step.setZero();
                }

                workspace->relax_param = relax_param_new;
            }

            // Clear the filter on outer steps
            filter.clear();
            last_outer_step_iter = iter;
        }
        // Otherwise, continue taking newton steps
        else {
            iters_inner++;

            auto bump_reg = [](double &r, double reg_scale) {
                r = (r == 0.0 ? 1e-10 : reg_scale * r);
                return true;
            };

            bool factorization_success = false;
            bool inertia_success = false;
            bool linesearch_success = false;

            kkt_system->update_kkt_system();

            // Initialize regularization value, clear step size before linesearch loop
            // TODO: find a better heuristic for initializing the regularizer, maybe based on
            // the s_comp_stationarity diagonal values
            double reg = 0.0;
            workspace->newton_step_size = 0.0;
            
            while (!linesearch_success && reg <= 1e8) {
                kkt_system->update_primal_regularizer(reg);

                ++factorizations;
                if (!kkt_system->numerical_factorization()) {
                    ++factorizations_ldlt;
                    bump_reg(reg, 10.0);
                    continue;
                }
                factorization_success = true;

                if (!kkt_system->check_inertia()) {
                    ++factorizations_inertia;
                    bump_reg(reg, 10.0);
                    continue;
                }
                inertia_success = true;

                // If inertia correction succeeds, compute the Newton step and try filter linesearch
                kkt_system->compute_step(workspace->newton_step, -kkt_system->residual);

                if (!filter_linesearch()) {
                    ++factorizations_linesearch;
                    bump_reg(reg, 10.0);
                    continue;
                }
                linesearch_success = true;
            }

            if (!factorization_success)
                throw std::runtime_error("Factorization failed for all values at iter " + std::to_string(iter) + " with inertia regularizer " + std::to_string(kkt_system->primal_regularizer));

            if (!inertia_success)
                throw std::runtime_error("Inertia correction failed for all values at iter " + std::to_string(iter) + " with inertia regularizer " + std::to_string(kkt_system->primal_regularizer));

            if (!linesearch_success)
                throw std::runtime_error("Linesearch failed for all values at iter " + std::to_string(iter) + " with inertia regularizer " + std::to_string(kkt_system->primal_regularizer));
        }
    }

    double solve_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    result.iters = iters_outer + iters_inner;
    result.iters_outer = iters_outer;
    result.iters_inner = iters_inner;
    result.factorizations = factorizations;
    result.factorizations_ldlt = factorizations_ldlt;
    result.factorizations_inertia = factorizations_inertia;
    result.factorizations_linesearch = factorizations_linesearch;
    result.solve_time_s = solve_time_s;
    result.z = Vec(workspace->z);
    result.s_ineq = Vec(workspace->s_ineq);
    result.s_comp = Vec(workspace->s_comp);
    result.m_eq = Vec(workspace->m_eq);
    result.m_ineq = Vec(workspace->m_ineq);
    result.m_comp_L = Vec(workspace->m_comp_L);
    result.m_comp_R = Vec(workspace->m_comp_R);

    return result;
}
