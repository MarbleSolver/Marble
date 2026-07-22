#include "solver.h"
#include <cstdlib>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/fmt/fmt.h>

namespace {
    Eigen::VectorXi safe_linspaced(int n, int start) {
        if (n == 0) return Eigen::VectorXi(0);
        return Eigen::VectorXi::LinSpaced(n, start, start + n - 1);
    }

    // Shared "marble" console logger; its level is set per-solve from Options::verbosity.
    // The pattern is just the (level-colored) message so the table stays aligned.
    std::shared_ptr<spdlog::logger> solver_logger() {
        auto lg = spdlog::get("marble");
        if (!lg) {
            lg = spdlog::stdout_color_mt("marble");
            lg->set_pattern("%^%v%$");
        }
        return lg;
    }

    // verbosity -> spdlog threshold:
    //   0 silent | 1 final summary | 2 + setup header | 3 + per-iteration table
    spdlog::level::level_enum level_from_verbosity(int verbosity) {
        switch (verbosity) {
            case 0:  return spdlog::level::off;
            case 1:
            case 2:  return spdlog::level::info;
            default: return spdlog::level::debug;
        }
    }

    // Format the inertia-regularizer column in scientific notation, or "---" (no value / NaN) / "0" (exactly zero).
    std::string fmt_reg(std::optional<double> regularizer) {
        if (!regularizer || std::isnan(*regularizer)) return "---";
        if (*regularizer == 0.0)                      return "0";
        return fmt::format("{:.1e}", *regularizer);
    }

    std::string table_header() {
        return fmt::format("{:<12}  {:>7}  {:>7}  {:>9}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
                           fmt::format("{:>5}  {:<5}", "iter", "type"),
                           "lg(ρ)", "lg(κ)", "reg",
                           "||kkt||", "||eq||", "||ineq||", "||comp||", "obj");
    }

    std::string table_row(const std::string& label,
                          double log_penalty,
                          double log_relaxation,
                          const std::string& regularizer,
                          double kkt,
                          double eq_vio,
                          double ineq_vio,
                          double comp_vio,
                          double obj) {
        return fmt::format("{:<12}  {:>7.1f}  {:>7.1f}  {:>9}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {:>12.3e}",
                           label, log_penalty, log_relaxation, regularizer,
                           kkt, eq_vio, ineq_vio, comp_vio, obj);
    }

    std::string outer_label(int iter) {
        return fmt::format("{:>5}  {:<5}", iter, "O");
    }

    std::string inner_label(int iter) {
        return fmt::format("{:>5}  {:<5}", iter, "I");
    }

    // Convergence / reporting metrics for the current workspace iterate:
    // (||kkt||_inf, ||eq||_inf, ||ineq^-||_inf, ||comp||_inf, objective).
    // Empty constraint sets report 0 for that violation.
    std::tuple<double, double, double, double, double>
    compute_metrics(const Workspace& ws, const Problem& prob) {
        double kkt      = ws.kkt_residual.lpNorm<Eigen::Infinity>();
        double eq_vio   = prob.n_eq   > 0 ? ws.residual_eq.lpNorm<Eigen::Infinity>() : 0.0;
        double ineq_vio = prob.n_ineq > 0 ? ws.residual_ineq.cwiseMin(0).lpNorm<Eigen::Infinity>() : 0.0;
        double comp_vio = prob.n_comp > 0
            ? ws.residual_comp_L.cwiseProduct(ws.residual_comp_R).lpNorm<Eigen::Infinity>()
            : 0.0;
        double obj = 0.5 * ws.z.dot(prob.cost_hessian * ws.z)
                   + prob.cost_gradient.dot(ws.z)
                   + prob.cost_const;
        return {kkt, eq_vio, ineq_vio, comp_vio, obj};
    }
}

Vec Solver::retract(const Eigen::Ref<const Vec>& s, double relax_param) const {
    // Vectorized retraction map for both inequality and complementarity slacks
    // p(s) = 0.5 * (s + sqrt(s^2 + 4*kappa)), kappa = relax_param
    if (options.retraction_type == 0) {
        const double four_kappa = 4.0 * relax_param;
        return (0.5 * (s.array() + (s.array().square() + four_kappa).sqrt())).matrix();
    }
    else if (options.retraction_type == 1) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return sqrt_kappa * s.array().exp();
    }
    else if (options.retraction_type == 2) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return sqrt_kappa * (s.array() / sqrt_kappa).exp();
    }
    else {
        throw std::runtime_error("Invalid retraction type");
    }
}

Vec Solver::retract_deriv(const Eigen::Ref<const Vec>& s, double relax_param) const {
    // Derivative of the vectorized retraction map above
    // p'(s) = 0.5 * (1 + s / sqrt(s^2 + 4*kappa))
    if (options.retraction_type == 0) {
        const double four_kappa = 4.0 * relax_param;
        return (0.5 * (1.0 + s.array() / (s.array().square() + four_kappa).sqrt())).matrix();
    }
    else if (options.retraction_type == 1) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return sqrt_kappa * s.array().exp();
    }
    else if (options.retraction_type == 2) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return (s.array() / sqrt_kappa).exp();
    }
    else {
        throw std::runtime_error("Invalid retraction type");
    }
}

Vec Solver::retract_second_deriv(const Eigen::Ref<const Vec>& s, double relax_param) const {
    // Second derivative of the vectorized retraction map above
    // p''(s) = 2*kappa / (s^2 + 4*kappa)^(3/2)
    if (options.retraction_type == 0) {
        const double kappa = relax_param;
        return (2.0 * kappa * (s.array().square() + 4.0 * kappa).pow(-1.5)).matrix();
    }
    else if (options.retraction_type == 1) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return sqrt_kappa * s.array().exp();
    }
    else if (options.retraction_type == 2) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return (s.array() / sqrt_kappa).exp() / sqrt_kappa;
    }
    else {
        throw std::runtime_error("Invalid retraction type");
    }
}

Vec Solver::retract_drelax(const Eigen::Ref<const Vec>& s, double relax_param) const {
    // Derivative of the retraction map wrt the relaxation parameter kappa = relax_param
    // dp/dkappa = 1 / sqrt(s^2 + 4*kappa)
    if (options.retraction_type == 0) {
        const double four_kappa = 4.0 * relax_param;
        return (s.array().square() + four_kappa).rsqrt().matrix();
    }
    else if (options.retraction_type == 1) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return 0.5 / sqrt_kappa * s.array().exp();
    }
    else if (options.retraction_type == 2) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return (0.5 / sqrt_kappa) * (1.0 - s.array() / sqrt_kappa) * (s.array() / sqrt_kappa).exp();
    }
    else {
        throw std::runtime_error("Invalid retraction type");
    }
}

Vec Solver::retract_deriv_drelax(const Eigen::Ref<const Vec>& s, double relax_param) const {
    // Mixed derivative of the retraction map wrt s and kappa = relax_param
    // d/dkappa p'(s) = -s / (s^2 + 4*kappa)^(3/2)
    if (options.retraction_type == 0) {
        const double four_kappa = 4.0 * relax_param;
        return (-s.array() * (s.array().square() + four_kappa).pow(-1.5)).matrix();
    }
    else if (options.retraction_type == 1) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return 0.5 / sqrt_kappa * s.array().exp();
    }
    else if (options.retraction_type == 2) {
        const double sqrt_kappa = std::sqrt(relax_param);
        return (-0.5 / (relax_param * sqrt_kappa)) * s.array() * (s.array() / sqrt_kappa).exp();
    }
    else {
        throw std::runtime_error("Invalid retraction type");
    }
}

void Solver::set_problem(Problem problem, Solver::Options& options) {
    this->prob = std::make_shared<Problem>(std::move(problem));

    const auto t0 = std::chrono::steady_clock::now();
    this->options = options;
    // Define subproblem dimensions
    n_primals =
        this->prob->nz + this->prob->n_ineq + this->prob->n_comp;  // Primal variables include original and slacks
    n_duals = this->prob->n_eq + this->prob->n_ineq + 2 * this->prob->n_comp;  // Dual variables for each constraint
    n_vars = n_primals + n_duals;                                              // Total variables

    // The inner subproblem solves for r(y) = 0 where r is the kkt residual
    // and y is the stacked vector of primal and dual variables.
    // Here we construct indices of where each set of variables is in y
    int total_inds = 0;

    z_inds       = safe_linspaced(this->prob->nz,     total_inds); total_inds += this->prob->nz;
    s_ineq_inds  = safe_linspaced(this->prob->n_ineq, total_inds); total_inds += this->prob->n_ineq;
    s_comp_inds  = safe_linspaced(this->prob->n_comp, total_inds); total_inds += this->prob->n_comp;
    m_eq_inds    = safe_linspaced(this->prob->n_eq,   total_inds); total_inds += this->prob->n_eq;
    m_ineq_inds  = safe_linspaced(this->prob->n_ineq, total_inds); total_inds += this->prob->n_ineq;
    m_comp_L_inds = safe_linspaced(this->prob->n_comp, total_inds); total_inds += this->prob->n_comp;
    m_comp_R_inds = safe_linspaced(this->prob->n_comp, total_inds); total_inds += this->prob->n_comp;

    // Allocate for solution vector, multiplier estimates, and KKT residual
    // Init workspace
    workspace = std::make_shared<Workspace>(*this->prob);

    // Compute scaling internally from problem data
    ruiz_equilibration(options.ruiz_iterations);

    // Construct initial KKT system and sparsity pattern
    initialize_kkt_sparsity();

    // Init filter
    filter = std::make_shared<Filter>(options.gamma_objective, options.gamma_constraint);

    setup_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void Solver::initialize_kkt_sparsity() {
    // Construct the general KKT matrix using the indices above to populate each block
    // Once the KKT matrix sparsity structure is established, we extract indicies into
    // valuePtr so that it can be efficiently update
    workspace->kkt_system = SMat(n_vars, n_vars);
    std::vector<Eigen::Triplet<double>> triplets;

    // Populate stationarity Hz + J_eq'*m_eq + J_ineq'*m_ineq + L_comp'*m_comp_L + R_comp'*m_comp_R
    workspace->appendBlockTriplets(triplets, prob->cost_hessian, z_inds[0], z_inds[0]);
    
    // Guard every block that touches an empty constraint set
    if (prob->n_eq > 0) {
        workspace->appendBlockTriplets(triplets, prob->J_eq.transpose(), z_inds[0], m_eq_inds[0]);
        workspace->appendBlockTriplets(triplets, prob->J_eq,             m_eq_inds[0], z_inds[0]);
    }
    if (prob->n_ineq > 0) {
        workspace->appendBlockTriplets(triplets, prob->J_ineq.transpose(), z_inds[0], m_ineq_inds[0]);
        workspace->appendBlockTriplets(triplets, prob->J_ineq,             m_ineq_inds[0], z_inds[0]);
    }
    if (prob->n_comp > 0) {
        workspace->appendBlockTriplets(triplets, prob->L_comp.transpose(), z_inds[0], m_comp_L_inds[0]);
        workspace->appendBlockTriplets(triplets, prob->L_comp,             m_comp_L_inds[0], z_inds[0]);
        workspace->appendBlockTriplets(triplets, prob->R_comp.transpose(), z_inds[0], m_comp_R_inds[0]);
        workspace->appendBlockTriplets(triplets, prob->R_comp,             m_comp_R_inds[0], z_inds[0]);
    }

    // Populate stationarity for inequality slacks with dummy variables
    // -p'(s_ineq)*(p(-s_ineq) - m_ineq))
    // which has two diagonal matrices on the blocks corresponding to s_ineq and m_ineq
    for (int i = 0; i < prob->n_ineq; i++) {
        triplets.emplace_back(s_ineq_inds[i], s_ineq_inds[i], 1.0);
        triplets.emplace_back(s_ineq_inds[i], m_ineq_inds[i], 1.0);
        triplets.emplace_back(m_ineq_inds[i], s_ineq_inds[i], 1.0);
    }

    // Populate stationarity for complementarity slacks with dummy variables
    // p'(s_comp)*m_comp_L - p'(-s_comp)*m_comp_R
    for (int i = 0; i < prob->n_comp; i++) {
        // Derivative with respect to the slacks (diagonal)
        triplets.emplace_back(s_comp_inds[i], s_comp_inds[i], 1.0);

        // Derivative with respect to the left and right multipliers
        triplets.emplace_back(s_comp_inds[i], m_comp_L_inds[i], 1.0);
        triplets.emplace_back(s_comp_inds[i], m_comp_R_inds[i], 1.0);
        triplets.emplace_back(m_comp_L_inds[i], s_comp_inds[i], 1.0);
        triplets.emplace_back(m_comp_R_inds[i], s_comp_inds[i], 1.0);
    }

    // Populate AL regularizer for multipliers
    for (int i = 0; i < prob->n_eq; i++) {
        triplets.emplace_back(m_eq_inds[i], m_eq_inds[i], 1.0);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
        triplets.emplace_back(m_ineq_inds[i], m_ineq_inds[i], 1.0);
    }
    for (int i = 0; i < prob->n_comp; i++) {
        triplets.emplace_back(m_comp_L_inds[i], m_comp_L_inds[i], 1.0);
        triplets.emplace_back(m_comp_R_inds[i], m_comp_R_inds[i], 1.0);
    }

    // TODO: more efficient way to make sure that every diagonal element exists
    for (int i = 0; i < n_vars; i++) {
        triplets.emplace_back(i, i, 0.0);
    }

    // Build sparse matrix
    workspace->kkt_system.setFromTriplets(triplets.begin(), triplets.end());

    // Apply diagonal scaling D*K*D to the KKT system
    for (int c = 0; c < workspace->kkt_system.outerSize(); c++) {
        for (typename SMat::InnerIterator it(workspace->kkt_system, c); it; ++it) {
            it.valueRef() = workspace->scaling[it.row()] * it.value() * workspace->scaling[it.col()];
        }
    }

    // Compute AMD ordering permutation to reduce fill-in during factorization
    compute_amd_ordering();
    SMat temp = workspace->amd_perm.transpose()*workspace->kkt_system*workspace->amd_perm; // Need temp to prevent aliasing
    workspace->kkt_system = temp.triangularView<Eigen::Upper>(); // QDLDL requires upper triangular storage
    workspace->kkt_system.makeCompressed();

    // Perform an analytical factorization
    analytical_factorization();

    // Determine indices into valuePtr for updating of nonlinear and penalty terms
    z_z_inds.resize(prob->nz);
    for (int i = 0; i < prob->nz; i++) {
        z_z_inds[i] = workspace->findValuePtrIndex(z_inds[i], z_inds[i]);
    }   
    s_ineq_s_ineq_inds.resize(prob->n_ineq);
    s_ineq_m_ineq_inds.resize(prob->n_ineq);
    for (int i = 0; i < prob->n_ineq; i++) {
        s_ineq_s_ineq_inds[i] = workspace->findValuePtrIndex(s_ineq_inds[i], s_ineq_inds[i]);
        s_ineq_m_ineq_inds[i] = workspace->findValuePtrIndex(s_ineq_inds[i], m_ineq_inds[i]);
    }
    
    s_comp_s_comp_inds.resize(prob->n_comp);
    s_comp_m_comp_L_inds.resize(prob->n_comp);
    s_comp_m_comp_R_inds.resize(prob->n_comp);
    for (int i = 0; i < prob->n_comp; i++) {
        s_comp_s_comp_inds[i] = workspace->findValuePtrIndex(s_comp_inds[i], s_comp_inds[i]);
        s_comp_m_comp_L_inds[i] = workspace->findValuePtrIndex(s_comp_inds[i], m_comp_L_inds[i]);
        s_comp_m_comp_R_inds[i] = workspace->findValuePtrIndex(s_comp_inds[i], m_comp_R_inds[i]);
    }

    penalty_inds.resize(prob->n_eq + prob->n_ineq + 2 * prob->n_comp);
    for (int i = 0; i < prob->n_eq; i++) {
        penalty_inds[i] = workspace->findValuePtrIndex(m_eq_inds[i], m_eq_inds[i]);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
        penalty_inds[prob->n_eq + i] = workspace->findValuePtrIndex(m_ineq_inds[i], m_ineq_inds[i]);
    }
    for (int i = 0; i < prob->n_comp; i++) {
        penalty_inds[prob->n_eq + prob->n_ineq + i] = workspace->findValuePtrIndex(m_comp_L_inds[i], m_comp_L_inds[i]);
    }
    for (int i = 0; i < prob->n_comp; i++) {
        penalty_inds[prob->n_eq + prob->n_ineq + prob->n_comp + i] = workspace->findValuePtrIndex(m_comp_R_inds[i], m_comp_R_inds[i]);
    }

    regularizer_inds.resize(n_primals);
    for (int i = 0; i < prob->nz; i++) {
        regularizer_inds[i] = workspace->findValuePtrIndex(z_inds[i], z_inds[i]);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
        regularizer_inds[prob->nz + i] = workspace->findValuePtrIndex(s_ineq_inds[i], s_ineq_inds[i]);
    }
    for (int i = 0; i < prob->n_comp; i++) {
        regularizer_inds[prob->nz + prob->n_ineq + i] = workspace->findValuePtrIndex(s_comp_inds[i], s_comp_inds[i]);
    }
}

void Solver::update_residuals() {
    // Constraint residuals at the current primal iterate z. Kept separate from
    // update_KKT_residual so they can be refreshed exactly where z changes (initial
    // point, filter linesearch, IFT relaxation predictor)
    workspace->residual_eq = prob->J_eq * workspace->z + prob->c_eq;
    workspace->residual_ineq = prob->J_ineq * workspace->z + prob->c_ineq;
    workspace->residual_comp_L = prob->L_comp * workspace->z + prob->l_comp;
    workspace->residual_comp_R = prob->R_comp * workspace->z + prob->r_comp;
}

void Solver::update_KKT_residual(double relax_param, double penalty_param) {
    // Assumes the constraint residuals are already up to date for the current z
    const double inv_penalty_param = 1.0 / penalty_param;

    // z stationarity
    workspace->kkt_residual(z_inds) = prob->cost_hessian * workspace->z + prob->cost_gradient +
                        prob->J_eq.transpose() * workspace->m_eq +
                        prob->J_ineq.transpose() * workspace->m_ineq +
                        prob->L_comp.transpose() * workspace->m_comp_L +
                        prob->R_comp.transpose() * workspace->m_comp_R;

    if (options.retraction_type == 0) { // Main path (optimized, uses softplus properties)
        // Inequality slack stationarity (p'(s_ineq) * (-m_ineq - p(-s_ineq)))
        Vec p_neg_ineq = retract(-workspace->s_ineq, relax_param);
        Vec d_p_ineq = retract_deriv(workspace->s_ineq, relax_param);
        workspace->kkt_residual(s_ineq_inds) = d_p_ineq.cwiseProduct(-workspace->m_ineq - p_neg_ineq);

        // Complementarity slack stationarity (-p'(s_comp)*m_comp_L + p'(-s_comp)*m_comp_R)
        Vec d_p_comp = retract_deriv(workspace->s_comp, relax_param);
        for (int i = 0; i < prob->n_comp; i++) {
            workspace->kkt_residual(s_comp_inds[i]) =
                -d_p_comp[i] * workspace->m_comp_L[i] + (1 - d_p_comp[i]) * workspace->m_comp_R[i];
        }

        // Equality primal feasibility (r_eq - inv_penalty_param * (m_eq - m_eq_est))
        workspace->kkt_residual(m_eq_inds) =
            workspace->residual_eq - inv_penalty_param * (workspace->m_eq - workspace->m_eq_est);

        // Inequality primal feasibility (r_ineq - (p(s_ineq) - inv_penalty_param * (m_ineq - m_ineq_est))
        workspace->kkt_residual(m_ineq_inds) = workspace->residual_ineq -
                                            (workspace->s_ineq + p_neg_ineq) -  // p(s) - p(-s) = s
                                            inv_penalty_param * (workspace->m_ineq - workspace->m_ineq_est);

        // Complementarity primal feasibility (r_comp_L - p(s_comp) - inv_penalty_param * (m_comp_L - m_comp_L_est)
        //                                    (r_comp_R - p(-s_comp) - inv_penalty_param * (m_comp_R - m_comp_R_est))
        Vec p_comp = retract(workspace->s_comp, relax_param);
        workspace->kkt_residual(m_comp_L_inds) = workspace->residual_comp_L - p_comp -
                                                inv_penalty_param * (workspace->m_comp_L - workspace->m_comp_L_est);
        workspace->kkt_residual(m_comp_R_inds) = workspace->residual_comp_R - 
                                                (p_comp - workspace->s_comp) -
                                                inv_penalty_param * (workspace->m_comp_R - workspace->m_comp_R_est);
    }
    else { // Alternatives for testing retraction maps
        // Inequality slack stationarity (p'(s_ineq) * (-m_ineq - p(-s_ineq)))
        Vec p_neg_ineq = retract(-workspace->s_ineq, relax_param);
        Vec d_p_ineq = retract_deriv(workspace->s_ineq, relax_param);
        workspace->kkt_residual(s_ineq_inds) = d_p_ineq.cwiseProduct(-workspace->m_ineq - p_neg_ineq);

        // Complementarity slack stationarity (-p'(s_comp)*m_comp_L + p'(-s_comp)*m_comp_R)
        Vec d_p_comp = retract_deriv(workspace->s_comp, relax_param);
        Vec d_p_neg_comp = retract_deriv(-workspace->s_comp, relax_param);
        for (int i = 0; i < prob->n_comp; i++) {
            workspace->kkt_residual(s_comp_inds[i]) =
                -d_p_comp[i] * workspace->m_comp_L[i] + d_p_neg_comp[i] * workspace->m_comp_R[i];
        }

        // Equality primal feasibility (r_eq - inv_penalty_param * (m_eq - m_eq_est))
        workspace->kkt_residual(m_eq_inds) =
            workspace->residual_eq - inv_penalty_param * (workspace->m_eq - workspace->m_eq_est);

        // Inequality primal feasibility (r_ineq - (p(s_ineq) - inv_penalty_param * (m_ineq - m_ineq_est))
        Vec p_ineq = retract(workspace->s_ineq, relax_param);
        workspace->kkt_residual(m_ineq_inds) = workspace->residual_ineq - p_ineq -  
                                            inv_penalty_param * (workspace->m_ineq - workspace->m_ineq_est);

        // Complementarity primal feasibility (r_comp_L - p(s_comp) - inv_penalty_param * (m_comp_L - m_comp_L_est)
        //                                    (r_comp_R - p(-s_comp) - inv_penalty_param * (m_comp_R - m_comp_R_est))
        Vec p_comp = retract(workspace->s_comp, relax_param);
        Vec p_neg_comp = retract(-workspace->s_comp, relax_param);
        workspace->kkt_residual(m_comp_L_inds) = workspace->residual_comp_L - p_comp -
                                                inv_penalty_param * (workspace->m_comp_L - workspace->m_comp_L_est);
        workspace->kkt_residual(m_comp_R_inds) = workspace->residual_comp_R - p_neg_comp -
                                                inv_penalty_param * (workspace->m_comp_R - workspace->m_comp_R_est);
    }
}

void Solver::update_dKKT_residual_drelax(double relax_param) {
    // Derivative of the KKT residual wrt the relaxation parameter kappa (= relax_param).
    workspace->dkkt_residual_drelax.setZero();

    if (options.retraction_type == 0) { // Main path, optimized with solver properties
        // Inequality slack stationarity: r = p'(s) * (-m_ineq - p(-s))
        // d/dkappa = p'_kappa(s) * (-m_ineq - p(-s)) - p'(s) * p_kappa(-s)
        // (p(-s) is even in s, so p_kappa(-s) = p_kappa(s) = retract_drelax(s))
        Vec p_neg_ineq = retract(-workspace->s_ineq, relax_param);
        Vec d_p_ineq = retract_deriv(workspace->s_ineq, relax_param);
        Vec dp_ineq_dk = retract_drelax(workspace->s_ineq, relax_param);
        Vec dd_p_ineq_dk = retract_deriv_drelax(workspace->s_ineq, relax_param);
        workspace->dkkt_residual_drelax(s_ineq_inds) =
            dd_p_ineq_dk.cwiseProduct(-workspace->m_ineq - p_neg_ineq) - d_p_ineq.cwiseProduct(dp_ineq_dk);

        // Complementarity slack stationarity: r = -p'(s)*m0 + (1 - p'(s))*m1
        // d/dkappa = -p'_kappa(s) * (m0 + m1)
        Vec dp_comp_dk = retract_drelax(workspace->s_comp, relax_param);
        Vec dd_p_comp_dk = retract_deriv_drelax(workspace->s_comp, relax_param);
        for (int i = 0; i < prob->n_comp; i++) {
            workspace->dkkt_residual_drelax(s_comp_inds[i]) =
                -dd_p_comp_dk[i] * (workspace->m_comp_L[i] + workspace->m_comp_R[i]);
        }

        // Inequality primal feasibility: r = ... - (s + p(-s)) - ...
        // d/dkappa = -p_kappa(-s) = -retract_drelax(s)
        workspace->dkkt_residual_drelax(m_ineq_inds) = -dp_ineq_dk;

        // Complementarity primal feasibility:
        //   row 2i   += -p(s)        -> d/dkappa = -p_kappa(s)
        //   row 2i+1 += -(p(s) - s)  -> d/dkappa = -p_kappa(s)
        for (int i = 0; i < prob->n_comp; i++) {
            workspace->dkkt_residual_drelax(m_comp_L_inds[i]) = -dp_comp_dk[i];
            workspace->dkkt_residual_drelax(m_comp_R_inds[i]) = -dp_comp_dk[i];
        }
    }
    else { // Alternative path for testing
        // Inequality slack stationarity: r = p'(s) * (-m_ineq - p(-s))
        // d/dkappa = p'_kappa(s) * (-m_ineq - p(-s)) - p'(s) * p_kappa(-s)
        Vec dd_p_ineq_dk = retract_deriv_drelax(workspace->s_ineq, relax_param);
        Vec p_neg_ineq = retract(-workspace->s_ineq, relax_param);
        Vec dp_ineq = retract_deriv(workspace->s_ineq, relax_param);
        Vec dp_neg_ineq_dk = retract_drelax(-workspace->s_ineq, relax_param);
        workspace->dkkt_residual_drelax(s_ineq_inds) =
            dd_p_ineq_dk.cwiseProduct(-workspace->m_ineq - p_neg_ineq) - dp_ineq.cwiseProduct(dp_neg_ineq_dk);

        // Complementarity slack stationarity: r = -p'(s)*m0 + p'(-s)*m1
        Vec dd_p_comp_dk = retract_deriv_drelax(workspace->s_comp, relax_param);
        Vec dd_p_neg_comp_dk = retract_deriv_drelax(-workspace->s_comp, relax_param);
        workspace->dkkt_residual_drelax(s_comp_inds) =
            -dd_p_comp_dk.cwiseProduct(workspace->m_comp_L) + dd_p_neg_comp_dk.cwiseProduct(workspace->m_comp_R);

        // Inequality primal feasibility: r = ... - p(s) - ...
        workspace->dkkt_residual_drelax(m_ineq_inds) = -retract_drelax(workspace->s_ineq, relax_param);

        // Complementarity primal feasibility
        workspace->dkkt_residual_drelax(m_comp_L_inds) = -retract_drelax(workspace->s_comp, relax_param);
        workspace->dkkt_residual_drelax(m_comp_R_inds) = -retract_drelax(-workspace->s_comp, relax_param);
    }
}

void Solver::update_KKT_system(double relax_param, double penalty_param) {
    // Update KKT system terms that depend on the solution, which are the nonlinear terms associated with the inequality
    // and complementarity slacks and multipliers, as well as the penalty terms
    update_KKT_ineq(workspace->s_ineq, relax_param);
    update_KKT_comp(workspace->s_comp, workspace->m_comp_L, workspace->m_comp_R, relax_param);
    update_KKT_penalty(penalty_param);
}

void Solver::update_KKT_ineq(const Eigen::Ref<const Vec>& s_ineq, double relax_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    Vec d_p = retract_deriv(s_ineq, relax_param);

    // Ineq stationarity wrt s_ineq is d_p*d_neg_p
    //                   wrt m_ineq is -d_p
    // Use identity that retract_deriv(-s_ineq, relax_param) = 1 - d_p
    if (options.retraction_type == 0) {
        workspace->s_ineq_stationarity = -d_p.cwiseProduct(d_p) + d_p; // -d_p*d_neg_p (stored for diag updating)
        for (int i = 0; i < prob->n_ineq; i++) {
            nzval(s_ineq_s_ineq_inds[i]) = scaling(s_ineq_inds[i]) * workspace->s_ineq_stationarity[i] * scaling(s_ineq_inds[i]);
            nzval(s_ineq_m_ineq_inds[i]) = scaling(s_ineq_inds[i]) * -d_p[i] * scaling(m_ineq_inds[i]);
        }
    }
    else {
        workspace->s_ineq_stationarity = d_p.cwiseProduct(retract_deriv(-s_ineq, relax_param)); // -d_p*d_neg_p (stored for diag updating)
        for (int i = 0; i < prob->n_ineq; i++) {
            nzval(s_ineq_s_ineq_inds[i]) = scaling(s_ineq_inds[i]) * workspace->s_ineq_stationarity[i] * scaling(s_ineq_inds[i]);
            nzval(s_ineq_m_ineq_inds[i]) = scaling(s_ineq_inds[i]) * -d_p[i] * scaling(m_ineq_inds[i]);
        }
    }
}

void Solver::update_KKT_comp(const Eigen::Ref<const Vec>& s_comp, const Eigen::Ref<const Vec>& m_comp_L,
                             const Eigen::Ref<const Vec>& m_comp_R, double relax_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    Vec d_p = retract_deriv(s_comp, relax_param);
    Vec dd_p = retract_second_deriv(s_comp, relax_param);

    if (options.retraction_type == 0) {
        for (int i = 0; i < prob->n_comp; i++) {
            // Derivative of -p'(s)*m_L + p'(-s)*m_R wrt s
            // is -(p''(s)*m_L + p''(s)*m_R) but p''(s) = p''(-s) so its -p''(s)*(m_L + m_R)
            workspace->s_comp_stationarity[i] = -dd_p[i]*(m_comp_L[i] + m_comp_R[i]); // stored for diag updating
            nzval(s_comp_s_comp_inds[i]) = scaling(s_comp_inds[i]) * workspace->s_comp_stationarity[i] * scaling(s_comp_inds[i]);

            // Derivative wrt to m_L and m_R is just -d_p and -d_neg_p respectively
            nzval(s_comp_m_comp_L_inds[i]) = scaling(s_comp_inds[i]) * -d_p[i] * scaling(m_comp_L_inds[i]);
            nzval(s_comp_m_comp_R_inds[i]) = scaling(s_comp_inds[i]) * (1 - d_p[i]) * scaling(m_comp_R_inds[i]);  // d_neg_p = 1 - d_p
        }
    }
    else {
        Vec d_neg_p = retract_deriv(-s_comp, relax_param);
        Vec dd_neg_p = retract_second_deriv(-s_comp, relax_param);
        for (int i = 0; i < prob->n_comp; i++) {
            // Derivative of -p'(s)*m_L + p'(-s)*m_R wrt s
            // is -(p''(s)*m_L + p''(-s)*m_R)
            workspace->s_comp_stationarity[i] = -dd_p[i]*m_comp_L[i] - dd_neg_p[i]*m_comp_R[i]; // stored for diag updating
            nzval(s_comp_s_comp_inds[i]) = scaling(s_comp_inds[i]) * workspace->s_comp_stationarity[i] * scaling(s_comp_inds[i]);

            // Derivative wrt to m_L and m_R is just -d_p and d_neg_p respectively
            nzval(s_comp_m_comp_L_inds[i]) = scaling(s_comp_inds[i]) * -d_p[i] * scaling(m_comp_L_inds[i]);
            nzval(s_comp_m_comp_R_inds[i]) = scaling(s_comp_inds[i]) * d_neg_p[i] * scaling(m_comp_R_inds[i]);
        }
    }
}

void Solver::update_KKT_penalty(const double penalty_param) {
    const double inv_penalty_param = 1.0 / penalty_param;
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    for (int k = 0; k < penalty_inds.size(); k++) {
        nzval(penalty_inds[k]) = -inv_penalty_param * scaling(n_primals + k) * scaling(n_primals + k);
    }
}

void Solver::update_KKT_primal_regularizer(const double reg) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    if (!options.clamp_hessian) {
        for (int k = 0; k < prob->nz; k++) {
            nzval(z_z_inds[k]) = (prob->cost_hessian_diag[k] + reg) * scaling(z_inds[k]) * scaling(z_inds[k]);
        }
        for (int k = 0; k < prob->n_ineq; k++) {
            nzval(s_ineq_s_ineq_inds[k]) = (workspace->s_ineq_stationarity[k] + reg)  * scaling(s_ineq_inds[k]) * scaling(s_ineq_inds[k]);
        }
        for (int k = 0; k < prob->n_comp; k++) {
            nzval(s_comp_s_comp_inds[k]) = (workspace->s_comp_stationarity[k] + reg) * scaling(s_comp_inds[k]) * scaling(s_comp_inds[k]);
        }
    }
    else {
        for (int k = 0; k < prob->nz; k++) {
            nzval(z_z_inds[k]) = std::max(prob->cost_hessian_diag[k] + reg, 1e-8) * scaling(z_inds[k]) * scaling(z_inds[k]);
        }
        for (int k = 0; k < prob->n_ineq; k++) {
            nzval(s_ineq_s_ineq_inds[k]) = std::max(workspace->s_ineq_stationarity[k] + reg, 1e-8)  * scaling(s_ineq_inds[k]) * scaling(s_ineq_inds[k]);
        }
        for (int k = 0; k < prob->n_comp; k++) {
            nzval(s_comp_s_comp_inds[k]) = std::max(workspace->s_comp_stationarity[k] + reg, 1e-8) * scaling(s_comp_inds[k]) * scaling(s_comp_inds[k]);
        }
    }
}

bool Solver::analytical_factorization() {
    const QDLDL_int* Ap = workspace->kkt_system.outerIndexPtr();
    const QDLDL_int* Ai = workspace->kkt_system.innerIndexPtr();

    workspace->sum_Lnz = QDLDL_etree(n_vars, Ap, Ai, workspace->iwork.data(), workspace->Lnz.data(), workspace->etree.data());

    if (workspace->sum_Lnz < 0) {
            std::cerr << "Error: Matrix A is not properly formatted (must be upper triangular CSC)." << std::endl;
            return false;
    }

    // Allocate memory for numerical values
    // TODO: upperbound this and re-use?
    workspace->Li.resize(workspace->sum_Lnz);
    workspace->Lx.resize(workspace->sum_Lnz);

    return true;
}

bool Solver::numerical_factorization() {
    // Extract raw CSC pointers
    const QDLDL_int* Ap = workspace->kkt_system.outerIndexPtr();
    const QDLDL_int* Ai = workspace->kkt_system.innerIndexPtr();
    const QDLDL_float* Ax = workspace->kkt_system.valuePtr();

    // Perform numerical factorization
    QDLDL_int factor_status = QDLDL_factor(n_vars, Ap, Ai, Ax,
                        workspace->Lp.data(), workspace->Li.data(), workspace->Lx.data(),
                        workspace->D.data(), workspace->Dinv.data(), workspace->Lnz.data(),
                        workspace->etree.data(),
                        workspace->bwork.data(), workspace->iwork.data(), workspace->fwork.data());

    // Check factor_status TODO: return/use
    if (factor_status == -1) {
        return false;
    }
    ++n_factorizations;
    return true;
}

bool Solver::check_inertia() {
    return (workspace->D.array() > 1e-10).count() == n_primals && (workspace->D.array() < -1e-10).count() == n_duals;
}

void Solver::backsolve(const Eigen::Ref<const Vec>& b, Vec& x) {
    // The factorization is of the scaled system D*K*D (D = diag(scaling)), so to
    // solve K x = b we solve (D*K*D) y = D*b for y and recover x = D*y.
    // The RHS is also permuted into AMD ordering; the result is unpermuted after.
    // b never aliases x, so scaling+permuting the RHS straight into x needs no temp
    x.noalias() = b.cwiseProduct(workspace->scaling)(workspace->amd_perm_vec);

    // Solve system
    QDLDL_solve(n_vars, workspace->Lp.data(), workspace->Li.data(), workspace->Lx.data(),
                workspace->Dinv.data(), x.data());

    workspace->backsolve_scratch = x(workspace->amd_iperm_vec).cwiseProduct(workspace->scaling);
    x.swap(workspace->backsolve_scratch);
}

void Solver::compute_amd_ordering() {
    Eigen::AMDOrdering<QDLDL_int> amd_ordering;
    amd_ordering(workspace->kkt_system, workspace->amd_perm);
    workspace->amd_perm_vec = workspace->amd_perm.indices();
    workspace->amd_iperm_vec.resize(n_vars);
    for (int i = 0; i < n_vars; ++i) {
        workspace->amd_iperm_vec[workspace->amd_perm_vec[i]] = i; 
    }
}

bool Solver::convergence() const {
    // KKT residual norm
    double kkt_norm = workspace->kkt_residual.lpNorm<Eigen::Infinity>();

    // Primal feasibility
    double eq_violation = workspace->residual_eq.lpNorm<Eigen::Infinity>();
    double ineq_violation = workspace->residual_ineq.cwiseMin(0).lpNorm<Eigen::Infinity>();

    double comp_violation = workspace->residual_comp_L.cwiseProduct(workspace->residual_comp_R).lpNorm<Eigen::Infinity>();

    return kkt_norm < options.convergence_kkt_norm &&
           eq_violation < options.convergence_eq_violation &&
           ineq_violation < options.convergence_ineq_violation &&
           comp_violation < options.convergence_comp_violation;
}

void Solver::log_solver_info() {
    // Bind (and lazily create) the shared logger, then set its level from the verbosity option
    log = solver_logger();
    log->set_level(level_from_verbosity(options.verbosity));

    // Solver info header (shown at verbosity >= 2)
    if (options.verbosity >= 2) {
        log->info("");
        log->info("Marble Solver");
        log->info("  variables   : {} total  (nz={}, +{} ineq slacks, +{} comp slacks; {} duals)",
                  n_vars, prob->nz, prob->n_ineq, prob->n_comp, n_duals);
        log->info("  constraints : {} equality, {} inequality, {} complementarity",
                  prob->n_eq, prob->n_ineq, prob->n_comp);
        log->info("  penalty     : init={:.1e}  max={:.1e}  scale={:.1f}",
                  options.penalty_initial, options.penalty_max, options.penalty_scaling);
        log->info("  relaxation  : init={:.1e}  min={:.1e}  scale={:.2f}",
                  options.relaxation_initial, options.relaxation_min, options.relaxation_scaling);
        log->info("  convergence : kkt<{:.1e}  eq<{:.1e}  ineq<{:.1e}  comp<{:.1e}  (max_iters={})",
                  options.convergence_kkt_norm, options.convergence_eq_violation,
                  options.convergence_ineq_violation, options.convergence_comp_violation, options.max_iters);
        log->info("");
    }

    // Per-iteration table header (shown at verbosity >= 3)
    const std::string header = table_header();
    log->debug("{}", header);
    log->debug("{}", std::string(header.size(), '-'));
}

void Solver::log_initial_step() {
    if (!log->should_log(spdlog::level::debug)) return;

    auto [kkt, eq_vio, ineq_vio, comp_vio, obj] = compute_metrics(*workspace, *prob);
    log->debug("{}",
               table_row("",
                         std::log10(workspace->penalty_param),
                         std::log10(workspace->relax_param),
                         fmt_reg(std::nullopt),
                         kkt, eq_vio, ineq_vio, comp_vio, obj));
}

void Solver::log_newton_step(int iter) {
    if (!log->should_log(spdlog::level::debug)) return;

    auto [kkt, eq_vio, ineq_vio, comp_vio, obj] = compute_metrics(*workspace, *prob);
    log->debug("{}",
               table_row(inner_label(iter),
                         std::log10(workspace->penalty_param),
                         std::log10(workspace->relax_param),
                         fmt_reg(last_regularizer),
                         kkt, eq_vio, ineq_vio, comp_vio, obj));
}

void Solver::log_outer_step(int iter) {
    if (!log->should_log(spdlog::level::debug)) return;

    auto [kkt, eq_vio, ineq_vio, comp_vio, obj] = compute_metrics(*workspace, *prob);
    log->debug("{}",
               table_row(outer_label(iter),
                         std::log10(workspace->penalty_param),
                         std::log10(workspace->relax_param),
                         fmt_reg(std::nullopt),  // outer rows carry no inertia regularizer
                         kkt, eq_vio, ineq_vio, comp_vio, obj));
}

void Solver::log_solve_summary(bool converged, int iterations_outer, int iterations_inner) {
    if (!log->should_log(spdlog::level::info)) return;

    const int iterations_total = iterations_outer + iterations_inner;
    log->info("");
    log->info("Marble {}", converged ? "converged" : "did not converge");
    log->info("  iterations    : {} total  ({} outer, {} inner)",
              iterations_total, iterations_outer, iterations_inner);
    log->info("  factorizations: {}", n_factorizations);
    log->info("  timing        : setup={:.3e}s  solve={:.3e}s  total={:.3e}s",
              setup_time_s, solve_time_s, setup_time_s + solve_time_s);
}

SolveResult Solver::solve() {
    const auto t0 = std::chrono::steady_clock::now();

    log_solver_info();

    // Initialize penalty and relaxation parameters
    workspace->relax_param = options.relaxation_initial;
    workspace->penalty_param = options.penalty_initial;

    // Reset per-solve state: solution, multiplier estimates, filter, regularizer, and counters
    n_factorizations = 0;
    last_regularizer = std::nullopt;

    // Initialize solution and multiplier estimates
    workspace->solution.setZero();
    if (options.comp_init_random) {
        if (options.comp_init_seed >= 0) std::srand(static_cast<unsigned int>(options.comp_init_seed));
        workspace->s_comp = 1e-3 * Vec::Random(prob->n_comp);
    }
    workspace->m_eq_est.setZero();
    workspace->m_ineq_est.setZero();
    workspace->m_comp_L_est.setZero();
    workspace->m_comp_R_est.setZero();

    update_residuals();
    update_KKT_residual(workspace->relax_param, workspace->penalty_param);

    // Clear the filter initially
    filter->clear();

    // Initialize flags and counters
    bool converged = false;
    int n_iter_outer = 0;
    int n_iter_inner = 0;

    // Log the initial residuals (before any steps are taken)
    log_initial_step();

    for (; n_iter_outer < options.max_iters; ++n_iter_outer) {
        // Refresh the KKT residual for the current iterate, then check for convergence
        if ((converged = convergence())) break;

        // Perform inner AL minimization
        double stat_tol = std::max(1e-14, 1e-1 * workspace->relax_param);
        n_iter_inner += minimize_augmented_lagrangian(stat_tol);
        
        if (workspace->penalty_param >= options.penalty_max) {
            // Use the existing factorization to get an estmate of d(solution*)/d(relaxation_param) using IFT
            update_dKKT_residual_drelax(workspace->relax_param);
            backsolve(-workspace->dkkt_residual_drelax, workspace->dsolution_drelax);
            
            // Update relaxation parameter
            double relax_param_new = std::max(options.relaxation_min, workspace->relax_param * options.relaxation_scaling);
            double delta_relax_param = relax_param_new - workspace->relax_param;

            workspace->solution += workspace->dsolution_drelax * delta_relax_param;
            update_residuals();

            // update_KKT_residual(relax_param_new, inv_penalty_param);

            // int predictor_backtracks = 0;
            // while (workspace->kkt_residual.lpNorm<Eigen::Infinity>() > 10.0 &&
            //        ++predictor_backtracks < options.max_iters_linesearch) {
            //     delta_relax_param *= 0.5;
            //     relax_param_new = workspace->relax_param + delta_relax_param;

            //     workspace->solution -= workspace->dsolution_drelax * delta_relax_param;
            //     update_KKT_residual(relax_param_new, inv_penalty_param);
            // }

            // If we are at the maximum penalty, we update multiplier estimates without increasing penalty
            workspace->m_eq_est = workspace->m_eq;
            workspace->m_ineq_est = workspace->m_ineq;
            workspace->m_comp_L_est = workspace->m_comp_L;
            workspace->m_comp_R_est = workspace->m_comp_R;

            // Scale relaxation parameter
            workspace->relax_param = relax_param_new;
        } else {
            // Increase penalty parameter
            workspace->penalty_param = std::min(workspace->penalty_param * options.penalty_scaling, options.penalty_max);
        }

        // Start each inner subproblem with a fresh filter (entries are penalty/relaxation dependent)
        // last_regularizer = std::nullopt;
        filter->clear();
        log_outer_step(n_iter_outer);
    }

    solve_time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    log_solve_summary(converged, n_iter_outer, n_iter_inner);

    return SolveResult{
        converged,
        n_iter_outer + n_iter_inner,
        n_iter_outer,
        n_iter_inner,
        n_factorizations,
        setup_time_s,
        solve_time_s,
        Vec(workspace->z),
        Vec(workspace->s_ineq),
        Vec(workspace->s_comp),
        Vec(workspace->m_eq),
        Vec(workspace->m_ineq),
        Vec(workspace->m_comp_L),
        Vec(workspace->m_comp_R),
    };
}

int Solver::minimize_augmented_lagrangian(double stat_tol) {
    for (int iter = 0; iter < options.max_iters; ++iter) {
        // Compute KKT residual and check the inner stationarity tolerance
        update_KKT_residual(workspace->relax_param, workspace->penalty_param);

        if (workspace->kkt_residual.lpNorm<Eigen::Infinity>() < stat_tol) {
            return iter;
        }

        newton_step(workspace->relax_param, workspace->penalty_param);
        log_newton_step(iter);
    }

    log->error("exceeded maximum iterations during inner AL minimize: {}", options.max_iters);
    throw std::runtime_error("Exceeded maximum iterations during inner AL minimize!");
}

void Solver::newton_step(double relax_param, double penalty_param) {
    bool factorization_succeeded = false;
    bool inertia_correction_succeeded = false;
    bool linesearch_succeeded = false;

    update_KKT_system(relax_param, penalty_param);

    // Warm-start the regularizer one decade below the last successful value so we keep
    // probing for a smaller regularizer each step (0 if that would drop to <= 1e-8).
    double regularizer_to_try = (options.inertia_warmstart && last_regularizer)
        ? (*last_regularizer <= 1e-8 ? 0.0 : *last_regularizer / 10.0)
        : 0.0;
    double inertia_regularizer = std::numeric_limits<double>::infinity();

    auto increase_regularizer = [](double& r) { r = (r == 0.0 ? 1e-8 : 10.0 * r); return true ; };

    while (!linesearch_succeeded && regularizer_to_try <= 1e8) {
        update_KKT_primal_regularizer(regularizer_to_try);

        if (!numerical_factorization() && increase_regularizer(regularizer_to_try)) continue;
        factorization_succeeded = true;

        if (!check_inertia() && increase_regularizer(regularizer_to_try)) continue;
        inertia_correction_succeeded = true;

        inertia_regularizer = std::min(regularizer_to_try, inertia_regularizer);

        backsolve(-workspace->kkt_residual, workspace->newton_step);

        if (!filter_linesearch(relax_param, penalty_param, options.max_iters_linesearch) && increase_regularizer(regularizer_to_try)) continue;
        linesearch_succeeded = true;
    }

    if (!factorization_succeeded) {
        log->error("newton step: numerical factorization failed for all regularization values");
        throw std::runtime_error("Numerical factorization failed for all regularization values!");
    } else if (!inertia_correction_succeeded) {
        log->error("newton step: inertia could not be corrected for any regularization value");
        throw std::runtime_error("Inertia correction failed for all regularization values!");
    } else if (!linesearch_succeeded) {
        log->error("newton step: linesearch failed for all regularization values");
        throw std::runtime_error("Linesearch failed for all regularization values!");
    }

    last_regularizer = inertia_regularizer <= 1e-8 ? 0.0 : inertia_regularizer;
}

Filter::Entry Solver::entry_from_solution(double relax_param, double penalty_param) const {
    const double inv_penalty_param = 1.0 / penalty_param;

    Vec m_eq_primal_feas = workspace->residual_eq - inv_penalty_param * (workspace->m_eq - workspace->m_eq_est);

    Vec p_ineq = retract(workspace->s_ineq, relax_param);
    Vec m_ineq_primal_feas = workspace->residual_ineq - p_ineq - 
                                inv_penalty_param * (workspace->m_ineq - workspace->m_ineq_est);
    Vec m_comp_L_primal_feas(prob->n_comp);
    Vec m_comp_R_primal_feas(prob->n_comp);

    if (options.retraction_type == 0) { // Main path, uses retraction map properties, optimized
        Vec p_comp = retract(workspace->s_comp, relax_param);
        m_comp_L_primal_feas =
            workspace->residual_comp_L - inv_penalty_param * (workspace->m_comp_L - workspace->m_comp_L_est);
        m_comp_R_primal_feas =
            workspace->residual_comp_R - inv_penalty_param * (workspace->m_comp_R - workspace->m_comp_R_est);

        for (int i = 0; i < prob->n_comp; i++) {
            m_comp_L_primal_feas[i] += -p_comp[i];                               // p(s)
            m_comp_R_primal_feas[i] += -(p_comp[i] - workspace->s_comp[i]);      // p(s) - p(-s) = s
        }
    }
    else { // Alternate path for testing
        Vec p_comp = retract(workspace->s_comp, relax_param);
        Vec p_neg_comp = retract(-workspace->s_comp, relax_param);
         m_comp_L_primal_feas =
            workspace->residual_comp_L - inv_penalty_param * (workspace->m_comp_L - workspace->m_comp_L_est);
        Vec m_comp_R_primal_feas =
            workspace->residual_comp_R - inv_penalty_param * (workspace->m_comp_R - workspace->m_comp_R_est);

        for (int i = 0; i < prob->n_comp; i++) {
            m_comp_L_primal_feas[i] += -p_comp[i];          // p(s)
            m_comp_R_primal_feas[i] += -p_neg_comp[i];      // p(s) - p(-s) = s
        }
    }

    double candidate_constraint_violation = n_duals == 0 ? 0.0 :
        (m_eq_primal_feas.lpNorm<1>() + m_ineq_primal_feas.lpNorm<1>() +
         m_comp_L_primal_feas.lpNorm<1>() + m_comp_R_primal_feas.lpNorm<1>()) / n_duals;

    double candidate_objective =
        0.5 * workspace->z.transpose() * prob->cost_hessian * workspace->z +
        prob->cost_gradient.dot(workspace->z) + prob->cost_const -
        relax_param * p_ineq.array().log().sum() +
        0.5 * inv_penalty_param *
            (workspace->m_eq.squaredNorm() + workspace->m_ineq.squaredNorm() +
             workspace->m_comp_L.squaredNorm() + workspace->m_comp_R.squaredNorm());

    return {
        candidate_constraint_violation,
        candidate_objective
    };
}

bool Solver::filter_linesearch(double relax_param, double penalty_param, int max_iters) {
    double step_size = 1.0;

    workspace->solution += step_size * workspace->newton_step;  // Candidate solution for full step
    update_residuals();

    // TODO: this can be done much more efficiently due to the linearity
    // we should compute a delta for each constraint and the cost from the newton_step
    // and use those scaled vectors to assembly the feasibility candidates
    // We should also re-use the terms in the bottom half of the KKT residual
    
    for (int i = 0; i < max_iters; ++i) {
        const Filter::Entry candidate = entry_from_solution(relax_param, penalty_param);

        if (filter->acceptable(candidate)) {
            filter->update(candidate);
            return true;
        }

        // If not acceptable, shrink step size and try again
        step_size *= 0.5;

        workspace->solution -= step_size * workspace->newton_step;
        update_residuals();
    }

    // Restore original solution before returning failure
    workspace->solution -= step_size * workspace->newton_step;
    update_residuals();

    return false;
}

void Solver::ruiz_equilibration(int niter) {
    SMat H_bar = prob->cost_hessian;
    SMat J_eq_bar = prob->J_eq;
    SMat J_ineq_bar = prob->J_ineq;
    SMat L_comp_bar = prob->L_comp;
    SMat R_comp_bar = prob->R_comp;

    const int nz = prob->nz;
    const int n_eq = prob->n_eq;
    const int n_ineq = prob->n_ineq;
    const int n_comp = prob->n_comp;

    workspace->scaling.setOnes(nz + n_ineq + n_comp + n_eq + n_ineq + 2*n_comp);
    Eigen::VectorBlock<Vec> d = workspace->scaling.head(nz);
    Eigen::VectorBlock<Vec> e_eq = workspace->scaling.segment(nz + n_ineq + n_comp, n_eq);
    Eigen::VectorBlock<Vec> e_ineq = workspace->scaling.segment(nz + n_ineq + n_comp + n_eq, n_ineq);
    Eigen::VectorBlock<Vec> e_comp_L = workspace->scaling.segment(nz + n_ineq + n_comp + n_eq + n_ineq, n_comp);
    Eigen::VectorBlock<Vec> e_comp_R = workspace->scaling.segment(nz + n_ineq + n_comp + n_eq + n_ineq + n_comp, n_comp);

    // Helper functions: Inf-norm column/row reductions using sparse nonzero iteration
    for (int iter = 0; iter < niter; ++iter) {
        // Matrix equilibration
        // Compute column-wise norms of first part of KKT system (stationarity)
        Vec d_temp = Vec::Zero(nz);
        for (int col = 0; col < H_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(H_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }
        for (int col = 0; col < J_eq_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(J_eq_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }
        for (int col = 0; col < J_ineq_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(J_ineq_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }
        for (int col = 0; col < L_comp_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(L_comp_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }
        for (int col = 0; col < R_comp_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(R_comp_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }

        // Compute row-wise norms of second part of KKT system (feasibility)
        Vec e_eq_temp = Vec::Zero(n_eq);
        Vec e_ineq_temp = Vec::Zero(n_ineq);
        Vec e_comp_L_temp = Vec::Zero(n_comp);
        Vec e_comp_R_temp = Vec::Zero(n_comp);
        for (int col = 0; col < J_eq_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(J_eq_bar, col); it; ++it) {
                e_eq_temp[it.row()] = std::max(e_eq_temp[it.row()], std::abs(it.value()));
            }
        }
        for (int col = 0; col < J_ineq_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(J_ineq_bar, col); it; ++it) {
                e_ineq_temp[it.row()] = std::max(e_ineq_temp[it.row()], std::abs(it.value()));
            }
        }
        for (int col = 0; col < L_comp_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(L_comp_bar, col); it; ++it) {
                e_comp_L_temp[it.row()] = std::max(e_comp_L_temp[it.row()], std::abs(it.value()));
            }
        }
        for (int col = 0; col < R_comp_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(R_comp_bar, col); it; ++it) {
                e_comp_R_temp[it.row()] = std::max(e_comp_R_temp[it.row()], std::abs(it.value()));
            }
        }

        // Clamp scaling values
        d_temp = d_temp.cwiseMax(1e-4).cwiseMin(1e4);
        e_eq_temp = e_eq_temp.cwiseMax(1e-4).cwiseMin(1e4);
        e_ineq_temp = e_ineq_temp.cwiseMax(1e-4).cwiseMin(1e4);
        e_comp_L_temp = e_comp_L_temp.cwiseMax(1e-4).cwiseMin(1e4);
        e_comp_R_temp = e_comp_R_temp.cwiseMax(1e-4).cwiseMin(1e4);

        // Take square roots, reciprocal, turn into diag vectors
        d_temp = d_temp.array().rsqrt().matrix();
        e_eq_temp = e_eq_temp.array().rsqrt().matrix();
        e_ineq_temp = e_ineq_temp.array().rsqrt().matrix();
        e_comp_L_temp = e_comp_L_temp.array().rsqrt().matrix();
        e_comp_R_temp = e_comp_R_temp.array().rsqrt().matrix();

        // Update problem data
        for (int col = 0; col < H_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(H_bar, col); it; ++it) {
                it.valueRef() *= d_temp[it.row()] * d_temp[col];
            }
        }
        for (int col = 0; col < J_eq_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(J_eq_bar, col); it; ++it) {
                it.valueRef() *= e_eq_temp[it.row()] * d_temp[col];
            }
        }
        for (int col = 0; col < J_ineq_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(J_ineq_bar, col); it; ++it) {
                it.valueRef() *= e_ineq_temp[it.row()] * d_temp[col];
            }
        }
        for (int col = 0; col < L_comp_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(L_comp_bar, col); it; ++it) {
                it.valueRef() *= e_comp_L_temp[it.row()] * d_temp[col];
            }
        }
        for (int col = 0; col < R_comp_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(R_comp_bar, col); it; ++it) {
                it.valueRef() *= e_comp_R_temp[it.row()] * d_temp[col];
            }
        }

        // Update eq matrices
        d = d.cwiseProduct(d_temp);
        e_eq = e_eq.cwiseProduct(e_eq_temp);
        e_ineq = e_ineq.cwiseProduct(e_ineq_temp);
        e_comp_L = e_comp_L.cwiseProduct(e_comp_L_temp);
        e_comp_R = e_comp_R.cwiseProduct(e_comp_R_temp);
    }

}
