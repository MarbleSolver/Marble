#include "solver.h"
#include <nlohmann/json.hpp>
#include <fstream>

Eigen::VectorXi safe_linspaced(int n, int start) {
    if (n == 0) return Eigen::VectorXi(0);
    return Eigen::VectorXi::LinSpaced(n, start, start + n - 1);
}

Vec Solver::retract(const Vec& s, double sqrt_relax_param) const {
    // Vectorized retraction map for both inequality and complementarity slacks
    // p(s) = sqrt(s^2 + relax_param)
    // Eigen::VectorXd p = 0.5*(s/sqrt_relax_param + (s.))
    auto& s_arr = s.array() / sqrt_relax_param;
    Vec p = 0.5 * (s_arr + (s_arr.square() + 4.0).sqrt());
    return sqrt_relax_param * p;
}

Vec Solver::retract_deriv(const Vec& s, double sqrt_relax_param) const {
    // Derivative of the vectorized retraction map above
    auto& s_arr = s.array() / sqrt_relax_param;
    Vec p_deriv = 0.5 * (s_arr / sqrt(s_arr.square() + 4.0) + 1.0);
    return p_deriv;
}

Vec Solver::retract_second_deriv(const Vec& s, double sqrt_relax_param) const {
    // Second derivative of the vectorized retraction map above
    auto& s_arr = s.array() / sqrt_relax_param;
    Vec p_second_deriv = 2.0 / (s_arr.square() + 4.0).pow(1.5);
    return p_second_deriv / sqrt_relax_param;
}

std::pair<Vec, Vec> Solver::ruiz_equilibration(const SMat& H, const SMat& A, int niter) const {
    SMat H_bar = H;
    SMat A_bar = A;

    Vec d = Vec::Ones(H.rows());
    Vec e = Vec::Ones(A.rows());

    // Helper functions: Inf-norm column/row reductions using sparse nonzero iteration
    for (int iter = 0; iter < niter; ++iter) {
        // Matrix equilibration
        // Compute column-wise norms of first part of KKT system (stationarity)
        Vec d_temp = Vec::Zero(H_bar.cols());
        for (int col = 0; col < H_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(H_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }
        for (int col = 0; col < A_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(A_bar, col); it; ++it) {
                d_temp[col] = std::max(d_temp[col], std::abs(it.value()));
            }
        }

        // Compute row-wise norms of second part of KKT system (feasibility)
        Vec e_temp = Vec::Zero(A_bar.rows());
        for (int col = 0; col < A_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(A_bar, col); it; ++it) {
                e_temp[it.row()] = std::max(e_temp[it.row()], std::abs(it.value()));
            }
        }

        // Clamp scaling values
        d_temp = d_temp.cwiseMax(1e-4).cwiseMin(1e4);
        e_temp = e_temp.cwiseMax(1e-4).cwiseMin(1e4);

        // Take square roots, reciprocal, turn into diag vectors
        d_temp = d_temp.array().rsqrt().matrix();
        e_temp = e_temp.array().rsqrt().matrix();

        // Update problem data
        for (int col = 0; col < H_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(H_bar, col); it; ++it) {
                it.valueRef() *= d_temp[it.row()] * d_temp[col];
            }
        }
        for (int col = 0; col < A_bar.outerSize(); ++col) {
            for (SMat::InnerIterator it(A_bar, col); it; ++it) {
                it.valueRef() *= e_temp[it.row()] * d_temp[col];
            }
        }

        // Update eq matrices
        d = d.cwiseProduct(d_temp);
        if (e.size() > 0) {
            e = e.cwiseProduct(e_temp);
        }
    }

    return {d, e};
}

void Solver::set_problem(Problem& prob, Vec scaling) {
    // Move problem
    this->prob = std::make_shared<Problem>(std::move(prob));

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
    m_comp_inds  = safe_linspaced(2 * this->prob->n_comp, total_inds); total_inds += 2 * this->prob->n_comp;

    // Indices into the complementarity residual for extracting the comp residual violation for filter evaluation
    if (this->prob->n_comp == 0) {
        comp_L_inds = Eigen::VectorXi(0);
        comp_R_inds = Eigen::VectorXi(0);
    } else {
        comp_L_inds = 2 * Eigen::VectorXi::LinSpaced(this->prob->n_comp, 0, this->prob->n_comp - 1);
        comp_R_inds = comp_L_inds.array() + 1;
    }

    // Allocate for solution vector, multiplier estimates, and KKT residual
    // Init workspace
    workspace = std::make_shared<Workspace>(*this->prob);
    workspace->scaling = std::move(scaling);

    // Construct initial KKT system and sparsity pattern
    initialize_kkt_sparsity();
}

void Solver::initialize_kkt_sparsity() {
    // Construct the general KKT matrix using the indices above to populate each block
    // Once the KKT matrix sparsity structure is established, we extract indicies into
    // valuePtr so that it can be efficiently update
    workspace->kkt_system = SMat(n_vars, n_vars);
    std::vector<Eigen::Triplet<double>> triplets;

    // Populate stationarity Hz + J_eq'*m_eq + J_ineq'*m_ineq + J_comp'*m_comp 
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
        workspace->appendBlockTriplets(triplets, prob->J_comp.transpose(), z_inds[0], m_comp_inds[0]);
        workspace->appendBlockTriplets(triplets, prob->J_comp,             m_comp_inds[0], z_inds[0]);
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
    // p'(s_comp)*m1 - p'(-s_comp)*m2
    // where m1 and m2 are subsets of m_comp corresponding to the left and right
    // side of the complementarity constraint.
    // TODO: define this indexing somewhere and/or make it more general
    for (int i = 0; i < prob->n_comp; i++) {
        // Derivative with respect to the slacks (diagonal)
        triplets.emplace_back(s_comp_inds[i], s_comp_inds[i], 1.0);

        // Derivative with respect to multipliers (n_comp by 2*n_comp)
        triplets.emplace_back(s_comp_inds[i], m_comp_inds[i*2], 1.0);
        triplets.emplace_back(s_comp_inds[i], m_comp_inds[i*2 + 1], 1.0);
        triplets.emplace_back(m_comp_inds[i*2], s_comp_inds[i], 1.0);
        triplets.emplace_back(m_comp_inds[i*2 + 1], s_comp_inds[i], 1.0);
    }

    // Populate AL regularizer for multipliers
    for (int i = 0; i < prob->n_eq; i++) {
        triplets.emplace_back(m_eq_inds[i], m_eq_inds[i], 1.0);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
        triplets.emplace_back(m_ineq_inds[i], m_ineq_inds[i], 1.0);
    }
    for (int i = 0; i < 2 * prob->n_comp; i++) {
        triplets.emplace_back(m_comp_inds[i], m_comp_inds[i], 1.0);
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
    s_comp_m_comp_inds.resize(2 * prob->n_comp);
    for (int i = 0; i < prob->n_comp; i++) {
        s_comp_s_comp_inds[i] = workspace->findValuePtrIndex(s_comp_inds[i], s_comp_inds[i]);
        s_comp_m_comp_inds[i * 2] = workspace->findValuePtrIndex(s_comp_inds[i], m_comp_inds[i * 2]);
        s_comp_m_comp_inds[i * 2 + 1] = workspace->findValuePtrIndex(s_comp_inds[i], m_comp_inds[i * 2 + 1]);
    }

    penalty_inds.resize(prob->n_eq + prob->n_ineq + 2 * prob->n_comp);
    for (int i = 0; i < prob->n_eq; i++) {
        penalty_inds[i] = workspace->findValuePtrIndex(m_eq_inds[i], m_eq_inds[i]);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
        penalty_inds[prob->n_eq + i] = workspace->findValuePtrIndex(m_ineq_inds[i], m_ineq_inds[i]);
    }
    for (int i = 0; i < 2 * prob->n_comp; i++) {
        penalty_inds[prob->n_eq + prob->n_ineq + i] = workspace->findValuePtrIndex(m_comp_inds[i], m_comp_inds[i]);
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

void Solver::update_KKT_residual(double sqrt_relax_param, double inv_penalty_param) {
    // Compute the constraint residuals

    // TODO: we don't need to recompute these; they will have been already computed in the linesearch beforehand
    // just need to compute these in the very beginning before we've run the first linesearch
    workspace->residual_eq = prob->J_eq * workspace->z + prob->c_eq;
    workspace->residual_ineq = prob->J_ineq * workspace->z + prob->c_ineq;
    workspace->residual_comp = prob->J_comp * workspace->z + prob->c_comp;

    // z stationarity
    workspace->kkt_residual(z_inds) = prob->cost_hessian * workspace->z + prob->cost_gradient + 
                        prob->J_eq.transpose() * workspace->m_eq +
                        prob->J_ineq.transpose() * workspace->m_ineq + 
                        prob->J_comp.transpose() * workspace->m_comp;

    // Inequality slack stationarity
    Vec p_neg_ineq = retract(-workspace->s_ineq, sqrt_relax_param);
    Vec d_p_ineq = retract_deriv(workspace->s_ineq, sqrt_relax_param);
    workspace->kkt_residual(s_ineq_inds) = d_p_ineq.cwiseProduct(-workspace->m_ineq - p_neg_ineq);

    // Complementarity slack stationarity
    Vec d_p_comp = retract_deriv(workspace->s_comp, sqrt_relax_param);
    for (int i = 0; i < prob->n_comp; i++) {
        workspace->kkt_residual(s_comp_inds[i]) =
            -d_p_comp[i] * workspace->m_comp[2 * i] + (1 - d_p_comp[i]) * workspace->m_comp[2 * i + 1];
    }

    // Equality primal feasibility
    workspace->kkt_residual(m_eq_inds) =
        workspace->residual_eq - inv_penalty_param * (workspace->m_eq - workspace->m_eq_est);

    // Inequality primal feasibility
    workspace->kkt_residual(m_ineq_inds) = workspace->residual_ineq -
                                           (workspace->s_ineq + p_neg_ineq) -  // p(s) - p(-s) = s
                                           inv_penalty_param * (workspace->m_ineq - workspace->m_ineq_est);

    // Complementarity primal feasibility
    Vec p_comp = retract(workspace->s_comp, sqrt_relax_param);
    workspace->kkt_residual(m_comp_inds) =
        workspace->residual_comp - inv_penalty_param * (workspace->m_comp - workspace->m_comp_est);
    for (int i = 0; i < prob->n_comp; i++) {
        workspace->kkt_residual(m_comp_inds[2 * i]) += -p_comp[i];                               // p(s)
        workspace->kkt_residual(m_comp_inds[2 * i + 1]) += -(p_comp[i] - workspace->s_comp[i]);  // p(s) - p(-s) = s
    }
}

void Solver::update_KKT_system(double sqrt_relax_param, double inv_penalty_param) {
    // Update KKT system terms that depend on the solution, which are the nonlinear terms associated with the inequality
    // and complementarity slacks and multipliers, as well as the penalty terms
    update_KKT_ineq(workspace->s_ineq, sqrt_relax_param);
    update_KKT_comp(workspace->s_comp, workspace->m_comp, sqrt_relax_param);
    update_KKT_penalty(inv_penalty_param);
}

void Solver::update_KKT_ineq(const Vec& s_ineq, double sqrt_relax_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    Vec d_p = retract_deriv(s_ineq, sqrt_relax_param);

    // Use identity that retract_deriv(-s_ineq, sqrt_relax_param) = 1 - d_p
    workspace->s_ineq_stationarity = -d_p.cwiseProduct(d_p) + d_p; // -d_p*d_neg_p (stored for diag updating)
    for (int i = 0; i < prob->n_ineq; i++) {
        nzval(s_ineq_s_ineq_inds[i]) = scaling(s_ineq_inds[i]) * workspace->s_ineq_stationarity[i] * scaling(s_ineq_inds[i]);
        nzval(s_ineq_m_ineq_inds[i]) = scaling(s_ineq_inds[i]) * -d_p[i] * scaling(m_ineq_inds[i]);
    }
}

// TODO: fix indexing into m_comp so that it can be a const reference
void Solver::update_KKT_comp(const Vec& s_comp, Vec m_comp, double sqrt_relax_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    Vec d_p = retract_deriv(s_comp, sqrt_relax_param);
    Vec dd_p = retract_second_deriv(s_comp, sqrt_relax_param);

    for (int i = 0; i < prob->n_comp; i++) {
        // Derivative of -p'(s)*m1 + p'(-s)*m2 wrt s
        // is -(p''(s)*m1 + p''(s)*m2) but p''(s) = p''(-s) so its -p''(s)*(m1 + m2)
        workspace->s_comp_stationarity[i] = -dd_p[i]*(m_comp[2*i] + m_comp[2*i + 1]); // stored for diag updating
        nzval(s_comp_s_comp_inds[i]) = scaling(s_comp_inds[i]) * workspace->s_comp_stationarity[i] * scaling(s_comp_inds[i]);

        // Derivative wrt to m1 and m2 is just -d_p and -d_neg_p respectively
        nzval(s_comp_m_comp_inds[2 * i]) = scaling(s_comp_inds[i]) * -d_p[i] * scaling(m_comp_inds[2 * i]);
        nzval(s_comp_m_comp_inds[2 * i + 1]) = scaling(s_comp_inds[i]) * (1 - d_p[i]) * scaling(m_comp_inds[2 * i + 1]);  // d_neg_p = 1 - d_p
    }
}

void Solver::update_KKT_penalty(const double inv_penalty_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
    for (int k = 0; k < penalty_inds.size(); k++) {
        nzval(penalty_inds[k]) = -inv_penalty_param * scaling(n_primals + k) * scaling(n_primals + k);
    }
}

void Solver::update_KKT_primal_regularizer(const double reg) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Eigen::Ref<Eigen::VectorXd> scaling = workspace->scaling;
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
    return true;
}

bool Solver::check_inertia() {
    return (workspace->D.array() > 1e-10).count() == n_primals && (workspace->D.array() < -1e-10).count() == n_duals;
}

void Solver::backsolve() {
    // Initialize step as -residual, permuted using AMD ordering
    for (int k = 0; k < n_vars; k++) {
        workspace->newton_step[k] = -workspace->kkt_residual(workspace->amd_perm_vec[k])*workspace->scaling(workspace->amd_perm_vec[k]); // Need to apply scaling to the residual for the backsolve
    }

    // Solve system
    QDLDL_solve(n_vars, workspace->Lp.data(), workspace->Li.data(), workspace->Lx.data(),
                workspace->Dinv.data(),  workspace->newton_step.data());

    // Unpermute solution 
    workspace->newton_step = workspace->newton_step(workspace->amd_iperm_vec).eval();
    workspace->newton_step = workspace->newton_step.cwiseProduct(workspace->scaling).eval(); // Need to unscale the step after the backsolve
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

bool Solver::convergence(const Solver::Options& options) {
    // KKT residual norm
    double kkt_norm = workspace->kkt_residual.lpNorm<Eigen::Infinity>();

    // Primal feasibility
    double eq_violation = workspace->residual_eq.lpNorm<Eigen::Infinity>();
    double ineq_violation = workspace->residual_ineq.cwiseMin(0).lpNorm<Eigen::Infinity>();

    double comp_violation = (workspace->residual_comp(comp_L_inds).cwiseProduct(workspace->residual_comp(comp_R_inds)))
                              .lpNorm<Eigen::Infinity>();

    return kkt_norm < options.convergence_kkt_norm && eq_violation < options.convergence_eq_violation &&
           ineq_violation < options.convergence_ineq_violation && comp_violation < options.convergence_comp_violation;
}

bool Solver::solve(const Solver::Options& options) {
    bool converged = false;

    workspace->relax_param = options.relaxation_initial;
    workspace->penalty_param = options.penalty_initial;

    int last_outer_step_iter = -1;
    double outer_step_kkt_norm_adjustment = 1.0;

    workspace->solution.setZero(); // TODO: warm-starting options

    int n_iter_outer = 0;
    int n_iter_inner = 0;

    for (int iter = 0; iter < options.max_iters; ++iter) {
        // Compute KKT residual and check convergence
        double sqrt_relaxation_param = sqrt(workspace->relax_param);
        double inv_penalty_param = 1.0 / workspace->penalty_param;

        update_KKT_residual(sqrt_relaxation_param, inv_penalty_param);

        if (convergence(options)) {
            converged = true;
            break;
        }

        // Check if we are performing an inner or outer step based on KKT residual norm
        if (workspace->kkt_residual.lpNorm<Eigen::Infinity>() < outer_step_kkt_norm_adjustment * options.outer_step_kkt_norm) {            
            // Outer step: update multiplier estimates and increase penalty
            n_iter_outer++;
            
            // Check if we need to decrease the outer step KKT norm requirement, which is done only if
            // there have been 2 consecutive outer steps without an inner step in between
            if (iter > 0 && last_outer_step_iter == iter - 1 && workspace->relax_param <= options.relaxation_min) {
                outer_step_kkt_norm_adjustment /= 10.0;
            }

            if (workspace->penalty_param >= options.penalty_max) {
                // If we are at the maximum penalty, we can just update multiplier estimates without increasing penalty
                workspace->m_eq_est = workspace->m_eq;
                workspace->m_ineq_est = workspace->m_ineq;
                workspace->m_comp_est = workspace->m_comp;

                // Scale relaxation parameter
                workspace->relax_param = std::max(workspace->relax_param * options.relaxation_scaling, options.relaxation_min);
            } else {
                // Otherwise, increase penalty and update multiplier estimates with scaling
                workspace->penalty_param = std::min(workspace->penalty_param * options.penalty_scaling, options.penalty_max);
            }

            // Clear the filter
            filter->clear();
            last_outer_step_iter = iter;
        } else {
            n_iter_inner++;
            bool linesearch_succeeded = false;
            bool factorization_succeeded = false;
            bool inertia_correction_succeeded = false;

            update_KKT_system(sqrt_relaxation_param, inv_penalty_param);

            for (double regularizer : kkt_system_regularizers) {
                update_KKT_primal_regularizer(regularizer);
                if (!numerical_factorization()) {
                    continue;
                }
                factorization_succeeded = true;

                if (!check_inertia()) {
                    continue;
                }
                inertia_correction_succeeded = true;

                backsolve(); // Computes Newton step

                linesearch_succeeded = filter_linesearch(sqrt_relaxation_param, inv_penalty_param, options.max_iters_linesearch);

                if (linesearch_succeeded) {
                    break;
                }

                factorization_succeeded = false;
                inertia_correction_succeeded = false;
            }

            if (!factorization_succeeded) {
                throw std::runtime_error("Numerical factorization failed for all regularization values!");
            }

            if (!inertia_correction_succeeded) {
                throw std::runtime_error("Inertia correction failed for all regularization values!");
            }

            if (!linesearch_succeeded) {
                throw std::runtime_error("Linesearch failed to find an acceptable point!");
            }
        }
    }

    // // Output final solution and solve information to JSON
    // nlohmann::json output_json;
    // output_json["converged"] = converged;
    // output_json["n_iter_outer"] = n_iter_outer;
    // output_json["n_iter_inner"] = n_iter_inner;
    // output_json["n_iter"] = n_iter_outer + n_iter_inner;
    
    // output_json["x_opt"] = std::vector<double>(workspace->z.data(), workspace->z.data() + prob->nz);
    // std::ofstream f(options.output_dir / "output.json");
    // f << output_json.dump(4);

    return converged;
}

// TODO: is maybe weird that this is part of Solver and not Filter... but need access to the workspace and problem
// to compute these quantities
Filter::Entry Solver::entry_from_solution(double sqrt_relax_param, double inv_penalty_param) const {
    Vec m_eq_primal_feas = workspace->residual_eq - inv_penalty_param * (workspace->m_eq - workspace->m_eq_est);

    Vec p_ineq = retract(workspace->s_ineq, sqrt_relax_param);
    Vec m_ineq_primal_feas = workspace->residual_ineq - p_ineq -  // p(s) - p(-s) = s
                                inv_penalty_param * (workspace->m_ineq - workspace->m_ineq_est);

    Vec p_comp = retract(workspace->s_comp, sqrt_relax_param);
    Vec m_comp_primal_feas =
        workspace->residual_comp - inv_penalty_param * (workspace->m_comp - workspace->m_comp_est);

    for (int i = 0; i < prob->n_comp; i++) {
        m_comp_primal_feas[2 * i] += -p_comp[i];                               // p(s)
        m_comp_primal_feas[2 * i + 1] += -(p_comp[i] - workspace->s_comp[i]);  // p(s) - p(-s) = s
    }

    double candidate_constraint_violation = n_duals == 0 ? 0.0 :
        (m_eq_primal_feas.lpNorm<1>() + m_ineq_primal_feas.lpNorm<1>() + m_comp_primal_feas.lpNorm<1>()) / n_duals;

    double candidate_objective =
        0.5 * workspace->z.transpose() * prob->cost_hessian * workspace->z +
        prob->cost_gradient.dot(workspace->z) + prob->cost_const -
        pow(sqrt_relax_param, 2) * p_ineq.array().log().sum() +
        0.5 * inv_penalty_param *
            (workspace->m_eq.squaredNorm() + workspace->m_ineq.squaredNorm() + workspace->m_comp.squaredNorm());

    return {
        candidate_constraint_violation,
        candidate_objective
    };
}

bool Solver::filter_linesearch(double sqrt_relax_param, double inv_penalty_param, int max_iters) {
    double step_size = 1.0;
    workspace->solution += step_size * workspace->newton_step;  // Candidate solution for full step

    // TODO: this can be done much more efficiently due to the linearity
    // we should compute a delta for each constraint and the cost from the newton_step
    // and use those scaled vectors to assembly the feasibility candidates
    // We should also re-use the terms in the bottom half of the KKT residual
    
    for (int i = 0; i < max_iters; ++i) {
        // Update constraint residuals from candidate solution
        workspace->residual_eq = prob->J_eq * workspace->z + prob->c_eq;
        workspace->residual_ineq = prob->J_ineq * workspace->z + prob->c_ineq;
        workspace->residual_comp = prob->J_comp * workspace->z + prob->c_comp;

        const Filter::Entry candidate = entry_from_solution(sqrt_relax_param, inv_penalty_param);

        if (filter->acceptable(candidate)) {
            filter->update(candidate);
            return true;
        }

        // If not acceptable, shrink step size and try again
        step_size *= 0.5;
        workspace->solution -= step_size * workspace->newton_step;
    }

    // Restore original solution before returning failure
    workspace->solution -= step_size * workspace->newton_step;

    return false;
}