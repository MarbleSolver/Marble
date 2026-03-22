#include "solver.h"

Solver::Solver() {
    // Init workspace
    workspace = std::make_shared<Workspace>();
};

Vec Solver::retract(const Vec& s, double sqrt_relax_param) {
    // Vectorized retraction map for both inequality and complementarity slacks
    // p(s) = sqrt(s^2 + relax_param)
    // Eigen::VectorXd p = 0.5*(s/sqrt_relax_param + (s.))
    auto& s_arr = s.array()/sqrt_relax_param;
    Vec p = 0.5*(s_arr + (s_arr.square() + 4.0).sqrt());
    return sqrt_relax_param * p;
}

Vec Solver::retract_deriv(const Vec& s, double sqrt_relax_param) {
    // Derivative of the vectorized retraction map above
    auto& s_arr = s.array()/sqrt_relax_param;
    Vec p_deriv = 0.5*(s_arr/sqrt(s_arr.square() + 4.0) + 1.0);
    return p_deriv;
}

Vec Solver::retract_second_deriv(const Vec& s, double sqrt_relax_param) {
    // Second derivative of the vectorized retraction map above
    auto& s_arr = s.array()/sqrt_relax_param;
    Vec p_second_deriv = 2.0/(s_arr.square() + 4.0).pow(1.5);
    return p_second_deriv/sqrt_relax_param;
}

void Solver::set_problem(Problem& prob) {
    // Move problem
    this->prob = std::make_shared<Problem>(std::move(prob));

    // Define subproblem dimensions
    n_primals = this->prob->nz + this->prob->n_ineq + this->prob->n_comp; // Primal variables include original and slacks
    n_duals = this->prob->n_eq + this->prob->n_ineq + 2*this->prob->n_comp; // Dual variables for each constraint
    n_vars = n_primals + n_duals; // Total variables

    // The inner subproblem solves for r(y) = 0 where r is the kkt residual
    // and y is the stacked vector of primal and dual variables.
    // Here we construct indices of where each set of variables is in y
    int total_inds = 0;
    z_inds = Eigen::VectorXi::LinSpaced(prob.nz, 0, prob.nz - 1);
    total_inds += prob.nz;
    s_ineq_inds = Eigen::VectorXi::LinSpaced(prob.n_ineq, total_inds, total_inds + prob.n_ineq - 1);
    total_inds += prob.n_ineq;
    s_comp_inds = Eigen::VectorXi::LinSpaced(prob.n_comp, total_inds, total_inds + prob.n_comp - 1);
    total_inds += prob.n_comp;
    m_eq_inds = Eigen::VectorXi::LinSpaced(prob.n_eq, total_inds, total_inds + prob.n_eq - 1);
    total_inds += prob.n_eq;
    m_ineq_inds = Eigen::VectorXi::LinSpaced(prob.n_ineq, total_inds, total_inds + prob.n_ineq - 1);
    total_inds += prob.n_ineq;
    m_comp_inds = Eigen::VectorXi::LinSpaced(2*prob.n_comp, total_inds, total_inds + 2*prob.n_comp - 1);
    total_inds += 2*prob.n_comp;

    // Allocate for solution vector, multiplier estimates, and KKT residual
    workspace->z.resize(prob.nz);
    workspace->s_ineq.resize(prob.n_ineq);
    workspace->s_comp.resize(prob.n_comp);
    workspace->m_eq.resize(prob.n_eq);
    workspace->m_ineq.resize(prob.n_ineq);
    workspace->m_comp.resize(2*prob.n_comp);
    workspace->m_eq_est.resize(prob.n_eq);
    workspace->m_ineq_est.resize(prob.n_ineq);
    workspace->m_comp_est.resize(2*prob.n_comp);
    workspace->kkt_residual.resize(n_vars);

    // Allocate for linear system solve
    workspace->etree.resize(n_vars);
    workspace->Lnz.resize(n_vars);
    workspace->iwork.resize(3 * n_vars);
    workspace->bwork.resize(n_vars);
    workspace->fwork.resize(n_vars);
    workspace->Lp.resize(n_vars + 1);

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
    workspace->appendBlockTriplets(triplets, prob->cost_hessian.triangularView<Eigen::Upper>(), z_inds[0], z_inds[0]);
    workspace->appendBlockTriplets(triplets, prob->J_eq.transpose(), z_inds[0], m_eq_inds[0]);
    workspace->appendBlockTriplets(triplets, prob->J_ineq.transpose(), z_inds[0], m_ineq_inds[0]);
    workspace->appendBlockTriplets(triplets, prob->J_comp.transpose(), z_inds[0], m_comp_inds[0]);

    // Populate stationarity for inequality slacks with dummy variables
    // -p'(s_ineq)*(p(-s_ineq) - m_ineq))
    // which has two diagonal matrices on the blocks corresponding to s_ineq and m_ineq
    for (int i = 0; i < prob->n_ineq; i++) {
        triplets.emplace_back(s_ineq_inds[i], s_ineq_inds[i], 1.0);
        triplets.emplace_back(s_ineq_inds[i], m_ineq_inds[i], 1.0);
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
    }

    // Populate AL regularizer for multipliers
    for (int i = 0; i < prob->n_eq; i++) {
        triplets.emplace_back(m_eq_inds[i], m_eq_inds[i], 1.0);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
        triplets.emplace_back(m_ineq_inds[i], m_ineq_inds[i], 1.0);
    }
    for (int i = 0; i < 2*prob->n_comp; i++) {
        triplets.emplace_back(m_comp_inds[i], m_comp_inds[i], 1.0);
    }

    // TODO: more efficient way to make sure that every diagonal element exists
    for (int i = 0; i < n_vars; i++) {
        triplets.emplace_back(i, i, 0.0);
    }

    // Build sparse matrix
    workspace->kkt_system.setFromTriplets(triplets.begin(), triplets.end());
    workspace->kkt_system.makeCompressed();

    // Determine indices into valuePtr for updating of nonlinear and penalty terms
    s_ineq_s_ineq_inds.resize(prob->n_ineq);
    s_ineq_m_ineq_inds.resize(prob->n_ineq);
    for (int i = 0; i < prob->n_ineq; i++) {
        s_ineq_s_ineq_inds[i] = workspace->findValuePtrIndex(s_ineq_inds[i], s_ineq_inds[i]);
        s_ineq_m_ineq_inds[i] = workspace->findValuePtrIndex(s_ineq_inds[i], m_ineq_inds[i]);
    }
    s_comp_s_comp_inds.resize(prob->n_comp);
    s_comp_m_comp_inds.resize(2*prob->n_comp);
    for (int i = 0; i < prob->n_comp; i++) {
        s_comp_s_comp_inds[i] = workspace->findValuePtrIndex(s_comp_inds[i], s_comp_inds[i]);
        s_comp_m_comp_inds[i*2] = workspace->findValuePtrIndex(s_comp_inds[i], m_comp_inds[i*2]);
        s_comp_m_comp_inds[i*2 + 1] = workspace->findValuePtrIndex(s_comp_inds[i], m_comp_inds[i*2 + 1]);
    }
    penalty_inds.resize(prob->n_eq + prob->n_ineq + 2*prob->n_comp);
    for (int i = 0; i < prob->n_eq; i++) {
        penalty_inds[i] = workspace->findValuePtrIndex(m_eq_inds[i], m_eq_inds[i]);
    }
    for (int i = 0; i < prob->n_ineq; i++) {
       penalty_inds[prob->n_eq + i] = workspace->findValuePtrIndex(m_ineq_inds[i], m_ineq_inds[i]);
    }
    for (int i = 0; i < 2*prob->n_comp; i++) {
        penalty_inds[prob->n_eq + prob->n_ineq + i] = workspace->findValuePtrIndex(m_comp_inds[i], m_comp_inds[i]);
    }
}

void Solver::update_KKT_residual(double sqrt_relax_param, double inv_penalty_param) {
    // z stationarity
    workspace->kkt_residual(z_inds) = prob->cost_hessian*workspace->z + prob->cost_gradient +
                           prob->J_eq.transpose()*workspace->m_eq + 
                           prob->J_ineq.transpose()*workspace->m_ineq + 
                           prob->J_comp.transpose()*workspace->m_comp;

    // Inequality slack stationarity
    Vec p_neg_ineq = retract(-workspace->s_ineq, sqrt_relax_param);
    Vec d_p_ineq = retract_deriv(workspace->s_ineq, sqrt_relax_param);
    workspace->kkt_residual(s_ineq_inds) = d_p_ineq.cwiseProduct(-workspace->m_ineq - p_neg_ineq);

    // Complementarity slack stationarity
    Vec d_p_comp = retract_deriv(workspace->s_comp, sqrt_relax_param);
    for (int i = 0; i < prob->n_comp; i++) {
        workspace->kkt_residual(s_comp_inds[i]) = -d_p_comp[i]*workspace->m_comp[2*i] + (1 - d_p_comp[i])*workspace->m_comp[2*i + 1];
    }

    // Equality primal feasibility
    workspace->kkt_residual(m_eq_inds) = prob->J_eq*workspace->z + prob->c_eq - 
                                            inv_penalty_param*(workspace->m_eq - workspace->m_eq_est);

    // Inequality primal feasibility
    workspace->kkt_residual(m_ineq_inds) = prob->J_ineq*workspace->z + prob->c_ineq - 
                                            (workspace->s_ineq + p_neg_ineq) - // p(s) - p(-s) = s
                                            inv_penalty_param*(workspace->m_ineq - workspace->m_ineq_est);

    // Complementarity primal feasibility
    Vec p_comp = retract(workspace->s_comp, sqrt_relax_param);
    workspace->kkt_residual(m_comp_inds) = prob->J_comp*workspace->z + prob->c_comp - 
                                            inv_penalty_param*(workspace->m_comp - workspace->m_comp_est);
    for (int i = 0; i < prob->n_comp; i++) {
        workspace->kkt_residual(m_comp_inds[2*i]) += -p_comp[i]; // p(s)
        workspace->kkt_residual(m_comp_inds[2*i + 1]) += -(p_comp[i] - workspace->s_comp[i]); // p(s) - p(-s) = s
    }
}

void Solver::update_KKT_system(double sqrt_relax_param, double inv_penalty_param) {
    // Update KKT system terms that depend on the solution, which are the nonlinear terms associated with the inequality and complementarity slacks and multipliers, as well as the penalty terms
    update_KKT_ineq(workspace->s_ineq, sqrt_relax_param);
    update_KKT_comp(workspace->s_comp, workspace->m_comp, sqrt_relax_param);
    update_KKT_penalty(inv_penalty_param);
}

void Solver::update_KKT_ineq(const Vec& s_ineq, double sqrt_relax_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Vec d_p = retract_deriv(s_ineq, sqrt_relax_param);

    // Use identity that retract_deriv(-s_ineq, sqrt_relax_param) = 1 - d_p
    nzval(s_ineq_s_ineq_inds) = -d_p.cwiseProduct(d_p) + d_p; // -d_p*d_neg_p
    nzval(s_ineq_m_ineq_inds) = -d_p;
}

// TODO: fix indexing into m_comp so that it can be a const reference
void Solver::update_KKT_comp(const Vec& s_comp, Vec m_comp, double sqrt_relax_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    Vec d_p = retract_deriv(s_comp, sqrt_relax_param);
    Vec dd_p = retract_second_deriv(s_comp, sqrt_relax_param);

    for (int i = 0; i < prob->n_comp; i++) {
        // Derivative of -p'(s)*m1 + p'(-s)*m2 wrt s
        // is -(p''(s)*m1 + p''(s)*m2) but p''(s) = p''(-s) so its -p''(s)*(m1 + m2)
        nzval(s_comp_s_comp_inds[i]) = -dd_p[i]*(m_comp[2*i] + m_comp[2*i + 1]);

        // Derivative wrt to m1 and m2 is just -d_p and -d_neg_p respectively
        nzval(s_comp_m_comp_inds[2*i]) = -d_p[i];
        nzval(s_comp_m_comp_inds[2*i + 1]) = (1 - d_p[i]); // d_neg_p = 1 - d_p
    } 
}

void Solver::update_KKT_penalty(const double inv_penalty_param) {
    Eigen::Map<Eigen::VectorXd> nzval(workspace->kkt_system.valuePtr(), workspace->kkt_system.nonZeros());
    nzval(penalty_inds) = -inv_penalty_param*Vec::Ones(prob->n_eq + prob->n_ineq + 2*prob->n_comp);
}

void Solver::analytical_factorization() {
    const QDLDL_int* Ap = workspace->kkt_system.outerIndexPtr();
    const QDLDL_int* Ai = workspace->kkt_system.innerIndexPtr();

    workspace->sum_Lnz = QDLDL_etree(n_vars, Ap, Ai, workspace->iwork.data(), workspace->Lnz.data(), workspace->etree.data());

    if (workspace->sum_Lnz < 0) {
            std::cerr << "Error: Matrix A is not properly formatted (must be upper triangular CSC)." << std::endl;
    }
}