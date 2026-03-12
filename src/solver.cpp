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

    // Initialize KKT system sparsirty
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
}