#include "solver.h"

Solver::Solver() {
    // Init workspace
    workspace = std::make_shared<Workspace>();
};

void Solver::set_problem(Problem& prob) {
    // Move problem
    this->prob = std::make_shared<Problem>(std::move(prob));
    auto& p = *this->prob;

    auto make_range = [](int start, int count) {
        Eigen::VectorXi inds = Eigen::VectorXi::LinSpaced(count, 0, count - 1);
        inds.array() += start;
        return inds;
    };

    // Construct indices
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

    // Init KKT system
    workspace->kkt_system = SMat(total_inds, total_inds);
    std::vector<Eigen::Triplet<double>> triplets;
    workspace->appendBlockTriplets(triplets, p.cost_hessian, z_inds[0], z_inds[0]);
    workspace->appendBlockTriplets(triplets, p.J_eq.transpose(), z_inds[0], m_eq_inds[0]);
    workspace->appendBlockTriplets(triplets, p.J_ineq.transpose(), z_inds[0], m_ineq_inds[0]);
    workspace->appendBlockTriplets(triplets, p.J_comp.transpose(), z_inds[0], m_comp_inds[0]);
    workspace->kkt_system.setFromTriplets(triplets.begin(), triplets.end());
}