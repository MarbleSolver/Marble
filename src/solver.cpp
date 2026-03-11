#include "solver.h"

Solver::Solver() {

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
}