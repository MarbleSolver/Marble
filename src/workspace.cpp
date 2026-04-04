#include "workspace.h"
#include <algorithm>

Workspace::Workspace(const Problem& prob) 
        :solution(prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq + prob.n_ineq + 2 * prob.n_comp),
      z(solution.data(), prob.nz),
      s_ineq(solution.data() + prob.nz, prob.n_ineq),
      s_comp(solution.data() + prob.nz + prob.n_ineq, prob.n_comp),
      m_eq(solution.data() + prob.nz + prob.n_ineq + prob.n_comp, prob.n_eq),
      m_ineq(solution.data() + prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq, prob.n_ineq),
      m_comp(solution.data() + prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq + prob.n_ineq, 2 * prob.n_comp)
{

    m_eq_est.resize(prob.n_eq);
    m_ineq_est.resize(prob.n_ineq);
    m_comp_est.resize(2 * prob.n_comp);

    int n_vars = prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq + prob.n_ineq + 2 * prob.n_comp;
    kkt_residual.resize(n_vars);
    s_comp_stationarity.resize(prob.n_comp);
    s_ineq_stationarity.resize(prob.n_ineq);
    newton_step.resize(n_vars);

    residual_eq.resize(prob.n_eq);
    residual_ineq.resize(prob.n_ineq);
    residual_comp.resize(2 * prob.n_comp);

    // Initialize solution and multiplier estimates to 0
    solution.setZero();
    m_eq_est.setZero();
    m_ineq_est.setZero();
    m_comp_est.setZero();

    // Initialize workspace constraint residuals
    residual_eq   = prob.J_eq * z + prob.c_eq;
    residual_ineq = prob.J_ineq * z + prob.c_ineq;
    residual_comp = prob.J_comp * z + prob.c_comp;

    // Allocate for linear system solve
    etree.resize(n_vars);
    Lnz.resize(n_vars);
    iwork.resize(3 * n_vars);
    bwork.resize(n_vars);
    fwork.resize(n_vars);
    Lp.resize(n_vars + 1);
    D.resize(n_vars);
    Dinv.resize(n_vars);
}

void Workspace::appendBlockTriplets(std::vector<Eigen::Triplet<double>>& triplets, const SMat& block, int row_start, int col_start) {
    for (int k=0; k<block.outerSize(); ++k) {
        for (SMat::InnerIterator it(block, k); it; ++it) {
            int row = row_start + it.row();
            int col = col_start + it.col();
            double value = it.value();
            triplets.emplace_back(row, col, value);
        }
    }
}

int Workspace::findValuePtrIndex(int row, int col) {
    // Map into permuted matrix
    row = amd_iperm_vec[row];
    col = amd_iperm_vec[col];
    if (row > col) {
        std::swap(row, col); // Ensure we are in the upper triangular part of the matrix, since it's symmetric
    }
    // Get indices for the start and end of the column
    int col_start = kkt_system.outerIndexPtr()[col];
    int col_end = kkt_system.outerIndexPtr()[col + 1];

    // Search over the row for the row index
    const QDLDL_int* inner = kkt_system.innerIndexPtr(); // This is like rowval in CSC format
    const QDLDL_int* it = std::lower_bound(inner + col_start, inner + col_end, row);
    if (it != inner + col_end && *it == row) {
        return it - inner;
    }
    return -1;
}