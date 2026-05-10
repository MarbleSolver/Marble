#include "problem.h"

static SMat build_J_comp(const SMat& L, const SMat& R) {
    int n_comp = L.rows();
    int nz     = L.cols();
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(L.nonZeros() + R.nonZeros());
    for (int j = 0; j < L.outerSize(); ++j)
        for (SMat::InnerIterator it(L, j); it; ++it)
            triplets.emplace_back(2 * it.row(), it.col(), it.value());
    for (int j = 0; j < R.outerSize(); ++j)
        for (SMat::InnerIterator it(R, j); it; ++it)
            triplets.emplace_back(2 * it.row() + 1, it.col(), it.value());
    SMat result(2 * n_comp, nz);
    result.setFromTriplets(triplets.begin(), triplets.end());
    return result;
}

static Vec build_c_comp(const Vec& l, const Vec& r) {
    int n = l.size();
    Vec c(2 * n);
    for (int i = 0; i < n; ++i) {
        c(2 * i)     = l(i);
        c(2 * i + 1) = r(i);
    }
    return c;
}

Problem::Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
                 SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
                 SMat L, Vec l, SMat R, Vec r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(L.rows()),
      cost_hessian(std::move(cost_hessian)), cost_gradient(std::move(cost_gradient)), cost_const(cost_const),
      J_eq(std::move(J_eq)), c_eq(std::move(c_eq)),
      J_ineq(std::move(J_ineq)), c_ineq(std::move(c_ineq)) {
    J_comp = build_J_comp(L, R);
    c_comp = build_c_comp(l, r);
    cost_hessian_diag = this->cost_hessian.diagonal();
}

Problem::Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
                 Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
                 Mat L, Vec l, Mat R, Vec r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(L.rows()),
      cost_hessian(cost_hessian.sparseView()), cost_gradient(std::move(cost_gradient)), cost_const(cost_const),
      J_eq(J_eq.sparseView()), c_eq(std::move(c_eq)),
      J_ineq(J_ineq.sparseView()), c_ineq(std::move(c_ineq)) {
    Mat J_comp_dense(2 * n_comp, nz);
    J_comp_dense(Eigen::seq(0, Eigen::last, 2), Eigen::all) = L;
    J_comp_dense(Eigen::seq(1, Eigen::last, 2), Eigen::all) = R;
    J_comp = J_comp_dense.sparseView();
    c_comp = build_c_comp(l, r);
    cost_hessian_diag = this->cost_hessian.diagonal();
}
