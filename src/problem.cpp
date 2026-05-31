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

void Problem::build_c_comp(const Vec& l, const Vec& r) {
    comp_L_inds = 2 * Eigen::VectorXi::LinSpaced(n_comp, 0, n_comp - 1);
    comp_R_inds = comp_L_inds.array() + 1;
    c_comp.resize(2 * n_comp);
    c_comp(comp_L_inds) = l;
    c_comp(comp_R_inds) = r;
}

Problem::Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
                 SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
                 SMat L, Vec l, SMat R, Vec r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(L.rows()),
      cost_hessian(std::move(cost_hessian)), cost_gradient(std::move(cost_gradient)), cost_const(cost_const),
      J_eq(std::move(J_eq)), c_eq(std::move(c_eq)),
      J_ineq(std::move(J_ineq)), c_ineq(std::move(c_ineq)) {
    J_comp = build_J_comp(L, R);
    build_c_comp(l, r);
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
    build_c_comp(l, r);
    cost_hessian_diag = this->cost_hessian.diagonal();
}

double Problem::obj(const Vec& z) const {
    return 0.5 * z.dot(cost_hessian * z) + cost_gradient.dot(z) + cost_const;
}

Vec Problem::residual_eq(const Vec& z) const {
    return J_eq * z + c_eq;
}

Vec Problem::residual_ineq(const Vec& z) const {
    return (J_ineq * z + c_ineq).cwiseMin(0);
}

Vec Problem::residual_comp(const Vec& z) const {
    Vec res = J_comp * z + c_comp;
    return res(comp_L_inds).cwiseProduct(res(comp_R_inds));
}