#include "problem.h"

#include <stdexcept>
#include <string>

namespace {

std::string dims2(long rows, long cols) {
    return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}

// Verify that every block has consistent dimensions before any indexing happens
void validate_problem_dims(const SMat& H,      const Vec& grad,
                           const SMat& J_eq,   const Vec& c_eq,
                           const SMat& J_ineq, const Vec& c_ineq,
                           const SMat& L,      const Vec& l,
                           const SMat& R,      const Vec& r) {
    if (H.rows() != H.cols())
        throw std::invalid_argument(
            "Marble problem: cost Hessian Q must be square, got " + dims2(H.rows(), H.cols()));
    const long nz = H.cols();
    if (grad.size() != nz)
        throw std::invalid_argument(
            "Marble problem: cost gradient q has length " + std::to_string(grad.size()) +
            " but Q is " + dims2(nz, nz) + "; expected length " + std::to_string(nz));

    if (c_eq.size() != J_eq.rows())
        throw std::invalid_argument(
            "Marble problem: J_eq has " + std::to_string(J_eq.rows()) +
            " rows but c_eq/b_eq has length " + std::to_string(c_eq.size()) + "; they must match");
    if (J_eq.rows() > 0 && J_eq.cols() != nz)
        throw std::invalid_argument(
            "Marble problem: J_eq has " + std::to_string(J_eq.cols()) +
            " columns but expected " + std::to_string(nz));

    if (c_ineq.size() != J_ineq.rows())
        throw std::invalid_argument(
            "Marble problem: J_ineq has " + std::to_string(J_ineq.rows()) +
            " rows but c_ineq/b_ineq has length " + std::to_string(c_ineq.size()) + "; they must match");
    if (J_ineq.rows() > 0 && J_ineq.cols() != nz)
        throw std::invalid_argument(
            "Marble problem: J_ineq has " + std::to_string(J_ineq.cols()) +
            " columns but expected " + std::to_string(nz));

    if (!(L.rows() == R.rows() && L.rows() == l.size() && L.rows() == r.size()))
        throw std::invalid_argument(
            "Marble problem: complementarity blocks L, l, R, r must share the same number of "
            "rows; got L rows=" + std::to_string(L.rows()) +
            ", l length=" + std::to_string(l.size()) +
            ", R rows=" + std::to_string(R.rows()) +
            ", r length=" + std::to_string(r.size()));
    if (L.rows() > 0) {
        if (L.cols() != nz)
            throw std::invalid_argument(
                "Marble problem: L has " + std::to_string(L.cols()) +
                " columns but expected " + std::to_string(nz));
        if (R.cols() != nz)
            throw std::invalid_argument(
                "Marble problem: R has " + std::to_string(R.cols()) +
                " columns but expected " + std::to_string(nz));
    }
}

} // namespace

Problem::Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
                 SMat J_eq, Vec c_eq, SMat J_ineq, Vec c_ineq,
                 SMat L, Vec l, SMat R, Vec r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(L.rows()),
      cost_hessian(std::move(cost_hessian)), cost_gradient(std::move(cost_gradient)), cost_const(cost_const),
      J_eq(std::move(J_eq)), c_eq(std::move(c_eq)),
      J_ineq(std::move(J_ineq)), c_ineq(std::move(c_ineq)),
      L_comp(std::move(L)), l_comp(std::move(l)), R_comp(std::move(R)), r_comp(std::move(r)) {
    cost_hessian_diag = this->cost_hessian.diagonal();
    validate_problem_dims(this->cost_hessian, this->cost_gradient,
                          this->J_eq, this->c_eq, this->J_ineq, this->c_ineq,
                          this->L_comp, this->l_comp, this->R_comp, this->r_comp);
}

Problem::Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
                 Mat J_eq, Vec c_eq, Mat J_ineq, Vec c_ineq,
                 Mat L, Vec l, Mat R, Vec r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(L.rows()),
      cost_hessian(cost_hessian.sparseView()), cost_gradient(std::move(cost_gradient)), cost_const(cost_const),
      J_eq(J_eq.sparseView()), c_eq(std::move(c_eq)),
      J_ineq(J_ineq.sparseView()), c_ineq(std::move(c_ineq)),
      L_comp(L.sparseView()), l_comp(std::move(l)), R_comp(R.sparseView()), r_comp(std::move(r)) {
    cost_hessian_diag = this->cost_hessian.diagonal();
    validate_problem_dims(this->cost_hessian, this->cost_gradient,
                          this->J_eq, this->c_eq, this->J_ineq, this->c_ineq,
                          this->L_comp, this->l_comp, this->R_comp, this->r_comp);
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
    return (L_comp * z + l_comp).cwiseProduct(R_comp * z + r_comp);
}