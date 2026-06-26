#include "problem.h"

#include <stdexcept>
#include <string>

namespace {
/**
 * Format matrix dimensions for error messages
 *
 * @param rows Row count
 * @param cols Column count
 * @return Dimension string
 */
std::string dims2(long rows, long cols) {
    return "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
}
} // namespace

Problem::Problem(SMat cost_hessian, Vec cost_gradient, double cost_const,
                 SMat J_eq, Vec b_eq, SMat J_ineq, Vec b_ineq,
                 SMat L, Vec l, SMat R, Vec r)
    : nz(cost_hessian.cols()), n_eq(J_eq.rows()), n_ineq(J_ineq.rows()), n_comp(L.rows()),
      cost_hessian(std::move(cost_hessian)), cost_gradient(std::move(cost_gradient)), cost_const(cost_const),
      J_eq(std::move(J_eq)), b_eq(std::move(b_eq)),
      J_ineq(std::move(J_ineq)), b_ineq(std::move(b_ineq)),
      L(std::move(L)), l(std::move(l)),
      R(std::move(R)), r(std::move(r)) {
    validate_dims();
}

Problem::Problem(Mat cost_hessian, Vec cost_gradient, double cost_const,
                 Mat J_eq, Vec b_eq, Mat J_ineq, Vec b_ineq,
                 Mat L, Vec l, Mat R, Vec r)
    : Problem(SMat(cost_hessian.sparseView()), std::move(cost_gradient), cost_const,
              SMat(J_eq.sparseView()),   std::move(b_eq),
              SMat(J_ineq.sparseView()), std::move(b_ineq),
              SMat(L.sparseView()),      std::move(l),
              SMat(R.sparseView()),      std::move(r)) {}

void Problem::validate_dims() const {
    if (cost_hessian.rows() != cost_hessian.cols())
        throw std::invalid_argument(
            "Marble problem: cost Hessian must be square, got " +
            dims2(cost_hessian.rows(), cost_hessian.cols()));
    if (cost_gradient.size() != nz)
        throw std::invalid_argument(
            "Marble problem: cost gradient has length " + std::to_string(cost_gradient.size()) +
            " but expected " + std::to_string(nz));

    if (b_eq.size() != n_eq)
        throw std::invalid_argument(
            "Marble problem: J_eq has " + std::to_string(n_eq) +
            " rows but b_eq has length " + std::to_string(b_eq.size()) + "; they must match");
    if (n_eq > 0 && J_eq.cols() != nz)
        throw std::invalid_argument(
            "Marble problem: J_eq has " + std::to_string(J_eq.cols()) +
            " columns but expected " + std::to_string(nz));

    if (b_ineq.size() != n_ineq)
        throw std::invalid_argument(
            "Marble problem: J_ineq has " + std::to_string(n_ineq) +
            " rows but b_ineq has length " + std::to_string(b_ineq.size()) + "; they must match");
    if (n_ineq > 0 && J_ineq.cols() != nz)
        throw std::invalid_argument(
            "Marble problem: J_ineq has " + std::to_string(J_ineq.cols()) +
            " columns but expected " + std::to_string(nz));

    if (!(R.rows() == n_comp && l.size() == n_comp && r.size() == n_comp))
        throw std::invalid_argument(
            "Marble problem: complementarity blocks L, l, R, r must share the same number of "
            "rows; got L rows=" + std::to_string(n_comp) +
            ", l length=" + std::to_string(l.size()) +
            ", R rows=" + std::to_string(R.rows()) +
            ", r length=" + std::to_string(r.size()));
    if (n_comp > 0) {
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

double Problem::obj(const Vec& z) const {
    return 0.5 * z.dot(cost_hessian * z) + cost_gradient.dot(z) + cost_const;
}

Vec Problem::residual_eq(const Vec& z) const {
    return J_eq * z + b_eq;
}

Vec Problem::residual_ineq(const Vec& z) const {
    return J_ineq * z + b_ineq;
}

Vec Problem::residual_comp_L(const Vec& z) const {
    return L * z + l;
}

Vec Problem::residual_comp_R(const Vec& z) const {
    return R * z + r;
}

Vec Problem::residual_comp(const Vec& z) const {
    return residual_comp_L(z).cwiseProduct(residual_comp_R(z));
}
