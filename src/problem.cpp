#include "problem.h"
#include "errors.h"

#include <cmath>
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
        throw InvalidInputError(
            "Marble problem: cost Hessian Q must be square, got " + dims2(H.rows(), H.cols()));
    const long nz = H.cols();
    if (nz == 0)
        throw InvalidInputError("Marble problem: the problem has no decision variables (Q is 0x0)");
    if (grad.size() != nz)
        throw InvalidInputError(
            "Marble problem: cost gradient q has length " + std::to_string(grad.size()) +
            " but Q is " + dims2(nz, nz) + "; expected length " + std::to_string(nz));

    if (c_eq.size() != J_eq.rows())
        throw InvalidInputError(
            "Marble problem: J_eq has " + std::to_string(J_eq.rows()) +
            " rows but c_eq/b_eq has length " + std::to_string(c_eq.size()) + "; they must match");
    if (J_eq.rows() > 0 && J_eq.cols() != nz)
        throw InvalidInputError(
            "Marble problem: J_eq has " + std::to_string(J_eq.cols()) +
            " columns but expected " + std::to_string(nz));

    if (c_ineq.size() != J_ineq.rows())
        throw InvalidInputError(
            "Marble problem: J_ineq has " + std::to_string(J_ineq.rows()) +
            " rows but c_ineq/b_ineq has length " + std::to_string(c_ineq.size()) + "; they must match");
    if (J_ineq.rows() > 0 && J_ineq.cols() != nz)
        throw InvalidInputError(
            "Marble problem: J_ineq has " + std::to_string(J_ineq.cols()) +
            " columns but expected " + std::to_string(nz));

    if (!(L.rows() == R.rows() && L.rows() == l.size() && L.rows() == r.size()))
        throw InvalidInputError(
            "Marble problem: complementarity blocks L, l, R, r must share the same number of "
            "rows; got L rows=" + std::to_string(L.rows()) +
            ", l length=" + std::to_string(l.size()) +
            ", R rows=" + std::to_string(R.rows()) +
            ", r length=" + std::to_string(r.size()));
    if (L.rows() > 0) {
        if (L.cols() != nz)
            throw InvalidInputError(
                "Marble problem: L has " + std::to_string(L.cols()) +
                " columns but expected " + std::to_string(nz));
        if (R.cols() != nz)
            throw InvalidInputError(
                "Marble problem: R has " + std::to_string(R.cols()) +
                " columns but expected " + std::to_string(nz));
    }
}

bool all_finite(const SMat& m) {
    for (int k = 0; k < m.outerSize(); ++k)
        for (SMat::InnerIterator it(m, k); it; ++it)
            if (!std::isfinite(it.value())) return false;
    return true;
}

// NaN/Inf data can't produce a meaningful solve, so reject it up front
void validate_problem_finite(const SMat& H,      const Vec& grad, double cost_const,
                             const SMat& J_eq,   const Vec& c_eq,
                             const SMat& J_ineq, const Vec& c_ineq,
                             const SMat& L,      const Vec& l,
                             const SMat& R,      const Vec& r) {
    const std::pair<const char*, bool> blocks[] = {
        {"Q", all_finite(H)}, {"q", grad.allFinite()}, {"c0", std::isfinite(cost_const)},
        {"J_eq", all_finite(J_eq)}, {"c_eq/b_eq", c_eq.allFinite()},
        {"J_ineq", all_finite(J_ineq)}, {"c_ineq/b_ineq", c_ineq.allFinite()},
        {"L", all_finite(L)}, {"l", l.allFinite()}, {"R", all_finite(R)}, {"r", r.allFinite()},
    };
    for (const auto& [name, finite] : blocks)
        if (!finite)
            throw InvalidInputError(std::string("Marble problem: ") + name + " contains NaN or Inf entries");
}

void check_primal_size(const Vec& z, int nz, const char* fn) {
    if (z.size() != nz)
        throw InvalidInputError(std::string("Marble problem: ") + fn + " expects z of length " +
                                std::to_string(nz) + ", got " + std::to_string(z.size()));
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
    for (SMat* block : {&this->cost_hessian, &this->J_eq, &this->J_ineq, &this->L_comp, &this->R_comp})
        block->makeCompressed();
    cost_hessian_diag = this->cost_hessian.diagonal();
    validate_problem_dims(this->cost_hessian, this->cost_gradient,
                          this->J_eq, this->c_eq, this->J_ineq, this->c_ineq,
                          this->L_comp, this->l_comp, this->R_comp, this->r_comp);
    validate_problem_finite(this->cost_hessian, this->cost_gradient, this->cost_const,
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
    for (SMat* block : {&this->cost_hessian, &this->J_eq, &this->J_ineq, &this->L_comp, &this->R_comp})
        block->makeCompressed();
    cost_hessian_diag = this->cost_hessian.diagonal();
    validate_problem_dims(this->cost_hessian, this->cost_gradient,
                          this->J_eq, this->c_eq, this->J_ineq, this->c_ineq,
                          this->L_comp, this->l_comp, this->R_comp, this->r_comp);
    validate_problem_finite(this->cost_hessian, this->cost_gradient, this->cost_const,
                            this->J_eq, this->c_eq, this->J_ineq, this->c_ineq,
                            this->L_comp, this->l_comp, this->R_comp, this->r_comp);
}

double Problem::obj(const Vec& z) const {
    check_primal_size(z, nz, "obj");
    return 0.5 * z.dot(cost_hessian * z) + cost_gradient.dot(z) + cost_const;
}

Vec Problem::residual_eq(const Vec& z) const {
    check_primal_size(z, nz, "residual_eq");
    return J_eq * z + c_eq;
}

Vec Problem::residual_ineq(const Vec& z) const {
    check_primal_size(z, nz, "residual_ineq");
    return (J_ineq * z + c_ineq).cwiseMin(0);
}

Vec Problem::residual_comp(const Vec& z) const {
    check_primal_size(z, nz, "residual_comp");
    return (L_comp * z + l_comp).cwiseProduct(R_comp * z + r_comp);
}