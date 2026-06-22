#pragma once

#include <Eigen/Core>

#include "utils.h"

/**
 * Cached values for the relaxation map evaluated at one slack vector.
 *
 * Storage lives in Workspace because KKT residual/Jacobian assembly reuses the
 * same b_kappa values and derivatives several times per iteration.
 */
struct RelaxedSlackValues {
    Vec b;                    // b_kappa(x)
    Vec b_neg;                // b_kappa(-x)
    Vec b_prime;              // b'_kappa(x)
    Vec b_neg_prime;          // b'_kappa(-x)
    Vec b_double_prime;       // b''_kappa(x)
    Vec b_neg_double_prime;   // b''_kappa(-x)
    Vec d_b_d_kappa;          // d/dkappa b_kappa(x)
    Vec d_b_neg_d_kappa;      // d/dkappa b_kappa(-x)
    Vec d_b_prime_d_kappa;    // d/dkappa b'_kappa(x)
    Vec d_b_neg_prime_d_kappa;// d/dkappa b'_kappa(-x)

    RelaxedSlackValues() = default;
    explicit RelaxedSlackValues(Eigen::Index n) { resize(n); }

    void resize(Eigen::Index n);
    void setZero();
};

/**
 * b_kappa(x) = 1/2 * (x + sqrt(x^2 + 4*kappa)).
 *
 * This is the smooth positive-part map currently used by Marble. Identities
 * like b_kappa(-x) = b_kappa(x) - x stay hidden in evaluate(), so KKT code can
 * still read equations as b(x) and b(-x).
 */
class RelaxationMap {
public:
    Vec b(const Eigen::Ref<const Vec>& x, double kappa) const;
    Vec b_prime(const Eigen::Ref<const Vec>& x, double kappa) const;
    Vec b_double_prime(const Eigen::Ref<const Vec>& x, double kappa) const;
    Vec d_b_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const;
    Vec d_b_prime_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const;
    Vec d_b_double_prime_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const;

    RelaxedSlackValues values(const Eigen::Ref<const Vec>& x, double kappa) const;
    void evaluate(const Eigen::Ref<const Vec>& x, double kappa,
                  RelaxedSlackValues& out) const;

private:
    Eigen::ArrayXd discriminant(const Eigen::Ref<const Vec>& x, double kappa) const;
    Eigen::ArrayXd sqrt_discriminant(const Eigen::Ref<const Vec>& x, double kappa) const;
};
