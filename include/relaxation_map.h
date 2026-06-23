#pragma once

#include <Eigen/Core>

#include "utils.h"

/**
 * Cached values for the relaxation map evaluated at one slack vector
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

    /**
     * Construct an empty cache
     */
    RelaxedSlackValues() = default;

    /**
     * Allocate all cached vectors
     *
     * @param n Number of slack entries
     */
    explicit RelaxedSlackValues(Eigen::Index n) { resize(n); }

    /**
     * Resize every cached vector
     *
     * @param n Number of slack entries
     */
    void resize(Eigen::Index n);

    /**
     * Set every cached vector to zero
     */
    void setZero();
};

/**
 * b_kappa(x) = 1/2 * (x + sqrt(x^2 + 4*kappa))
 *
 * This is the smooth positive-part map currently used by Marble. Identities
 * like b_kappa(-x) = b_kappa(x) - x stay hidden in evaluate(), so KKT code can
 * still read equations as b(x) and b(-x)
 */
class RelaxationMap {
public:
    /**
     * Evaluate b_kappa(x)
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Value of b_kappa at x
     */
    Vec b(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate the first derivative of b_kappa(x) with respect to x
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return First derivative values
     */
    Vec b_prime(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate the second derivative of b_kappa(x) with respect to x
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Second derivative values
     */
    Vec b_double_prime(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate the derivative of b_kappa(x) with respect to kappa
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Kappa derivative values
     */
    Vec d_b_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate the kappa derivative of b_prime
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Kappa derivative of b_prime
     */
    Vec d_b_prime_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate the kappa derivative of b_double_prime
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Kappa derivative of b_double_prime
     */
    Vec d_b_double_prime_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate and return all cached relaxation values
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Cached values for x and -x
     */
    RelaxedSlackValues values(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Evaluate all relaxation values into existing storage
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @param out Cache to resize and fill
     */
    void evaluate(const Eigen::Ref<const Vec>& x, double kappa,
                  RelaxedSlackValues& out) const;

private:
    /**
     * Compute x squared plus 4*kappa
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Discriminant array
     */
    Eigen::ArrayXd discriminant(const Eigen::Ref<const Vec>& x, double kappa) const;

    /**
     * Compute the square root of the discriminant
     *
     * @param x Input vector
     * @param kappa Relaxation parameter
     * @return Square-root discriminant array
     */
    Eigen::ArrayXd sqrt_discriminant(const Eigen::Ref<const Vec>& x, double kappa) const;
};
