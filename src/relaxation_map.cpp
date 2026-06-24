#include "relaxation_map.h"

void RelaxedSlackCache::resize(Eigen::Index n) {
    b.resize(n);
    b_neg.resize(n);
    b_prime.resize(n);
    b_neg_prime.resize(n);
    b_double_prime.resize(n);
    b_neg_double_prime.resize(n);
    d_b_d_kappa.resize(n);
    d_b_neg_d_kappa.resize(n);
    d_b_prime_d_kappa.resize(n);
    d_b_neg_prime_d_kappa.resize(n);
}

void RelaxedSlackCache::setZero() {
    b.setZero();
    b_neg.setZero();
    b_prime.setZero();
    b_neg_prime.setZero();
    b_double_prime.setZero();
    b_neg_double_prime.setZero();
    d_b_d_kappa.setZero();
    d_b_neg_d_kappa.setZero();
    d_b_prime_d_kappa.setZero();
    d_b_neg_prime_d_kappa.setZero();
}

RelaxedSlackCache RelaxationMap::values(const Eigen::Ref<const Vec>& x, double kappa) const {
    RelaxedSlackCache out(x.size());
    evaluate(x, kappa, out);
    return out;
}

void RelaxationMap::evaluate(const Eigen::Ref<const Vec>& x, double kappa,
                             RelaxedSlackCache& out) const {
    out.resize(x.size());

    const auto disc = discriminant(x, kappa);
    const auto sqrt_disc = disc.sqrt();

    out.b = (0.5 * (x.array() + sqrt_disc)).matrix();
    out.b_neg = out.b - x;  // b(x) - b(-x) = x

    out.b_prime = (0.5 * (1.0 + x.array() / sqrt_disc)).matrix();
    out.b_neg_prime = Vec::Ones(x.size()) - out.b_prime;  // b'(x) + b'(-x) = 1

    out.b_double_prime = (2.0 * kappa * disc.pow(-1.5)).matrix();
    out.b_neg_double_prime = out.b_double_prime;

    out.d_b_d_kappa = sqrt_disc.inverse().matrix();
    out.d_b_neg_d_kappa = out.d_b_d_kappa;
    out.d_b_prime_d_kappa = (-x.array() * disc.pow(-1.5)).matrix();
    out.d_b_neg_prime_d_kappa = -out.d_b_prime_d_kappa;
}

Vec RelaxationMap::b(const Eigen::Ref<const Vec>& x, double kappa) const {
    return (0.5 * (x.array() + sqrt_discriminant(x, kappa))).matrix();
}

Vec RelaxationMap::b_prime(const Eigen::Ref<const Vec>& x, double kappa) const {
    return (0.5 * (1.0 + x.array() / sqrt_discriminant(x, kappa))).matrix();
}

Vec RelaxationMap::b_double_prime(const Eigen::Ref<const Vec>& x, double kappa) const {
    return (2.0 * kappa * discriminant(x, kappa).pow(-1.5)).matrix();
}

Vec RelaxationMap::d_b_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const {
    return sqrt_discriminant(x, kappa).inverse().matrix();
}

Vec RelaxationMap::d_b_prime_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const {
    return (-x.array() * discriminant(x, kappa).pow(-1.5)).matrix();
}

Vec RelaxationMap::d_b_double_prime_d_kappa(const Eigen::Ref<const Vec>& x, double kappa) const {
    const auto disc = discriminant(x, kappa);
    return (2.0 * (x.array().square() - 2.0 * kappa) * disc.pow(-2.5)).matrix();
}

Eigen::ArrayXd RelaxationMap::discriminant(const Eigen::Ref<const Vec>& x, double kappa) const {
    return x.array().square() + 4.0 * kappa;
}

Eigen::ArrayXd RelaxationMap::sqrt_discriminant(const Eigen::Ref<const Vec>& x, double kappa) const {
    return discriminant(x, kappa).sqrt();
}
