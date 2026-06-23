#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

using Mat  = Eigen::MatrixXd;
using Vec  = Eigen::VectorXd;
using SMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

/**
 * Build an integer range or an empty vector
 *
 * @param n Number of entries
 * @param start First value in the range
 * @return Vector [start, start + n - 1], or an empty vector when n is zero
 */
inline Eigen::VectorXi safe_linspaced(int n, int start) {
    if (n == 0) return Eigen::VectorXi(0);
    return Eigen::VectorXi::LinSpaced(n, start, start + n - 1);
}
