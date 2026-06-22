#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

// Common dense/sparse aliases used across Marble
using Mat  = Eigen::MatrixXd;
using Vec  = Eigen::VectorXd;
using SMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

inline Eigen::VectorXi safe_linspaced(int n, int start) {
    if (n == 0) return Eigen::VectorXi(0);
    return Eigen::VectorXi::LinSpaced(n, start, start + n - 1);
}
