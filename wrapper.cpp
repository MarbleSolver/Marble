#include <jlcxx/jlcxx.hpp>
#include "problem.h"

// Helper functions for converting between Eigen types and Julia types
Eigen::VectorXd to_eigen(jlcxx::ArrayRef<double, 1>& arr) {
    return Eigen::VectorXd::Map(arr.data(), arr.size());
}

Eigen::MatrixXd to_eigen(jlcxx::ArrayRef<double, 2>& arr, int rows, int cols) {
    return Eigen::MatrixXd::Map(arr.data(), rows, cols);
}

jlcxx::ArrayRef<double, 1> to_julia(Eigen::VectorXd& vec) {
    return jlcxx::make_julia_array(vec.data(), vec.size());
}
jlcxx::ArrayRef<double, 2> to_julia(Eigen::MatrixXd& mat) {
    return jlcxx::make_julia_array(mat.data(), mat.rows(), mat.cols());
}

Eigen::MatrixXd eigen_mat = Eigen::MatrixXd::Random(3, 3);
Eigen::VectorXd eigen_vec = Eigen::VectorXd::Random(5);

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.method("test_array", [](jlcxx::ArrayRef<double, 1> arr) {
        std::cout << "here" << std::endl;
        eigen_vec = to_eigen(arr);
        std::cout << "Received array of size: " << eigen_vec.size() << std::endl;
        std::cout << "Contents: " << eigen_vec.transpose() << std::endl;
        eigen_vec[2] += 5;
        return to_julia(eigen_vec);
    });
    mod.method("test_matrix", [](jlcxx::ArrayRef<double, 2> arr, int rows, int cols) {
        std::cout << "here" << std::endl;
        eigen_mat = to_eigen(arr, rows, cols);
        std::cout << "Received matrix of size: " << eigen_mat.rows() << "x" << eigen_mat.cols() << std::endl;
        std::cout << "Contents:\n" << eigen_mat << std::endl;
        eigen_mat(0, 0) += 5;
        return to_julia(eigen_mat);
    });
}