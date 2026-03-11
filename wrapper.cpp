#include <jlcxx/jlcxx.hpp>
#include "problem.h"

using jl_Vec = jlcxx::ArrayRef<double, 1>;
using jl_Mat = jlcxx::ArrayRef<double, 2>;

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
    // Problem class bindings
    mod.add_type<Problem>("Problem")
        .constructor([](jl_Mat cost_hessian, jl_Vec cost_gradient,
                    jl_Mat J_eq, jl_Vec c_eq, jl_Mat J_ineq, jl_Vec c_ineq,
                    jl_Mat J_comp_l, jl_Vec c_comp_l, jl_Mat J_comp_r, jl_Vec c_comp_r) {
        int nz = cost_gradient.size();
        int n_eq = c_eq.size();
        int n_ineq = c_ineq.size();
        int n_comp = c_comp_l.size();
        return new Problem(to_eigen(cost_hessian, nz, nz), to_eigen(cost_gradient),
                           to_eigen(J_eq, n_eq, nz), to_eigen(c_eq), 
                           to_eigen(J_ineq, n_ineq, nz), to_eigen(c_ineq),
                           to_eigen(J_comp_l, n_comp, nz), to_eigen(c_comp_l),
                           to_eigen(J_comp_r, n_comp, nz), to_eigen(c_comp_r));
        })
        .method("nz", [](Problem& p) -> const int { return p.nz; })
        .method("n_eq", [](Problem& p) -> const int { return p.n_eq; })
        .method("n_ineq", [](Problem& p) -> const int { return p.n_ineq; })
        .method("n_comp", [](Problem& p) -> const int { return p.n_comp; });

    
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