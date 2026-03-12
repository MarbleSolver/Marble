#include <jlcxx/jlcxx.hpp>
#include "solver.h"

using jl_VecXd = jlcxx::ArrayRef<double, 1>;
using jl_MatXd = jlcxx::ArrayRef<double, 2>;
using jl_VecXi = jlcxx::ArrayRef<int, 1>;
using jl_MatXi = jlcxx::ArrayRef<int, 2>;

// Helper functions for converting between Eigen types and Julia types
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> to_eigen(jlcxx::ArrayRef<T, 1>& arr) {
    return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(arr.data(), arr.size());
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to_eigen(jlcxx::ArrayRef<T, 2>& arr, int rows, int cols) {
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Map(arr.data(), rows, cols);
}

template <typename T>
jlcxx::ArrayRef<T, 1> to_julia(Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
    return jlcxx::make_julia_array(vec.data(), vec.size());
}

template <typename T>
jlcxx::ArrayRef<T, 2> to_julia(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat) {
    return jlcxx::make_julia_array(mat.data(), mat.rows(), mat.cols());
}

Eigen::MatrixXd eigen_mat = Eigen::MatrixXd::Random(3, 3);
Eigen::VectorXd eigen_vec = Eigen::VectorXd::Random(5);

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    // Problem class bindings
    mod.add_type<Problem>("Problem")
        .constructor([](jl_MatXd cost_hessian, jl_VecXd cost_gradient,
                    jl_MatXd J_eq, jl_VecXd c_eq, jl_MatXd J_ineq, jl_VecXd c_ineq,
                    jl_MatXd J_comp, jl_VecXd c_comp) {
        int nz = cost_gradient.size();
        int n_eq = c_eq.size();
        int n_ineq = c_ineq.size();
        int n_comp = c_comp.size();
        return new Problem(to_eigen(cost_hessian, nz, nz), to_eigen(cost_gradient),
                           to_eigen(J_eq, n_eq, nz), to_eigen(c_eq), 
                           to_eigen(J_ineq, n_ineq, nz), to_eigen(c_ineq),
                           to_eigen(J_comp, n_comp, nz), to_eigen(c_comp));
        })
        .method("nz", [](Problem& p) -> const int { return p.nz; })
        .method("n_eq", [](Problem& p) -> const int { return p.n_eq; })
        .method("n_ineq", [](Problem& p) -> const int { return p.n_ineq; })
        .method("n_comp", [](Problem& p) -> const int { return p.n_comp; });

    // Workspace class bindings
    mod.add_type<Workspace>("Workspace")
        .constructor()
        .method("z", [](Workspace& w) { return to_julia(w.z); })
        .method("s_ineq", [](Workspace& w) { return to_julia(w.s_ineq); })
        .method("s_comp", [](Workspace& w) { return to_julia(w.s_comp); })
        .method("m_eq", [](Workspace& w) { return to_julia(w.m_eq); })
        .method("m_ineq", [](Workspace& w) { return to_julia(w.m_ineq); })
        .method("m_comp", [](Workspace& w) { return to_julia(w.m_comp); })
        .method("m_eq_est", [](Workspace& w) { return to_julia(w.m_eq_est); })
        .method("m_ineq_est", [](Workspace& w) { return to_julia(w.m_ineq_est); })
        .method("m_comp_est", [](Workspace& w) { return to_julia(w.m_comp_est); })
        .method("kkt_residual", [](Workspace& w) { return to_julia(w.kkt_residual); })
        .method("kkt_system", [](Workspace& w) {
            Eigen::SparseMatrix<double>& kkt = w.kkt_system;
            kkt.makeCompressed(); 
            jl_VecXi colptr = jlcxx::make_julia_array(kkt.outerIndexPtr(), kkt.outerSize() + 1);
            jl_VecXi rowval = jlcxx::make_julia_array(kkt.innerIndexPtr(), kkt.nonZeros());
            jl_VecXd nzval = jlcxx::make_julia_array(kkt.valuePtr(), kkt.nonZeros());

            return std::make_tuple(kkt.rows(), kkt.cols(), colptr, rowval, nzval); 
        });

    // Solver class bindings
    mod.add_type<Solver>("Solver")
        .constructor()
        .method("retract", [](Solver& solver, jl_VecXd s, double sqrt_relax_param) {
            Vec p = solver.retract(to_eigen(s), sqrt_relax_param);
            return std::vector<double>(p.data(), p.data() + p.size());
        })
        .method("retract_deriv", [](Solver& solver, jl_VecXd s, double sqrt_relax_param) {
            Vec p_deriv = solver.retract_deriv(to_eigen(s), sqrt_relax_param);
            return std::vector<double>(p_deriv.data(), p_deriv.data() + p_deriv.size());
        })
        .method("retract_second_deriv", [](Solver& solver, jl_VecXd s, double sqrt_relax_param) {
            Vec p_second_deriv = solver.retract_second_deriv(to_eigen(s), sqrt_relax_param);
            return std::vector<double>(p_second_deriv.data(), p_second_deriv.data() + p_second_deriv.size());
        })
        .method("update_KKT_residual", &Solver::update_KKT_residual)
        .method("update_KKT_system", &Solver::update_KKT_system)
        .method("update_KKT_ineq", [](Solver& solver, jl_VecXd s_ineq, double sqrt_relax_param) {
            solver.update_KKT_ineq(to_eigen(s_ineq), sqrt_relax_param);
        })
        .method("update_KKT_comp", [](Solver& solver, jl_VecXd s_comp, jl_VecXd m_comp, double sqrt_relax_param) {
            solver.update_KKT_comp(to_eigen(s_comp), to_eigen(m_comp), sqrt_relax_param);
        })
        .method("update_KKT_penalty", [](Solver& solver, double inv_penalty_param) {
            solver.update_KKT_penalty(inv_penalty_param);
        })
        .method("set_problem", &Solver::set_problem)
        .method("get_problem", &Solver::get_problem)
        .method("z_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.z_inds); })
        .method("s_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_ineq_inds); })
        .method("s_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_comp_inds); })
        .method("m_eq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.m_eq_inds); })
        .method("m_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.m_ineq_inds); })
        .method("m_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.m_comp_inds); })
        .method("s_ineq_s_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_ineq_s_ineq_inds); })
        .method("s_ineq_m_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_ineq_m_ineq_inds); })
        .method("s_comp_s_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_comp_s_comp_inds); })
        .method("s_comp_m_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_comp_m_comp_inds); })
        .method("get_workspace", &Solver::get_workspace);


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