#include <jlcxx/jlcxx.hpp>
#include "solver.h"

using jl_VecXd = jlcxx::ArrayRef<double, 1>;
using jl_MatXd = jlcxx::ArrayRef<double, 2>;
using jl_VecXi = jlcxx::ArrayRef<int, 1>;
using jl_MatXi = jlcxx::ArrayRef<int, 2>;

// Helper functions for converting between Eigen types and Julia types
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to_eigen(jlcxx::ArrayRef<T, 2>& arr, int rows, int cols) {
    if (rows == 0 || cols == 0)
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(rows, cols);
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Map(arr.data(), rows, cols);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> to_eigen(jlcxx::ArrayRef<T, 1>& arr) {
    if (arr.size() == 0)
        return Eigen::Matrix<T, Eigen::Dynamic, 1>(0);
    return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(arr.data(), arr.size());
}

template <typename T>
jlcxx::ArrayRef<T, 1> to_julia(Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
    return jlcxx::make_julia_array(vec.data(), vec.size());
}

template <typename T>
jlcxx::ArrayRef<T, 1> to_julia(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>& vec) {
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
        .constructor([](jl_MatXd cost_hessian, jl_VecXd cost_gradient, double cost_const,
                    jl_MatXd J_eq, jl_VecXd c_eq,
                    jl_MatXd J_ineq, jl_VecXd c_ineq,
                    jl_MatXd J_comp, jl_VecXd c_comp) {
            int nz     = cost_gradient.size();
            int n_eq   = c_eq.size();
            int n_ineq = c_ineq.size();
            int n_comp = c_comp.size();
            
            return new Problem(
                to_eigen(cost_hessian, nz, nz),  to_eigen(cost_gradient), cost_const,
                to_eigen(J_eq,   n_eq,   nz),    to_eigen(c_eq),
                to_eigen(J_ineq, n_ineq, nz),    to_eigen(c_ineq),
                to_eigen(J_comp, n_comp, nz),    to_eigen(c_comp)
            );
        })
        .method("nz", [](Problem& p) -> const int { return p.nz; })
        .method("n_eq", [](Problem& p) -> const int { return p.n_eq; })
        .method("n_ineq", [](Problem& p) -> const int { return p.n_ineq; })
        .method("n_comp", [](Problem& p) -> const int { return p.n_comp; });
    
    #define OPTION_SETTER(name, type) \
        .method(#name "!", [](Solver::Options& o, type v) { o.name = v; }) \
        .method(#name,     [](Solver::Options& o)         { return o.name; })

    mod.add_type<Solver::Options>("SolverOptions")
        .constructor()
        OPTION_SETTER(convergence_kkt_norm,       double)
        OPTION_SETTER(convergence_eq_violation,   double)
        OPTION_SETTER(convergence_ineq_violation, double)
        OPTION_SETTER(convergence_comp_violation, double)
        OPTION_SETTER(outer_step_kkt_norm,        double)
        OPTION_SETTER(penalty_initial,            double)
        OPTION_SETTER(penalty_max,                double)
        OPTION_SETTER(penalty_scaling,            double)
        OPTION_SETTER(relaxation_initial,         double)
        OPTION_SETTER(relaxation_min,             double)
        OPTION_SETTER(relaxation_scaling,         double)
        OPTION_SETTER(max_iters,                  int)
        OPTION_SETTER(max_iters_linesearch,       int)
        OPTION_SETTER(gamma_objective,            double)
        OPTION_SETTER(gamma_constraint,           double)
        .method("output_dir!", [](Solver::Options& o, const std::string& v) { o.output_dir = v; })
        .method("output_dir",  [](Solver::Options& o) { return o.output_dir.string(); });

    #undef OPTION_SETTER

    // Filter class bindings
    mod.add_type<Filter>("Filter")
        .constructor()
        .method("filter_clear", &Filter::clear)
        .method("filter_size", []( Filter& f ) { return f.entries.size(); })
        .method("filter_entries", [](Filter& f) {
            std::vector<double> entries;
            entries.reserve(f.entries.size() * 2);
            for (const auto& entry : f.entries) {
                entries.push_back(entry.feas);
                entries.push_back(entry.merit);
            }
            return entries;
        });

    // Workspace class bindings
    mod.add_type<Workspace>("Workspace")
        .constructor([](const Problem& prob) { return new Workspace(prob); })
        .method("solution", [](Workspace& w) { return to_julia(w.solution); })
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
        .method("residual_eq", [](Workspace& w) { return to_julia(w.residual_eq); })
        .method("residual_ineq", [](Workspace& w) { return to_julia(w.residual_ineq); })
        .method("residual_comp", [](Workspace& w) { return to_julia(w.residual_comp); })
        .method("relax_param", [](Workspace& w) { return w.relax_param; })
        .method("penalty_param", [](Workspace& w) { return w.penalty_param; })
        .method("kkt_system", [](Workspace& w) {
            SMat& kkt = w.kkt_system;
            kkt.makeCompressed(); 
            jlcxx::ArrayRef<QDLDL_int, 1> colptr = jlcxx::make_julia_array(kkt.outerIndexPtr(), kkt.outerSize() + 1);
            jlcxx::ArrayRef<QDLDL_int, 1> rowval = jlcxx::make_julia_array(kkt.innerIndexPtr(), kkt.nonZeros());
            jl_VecXd nzval = jlcxx::make_julia_array(kkt.valuePtr(), kkt.nonZeros());

            return std::make_tuple(kkt.rows(), kkt.cols(), colptr, rowval, nzval); 
        })
        .method("newton_step", [](Workspace& w) { return to_julia(w.newton_step); })
        .method("D", [](Workspace& w) { return to_julia(w.D); })
        .method("amd_perm_vec", [](Workspace& w) { return to_julia(w.amd_perm_vec); })
        .method("amd_iperm_vec", [](Workspace& w) { return to_julia(w.amd_iperm_vec); });
        // .method("amd_perm", [](Workspace& w) { return to_julia(w.amd_perm.toDense()); })
;
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
        .method("update_KKT_primal_regularizer", &Solver::update_KKT_primal_regularizer)
        .method("analytical_factorization", &Solver::analytical_factorization)
        .method("numerical_factorization", &Solver::numerical_factorization)
        .method("backsolve", &Solver::backsolve)
        .method("check_inertia", &Solver::check_inertia)
        .method("compute_amd_ordering", &Solver::compute_amd_ordering)
        .method("solve", &Solver::solve)
        .method("ruiz_equilibration", [](Solver& solver,
                                          jl_MatXd H, int H_rows, int H_cols,
                                          jl_MatXd A, int A_rows, int A_cols,
                                          int niter) {
            Mat H_eigen = to_eigen(H, H_rows, H_cols);
            Mat A_eigen = to_eigen(A, A_rows, A_cols);

            SMat H_sparse = H_eigen.sparseView();
            SMat A_sparse = A_eigen.sparseView();

            auto [D, E] = solver.ruiz_equilibration(H_sparse, A_sparse, niter);
            return std::make_tuple(
                std::vector<double>(D.data(), D.data() + D.size()),
                std::vector<double>(E.data(), E.data() + E.size())
            );
        })
        .method("set_problem", [](Solver& solver, Problem& prob, jl_VecXd scaling) {
            solver.set_problem(prob, to_eigen(scaling));
        })
        .method("get_problem", &Solver::get_problem)
        .method("z_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.z_inds); })
        .method("s_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_ineq_inds); })
        .method("s_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_comp_inds); })
        .method("m_eq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.m_eq_inds); })
        .method("m_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.m_ineq_inds); })
        .method("m_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.m_comp_inds); })
        .method("comp_L_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.comp_L_inds); })
        .method("comp_R_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.comp_R_inds); })
        .method("z_z_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.z_z_inds); })
        .method("s_ineq_s_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_ineq_s_ineq_inds); })
        .method("s_ineq_m_ineq_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_ineq_m_ineq_inds); })
        .method("s_comp_s_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_comp_s_comp_inds); })
        .method("s_comp_m_comp_inds", [](Solver& s) -> jl_VecXi { return to_julia(s.s_comp_m_comp_inds); })
        .method("filter_linesearch", &Solver::filter_linesearch)
        .method("entry_from_solution", [](Solver& solver, double sqrt_relax_param, double inv_penalty_param) {
            Filter::Entry entry = solver.entry_from_solution(sqrt_relax_param, inv_penalty_param);
            return std::make_tuple(entry.feas, entry.merit);
        })
        .method("convergence", &Solver::convergence)
        .method("get_workspace", &Solver::get_workspace)
        .method("get_filter", &Solver::get_filter);
    
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