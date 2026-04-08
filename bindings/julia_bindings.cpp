#include <jlcxx/jlcxx.hpp>
#include "solver.h"
#include <cmath>

// Filter::Entry is a plain struct; tell CxxWrap not to treat it as a mirrored type
// so we can register it with add_type and attach methods to it.
namespace jlcxx {
    template<> struct IsMirroredType<Filter::Entry> : std::false_type {};
}

// ---------------------------------------------------------------------------
// Eigen ↔ Julia array helpers
// ---------------------------------------------------------------------------

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
to_eigen(jlcxx::ArrayRef<T, 2>& arr, int rows, int cols) {
    if (rows == 0 || cols == 0)
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(rows, cols);
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Map(arr.data(), rows, cols);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
to_eigen(jlcxx::ArrayRef<T, 1>& arr) {
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

// Build an Eigen sparse matrix from Julia SparseMatrixCSC components.
// Julia uses 1-based indices; Eigen uses 0-based.
// Accepts Int64 colptr/rowval (Julia's default index type) and casts to QDLDL_int internally.
static SMat csc_to_smat(int rows, int cols,
                        jlcxx::ArrayRef<int64_t, 1> colptr,
                        jlcxx::ArrayRef<int64_t, 1> rowval,
                        jlcxx::ArrayRef<double, 1>  nzval) {
    int nnz = (int)nzval.size();
    std::vector<QDLDL_int> cp(cols + 1), ri(nnz);
    for (int i = 0; i <= cols; i++) cp[i] = (QDLDL_int)(colptr[i] - 1);
    for (int i = 0; i < nnz;  i++) ri[i] = (QDLDL_int)(rowval[i] - 1);
    Eigen::Map<SMat> mapped(rows, cols, nnz,
                            cp.data(), ri.data(),
                            const_cast<double*>(nzval.data()));
    SMat result = mapped;
    result.makeCompressed();
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    // -----------------------------------------------------------------------
    // Problem
    // -----------------------------------------------------------------------
    mod.add_type<Problem>("Problem")
        .constructor([](jlcxx::ArrayRef<double, 2> cost_hessian,
                        jlcxx::ArrayRef<double, 1> cost_gradient,
                        double cost_const,
                        jlcxx::ArrayRef<double, 2> J_eq,   jlcxx::ArrayRef<double, 1> c_eq,
                        jlcxx::ArrayRef<double, 2> J_ineq, jlcxx::ArrayRef<double, 1> c_ineq,
                        jlcxx::ArrayRef<double, 2> J_comp, jlcxx::ArrayRef<double, 1> c_comp) {
            int nz     = cost_gradient.size();
            int n_eq   = c_eq.size();
            int n_ineq = c_ineq.size();
            int n_comp = c_comp.size();
            return new Problem(
                to_eigen(cost_hessian, nz, nz), to_eigen(cost_gradient), cost_const,
                to_eigen(J_eq,   n_eq,   nz),   to_eigen(c_eq),
                to_eigen(J_ineq, n_ineq, nz),   to_eigen(c_ineq),
                to_eigen(J_comp, n_comp, nz),   to_eigen(c_comp));
        })
        // Sparse constructor: accepts SparseMatrixCSC components from Julia.
        // For each sparse matrix pass (colptr, rowval, nzval) from the Julia struct.
        // Julia's colptr/rowval are Int64 (1-based); conversion to 0-based happens internally.
        // Usage from Julia:
        //   Problem(nz,
        //           H.colptr, H.rowval, H.nzval, q, cost_const,
        //           n_eq,   J_eq.colptr,   J_eq.rowval,   J_eq.nzval,   c_eq,
        //           n_ineq, J_ineq.colptr, J_ineq.rowval, J_ineq.nzval, c_ineq,
        //           n_comp, J_comp.colptr, J_comp.rowval, J_comp.nzval, c_comp)
        .constructor([](int nz,
                        jlcxx::ArrayRef<int64_t,1> Hcp,  jlcxx::ArrayRef<int64_t,1> Hrv,  jlcxx::ArrayRef<double,1> Hnz,
                        jlcxx::ArrayRef<double,1> grad, double cost_const,
                        int n_eq,
                        jlcxx::ArrayRef<int64_t,1> Ecp,  jlcxx::ArrayRef<int64_t,1> Erv,  jlcxx::ArrayRef<double,1> Enz,
                        jlcxx::ArrayRef<double,1> c_eq,
                        int n_ineq,
                        jlcxx::ArrayRef<int64_t,1> Icp,  jlcxx::ArrayRef<int64_t,1> Irv,  jlcxx::ArrayRef<double,1> Inz,
                        jlcxx::ArrayRef<double,1> c_ineq,
                        int n_comp,
                        jlcxx::ArrayRef<int64_t,1> Ccp,  jlcxx::ArrayRef<int64_t,1> Crv,  jlcxx::ArrayRef<double,1> Cnz,
                        jlcxx::ArrayRef<double,1> c_comp) {
            return new Problem(
                csc_to_smat(nz,     nz, Hcp, Hrv, Hnz), to_eigen(grad), cost_const,
                csc_to_smat(n_eq,   nz, Ecp, Erv, Enz), to_eigen(c_eq),
                csc_to_smat(n_ineq, nz, Icp, Irv, Inz), to_eigen(c_ineq),
                csc_to_smat(n_comp, nz, Ccp, Crv, Cnz), to_eigen(c_comp));
        })
        .method("nz",     [](const Problem& p) { return p.nz; })
        .method("n_eq",   [](const Problem& p) { return p.n_eq; })
        .method("n_ineq", [](const Problem& p) { return p.n_ineq; })
        .method("n_comp", [](const Problem& p) { return p.n_comp; });

    // -----------------------------------------------------------------------
    // SolverOptions
    // -----------------------------------------------------------------------
    #define OPTION_RW(name, T) \
        .method(#name,       [](const Solver::Options& o)    { return o.name; }) \
        .method(#name "!",   [](Solver::Options& o, T v)     { o.name = v; })

    mod.add_type<Solver::Options>("SolverOptions")
        .constructor()
        OPTION_RW(convergence_kkt_norm,       double)
        OPTION_RW(convergence_eq_violation,   double)
        OPTION_RW(convergence_ineq_violation, double)
        OPTION_RW(convergence_comp_violation, double)
        OPTION_RW(outer_step_kkt_norm,        double)
        OPTION_RW(penalty_initial,            double)
        OPTION_RW(penalty_max,                double)
        OPTION_RW(penalty_scaling,            double)
        OPTION_RW(relaxation_initial,         double)
        OPTION_RW(relaxation_min,             double)
        OPTION_RW(relaxation_scaling,         double)
        OPTION_RW(max_iters,                  int)
        OPTION_RW(max_iters_linesearch,       int)
        OPTION_RW(gamma_objective,            double)
        OPTION_RW(gamma_constraint,           double)
        OPTION_RW(ruiz_iterations,            int)
        .method("output_dir",  [](const Solver::Options& o) { return o.output_dir.string(); })
        .method("output_dir!", [](Solver::Options& o, const std::string& v) { o.output_dir = v; });

    #undef OPTION_RW

    // -----------------------------------------------------------------------
    // FilterEntry
    // -----------------------------------------------------------------------
    mod.add_type<Filter::Entry>("FilterEntry")
        .constructor()
        .constructor([](double feas, double merit) {
            Filter::Entry* e = new Filter::Entry(); e->feas = feas; e->merit = merit; return e;
        })
        .method("feas",   [](const Filter::Entry& e) { return e.feas; })
        .method("feas!",  [](Filter::Entry& e, double v) { e.feas = v; })
        .method("merit",  [](const Filter::Entry& e) { return e.merit; })
        .method("merit!", [](Filter::Entry& e, double v) { e.merit = v; });

    // -----------------------------------------------------------------------
    // Filter
    // -----------------------------------------------------------------------
    mod.add_type<Filter>("Filter")
        .constructor()
        .constructor([](double gamma_objective, double gamma_constraint) {
            return new Filter(gamma_objective, gamma_constraint);
        })
        .method("clear",   &Filter::clear)
        .method("size",    [](const Filter& f) { return f.entries.size(); })
        // Returns a flat vector [feas0, merit0, feas1, merit1, ...]
        // Use reshape(entries(f), 2, :) in Julia to get a 2×n matrix
        .method("entries", [](const Filter& f) {
            std::vector<double> out;
            out.reserve(f.entries.size() * 2);
            for (const auto& e : f.entries) {
                out.push_back(e.feas);
                out.push_back(e.merit);
            }
            return out;
        })
        // sufficient_progress returns std::pair<bool,bool> which CxxWrap can't wrap directly,
        // so expose each component as its own method
        .method("sufficient_feas_progress", [](Filter& f, const Filter::Entry& c, const Filter::Entry& e) {
            return f.sufficient_progress(c, e).first;
        })
        .method("sufficient_merit_progress", [](Filter& f, const Filter::Entry& c, const Filter::Entry& e) {
            return f.sufficient_progress(c, e).second;
        })
        .method("candidate_acceptable", &Filter::candidate_acceptable)
        .method("candidate_dominated",  &Filter::candidate_dominated)
        .method("acceptable",           &Filter::acceptable)
        .method("update",               &Filter::update);

    // -----------------------------------------------------------------------
    // Workspace  (read-only accessors)
    // -----------------------------------------------------------------------
    mod.add_type<Workspace>("Workspace")
        // Full solution and decomposed views
        .method("solution", [](Workspace& w) { return to_julia(w.solution); })
        .method("z",        [](Workspace& w) { return to_julia(w.z); })
        .method("s_ineq",   [](Workspace& w) { return to_julia(w.s_ineq); })
        .method("s_comp",   [](Workspace& w) { return to_julia(w.s_comp); })
        .method("m_eq",     [](Workspace& w) { return to_julia(w.m_eq); })
        .method("m_ineq",   [](Workspace& w) { return to_julia(w.m_ineq); })
        .method("m_comp",   [](Workspace& w) { return to_julia(w.m_comp); })
        // Multiplier estimates
        .method("m_eq_est",   [](Workspace& w) { return to_julia(w.m_eq_est); })
        .method("m_ineq_est", [](Workspace& w) { return to_julia(w.m_ineq_est); })
        .method("m_comp_est", [](Workspace& w) { return to_julia(w.m_comp_est); })
        // Residuals
        .method("kkt_residual",  [](Workspace& w) { return to_julia(w.kkt_residual); })
        .method("residual_eq",   [](Workspace& w) { return to_julia(w.residual_eq); })
        .method("residual_ineq", [](Workspace& w) { return to_julia(w.residual_ineq); })
        .method("residual_comp", [](Workspace& w) { return to_julia(w.residual_comp); })
        // Scalar parameters
        .method("relax_param",   [](const Workspace& w) { return w.relax_param; })
        .method("penalty_param", [](const Workspace& w) { return w.penalty_param; })
        // Newton step
        .method("newton_step", [](Workspace& w) { return to_julia(w.newton_step); })
        // KKT system: returns (rows, cols, colptr, rowval, nzval) tuple
        .method("kkt_system", [](Workspace& w) {
            SMat& kkt = w.kkt_system;
            kkt.makeCompressed();
            auto colptr = jlcxx::make_julia_array(kkt.outerIndexPtr(), kkt.outerSize() + 1);
            auto rowval = jlcxx::make_julia_array(kkt.innerIndexPtr(), kkt.nonZeros());
            auto nzval  = jlcxx::make_julia_array(kkt.valuePtr(),      kkt.nonZeros());
            return std::make_tuple((int)kkt.rows(), (int)kkt.cols(), colptr, rowval, nzval);
        })
        // QDLDL factorization data
        .method("D",            [](Workspace& w) { return to_julia(w.D); })
        .method("amd_perm_vec", [](Workspace& w) { return to_julia(w.amd_perm_vec); })
        .method("amd_iperm_vec", [](Workspace& w) { return to_julia(w.amd_iperm_vec); })
        .method("scaling", [](Workspace& w) { return to_julia(w.scaling); });

    // -----------------------------------------------------------------------
    // Solver
    // -----------------------------------------------------------------------
    mod.add_type<Solver>("Solver")
        .constructor()
        .constructor([](const Solver::Options& opts) { return new Solver(opts); })
        // Problem setup
        .method("ruiz_equilibration!", [](Solver& solver, int niter) {
            solver.ruiz_equilibration(niter);
            Workspace& workspace = solver.get_workspace();
            return std::vector<double>(workspace.scaling.data(), workspace.scaling.data() + workspace.scaling.size());
        })
        .method("set_problem!", [](Solver& s, Problem& prob) {
            s.set_problem(prob);
        })
        .method("get_problem",   &Solver::get_problem)
        // Main solve
        .method("solve!",       &Solver::solve)
        .method("convergence", &Solver::convergence)
        // Workspace / filter access
        .method("get_workspace", &Solver::get_workspace)
        .method("get_filter",    &Solver::get_filter)
        // Retraction maps
        .method("retract", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double sqrt_relax_param) {
            Vec r = s.retract(to_eigen(v), sqrt_relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        .method("retract_deriv", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double sqrt_relax_param) {
            Vec r = s.retract_deriv(to_eigen(v), sqrt_relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        .method("retract_second_deriv", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double sqrt_relax_param) {
            Vec r = s.retract_second_deriv(to_eigen(v), sqrt_relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        // KKT updates
        .method("update_KKT_residual!",          &Solver::update_KKT_residual)
        .method("update_KKT_system!",            &Solver::update_KKT_system)
        .method("update_KKT_ineq!", [](Solver& s, jlcxx::ArrayRef<double, 1> s_ineq, double sqrt_relax_param) {
            s.update_KKT_ineq(to_eigen(s_ineq), sqrt_relax_param);
        })
        .method("update_KKT_comp!", [](Solver& s, jlcxx::ArrayRef<double, 1> s_comp, jlcxx::ArrayRef<double, 1> m_comp, double sqrt_relax_param) {
            s.update_KKT_comp(to_eigen(s_comp), to_eigen(m_comp), sqrt_relax_param);
        })
        .method("update_KKT_penalty!",            &Solver::update_KKT_penalty)
        .method("update_KKT_primal_regularizer!", &Solver::update_KKT_primal_regularizer)
        // Factorization and backsolve
        .method("initialize_kkt_sparsity!",  &Solver::initialize_kkt_sparsity)
        .method("compute_amd_ordering!",     &Solver::compute_amd_ordering)
        .method("analytical_factorization!", &Solver::analytical_factorization)
        .method("numerical_factorization!",  &Solver::numerical_factorization)
        .method("check_inertia",            &Solver::check_inertia)
        .method("backsolve!",                &Solver::backsolve)
        // Linesearch
        .method("filter_linesearch!", &Solver::filter_linesearch)
        .method("entry_from_solution", [](const Solver& s, double sqrt_relax_param, double inv_penalty_param) {
            Filter::Entry e = s.entry_from_solution(sqrt_relax_param, inv_penalty_param);
            return std::make_tuple(e.feas, e.merit);
        })
        // Dimensions
        .method("n_primals", [](const Solver& s) { return s.n_primals; })
        .method("n_duals",   [](const Solver& s) { return s.n_duals; })
        .method("n_vars",    [](const Solver& s) { return s.n_vars; })
        // KKT index vectors
        .method("z_inds",             [](Solver& s) { return to_julia(s.z_inds); })
        .method("s_ineq_inds",        [](Solver& s) { return to_julia(s.s_ineq_inds); })
        .method("s_comp_inds",        [](Solver& s) { return to_julia(s.s_comp_inds); })
        .method("m_eq_inds",          [](Solver& s) { return to_julia(s.m_eq_inds); })
        .method("m_ineq_inds",        [](Solver& s) { return to_julia(s.m_ineq_inds); })
        .method("m_comp_inds",        [](Solver& s) { return to_julia(s.m_comp_inds); })
        .method("comp_L_inds",        [](Solver& s) { return to_julia(s.comp_L_inds); })
        .method("comp_R_inds",        [](Solver& s) { return to_julia(s.comp_R_inds); })
        .method("z_z_inds",           [](Solver& s) { return to_julia(s.z_z_inds); })
        .method("s_ineq_s_ineq_inds", [](Solver& s) { return to_julia(s.s_ineq_s_ineq_inds); })
        .method("s_ineq_m_ineq_inds", [](Solver& s) { return to_julia(s.s_ineq_m_ineq_inds); })
        .method("s_comp_s_comp_inds", [](Solver& s) { return to_julia(s.s_comp_s_comp_inds); })
        .method("s_comp_m_comp_inds", [](Solver& s) { return to_julia(s.s_comp_m_comp_inds); });
}
