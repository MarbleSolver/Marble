#include <jlcxx/jlcxx.hpp>
#include "solver.h"
#include <cmath>
#include <string>

// Plain structs: tell CxxWrap not to treat them as mirrored types
// so we can register them with add_type and attach methods.
namespace jlcxx {
    template<> struct IsMirroredType<Filter::Entry> : std::false_type {};
    template<> struct IsMirroredType<SolveResult>  : std::false_type {};
}

// CxxWrap turns a C++ exception into a Julia error only if it can be caught as
// std::exception; anything else aborts the Julia process. Inside Julia, std::invalid_argument,
// std::out_of_range and the other std types below resolve to the RTTI of Julia's bundled
// libstdc++, so they don't match std::exception. Catching them by exact type still works, so
// they (and anything unknown) are rethrown as std::runtime_error.
[[noreturn]] static void rethrow_for_julia() {
    try { throw; }
    catch (const std::invalid_argument& e) { throw std::runtime_error(e.what()); }
    catch (const std::domain_error& e)     { throw std::runtime_error(e.what()); }
    catch (const std::length_error& e)     { throw std::runtime_error(e.what()); }
    catch (const std::out_of_range& e)     { throw std::runtime_error(e.what()); }
    catch (const std::range_error& e)      { throw std::runtime_error(e.what()); }
    catch (const std::overflow_error& e)   { throw std::runtime_error(e.what()); }
    catch (const std::underflow_error& e)  { throw std::runtime_error(e.what()); }
    catch (const std::exception&)          { throw; }
    catch (...) { throw std::runtime_error("Marble: unknown C++ exception"); }
}

// Wrap a lambda or member function so every exception leaving it is one CxxWrap can catch.
// The result has a concrete (non-template) call operator, as CxxWrap needs to deduce types.
template <typename F, typename R, typename... Args>
static auto guard_call(F f, R (F::*)(Args...) const) {
    return [f](Args... args) -> R {
        try { return f(std::forward<Args>(args)...); }
        catch (...) { rethrow_for_julia(); }
    };
}

template <typename F>
static auto guard(F f) { return guard_call(std::move(f), &F::operator()); }

template <typename R, typename C, typename... Args>
static auto guard(R (C::*f)(Args...)) {
    return guard([f](C& obj, Args... args) -> R { return (obj.*f)(std::forward<Args>(args)...); });
}

template <typename R, typename C, typename... Args>
static auto guard(R (C::*f)(Args...) const) {
    return guard([f](const C& obj, Args... args) -> R { return (obj.*f)(std::forward<Args>(args)...); });
}

// Registers every constructor and method of a wrapped type through guard()
template <typename T>
struct GuardedType {
    jlcxx::TypeWrapper<T> type;

    GuardedType& constructor() { type.template constructor<>(); return *this; }

    template <typename F>
    GuardedType& constructor(F f) { type.constructor(guard(std::move(f))); return *this; }

    template <typename F>
    GuardedType& method(const std::string& name, F f) { type.method(name, guard(std::move(f))); return *this; }
};

template <typename T>
static GuardedType<T> guarded(jlcxx::TypeWrapper<T> type) { return {type}; }

// Map a Julia matrix to an Eigen matrix using its *true* dimensions (Julia is
// column-major, matching Eigen's default). Reading the real shape — rather than
// reshaping to caller-supplied dimensions — lets a misshapen block reach the
// Problem constructor, which then validates and throws instead of silently
// truncating the data.
template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
to_eigen(jlcxx::ArrayRef<T, 2>& arr) {
    const long rows = jl_array_dim(arr.wrapped(), 0);
    const long cols = jl_array_dim(arr.wrapped(), 1);
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

// Views into solver-owned memory (the workspace): valid until the solver is freed or set
// up with a new problem
template <typename T>
jlcxx::ArrayRef<T, 1> to_julia(Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
    return jlcxx::make_julia_array(vec.data(), vec.size());
}

template <typename T>
jlcxx::ArrayRef<T, 1> to_julia(Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>& vec) {
    return jlcxx::make_julia_array(vec.data(), vec.size());
}

// Julia-owned copies, for data whose C++ owner may be freed while Julia still holds the array
template <typename T>
jlcxx::Array<T> make_julia_owned(const T* data, std::size_t n) {
    jlcxx::Array<T> arr(n);

#if JULIA_VERSION_MAJOR > 1 || (JULIA_VERSION_MAJOR == 1 && JULIA_VERSION_MINOR >= 11)
    // Julia 1.11 and newer: macro takes 2 arguments
    T* dest_ptr = jl_array_data(arr.wrapped(), T);
#else
    // Julia 1.10 LTS and older: macro takes 1 argument and returns void*
    T* dest_ptr = reinterpret_cast<T*>(jl_array_data(arr.wrapped()));
#endif

    std::copy(data, data + n, dest_ptr);
    return arr;
}

template <typename T>
jlcxx::Array<T> make_julia_owned(const Eigen::Matrix<T, Eigen::Dynamic, 1>& vec) {
    return make_julia_owned(vec.data(), static_cast<std::size_t>(vec.size()));
}

// (rows, cols, colptr, rowval, nzval) copies of a compressed sparse matrix, 0-based indices
static auto sparse_to_julia(const SMat& m) {
    return std::make_tuple((int)m.rows(), (int)m.cols(),
                           make_julia_owned(m.outerIndexPtr(), m.outerSize() + 1),
                           make_julia_owned(m.innerIndexPtr(), m.nonZeros()),
                           make_julia_owned(m.valuePtr(), m.nonZeros()));
}

// Build an Eigen sparse matrix from Julia SparseMatrixCSC components (1-based Int64 indices).
// The column count comes from colptr, so a block of the wrong width reaches the Problem
// constructor's validation; malformed CSC arrays are rejected before anything is indexed.
static SMat csc_to_smat(int rows,
                        jlcxx::ArrayRef<int64_t, 1> colptr,
                        jlcxx::ArrayRef<int64_t, 1> rowval,
                        jlcxx::ArrayRef<double, 1>  nzval) {
    auto malformed = [](const std::string& why) {
        return InvalidInputError("Marble problem: malformed sparse matrix (" + why + ")");
    };
    if (rows < 0) throw malformed("negative row count");
    if (colptr.size() == 0) throw malformed("empty colptr");
    const std::size_t cols = colptr.size() - 1;
    const std::size_t nnz = nzval.size();
    if (rowval.size() != nnz) throw malformed("rowval and nzval lengths differ");
    if (colptr[0] != 1 || colptr[cols] != (int64_t)nnz + 1) throw malformed("colptr does not span nzval");

    std::vector<QDLDL_int> cp(cols + 1), ri(nnz);
    for (std::size_t j = 0; j <= cols; j++) {
        cp[j] = (QDLDL_int)(colptr[j] - 1);
        if (j > 0 && cp[j] < cp[j - 1]) throw malformed("colptr is not nondecreasing");
    }
    for (std::size_t k = 0; k < nnz; k++) {
        if (rowval[k] < 1 || rowval[k] > rows) throw malformed("row index out of range");
        ri[k] = (QDLDL_int)(rowval[k] - 1);
    }
    Eigen::Map<SMat> mapped(rows, (Eigen::Index)cols, (Eigen::Index)nnz,
                            cp.data(), ri.data(),
                            const_cast<double*>(nzval.data()));
    SMat result = mapped;
    result.makeCompressed();
    return result;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {

    mod.add_bits<RetractionType>("RetractionType", jlcxx::julia_type("CppEnum"));
    mod.set_const("Softplus",  RetractionType::Softplus);
    mod.set_const("Exp",       RetractionType::Exp);
    mod.set_const("ScaledExp", RetractionType::ScaledExp);

    mod.add_bits<SolveStatus>("SolveStatus", jlcxx::julia_type("CppEnum"));
    mod.set_const("Converged",               SolveStatus::Converged);
    mod.set_const("MaxIterations",           SolveStatus::MaxIterations);
    mod.set_const("FactorizationFailed",     SolveStatus::FactorizationFailed);
    mod.set_const("InertiaCorrectionFailed", SolveStatus::InertiaCorrectionFailed);
    mod.set_const("LinesearchFailed",        SolveStatus::LinesearchFailed);

    guarded(mod.add_type<Problem>("Problem"))
        .constructor([](jlcxx::ArrayRef<double, 2> cost_hessian,
                        jlcxx::ArrayRef<double, 1> cost_gradient,
                        double cost_const,
                        jlcxx::ArrayRef<double, 2> J_eq,   jlcxx::ArrayRef<double, 1> c_eq,
                        jlcxx::ArrayRef<double, 2> J_ineq, jlcxx::ArrayRef<double, 1> c_ineq,
                        jlcxx::ArrayRef<double, 2> L,      jlcxx::ArrayRef<double, 1> l,
                        jlcxx::ArrayRef<double, 2> R,      jlcxx::ArrayRef<double, 1> r) {
            return new Problem(
                to_eigen(cost_hessian), to_eigen(cost_gradient), cost_const,
                to_eigen(J_eq),   to_eigen(c_eq),
                to_eigen(J_ineq), to_eigen(c_ineq),
                to_eigen(L),      to_eigen(l),
                to_eigen(R),      to_eigen(r));
        })
        // Sparse constructor: accepts SparseMatrixCSC components from Julia.
        // For each sparse matrix pass (colptr, rowval, nzval) from the Julia struct; its column
        // count is taken from colptr. Julia's colptr/rowval are Int64 (1-based); conversion to
        // 0-based happens internally.
        // Usage from Julia:
        //   Problem(size(H, 1),
        //           H.colptr, H.rowval, H.nzval, q, cost_const,
        //           n_eq,   J_eq.colptr,   J_eq.rowval,   J_eq.nzval,   c_eq,
        //           n_ineq, J_ineq.colptr, J_ineq.rowval, J_ineq.nzval, c_ineq,
        //           n_comp, L.colptr, L.rowval, L.nzval, l,
        //                   R.colptr, R.rowval, R.nzval, r)
        .constructor([](int n_rows_Q,
                        jlcxx::ArrayRef<int64_t,1> Hcp,  jlcxx::ArrayRef<int64_t,1> Hrv,  jlcxx::ArrayRef<double,1> Hnz,
                        jlcxx::ArrayRef<double,1> grad, double cost_const,
                        int n_eq,
                        jlcxx::ArrayRef<int64_t,1> Ecp,  jlcxx::ArrayRef<int64_t,1> Erv,  jlcxx::ArrayRef<double,1> Enz,
                        jlcxx::ArrayRef<double,1> c_eq,
                        int n_ineq,
                        jlcxx::ArrayRef<int64_t,1> Icp,  jlcxx::ArrayRef<int64_t,1> Irv,  jlcxx::ArrayRef<double,1> Inz,
                        jlcxx::ArrayRef<double,1> c_ineq,
                        int n_comp,
                        jlcxx::ArrayRef<int64_t,1> Lcp,  jlcxx::ArrayRef<int64_t,1> Lrv,  jlcxx::ArrayRef<double,1> Lnz,
                        jlcxx::ArrayRef<double,1> l,
                        jlcxx::ArrayRef<int64_t,1> Rcp,  jlcxx::ArrayRef<int64_t,1> Rrv,  jlcxx::ArrayRef<double,1> Rnz,
                        jlcxx::ArrayRef<double,1> r) {
            return new Problem(
                csc_to_smat(n_rows_Q, Hcp, Hrv, Hnz), to_eigen(grad), cost_const,
                csc_to_smat(n_eq,     Ecp, Erv, Enz), to_eigen(c_eq),
                csc_to_smat(n_ineq,   Icp, Irv, Inz), to_eigen(c_ineq),
                csc_to_smat(n_comp,   Lcp, Lrv, Lnz), to_eigen(l),
                csc_to_smat(n_comp,   Rcp, Rrv, Rnz), to_eigen(r));
        })
        .method("nz",     [](const Problem& p) { return p.nz; })
        .method("n_eq",   [](const Problem& p) { return p.n_eq; })
        .method("n_ineq", [](const Problem& p) { return p.n_ineq; })
        .method("n_comp", [](const Problem& p) { return p.n_comp; })
        .method("obj",    [](const Problem& p, jlcxx::ArrayRef<double, 1> z) { return p.obj(to_eigen(z)); })
        .method("residual_eq",   [](const Problem& p, jlcxx::ArrayRef<double, 1> z) { return make_julia_owned(p.residual_eq(to_eigen(z))); })
        .method("residual_ineq", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) { return make_julia_owned(p.residual_ineq(to_eigen(z))); })
        .method("residual_comp", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) { return make_julia_owned(p.residual_comp(to_eigen(z))); })
        // Copies, not views: a Problem can be freed (or replaced by a new setup) while Julia
        // still holds the arrays
        .method("cost_hessian",  [](const Problem& p) { return sparse_to_julia(p.cost_hessian); })
        .method("cost_gradient", [](const Problem& p) { return make_julia_owned(p.cost_gradient); })
        .method("cost_const",    [](const Problem& p) { return p.cost_const; })
        .method("J_eq",          [](const Problem& p) { return sparse_to_julia(p.J_eq); })
        .method("c_eq",          [](const Problem& p) { return make_julia_owned(p.c_eq); })
        .method("J_ineq",        [](const Problem& p) { return sparse_to_julia(p.J_ineq); })
        .method("c_ineq",        [](const Problem& p) { return make_julia_owned(p.c_ineq); })
        .method("L_comp",        [](const Problem& p) { return sparse_to_julia(p.L_comp); })
        .method("l_comp",        [](const Problem& p) { return make_julia_owned(p.l_comp); })
        .method("R_comp",        [](const Problem& p) { return sparse_to_julia(p.R_comp); })
        .method("r_comp",        [](const Problem& p) { return make_julia_owned(p.r_comp); });

    #define OPTION_RW(name, T) \
        .method(#name,       [](const Solver::Options& o)    { return o.name; }) \
        .method(#name "!",   [](Solver::Options& o, T v)     { o.name = v; })

    guarded(mod.add_type<Solver::Options>("SolverOptions"))
        .constructor()
        OPTION_RW(convergence_kkt_norm,       double)
        OPTION_RW(convergence_eq_violation,   double)
        OPTION_RW(convergence_ineq_violation, double)
        OPTION_RW(convergence_comp_violation, double)
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
        OPTION_RW(inertia_warmstart,          bool)
        OPTION_RW(comp_init_random,           bool)
        OPTION_RW(comp_init_seed,             int)
        OPTION_RW(verbosity,                  int)
        OPTION_RW(retraction_type,            RetractionType)
        OPTION_RW(clamp_hessian,              bool)
        OPTION_RW(warmstart,                  bool);

    #undef OPTION_RW

    guarded(mod.add_type<Filter::Entry>("FilterEntry"))
        .constructor()
        .constructor([](double feas, double merit) {
            Filter::Entry* e = new Filter::Entry(); e->feas = feas; e->merit = merit; return e;
        })
        .method("feas",   [](const Filter::Entry& e) { return e.feas; })
        .method("feas!",  [](Filter::Entry& e, double v) { e.feas = v; })
        .method("merit",  [](const Filter::Entry& e) { return e.merit; })
        .method("merit!", [](Filter::Entry& e, double v) { e.merit = v; });

    guarded(mod.add_type<Filter>("Filter"))
        .constructor()
        .constructor([](double gamma_objective, double gamma_constraint) {
            return new Filter(gamma_objective, gamma_constraint);
        })
        .method("clear",   &Filter::clear)
        .method("num_entries", [](const Filter& f) { return f.entries.size(); })
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

    guarded(mod.add_type<Workspace>("Workspace"))
        // Full solution and decomposed views
        .method("solution", [](Workspace& w) { return to_julia(w.solution); })
        .method("z",        [](Workspace& w) { return to_julia(w.z); })
        .method("s_ineq",   [](Workspace& w) { return to_julia(w.s_ineq); })
        .method("s_comp",   [](Workspace& w) { return to_julia(w.s_comp); })
        .method("m_eq",     [](Workspace& w) { return to_julia(w.m_eq); })
        .method("m_ineq",   [](Workspace& w) { return to_julia(w.m_ineq); })
        .method("m_comp_L", [](Workspace& w) { return to_julia(w.m_comp_L); })
        .method("m_comp_R", [](Workspace& w) { return to_julia(w.m_comp_R); })
        // Multiplier estimates
        .method("m_eq_est",   [](Workspace& w) { return to_julia(w.m_eq_est); })
        .method("m_ineq_est", [](Workspace& w) { return to_julia(w.m_ineq_est); })
        .method("m_comp_L_est", [](Workspace& w) { return to_julia(w.m_comp_L_est); })
        .method("m_comp_R_est", [](Workspace& w) { return to_julia(w.m_comp_R_est); })
        // Residuals
        .method("kkt_residual",  [](Workspace& w) { return to_julia(w.kkt_residual); })
        .method("residual_eq",   [](Workspace& w) { return to_julia(w.residual_eq); })
        .method("residual_ineq", [](Workspace& w) { return to_julia(w.residual_ineq); })
        .method("residual_comp_L", [](Workspace& w) { return to_julia(w.residual_comp_L); })
        .method("residual_comp_R", [](Workspace& w) { return to_julia(w.residual_comp_R); })
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
        .method("dkkt_residual_drelax", [](Workspace& w) { return to_julia(w.dkkt_residual_drelax); })
        // QDLDL factorization data
        .method("D",            [](Workspace& w) { return to_julia(w.D); })
        .method("amd_perm_vec", [](Workspace& w) { return to_julia(w.amd_perm_vec); })
        .method("amd_iperm_vec", [](Workspace& w) { return to_julia(w.amd_iperm_vec); })
        .method("scaling", [](Workspace& w) { return to_julia(w.scaling); });

    // Vectors are copied out: a SolveResult returned by value is freed by its Julia
    // finalizer, so a view into it would dangle once the result is collected
    guarded(mod.add_type<SolveResult>("SolveResult"))
        .method("converged",        [](const SolveResult& r) { return r.converged; })
        .method("status",           [](const SolveResult& r) { return r.status; })
        .method("iterations",       [](const SolveResult& r) { return r.iterations; })
        .method("iterations_outer", [](const SolveResult& r) { return r.iterations_outer; })
        .method("iterations_inner", [](const SolveResult& r) { return r.iterations_inner; })
        .method("factorizations",   [](const SolveResult& r) { return r.factorizations; })
        .method("z",            [](const SolveResult& r) { return make_julia_owned(r.z); })
        .method("s_ineq",       [](const SolveResult& r) { return make_julia_owned(r.s_ineq); })
        .method("s_comp",       [](const SolveResult& r) { return make_julia_owned(r.s_comp); })
        .method("m_eq",         [](const SolveResult& r) { return make_julia_owned(r.m_eq); })
        .method("m_ineq",       [](const SolveResult& r) { return make_julia_owned(r.m_ineq); })
        .method("m_comp_L",     [](const SolveResult& r) { return make_julia_owned(r.m_comp_L); })
        .method("m_comp_R",     [](const SolveResult& r) { return make_julia_owned(r.m_comp_R); })
        .method("setup_time_s", [](const SolveResult& r) { return r.setup_time_s; })
        .method("solve_time_s", [](const SolveResult& r) { return r.solve_time_s; });

    guarded(mod.add_type<Solver>("Solver"))
        .constructor()
        // Problem setup
        .method("ruiz_equilibration!", [](Solver& solver, int niter) {
            solver.ruiz_equilibration(niter);
            Workspace& workspace = solver.get_workspace();
            return std::vector<double>(workspace.scaling.data(), workspace.scaling.data() + workspace.scaling.size());
        })
        // Problem setup: takes a Problem (built from dense or sparse data via the
        // Problem constructors, which validate dimensions).
        .method("set_problem!", [](Solver& s, Problem& problem, Solver::Options& opts) {
            s.set_problem(problem, opts);
        })
        .method("get_problem",   &Solver::get_problem)
        // Main solve
        .method("solve!", [](Solver& s) {
            return s.solve();
        })
        .method("convergence", &Solver::convergence)
        // Workspace / filter access
        .method("get_workspace", &Solver::get_workspace)
        .method("get_filter",    &Solver::get_filter)
        // Retraction maps
        .method("retract", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double relax_param) {
            Vec r = s.retract(to_eigen(v), relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        .method("retract_deriv", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double relax_param) {
            Vec r = s.retract_deriv(to_eigen(v), relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        .method("retract_second_deriv", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double relax_param) {
            Vec r = s.retract_second_deriv(to_eigen(v), relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        .method("retract_drelax", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double relax_param) {
            Vec r = s.retract_drelax(to_eigen(v), relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        .method("retract_deriv_drelax", [](Solver& s, jlcxx::ArrayRef<double, 1> v, double relax_param) {
            Vec r = s.retract_deriv_drelax(to_eigen(v), relax_param);
            return std::vector<double>(r.data(), r.data() + r.size());
        })
        // KKT updates
        .method("update_residuals!",             &Solver::update_residuals)
        .method("update_KKT_residual!",          &Solver::update_KKT_residual)
        .method("update_KKT_system!",            &Solver::update_KKT_system)
        .method("update_KKT_ineq!", [](Solver& s, jlcxx::ArrayRef<double, 1> s_ineq, double relax_param) {
            s.update_KKT_ineq(to_eigen(s_ineq), relax_param);
        })
        .method("update_KKT_comp!", [](Solver& s, jlcxx::ArrayRef<double, 1> s_comp, jlcxx::ArrayRef<double, 1> m_comp_L, jlcxx::ArrayRef<double, 1> m_comp_R, double relax_param) {
            s.update_KKT_comp(to_eigen(s_comp), to_eigen(m_comp_L), to_eigen(m_comp_R), relax_param);
        })
        .method("update_KKT_penalty!",            &Solver::update_KKT_penalty)
        .method("update_KKT_primal_regularizer!", &Solver::update_KKT_primal_regularizer)
        .method("update_dKKT_residual_drelax!", &Solver::update_dKKT_residual_drelax)
        // Factorization and backsolve
        .method("initialize_kkt_sparsity!",  &Solver::initialize_kkt_sparsity)
        .method("compute_amd_ordering!",     &Solver::compute_amd_ordering)
        .method("analytical_factorization!", &Solver::analytical_factorization)
        .method("numerical_factorization!",  &Solver::numerical_factorization)
        .method("check_inertia",            &Solver::check_inertia)
        .method("backsolve!", [](Solver& s, jlcxx::ArrayRef<double, 1> b, jlcxx::ArrayRef<double, 1> x) {
            Vec xv;
            s.backsolve(to_eigen(b), xv);
            if ((long)x.size() != xv.size())
                throw InvalidInputError("backsolve!: x has length " + std::to_string(x.size()) +
                                        ", expected " + std::to_string(xv.size()));
            for (int i = 0; i < xv.size(); i++) x[i] = xv[i];
        })
        // Linesearch
        .method("filter_linesearch!", &Solver::filter_linesearch)
        .method("entry_from_solution", [](const Solver& s, double relax_param, double penalty_param) {
            Filter::Entry e = s.entry_from_solution(relax_param, penalty_param);
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
        .method("m_comp_L_inds",      [](Solver& s) { return to_julia(s.m_comp_L_inds); })
        .method("m_comp_R_inds",      [](Solver& s) { return to_julia(s.m_comp_R_inds); })
        .method("z_z_inds",           [](Solver& s) { return to_julia(s.z_z_inds); })
        .method("s_ineq_s_ineq_inds", [](Solver& s) { return to_julia(s.s_ineq_s_ineq_inds); })
        .method("s_ineq_m_ineq_inds", [](Solver& s) { return to_julia(s.s_ineq_m_ineq_inds); })
        .method("s_comp_s_comp_inds", [](Solver& s) { return to_julia(s.s_comp_s_comp_inds); })
        .method("s_comp_m_comp_L_inds", [](Solver& s) { return to_julia(s.s_comp_m_comp_L_inds); })
        .method("s_comp_m_comp_R_inds", [](Solver& s) { return to_julia(s.s_comp_m_comp_R_inds); })
        .method("options", [](Solver& s) -> Solver::Options& { return s.options; });
}
