#include <jlcxx/jlcxx.hpp>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

#include "solver.h"

namespace jlcxx {
    template<> struct IsMirroredType<Filter::Entry> : std::false_type {};
    template<> struct IsMirroredType<RelaxationMap> : std::false_type {};
    template<> struct IsMirroredType<Solver::Result> : std::false_type {};
}

template <typename F>
auto julia_call(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::invalid_argument& e) {
        jl_error(e.what());
    } catch (const std::exception& e) {
        jl_error(e.what());
    } catch (const std::string& e) {
        jl_error(e.c_str());
    } catch (const char* e) {
        jl_error(e);
    } catch (...) {
        jl_error("Unknown C++ exception in Marble");
    }
    throw std::runtime_error("unreachable after jl_error");
}

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

template <typename T>
jlcxx::Array<T> make_julia_owned(Eigen::Matrix<T, Eigen::Dynamic, 1> vec) {
    jlcxx::Array<T> arr(vec.size());
#if JULIA_VERSION_MAJOR > 1 || (JULIA_VERSION_MAJOR == 1 && JULIA_VERSION_MINOR >= 11)
    T* dest_ptr = jl_array_data(arr.wrapped(), T);
#else
    T* dest_ptr = reinterpret_cast<T*>(jl_array_data(arr.wrapped()));
#endif
    std::copy(vec.data(), vec.data() + vec.size(), dest_ptr);
    return arr;
}

static Vec combine_vectors(const Vec& a, const Vec& b) {
    Vec out(a.size() + b.size());
    out << a, b;
    return out;
}

static SMat csc_to_smat(int rows, int cols,
                        jlcxx::ArrayRef<int64_t, 1> colptr,
                        jlcxx::ArrayRef<int64_t, 1> rowval,
                        jlcxx::ArrayRef<double, 1> nzval) {
    const int nnz = static_cast<int>(nzval.size());
    std::vector<int> cp(cols + 1), ri(nnz);
    for (int i = 0; i <= cols; ++i) cp[i] = static_cast<int>(colptr[i] - 1);
    for (int i = 0; i < nnz; ++i) ri[i] = static_cast<int>(rowval[i] - 1);

    Eigen::Map<SMat> mapped(rows, cols, nnz, cp.data(), ri.data(),
                            const_cast<double*>(nzval.data()));
    SMat result = mapped;
    result.makeCompressed();
    return result;
}

static auto smat_to_julia_tuple(SMat m) {
    m.makeCompressed();
    Eigen::Matrix<int64_t, Eigen::Dynamic, 1> colptr(m.outerSize() + 1);
    Eigen::Matrix<int64_t, Eigen::Dynamic, 1> rowval(m.nonZeros());
    Vec nzval(m.nonZeros());

    for (int i = 0; i <= m.outerSize(); ++i)
        colptr[i] = static_cast<int64_t>(m.outerIndexPtr()[i]) + 1;
    for (int i = 0; i < m.nonZeros(); ++i)
        rowval[i] = static_cast<int64_t>(m.innerIndexPtr()[i]) + 1;
    std::copy(m.valuePtr(), m.valuePtr() + m.nonZeros(), nzval.data());

    return std::make_tuple((int)m.rows(), (int)m.cols(),
                           make_julia_owned<int64_t>(std::move(colptr)),
                           make_julia_owned<int64_t>(std::move(rowval)),
                           make_julia_owned<double>(std::move(nzval)));
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<Problem>("Problem")
        .constructor([](jlcxx::ArrayRef<double, 2> cost_hessian,
                        jlcxx::ArrayRef<double, 1> cost_gradient,
                        double cost_const,
                        jlcxx::ArrayRef<double, 2> J_eq, jlcxx::ArrayRef<double, 1> b_eq,
                        jlcxx::ArrayRef<double, 2> J_ineq, jlcxx::ArrayRef<double, 1> b_ineq,
                        jlcxx::ArrayRef<double, 2> L, jlcxx::ArrayRef<double, 1> l,
                        jlcxx::ArrayRef<double, 2> R, jlcxx::ArrayRef<double, 1> r) {
            return julia_call([&]() {
                return new Problem(
                    to_eigen(cost_hessian), to_eigen(cost_gradient), cost_const,
                    to_eigen(J_eq), to_eigen(b_eq),
                    to_eigen(J_ineq), to_eigen(b_ineq),
                    to_eigen(L), to_eigen(l),
                    to_eigen(R), to_eigen(r));
            });
        })
        .constructor([](int nz,
                        jlcxx::ArrayRef<int64_t, 1> Hcp, jlcxx::ArrayRef<int64_t, 1> Hrv,
                        jlcxx::ArrayRef<double, 1> Hnz,
                        jlcxx::ArrayRef<double, 1> grad, double cost_const,
                        int n_eq,
                        jlcxx::ArrayRef<int64_t, 1> Ecp, jlcxx::ArrayRef<int64_t, 1> Erv,
                        jlcxx::ArrayRef<double, 1> Enz,
                        jlcxx::ArrayRef<double, 1> b_eq,
                        int n_ineq,
                        jlcxx::ArrayRef<int64_t, 1> Icp, jlcxx::ArrayRef<int64_t, 1> Irv,
                        jlcxx::ArrayRef<double, 1> Inz,
                        jlcxx::ArrayRef<double, 1> b_ineq,
                        int n_comp,
                        jlcxx::ArrayRef<int64_t, 1> Lcp, jlcxx::ArrayRef<int64_t, 1> Lrv,
                        jlcxx::ArrayRef<double, 1> Lnz,
                        jlcxx::ArrayRef<double, 1> l,
                        jlcxx::ArrayRef<int64_t, 1> Rcp, jlcxx::ArrayRef<int64_t, 1> Rrv,
                        jlcxx::ArrayRef<double, 1> Rnz,
                        jlcxx::ArrayRef<double, 1> r) {
            return julia_call([&]() {
                return new Problem(
                    csc_to_smat(nz, nz, Hcp, Hrv, Hnz), to_eigen(grad), cost_const,
                    csc_to_smat(n_eq, nz, Ecp, Erv, Enz), to_eigen(b_eq),
                    csc_to_smat(n_ineq, nz, Icp, Irv, Inz), to_eigen(b_ineq),
                    csc_to_smat(n_comp, nz, Lcp, Lrv, Lnz), to_eigen(l),
                    csc_to_smat(n_comp, nz, Rcp, Rrv, Rnz), to_eigen(r));
            });
        })
        .method("nz", [](const Problem& p) { return p.nz; })
        .method("n_eq", [](const Problem& p) { return p.n_eq; })
        .method("n_ineq", [](const Problem& p) { return p.n_ineq; })
        .method("n_comp", [](const Problem& p) { return p.n_comp; })
        .method("cost_hessian", [](const Problem& p) { return smat_to_julia_tuple(p.cost_hessian); })
        .method("cost_gradient", [](const Problem& p) { return make_julia_owned<double>(p.cost_gradient); })
        .method("cost_const", [](const Problem& p) { return p.cost_const; })
        .method("J_eq", [](const Problem& p) { return smat_to_julia_tuple(p.J_eq); })
        .method("b_eq", [](const Problem& p) { return make_julia_owned<double>(p.b_eq); })
        .method("J_ineq", [](const Problem& p) { return smat_to_julia_tuple(p.J_ineq); })
        .method("b_ineq", [](const Problem& p) { return make_julia_owned<double>(p.b_ineq); })
        .method("L", [](const Problem& p) { return smat_to_julia_tuple(p.L); })
        .method("l", [](const Problem& p) { return make_julia_owned<double>(p.l); })
        .method("R", [](const Problem& p) { return smat_to_julia_tuple(p.R); })
        .method("r", [](const Problem& p) { return make_julia_owned<double>(p.r); })
        .method("obj", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) {
            return p.obj(to_eigen(z));
        })
        .method("residual_eq", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) {
            return make_julia_owned<double>(p.residual_eq(to_eigen(z)));
        })
        .method("residual_ineq", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) {
            return make_julia_owned<double>(p.residual_ineq(to_eigen(z)));
        })
        .method("residual_comp_L", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) {
            return make_julia_owned<double>(p.residual_comp_L(to_eigen(z)));
        })
        .method("residual_comp_R", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) {
            return make_julia_owned<double>(p.residual_comp_R(to_eigen(z)));
        })
        .method("residual_comp", [](const Problem& p, jlcxx::ArrayRef<double, 1> z) {
            return make_julia_owned<double>(p.residual_comp(to_eigen(z)));
        });

    #define OPTION_RW(name, T) \
        .method(#name, [](const Solver::Options& o) { return o.name; }) \
        .method(#name "!", [](Solver::Options& o, T v) { o.name = v; })

    mod.add_type<Solver::Options>("SolverOptions")
        .constructor()
        OPTION_RW(convergence_kkt_norm, double)
        OPTION_RW(convergence_eq_violation, double)
        OPTION_RW(convergence_ineq_violation, double)
        OPTION_RW(convergence_comp_violation, double)
        OPTION_RW(outer_step_kkt_norm, double)
        OPTION_RW(penalty_initial, double)
        OPTION_RW(penalty_max, double)
        OPTION_RW(penalty_scaling, double)
        OPTION_RW(relax_initial, double)
        OPTION_RW(relax_min, double)
        OPTION_RW(relax_scaling, double)
        OPTION_RW(use_relax_correction, bool)
        OPTION_RW(max_iters, int)
        OPTION_RW(max_iters_linesearch, int)
        OPTION_RW(gamma_objective, double)
        OPTION_RW(gamma_constraint, double)
        OPTION_RW(ruiz_iters, int)
        OPTION_RW(verbosity, int);

    #undef OPTION_RW

    mod.add_type<Filter::Entry>("FilterEntry")
        .constructor()
        .constructor([](double feas, double merit) {
            auto* entry = new Filter::Entry();
            entry->feas = feas;
            entry->merit = merit;
            return entry;
        })
        .method("feas", [](const Filter::Entry& e) { return e.feas; })
        .method("feas!", [](Filter::Entry& e, double v) { e.feas = v; })
        .method("merit", [](const Filter::Entry& e) { return e.merit; })
        .method("merit!", [](Filter::Entry& e, double v) { e.merit = v; });

    mod.add_type<Filter>("Filter")
        .constructor()
        .constructor([](double gamma_objective, double gamma_constraint) {
            return new Filter(gamma_objective, gamma_constraint);
        })
        .method("clear", &Filter::clear)
        .method("num_entries", [](const Filter& f) { return f.entries.size(); })
        .method("entries", [](const Filter& f) {
            Vec out(2 * static_cast<Eigen::Index>(f.entries.size()));
            Eigen::Index i = 0;
            for (const auto& e : f.entries) {
                out[i++] = e.feas;
                out[i++] = e.merit;
            }
            return make_julia_owned<double>(std::move(out));
        })
        .method("sufficient_feas_progress", [](Filter& f, const Filter::Entry& c,
                                               const Filter::Entry& e) {
            return f.sufficient_progress(c, e).first;
        })
        .method("sufficient_merit_progress", [](Filter& f, const Filter::Entry& c,
                                                const Filter::Entry& e) {
            return f.sufficient_progress(c, e).second;
        })
        .method("candidate_acceptable", &Filter::candidate_acceptable)
        .method("candidate_dominated", &Filter::candidate_dominated)
        .method("acceptable", &Filter::acceptable)
        .method("update", &Filter::update);

    mod.add_type<RelaxationMap>("RelaxationMap")
        .method("b", [](const RelaxationMap& m, jlcxx::ArrayRef<double, 1> x, double kappa) {
            return make_julia_owned<double>(m.b(to_eigen(x), kappa));
        })
        .method("b_prime", [](const RelaxationMap& m, jlcxx::ArrayRef<double, 1> x, double kappa) {
            return make_julia_owned<double>(m.b_prime(to_eigen(x), kappa));
        })
        .method("b_double_prime", [](const RelaxationMap& m, jlcxx::ArrayRef<double, 1> x, double kappa) {
            return make_julia_owned<double>(m.b_double_prime(to_eigen(x), kappa));
        })
        .method("d_b_d_kappa", [](const RelaxationMap& m, jlcxx::ArrayRef<double, 1> x, double kappa) {
            return make_julia_owned<double>(m.d_b_d_kappa(to_eigen(x), kappa));
        })
        .method("d_b_prime_d_kappa", [](const RelaxationMap& m, jlcxx::ArrayRef<double, 1> x, double kappa) {
            return make_julia_owned<double>(m.d_b_prime_d_kappa(to_eigen(x), kappa));
        })
        .method("d_b_double_prime_d_kappa", [](const RelaxationMap& m, jlcxx::ArrayRef<double, 1> x, double kappa) {
            return make_julia_owned<double>(m.d_b_double_prime_d_kappa(to_eigen(x), kappa));
        });

    mod.add_type<Workspace>("Workspace")
        .method("solution", [](const Workspace& w) { return make_julia_owned<double>(w.solution); })
        .method("z", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.z)); })
        .method("s_ineq", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.s_ineq)); })
        .method("s_comp", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.s_comp)); })
        .method("m_eq", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.m_eq)); })
        .method("m_ineq", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.m_ineq)); })
        .method("m_comp_L", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.m_comp_L)); })
        .method("m_comp_R", [](const Workspace& w) { return make_julia_owned<double>(Vec(w.m_comp_R)); })
        .method("m_eq_est", [](const Workspace& w) { return make_julia_owned<double>(w.m_eq_est); })
        .method("m_ineq_est", [](const Workspace& w) { return make_julia_owned<double>(w.m_ineq_est); })
        .method("m_comp_L_est", [](const Workspace& w) { return make_julia_owned<double>(w.m_comp_L_est); })
        .method("m_comp_R_est", [](const Workspace& w) { return make_julia_owned<double>(w.m_comp_R_est); })
        .method("residual_eq", [](const Workspace& w) { return make_julia_owned<double>(w.residual_eq); })
        .method("residual_ineq", [](const Workspace& w) { return make_julia_owned<double>(w.residual_ineq); })
        .method("residual_comp_L", [](const Workspace& w) { return make_julia_owned<double>(w.residual_comp_L); })
        .method("residual_comp_R", [](const Workspace& w) { return make_julia_owned<double>(w.residual_comp_R); })
        .method("relax_param", [](const Workspace& w) { return w.relax_param; })
        .method("penalty_param", [](const Workspace& w) { return w.penalty_param; })
        .method("newton_step_size", [](const Workspace& w) { return w.newton_step_size; })
        .method("newton_step", [](const Workspace& w) { return make_julia_owned<double>(w.newton_step); })
        .method("relax_correction_step", [](const Workspace& w) {
            return make_julia_owned<double>(w.relax_correction_step);
        });

    mod.add_type<LdltSystem>("LdltSystem")
        .constructor([](int n) { return new LdltSystem(n); })
        .method("built", &LdltSystem::built)
        .method("matrix", [](const LdltSystem& s) { return smat_to_julia_tuple(s.matrix()); })
        .method("perm", [](const LdltSystem& s) { return make_julia_owned<int>(s.perm()); })
        .method("iperm", [](const LdltSystem& s) { return make_julia_owned<int>(s.iperm()); })
        .method("scaling", [](const LdltSystem& s) { return make_julia_owned<double>(s.scaling()); })
        .method("factorize!", [](LdltSystem& s) { return s.factorize(); })
        .method("check_inertia", &LdltSystem::check_inertia)
        .method("solve", [](LdltSystem& s, jlcxx::ArrayRef<double, 1> rhs) {
            Vec b = to_eigen(rhs);
            Vec x = Vec::Zero(b.size());
            s.solve(x, b);
            return make_julia_owned<double>(std::move(x));
        })
        .method("problem_to_internal", [](const LdltSystem& s, jlcxx::ArrayRef<double, 1> x_problem) {
            Vec in = to_eigen(x_problem);
            Vec out = Vec::Zero(in.size());
            s.problem_to_internal(in, out);
            return make_julia_owned<double>(std::move(out));
        })
        .method("internal_to_problem", [](const LdltSystem& s, jlcxx::ArrayRef<double, 1> x_internal) {
            Vec in = to_eigen(x_internal);
            Vec out = Vec::Zero(in.size());
            s.internal_to_problem(in, out);
            return make_julia_owned<double>(std::move(out));
        });

    mod.add_type<KKTSystem>("KKTSystem")
        .method("n_primals", [](const KKTSystem& k) { return k.n_primals; })
        .method("n_duals", [](const KKTSystem& k) { return k.n_duals; })
        .method("n_vars", [](const KKTSystem& k) { return k.n_vars; })
        .method("primal_regularizer", [](const KKTSystem& k) { return k.primal_regularizer; })
        .method("residual", [](const KKTSystem& k) { return make_julia_owned<double>(k.residual); })
        .method("grad_residual_relax_param", [](const KKTSystem& k) {
            return make_julia_owned<double>(k.grad_residual_relax_param);
        })
        .method("ldlt", [](KKTSystem& k) -> LdltSystem& { return k.kkt; })
        .method("matrix", [](const KKTSystem& k) { return smat_to_julia_tuple(k.kkt.matrix()); })
        .method("perm", [](const KKTSystem& k) { return make_julia_owned<int>(k.kkt.perm()); })
        .method("iperm", [](const KKTSystem& k) { return make_julia_owned<int>(k.kkt.iperm()); })
        .method("scaling", [](const KKTSystem& k) { return make_julia_owned<double>(k.kkt.scaling()); })
        .method("ruiz_equilibration", [](const KKTSystem& k, int niter) {
            return make_julia_owned<double>(k.ruiz_equilibration(niter));
        })
        .method("update_residual!", &KKTSystem::update_residual)
        .method("update_kkt_system!", &KKTSystem::update_kkt_system)
        .method("update_primal_regularizer!", &KKTSystem::update_primal_regularizer)
        .method("update_residual_relax_grad!", &KKTSystem::update_residual_relax_grad)
        .method("numerical_factorization!", &KKTSystem::numerical_factorization)
        .method("check_inertia", &KKTSystem::check_inertia)
        .method("compute_step", [](KKTSystem& k, jlcxx::ArrayRef<double, 1> rhs) {
            Vec b = to_eigen(rhs);
            Vec step = Vec::Zero(b.size());
            k.compute_step(step, b);
            return make_julia_owned<double>(std::move(step));
        });

    mod.add_type<Solver::Result>("SolveResult")
        .method("converged", [](const Solver::Result& r) { return r.converged; })
        .method("iters", [](const Solver::Result& r) { return r.iters; })
        .method("iters_outer", [](const Solver::Result& r) { return r.iters_outer; })
        .method("iters_inner", [](const Solver::Result& r) { return r.iters_inner; })
        .method("factorizations", [](const Solver::Result& r) { return r.factorizations; })
        .method("factorizations_ldlt", [](const Solver::Result& r) { return r.factorizations_ldlt; })
        .method("factorizations_inertia", [](const Solver::Result& r) { return r.factorizations_inertia; })
        .method("factorizations_linesearch", [](const Solver::Result& r) { return r.factorizations_linesearch; })
        .method("setup_time_s", [](const Solver::Result& r) { return r.setup_time_s; })
        .method("solve_time_s", [](const Solver::Result& r) { return r.solve_time_s; })
        .method("z", [](const Solver::Result& r) { return make_julia_owned<double>(r.z); })
        .method("s_ineq", [](const Solver::Result& r) { return make_julia_owned<double>(r.s_ineq); })
        .method("s_comp", [](const Solver::Result& r) { return make_julia_owned<double>(r.s_comp); })
        .method("m_eq", [](const Solver::Result& r) { return make_julia_owned<double>(r.m_eq); })
        .method("m_ineq", [](const Solver::Result& r) { return make_julia_owned<double>(r.m_ineq); })
        .method("m_comp_L", [](const Solver::Result& r) { return make_julia_owned<double>(r.m_comp_L); })
        .method("m_comp_R", [](const Solver::Result& r) { return make_julia_owned<double>(r.m_comp_R); })
        .method("m_comp", [](const Solver::Result& r) {
            return make_julia_owned<double>(combine_vectors(r.m_comp_L, r.m_comp_R));
        });

    mod.add_type<Solver>("Solver")
        .constructor()
        .method("set_problem!", [](Solver& s, Problem& problem, Solver::Options& opts) {
            s.set_problem(problem, opts);
        })
        .method("solve!", [](Solver& s) {
            return julia_call([&]() {
                return s.solve();
            });
        })
        .method("convergence", &Solver::convergence)
        .method("get_problem", &Solver::get_problem)
        .method("get_workspace", &Solver::get_workspace)
        .method("get_filter", &Solver::get_filter)
        .method("get_kkt_system", [](Solver& s) -> KKTSystem& {
            return julia_call([&]() -> KKTSystem& {
                s.require_problem_set("get_kkt_system");
                return *s.kkt_system;
            });
        })
        .method("get_relaxation_map", &Solver::get_relaxation_map)
        .method("update_relaxed_slack_values!", &Solver::update_relaxed_slack_values)
        .method("update_primal_residuals!", &Solver::update_primal_residuals)
        .method("apply_newton_step!", &Solver::apply_newton_step)
        .method("filter_linesearch!", &Solver::filter_linesearch)
        .method("entry_from_solution", [](const Solver& s) {
            Filter::Entry entry = s.entry_from_solution();
            return std::make_tuple(entry.feas, entry.merit);
        })
        .method("n_primals", [](const Solver& s) { return s.kkt_system ? s.kkt_system->n_primals : 0; })
        .method("n_duals", [](const Solver& s) { return s.kkt_system ? s.kkt_system->n_duals : 0; })
        .method("n_vars", [](const Solver& s) { return s.kkt_system ? s.kkt_system->n_vars : 0; })
        .method("options", [](Solver& s) -> Solver::Options& { return s.options; });
}
