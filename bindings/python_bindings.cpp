#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

#include "solver.h"

namespace py = pybind11;

static Vec copy_map(const Eigen::Map<Vec>& m) {
    return Vec(m);
}

static Vec combine_vectors(const Vec& a, const Vec& b) {
    Vec out(a.size() + b.size());
    out << a, b;
    return out;
}

static py::tuple smat_to_tuple(SMat m) {
    m.makeCompressed();
    Eigen::VectorXi colptr(m.outerSize() + 1), rowval(m.nonZeros());
    Vec nzval(m.nonZeros());
    std::copy(m.outerIndexPtr(), m.outerIndexPtr() + m.outerSize() + 1, colptr.data());
    std::copy(m.innerIndexPtr(), m.innerIndexPtr() + m.nonZeros(), rowval.data());
    std::copy(m.valuePtr(), m.valuePtr() + m.nonZeros(), nzval.data());
    return py::make_tuple((int)m.rows(), (int)m.cols(), colptr, rowval, nzval);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Marble compiled core";

    py::class_<Problem>(m, "Problem")
        .def(py::init([](const Mat& cost_hessian, const Vec& cost_gradient, double cost_const,
                         const Mat& J_eq, const Vec& c_eq,
                         const Mat& J_ineq, const Vec& c_ineq,
                         const Mat& L, const Vec& l,
                         const Mat& R, const Vec& r) {
            return new Problem(cost_hessian, cost_gradient, cost_const,
                               J_eq, c_eq, J_ineq, c_ineq, L, l, R, r);
        }),
             py::arg("cost_hessian"), py::arg("cost_gradient"), py::arg("cost_const"),
             py::arg("J_eq"), py::arg("c_eq"),
             py::arg("J_ineq"), py::arg("c_ineq"),
             py::arg("L"), py::arg("l"),
             py::arg("R"), py::arg("r"))
        .def(py::init([](const SMat& cost_hessian, const Vec& cost_gradient, double cost_const,
                         const SMat& J_eq, const Vec& c_eq,
                         const SMat& J_ineq, const Vec& c_ineq,
                         const SMat& L, const Vec& l,
                         const SMat& R, const Vec& r) {
            return new Problem(cost_hessian, cost_gradient, cost_const,
                               J_eq, c_eq, J_ineq, c_ineq, L, l, R, r);
        }),
             py::arg("cost_hessian"), py::arg("cost_gradient"), py::arg("cost_const"),
             py::arg("J_eq"), py::arg("c_eq"),
             py::arg("J_ineq"), py::arg("c_ineq"),
             py::arg("L"), py::arg("l"),
             py::arg("R"), py::arg("r"))
        .def_readonly("nz", &Problem::nz)
        .def_readonly("n_eq", &Problem::n_eq)
        .def_readonly("n_ineq", &Problem::n_ineq)
        .def_readonly("n_comp", &Problem::n_comp)
        .def_property_readonly("cost_hessian", [](const Problem& p) { return smat_to_tuple(p.cost_hessian); })
        .def_property_readonly("cost_gradient", [](const Problem& p) { return p.cost_gradient; })
        .def_readonly("cost_const", &Problem::cost_const)
        .def_property_readonly("J_eq", [](const Problem& p) { return smat_to_tuple(p.J_eq); })
        .def_property_readonly("c_eq", [](const Problem& p) { return p.c_eq; })
        .def_property_readonly("J_ineq", [](const Problem& p) { return smat_to_tuple(p.J_ineq); })
        .def_property_readonly("c_ineq", [](const Problem& p) { return p.c_ineq; })
        .def_property_readonly("L", [](const Problem& p) { return smat_to_tuple(p.L); })
        .def_property_readonly("l", [](const Problem& p) { return p.l; })
        .def_property_readonly("R", [](const Problem& p) { return smat_to_tuple(p.R); })
        .def_property_readonly("r", [](const Problem& p) { return p.r; })
        .def("obj", &Problem::obj, py::arg("z"))
        .def("residual_eq", &Problem::residual_eq, py::arg("z"))
        .def("residual_ineq", &Problem::residual_ineq, py::arg("z"))
        .def("residual_comp_L", &Problem::residual_comp_L, py::arg("z"))
        .def("residual_comp_R", &Problem::residual_comp_R, py::arg("z"))
        .def("residual_comp", &Problem::residual_comp, py::arg("z"));

    py::class_<Solver::Options>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("convergence_kkt_norm", &Solver::Options::convergence_kkt_norm)
        .def_readwrite("convergence_eq_violation", &Solver::Options::convergence_eq_violation)
        .def_readwrite("convergence_ineq_violation", &Solver::Options::convergence_ineq_violation)
        .def_readwrite("convergence_comp_violation", &Solver::Options::convergence_comp_violation)
        .def_readwrite("outer_step_kkt_norm", &Solver::Options::outer_step_kkt_norm)
        .def_readwrite("penalty_initial", &Solver::Options::penalty_initial)
        .def_readwrite("penalty_max", &Solver::Options::penalty_max)
        .def_readwrite("penalty_scaling", &Solver::Options::penalty_scaling)
        .def_readwrite("relax_initial", &Solver::Options::relax_initial)
        .def_readwrite("relax_min", &Solver::Options::relax_min)
        .def_readwrite("relax_scaling", &Solver::Options::relax_scaling)
        .def_readwrite("use_relax_correction", &Solver::Options::use_relax_correction)
        .def_readwrite("max_iters", &Solver::Options::max_iters)
        .def_readwrite("max_iters_linesearch", &Solver::Options::max_iters_linesearch)
        .def_readwrite("gamma_objective", &Solver::Options::gamma_objective)
        .def_readwrite("gamma_constraint", &Solver::Options::gamma_constraint)
        .def_readwrite("ruiz_iters", &Solver::Options::ruiz_iters)
        .def_readwrite("verbosity", &Solver::Options::verbosity)
        .def("__repr__", [](const Solver::Options& o) {
            return "<SolverOptions max_iters=" + std::to_string(o.max_iters) +
                   " convergence_kkt_norm=" + std::to_string(o.convergence_kkt_norm) + ">";
        });

    py::class_<Filter::Entry>(m, "FilterEntry")
        .def(py::init<>())
        .def(py::init([](double feas, double merit) {
            Filter::Entry entry;
            entry.feas = feas;
            entry.merit = merit;
            return entry;
        }), py::arg("feas"), py::arg("merit"))
        .def_readwrite("feas", &Filter::Entry::feas)
        .def_readwrite("merit", &Filter::Entry::merit);

    py::class_<Filter>(m, "Filter")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("gamma_objective"), py::arg("gamma_constraint"))
        .def("clear", &Filter::clear)
        .def_property_readonly("num_entries", [](const Filter& f) { return f.entries.size(); })
        .def_property_readonly("entries", [](const Filter& f) {
            Vec out(2 * static_cast<Eigen::Index>(f.entries.size()));
            Eigen::Index i = 0;
            for (const auto& e : f.entries) {
                out[i++] = e.feas;
                out[i++] = e.merit;
            }
            return out;
        })
        .def("sufficient_progress", &Filter::sufficient_progress,
             py::arg("candidate"), py::arg("entry"))
        .def("sufficient_feas_progress", [](const Filter& f, const Filter::Entry& c,
                                            const Filter::Entry& e) {
            return f.sufficient_progress(c, e).first;
        }, py::arg("candidate"), py::arg("entry"))
        .def("sufficient_merit_progress", [](const Filter& f, const Filter::Entry& c,
                                             const Filter::Entry& e) {
            return f.sufficient_progress(c, e).second;
        }, py::arg("candidate"), py::arg("entry"))
        .def("candidate_acceptable", &Filter::candidate_acceptable,
             py::arg("candidate"), py::arg("entry"))
        .def("candidate_dominated", &Filter::candidate_dominated,
             py::arg("candidate"), py::arg("entry"))
        .def("acceptable", &Filter::acceptable, py::arg("candidate"))
        .def("update", &Filter::update, py::arg("new_entry"));

    py::class_<RelaxationMap>(m, "RelaxationMap")
        .def("b", &RelaxationMap::b, py::arg("x"), py::arg("kappa"))
        .def("b_prime", &RelaxationMap::b_prime, py::arg("x"), py::arg("kappa"))
        .def("b_double_prime", &RelaxationMap::b_double_prime, py::arg("x"), py::arg("kappa"))
        .def("d_b_d_kappa", &RelaxationMap::d_b_d_kappa, py::arg("x"), py::arg("kappa"))
        .def("d_b_prime_d_kappa", &RelaxationMap::d_b_prime_d_kappa, py::arg("x"), py::arg("kappa"))
        .def("d_b_double_prime_d_kappa", &RelaxationMap::d_b_double_prime_d_kappa,
             py::arg("x"), py::arg("kappa"));

    py::class_<Workspace>(m, "Workspace")
        .def_property_readonly("solution", [](const Workspace& w) { return w.solution; })
        .def_property_readonly("z", [](const Workspace& w) { return copy_map(w.z); })
        .def_property_readonly("s_ineq", [](const Workspace& w) { return copy_map(w.s_ineq); })
        .def_property_readonly("s_comp", [](const Workspace& w) { return copy_map(w.s_comp); })
        .def_property_readonly("m_eq", [](const Workspace& w) { return copy_map(w.m_eq); })
        .def_property_readonly("m_ineq", [](const Workspace& w) { return copy_map(w.m_ineq); })
        .def_property_readonly("m_comp_L", [](const Workspace& w) { return copy_map(w.m_comp_L); })
        .def_property_readonly("m_comp_R", [](const Workspace& w) { return copy_map(w.m_comp_R); })
        .def_property_readonly("m_eq_est", [](const Workspace& w) { return w.m_eq_est; })
        .def_property_readonly("m_ineq_est", [](const Workspace& w) { return w.m_ineq_est; })
        .def_property_readonly("m_comp_L_est", [](const Workspace& w) { return w.m_comp_L_est; })
        .def_property_readonly("m_comp_R_est", [](const Workspace& w) { return w.m_comp_R_est; })
        .def_property_readonly("residual_eq", [](const Workspace& w) { return w.residual_eq; })
        .def_property_readonly("residual_ineq", [](const Workspace& w) { return w.residual_ineq; })
        .def_property_readonly("residual_comp_L", [](const Workspace& w) { return w.residual_comp_L; })
        .def_property_readonly("residual_comp_R", [](const Workspace& w) { return w.residual_comp_R; })
        .def_property_readonly("relax_param", [](const Workspace& w) { return w.relax_param; })
        .def_property_readonly("penalty_param", [](const Workspace& w) { return w.penalty_param; })
        .def_property_readonly("newton_step_size", [](const Workspace& w) { return w.newton_step_size; })
        .def_property_readonly("newton_step", [](const Workspace& w) { return w.newton_step; })
        .def_property_readonly("relax_correction_step", [](const Workspace& w) {
            return w.relax_correction_step;
        });

    py::class_<LdltSystem>(m, "LdltSystem")
        .def(py::init<int>(), py::arg("n"))
        .def_property_readonly("built", &LdltSystem::built)
        .def_property_readonly("matrix", [](const LdltSystem& s) { return smat_to_tuple(s.matrix()); })
        .def_property_readonly("perm", [](const LdltSystem& s) { return s.perm(); })
        .def_property_readonly("iperm", [](const LdltSystem& s) { return s.iperm(); })
        .def_property_readonly("scaling", [](const LdltSystem& s) { return s.scaling(); })
        .def("factorize", [](LdltSystem& s) { return s.factorize(); })
        .def("check_inertia", &LdltSystem::check_inertia,
             py::arg("n_pos"), py::arg("n_neg"), py::arg("atol") = 1e-10)
        .def("solve", [](LdltSystem& s, const Vec& rhs) {
            Vec x = Vec::Zero(rhs.size());
            s.solve(x, rhs);
            return x;
        }, py::arg("rhs"))
        .def("problem_to_internal", [](const LdltSystem& s, const Vec& x_problem) {
            Vec out = Vec::Zero(x_problem.size());
            s.problem_to_internal(x_problem, out);
            return out;
        }, py::arg("x_problem"))
        .def("internal_to_problem", [](const LdltSystem& s, const Vec& x_internal) {
            Vec out = Vec::Zero(x_internal.size());
            s.internal_to_problem(x_internal, out);
            return out;
        }, py::arg("x_internal"));

    py::class_<KKTSystem>(m, "KKTSystem")
        .def_property_readonly("n_primals", [](const KKTSystem& k) { return k.n_primals; })
        .def_property_readonly("n_duals", [](const KKTSystem& k) { return k.n_duals; })
        .def_property_readonly("n_vars", [](const KKTSystem& k) { return k.n_vars; })
        .def_property_readonly("primal_regularizer", [](const KKTSystem& k) { return k.primal_regularizer; })
        .def_property_readonly("residual", [](const KKTSystem& k) { return k.residual; })
        .def_property_readonly("grad_residual_relax_param", [](const KKTSystem& k) {
            return k.grad_residual_relax_param;
        })
        .def_property_readonly("ldlt", [](KKTSystem& k) -> LdltSystem& { return k.kkt; },
             py::return_value_policy::reference_internal)
        .def_property_readonly("matrix", [](const KKTSystem& k) { return smat_to_tuple(k.kkt.matrix()); })
        .def_property_readonly("perm", [](const KKTSystem& k) { return k.kkt.perm(); })
        .def_property_readonly("iperm", [](const KKTSystem& k) { return k.kkt.iperm(); })
        .def_property_readonly("scaling", [](const KKTSystem& k) { return k.kkt.scaling(); })
        .def("ruiz_equilibration", &KKTSystem::ruiz_equilibration, py::arg("niter"))
        .def("update_residual", &KKTSystem::update_residual)
        .def("update_kkt_system", &KKTSystem::update_kkt_system)
        .def("update_primal_regularizer", &KKTSystem::update_primal_regularizer,
             py::arg("regularizer"))
        .def("update_residual_relax_grad", &KKTSystem::update_residual_relax_grad)
        .def("numerical_factorization", &KKTSystem::numerical_factorization)
        .def("check_inertia", &KKTSystem::check_inertia)
        .def("compute_step", [](KKTSystem& k, const Vec& rhs) {
            Vec step = Vec::Zero(rhs.size());
            k.compute_step(step, rhs);
            return step;
        }, py::arg("rhs"));

    py::class_<Solver::Result>(m, "SolveResult")
        .def_readonly("converged", &Solver::Result::converged)
        .def_readonly("iters", &Solver::Result::iters)
        .def_readonly("iters_outer", &Solver::Result::iters_outer)
        .def_readonly("iters_inner", &Solver::Result::iters_inner)
        .def_readonly("factorizations", &Solver::Result::factorizations)
        .def_readonly("factorizations_ldlt", &Solver::Result::factorizations_ldlt)
        .def_readonly("factorizations_inertia", &Solver::Result::factorizations_inertia)
        .def_readonly("factorizations_linesearch", &Solver::Result::factorizations_linesearch)
        .def_readonly("setup_time_s", &Solver::Result::setup_time_s)
        .def_readonly("solve_time_s", &Solver::Result::solve_time_s)
        .def_property_readonly("z", [](const Solver::Result& r) { return r.z; })
        .def_property_readonly("s_ineq", [](const Solver::Result& r) { return r.s_ineq; })
        .def_property_readonly("s_comp", [](const Solver::Result& r) { return r.s_comp; })
        .def_property_readonly("m_eq", [](const Solver::Result& r) { return r.m_eq; })
        .def_property_readonly("m_ineq", [](const Solver::Result& r) { return r.m_ineq; })
        .def_property_readonly("m_comp_L", [](const Solver::Result& r) { return r.m_comp_L; })
        .def_property_readonly("m_comp_R", [](const Solver::Result& r) { return r.m_comp_R; })
        .def_property_readonly("m_comp", [](const Solver::Result& r) {
            return combine_vectors(r.m_comp_L, r.m_comp_R);
        })
        .def("__repr__", [](const Solver::Result& r) {
            return "<SolveResult converged=" + std::string(r.converged ? "True" : "False") +
                   " iters=" + std::to_string(r.iters) +
                   " factorizations=" + std::to_string(r.factorizations) + ">";
        });

    py::class_<Solver>(m, "Solver")
        .def(py::init<>())
        .def("set_problem", [](Solver& s, Problem& problem, Solver::Options& options) {
            s.set_problem(problem, options);
        }, py::arg("problem"), py::arg("options"))
        .def("solve", [](Solver& s) { return s.solve(); })
        .def("convergence", &Solver::convergence)
        .def("get_problem", &Solver::get_problem, py::return_value_policy::reference_internal)
        .def("get_workspace", &Solver::get_workspace, py::return_value_policy::reference_internal)
        .def("get_filter", &Solver::get_filter, py::return_value_policy::reference_internal)
        .def("get_kkt_system", [](Solver& s) -> KKTSystem& {
            s.require_problem_set("get_kkt_system");
            return *s.kkt_system;
        }, py::return_value_policy::reference_internal)
        .def("get_relaxation_map", &Solver::get_relaxation_map,
             py::return_value_policy::reference_internal)
        .def("update_relaxed_slack_values", &Solver::update_relaxed_slack_values)
        .def("update_primal_residuals", &Solver::update_primal_residuals)
        .def("apply_newton_step", &Solver::apply_newton_step, py::arg("step_size"))
        .def("filter_linesearch", &Solver::filter_linesearch)
        .def("relax_correction_linesearch", &Solver::relax_correction_linesearch,
             py::arg("relax_param_new"))
        .def("entry_from_solution", [](const Solver& s) {
            Filter::Entry entry = s.entry_from_solution();
            return py::make_tuple(entry.feas, entry.merit);
        })
        .def_property_readonly("n_primals", [](const Solver& s) {
            return s.kkt_system ? s.kkt_system->n_primals : 0;
        })
        .def_property_readonly("n_duals", [](const Solver& s) {
            return s.kkt_system ? s.kkt_system->n_duals : 0;
        })
        .def_property_readonly("n_vars", [](const Solver& s) {
            return s.kkt_system ? s.kkt_system->n_vars : 0;
        })
        .def("get_options", [](Solver& s) -> Solver::Options& { return s.options; },
             py::return_value_policy::reference_internal)
        .def_property_readonly("options", [](Solver& s) -> Solver::Options& { return s.options; },
             py::return_value_policy::reference_internal);
}
