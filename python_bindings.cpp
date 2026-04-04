#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "solver.h"

namespace py = pybind11;

// Helper: copy Eigen::Map<Vec> → Vec so pybind11/eigen can convert to numpy
static Vec copy_map(const Eigen::Map<Vec>& m) { return Vec(m); }

PYBIND11_MODULE(rcqp, m) {
    m.doc() = "RCQP: constrained optimization solver with complementarity constraints";

    // -------------------------------------------------------------------------
    // Problem
    // -------------------------------------------------------------------------
    py::class_<Problem>(m, "Problem")
        // Dense numpy-array constructor (mirrors Julia wrapper)
        .def(py::init([](const Mat& cost_hessian, const Vec& cost_gradient, double cost_const,
                         const Mat& J_eq,   const Vec& c_eq,
                         const Mat& J_ineq, const Vec& c_ineq,
                         const Mat& J_comp, const Vec& c_comp) {
                 return new Problem(cost_hessian, cost_gradient, cost_const,
                                    J_eq, c_eq, J_ineq, c_ineq, J_comp, c_comp);
             }),
             py::arg("cost_hessian"), py::arg("cost_gradient"), py::arg("cost_const"),
             py::arg("J_eq"),   py::arg("c_eq"),
             py::arg("J_ineq"), py::arg("c_ineq"),
             py::arg("J_comp"), py::arg("c_comp"))
        // Sparse constructor – accepts Eigen::SparseMatrix which pybind11 converts
        // from scipy.sparse.csc_matrix automatically
        .def(py::init([](const SMat& cost_hessian, const Vec& cost_gradient, double cost_const,
                         const SMat& J_eq,   const Vec& c_eq,
                         const SMat& J_ineq, const Vec& c_ineq,
                         const SMat& J_comp, const Vec& c_comp) {
                 return new Problem(cost_hessian, cost_gradient, cost_const,
                                    J_eq, c_eq, J_ineq, c_ineq, J_comp, c_comp);
             }),
             py::arg("cost_hessian"), py::arg("cost_gradient"), py::arg("cost_const"),
             py::arg("J_eq"),   py::arg("c_eq"),
             py::arg("J_ineq"), py::arg("c_ineq"),
             py::arg("J_comp"), py::arg("c_comp"))
        .def_readonly("nz",     &Problem::nz)
        .def_readonly("n_eq",   &Problem::n_eq)
        .def_readonly("n_ineq", &Problem::n_ineq)
        .def_readonly("n_comp", &Problem::n_comp)
        .def_property_readonly("cost_gradient", [](const Problem& p) { return p.cost_gradient; })
        .def_readonly("cost_const", &Problem::cost_const);

    // -------------------------------------------------------------------------
    // Solver::Options
    // -------------------------------------------------------------------------
    py::class_<Solver::Options>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("convergence_kkt_norm",       &Solver::Options::convergence_kkt_norm)
        .def_readwrite("convergence_eq_violation",   &Solver::Options::convergence_eq_violation)
        .def_readwrite("convergence_ineq_violation", &Solver::Options::convergence_ineq_violation)
        .def_readwrite("convergence_comp_violation", &Solver::Options::convergence_comp_violation)
        .def_readwrite("outer_step_kkt_norm",        &Solver::Options::outer_step_kkt_norm)
        .def_readwrite("penalty_initial",            &Solver::Options::penalty_initial)
        .def_readwrite("penalty_max",                &Solver::Options::penalty_max)
        .def_readwrite("penalty_scaling",            &Solver::Options::penalty_scaling)
        .def_readwrite("relaxation_initial",         &Solver::Options::relaxation_initial)
        .def_readwrite("relaxation_min",             &Solver::Options::relaxation_min)
        .def_readwrite("relaxation_scaling",         &Solver::Options::relaxation_scaling)
        .def_readwrite("max_iters",                  &Solver::Options::max_iters)
        .def_readwrite("max_iters_linesearch",       &Solver::Options::max_iters_linesearch)
        .def_readwrite("gamma_objective",            &Solver::Options::gamma_objective)
        .def_readwrite("gamma_constraint",           &Solver::Options::gamma_constraint)
        .def_property("output_dir",
            [](const Solver::Options& o) { return o.output_dir.string(); },
            [](Solver::Options& o, const std::string& v) { o.output_dir = v; })
        .def("__repr__", [](const Solver::Options& o) {
            return "<SolverOptions convergence_kkt_norm=" + std::to_string(o.convergence_kkt_norm)
                 + " max_iters=" + std::to_string(o.max_iters) + ">";
        });

    // -------------------------------------------------------------------------
    // Filter::Entry
    // -------------------------------------------------------------------------
    py::class_<Filter::Entry>(m, "FilterEntry")
        .def(py::init<>())
        .def_readwrite("feas",  &Filter::Entry::feas)
        .def_readwrite("merit", &Filter::Entry::merit)
        .def("__repr__", [](const Filter::Entry& e) {
            return "<FilterEntry feas=" + std::to_string(e.feas)
                 + " merit=" + std::to_string(e.merit) + ">";
        });

    // -------------------------------------------------------------------------
    // Filter
    // -------------------------------------------------------------------------
    py::class_<Filter>(m, "Filter")
        .def(py::init<>())
        .def(py::init<double, double>(),
             py::arg("gamma_objective"), py::arg("gamma_constraint"))
        .def("clear", &Filter::clear)
        .def_property_readonly("size",
            [](const Filter& f) { return f.entries.size(); })
        .def_property_readonly("entries",
            [](const Filter& f) {
                std::vector<std::pair<double, double>> out;
                out.reserve(f.entries.size());
                for (const auto& e : f.entries)
                    out.emplace_back(e.feas, e.merit);
                return out;
            })
        .def("sufficient_progress", &Filter::sufficient_progress,
             py::arg("candidate"), py::arg("entry"))
        .def("candidate_acceptable", &Filter::candidate_acceptable,
             py::arg("candidate"), py::arg("entry"))
        .def("candidate_dominated", &Filter::candidate_dominated,
             py::arg("candidate"), py::arg("entry"))
        .def("acceptable", &Filter::acceptable, py::arg("candidate"))
        .def("update", &Filter::update, py::arg("new_entry"));

    // -------------------------------------------------------------------------
    // Workspace  (read-only view into solver internals)
    // -------------------------------------------------------------------------
    py::class_<Workspace>(m, "Workspace")
        // Solution components – copy Maps to plain Vec so numpy owns the data
        .def_property_readonly("solution",
            [](const Workspace& w) { return w.solution; })
        .def_property_readonly("z",
            [](const Workspace& w) { return copy_map(w.z); })
        .def_property_readonly("s_ineq",
            [](const Workspace& w) { return copy_map(w.s_ineq); })
        .def_property_readonly("s_comp",
            [](const Workspace& w) { return copy_map(w.s_comp); })
        .def_property_readonly("m_eq",
            [](const Workspace& w) { return copy_map(w.m_eq); })
        .def_property_readonly("m_ineq",
            [](const Workspace& w) { return copy_map(w.m_ineq); })
        .def_property_readonly("m_comp",
            [](const Workspace& w) { return copy_map(w.m_comp); })
        // Multiplier estimates
        .def_property_readonly("m_eq_est",
            [](const Workspace& w) { return w.m_eq_est; })
        .def_property_readonly("m_ineq_est",
            [](const Workspace& w) { return w.m_ineq_est; })
        .def_property_readonly("m_comp_est",
            [](const Workspace& w) { return w.m_comp_est; })
        // Residuals
        .def_property_readonly("kkt_residual",
            [](const Workspace& w) { return w.kkt_residual; })
        .def_property_readonly("residual_eq",
            [](const Workspace& w) { return w.residual_eq; })
        .def_property_readonly("residual_ineq",
            [](const Workspace& w) { return w.residual_ineq; })
        .def_property_readonly("residual_comp",
            [](const Workspace& w) { return w.residual_comp; })
        // Scalar parameters
        .def_property_readonly("relax_param",
            [](const Workspace& w) { return w.relax_param; })
        .def_property_readonly("penalty_param",
            [](const Workspace& w) { return w.penalty_param; })
        // Newton step
        .def_property_readonly("newton_step",
            [](const Workspace& w) { return w.newton_step; });

    // -------------------------------------------------------------------------
    // Solver
    // -------------------------------------------------------------------------
    py::class_<Solver>(m, "Solver")
        .def(py::init<>())
        .def(py::init<const Solver::Options&>(), py::arg("options"))
        // Problem setup
        .def("set_problem", [](Solver& s, Problem& prob, const Vec& scaling) {
                 s.set_problem(prob, scaling);
             },
             py::arg("problem"), py::arg("scaling"))
        .def("get_problem", &Solver::get_problem,
             py::return_value_policy::reference_internal)
        // Main solve
        .def("solve", &Solver::solve, py::arg("options"))
        .def("convergence", &Solver::convergence, py::arg("options"))
        // Workspace / filter access
        .def("get_workspace", &Solver::get_workspace,
             py::return_value_policy::reference_internal)
        .def("get_filter", &Solver::get_filter,
             py::return_value_policy::reference_internal)
        // Retraction maps
        .def("retract", &Solver::retract,
             py::arg("s"), py::arg("sqrt_relax_param"))
        .def("retract_deriv", &Solver::retract_deriv,
             py::arg("s"), py::arg("sqrt_relax_param"))
        .def("retract_second_deriv", &Solver::retract_second_deriv,
             py::arg("s"), py::arg("sqrt_relax_param"))
        // KKT updates
        .def("update_KKT_residual", &Solver::update_KKT_residual,
             py::arg("sqrt_relax_param"), py::arg("inv_penalty_param"))
        .def("update_KKT_system", &Solver::update_KKT_system,
             py::arg("sqrt_relax_param"), py::arg("inv_penalty_param"))
        .def("update_KKT_ineq", [](Solver& s, const Vec& s_ineq, double sqrt_relax_param) {
                 s.update_KKT_ineq(s_ineq, sqrt_relax_param);
             },
             py::arg("s_ineq"), py::arg("sqrt_relax_param"))
        .def("update_KKT_comp", [](Solver& s, const Vec& s_comp, const Vec& m_comp, double sqrt_relax_param) {
                 s.update_KKT_comp(s_comp, m_comp, sqrt_relax_param);
             },
             py::arg("s_comp"), py::arg("m_comp"), py::arg("sqrt_relax_param"))
        .def("update_KKT_penalty", &Solver::update_KKT_penalty,
             py::arg("inv_penalty_param"))
        .def("update_KKT_primal_regularizer", &Solver::update_KKT_primal_regularizer,
             py::arg("reg"))
        // Factorization and solve
        .def("initialize_kkt_sparsity", &Solver::initialize_kkt_sparsity)
        .def("compute_amd_ordering",    &Solver::compute_amd_ordering)
        .def("analytical_factorization",&Solver::analytical_factorization)
        .def("numerical_factorization", &Solver::numerical_factorization)
        .def("check_inertia",           &Solver::check_inertia)
        .def("backsolve",               &Solver::backsolve)
        // Linesearch
        .def("filter_linesearch", &Solver::filter_linesearch,
             py::arg("sqrt_relax_param"), py::arg("inv_penalty_param"), py::arg("max_iters"))
        .def("entry_from_solution", [](const Solver& s, double sqrt_relax_param, double inv_penalty_param) {
                 Filter::Entry e = s.entry_from_solution(sqrt_relax_param, inv_penalty_param);
                 return py::make_tuple(e.feas, e.merit);
             },
             py::arg("sqrt_relax_param"), py::arg("inv_penalty_param"))
        // Dimension info
        .def_readonly("n_primals", &Solver::n_primals)
        .def_readonly("n_duals",   &Solver::n_duals)
        .def_readonly("n_vars",    &Solver::n_vars)
        // KKT index vectors (returned as numpy int arrays)
        .def_property_readonly("z_inds",
            [](const Solver& s) { return s.z_inds; })
        .def_property_readonly("s_ineq_inds",
            [](const Solver& s) { return s.s_ineq_inds; })
        .def_property_readonly("s_comp_inds",
            [](const Solver& s) { return s.s_comp_inds; })
        .def_property_readonly("m_eq_inds",
            [](const Solver& s) { return s.m_eq_inds; })
        .def_property_readonly("m_ineq_inds",
            [](const Solver& s) { return s.m_ineq_inds; })
        .def_property_readonly("m_comp_inds",
            [](const Solver& s) { return s.m_comp_inds; })
        .def_property_readonly("comp_L_inds",
            [](const Solver& s) { return s.comp_L_inds; })
        .def_property_readonly("comp_R_inds",
            [](const Solver& s) { return s.comp_R_inds; });
}
