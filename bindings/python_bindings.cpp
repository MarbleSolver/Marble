#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "solver.h"

namespace py = pybind11;

// Helper: copy Eigen::Map<Vec> → Vec so pybind11/eigen can convert to numpy
static Vec copy_map(const Eigen::Map<Vec>& m) { return Vec(m); }

// (rows, cols, colptr, rowval, nzval) copies of a compressed sparse matrix, 0-based indices
static py::tuple sparse_to_tuple(const SMat& m) {
    Eigen::VectorXi cp(m.outerSize() + 1), rv(m.nonZeros());
    std::copy(m.outerIndexPtr(), m.outerIndexPtr() + m.outerSize() + 1, cp.data());
    std::copy(m.innerIndexPtr(), m.innerIndexPtr() + m.nonZeros(),     rv.data());
    Vec nz = Eigen::Map<const Vec>(m.valuePtr(), m.nonZeros());
    return py::make_tuple((int)m.rows(), (int)m.cols(), cp, rv, nz);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Marble compiled core: constrained optimization solver with complementarity constraints";

    // Registered base first: pybind11 tries the most recently registered translator first
    auto& marble_error = py::register_exception<MarbleError>(m, "MarbleError", PyExc_RuntimeError);
    py::register_exception<InvalidInputError>(m, "InvalidInputError",
        py::make_tuple(marble_error, py::handle(PyExc_ValueError)));

    py::enum_<SolveStatus>(m, "SolveStatus")
        .value("Converged",               SolveStatus::Converged)
        .value("MaxIterations",           SolveStatus::MaxIterations,           "Outer or inner iteration limit reached")
        .value("FactorizationFailed",     SolveStatus::FactorizationFailed,     "KKT factorization failed for every regularizer")
        .value("InertiaCorrectionFailed", SolveStatus::InertiaCorrectionFailed, "KKT inertia could not be corrected for any regularizer")
        .value("LinesearchFailed",        SolveStatus::LinesearchFailed,        "Filter linesearch failed for every regularizer");

    py::class_<Problem, std::shared_ptr<Problem>>(m, "Problem")
        // Dense numpy-array constructor (mirrors Julia wrapper)
        .def(py::init([](const Mat& cost_hessian, const Vec& cost_gradient, double cost_const,
                         const Mat& J_eq,   const Vec& c_eq,
                         const Mat& J_ineq, const Vec& c_ineq,
                         const Mat& L,      const Vec& l,
                         const Mat& R,      const Vec& r) {
                 return new Problem(cost_hessian, cost_gradient, cost_const,
                                    J_eq, c_eq, J_ineq, c_ineq, L, l, R, r);
             }),
             py::arg("cost_hessian"), py::arg("cost_gradient"), py::arg("cost_const"),
             py::arg("J_eq"),   py::arg("c_eq"),
             py::arg("J_ineq"), py::arg("c_ineq"),
             py::arg("L"), py::arg("l"),
             py::arg("R"), py::arg("r"))
        // Sparse constructor – accepts Eigen::SparseMatrix which pybind11 converts
        // from scipy.sparse.csc_matrix automatically
        .def(py::init([](const SMat& cost_hessian, const Vec& cost_gradient, double cost_const,
                         const SMat& J_eq,   const Vec& c_eq,
                         const SMat& J_ineq, const Vec& c_ineq,
                         const SMat& L,      const Vec& l,
                         const SMat& R,      const Vec& r) {
                 return new Problem(cost_hessian, cost_gradient, cost_const,
                                    J_eq, c_eq, J_ineq, c_ineq, L, l, R, r);
             }),
             py::arg("cost_hessian"), py::arg("cost_gradient"), py::arg("cost_const"),
             py::arg("J_eq"),   py::arg("c_eq"),
             py::arg("J_ineq"), py::arg("c_ineq"),
             py::arg("L"), py::arg("l"),
             py::arg("R"), py::arg("r"))
        .def_readonly("nz",     &Problem::nz)
        .def_readonly("n_eq",   &Problem::n_eq)
        .def_readonly("n_ineq", &Problem::n_ineq)
        .def_readonly("n_comp", &Problem::n_comp)
        .def("obj",           [](const Problem& p, const Vec& z) { return p.obj(z); }, py::arg("z"))
        .def("residual_eq",   [](const Problem& p, const Vec& z) { return p.residual_eq(z); },   py::arg("z"))
        .def("residual_ineq", [](const Problem& p, const Vec& z) { return p.residual_ineq(z); }, py::arg("z"))
        .def("residual_comp", [](const Problem& p, const Vec& z) { return p.residual_comp(z); }, py::arg("z"))
        .def_property_readonly("cost_gradient", [](const Problem& p) { return p.cost_gradient; })
        .def_readonly("cost_const", &Problem::cost_const)
        // Constraint matrices (returned as scipy-compatible CSC tuple: rows, cols, colptr, rowval, nzval)
        .def_property_readonly("J_eq", [](const Problem& p) { return sparse_to_tuple(p.J_eq); })
        .def_property_readonly("c_eq",   [](const Problem& p) { return p.c_eq; })
        .def_property_readonly("J_ineq", [](const Problem& p) { return sparse_to_tuple(p.J_ineq); })
        .def_property_readonly("c_ineq", [](const Problem& p) { return p.c_ineq; })
        .def_property_readonly("L_comp", [](const Problem& p) { return sparse_to_tuple(p.L_comp); })
        .def_property_readonly("l_comp", [](const Problem& p) { return p.l_comp; })
        .def_property_readonly("R_comp", [](const Problem& p) { return sparse_to_tuple(p.R_comp); })
        .def_property_readonly("r_comp", [](const Problem& p) { return p.r_comp; })
        // Cost hessian (same CSC tuple format)
        .def_property_readonly("cost_hessian", [](const Problem& p) { return sparse_to_tuple(p.cost_hessian); });

    py::class_<SolveResult>(m, "SolveResult")
        .def_readonly("converged",         &SolveResult::converged)
        .def_readonly("status",            &SolveResult::status)
        .def_readonly("iterations",        &SolveResult::iterations)
        .def_readonly("iterations_outer",  &SolveResult::iterations_outer)
        .def_readonly("iterations_inner",  &SolveResult::iterations_inner)
        .def_readonly("factorizations",    &SolveResult::factorizations)
        .def_readonly("setup_time_s",      &SolveResult::setup_time_s)
        .def_readonly("solve_time_s",      &SolveResult::solve_time_s)
        .def_property_readonly("z",        [](const SolveResult& r) { return r.z; })
        .def_property_readonly("s_ineq",   [](const SolveResult& r) { return r.s_ineq; })
        .def_property_readonly("s_comp",   [](const SolveResult& r) { return r.s_comp; })
        .def_property_readonly("m_eq",     [](const SolveResult& r) { return r.m_eq; })
        .def_property_readonly("m_ineq",   [](const SolveResult& r) { return r.m_ineq; })
        .def_property_readonly("m_comp_L", [](const SolveResult& r) { return r.m_comp_L; })
        .def_property_readonly("m_comp_R", [](const SolveResult& r) { return r.m_comp_R; })
        .def("__repr__", [](const SolveResult& r) {
            return "<SolveResult converged=" + std::string(r.converged ? "True" : "False")
                 + " status=" + py::str(py::cast(r.status)).cast<std::string>()
                 + " iterations=" + std::to_string(r.iterations)
                 + " factorizations=" + std::to_string(r.factorizations) + ">";
        });

    py::enum_<RetractionType>(m, "RetractionType")
        .value("Softplus",  RetractionType::Softplus,  "p(s) = 0.5 * (s + sqrt(s^2 + 4*kappa))")
        .value("Exp",       RetractionType::Exp,       "p(s) = sqrt(kappa) * exp(s)")
        .value("ScaledExp", RetractionType::ScaledExp, "p(s) = sqrt(kappa) * exp(s / sqrt(kappa))");

    py::class_<Solver::Options>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("convergence_kkt_norm",       &Solver::Options::convergence_kkt_norm)
        .def_readwrite("convergence_eq_violation",   &Solver::Options::convergence_eq_violation)
        .def_readwrite("convergence_ineq_violation", &Solver::Options::convergence_ineq_violation)
        .def_readwrite("convergence_comp_violation", &Solver::Options::convergence_comp_violation)
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
        .def_readwrite("ruiz_iterations",            &Solver::Options::ruiz_iterations)
        .def_readwrite("inertia_warmstart",          &Solver::Options::inertia_warmstart)
        .def_readwrite("comp_init_random",           &Solver::Options::comp_init_random)
        .def_readwrite("comp_init_seed",             &Solver::Options::comp_init_seed)
        .def_readwrite("verbosity",                  &Solver::Options::verbosity)
        .def_readwrite("retraction_type",            &Solver::Options::retraction_type)
        .def_readwrite("clamp_hessian",              &Solver::Options::clamp_hessian)
        .def_readwrite("warmstart",                  &Solver::Options::warmstart)
        .def("__repr__", [](const Solver::Options& o) {
            return "<SolverOptions convergence_kkt_norm=" + std::to_string(o.convergence_kkt_norm)
                 + " max_iters=" + std::to_string(o.max_iters) + ">";
        });

    py::class_<Filter::Entry>(m, "FilterEntry")
        .def(py::init<>())
        .def_readwrite("feas",  &Filter::Entry::feas)
        .def_readwrite("merit", &Filter::Entry::merit)
        .def("__repr__", [](const Filter::Entry& e) {
            return "<FilterEntry feas=" + std::to_string(e.feas)
                 + " merit=" + std::to_string(e.merit) + ">";
        });

    py::class_<Filter, std::shared_ptr<Filter>>(m, "Filter")
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

    py::class_<Workspace, std::shared_ptr<Workspace>>(m, "Workspace")
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
        .def_property_readonly("m_comp_L",
            [](const Workspace& w) { return copy_map(w.m_comp_L); })
        .def_property_readonly("m_comp_R",
            [](const Workspace& w) { return copy_map(w.m_comp_R); })
        // Multiplier estimates
        .def_property_readonly("m_eq_est",
            [](const Workspace& w) { return w.m_eq_est; })
        .def_property_readonly("m_ineq_est",
            [](const Workspace& w) { return w.m_ineq_est; })
        .def_property_readonly("m_comp_L_est",
            [](const Workspace& w) { return w.m_comp_L_est; })
        .def_property_readonly("m_comp_R_est",
            [](const Workspace& w) { return w.m_comp_R_est; })
        // Residuals
        .def_property_readonly("kkt_residual",
            [](const Workspace& w) { return w.kkt_residual; })
        .def_property_readonly("dkkt_residual_drelax",
            [](const Workspace& w) { return w.dkkt_residual_drelax; })
        .def_property_readonly("residual_eq",
            [](const Workspace& w) { return w.residual_eq; })
        .def_property_readonly("residual_ineq",
            [](const Workspace& w) { return w.residual_ineq; })
        .def_property_readonly("residual_comp_L",
            [](const Workspace& w) { return w.residual_comp_L; })
        .def_property_readonly("residual_comp_R",
            [](const Workspace& w) { return w.residual_comp_R; })
        // Scalar parameters
        .def_property_readonly("relax_param",
            [](const Workspace& w) { return w.relax_param; })
        .def_property_readonly("penalty_param",
            [](const Workspace& w) { return w.penalty_param; })
        // Newton step
        .def_property_readonly("newton_step",
            [](const Workspace& w) { return w.newton_step; })
        // KKT system: returns (rows, cols, colptr, rowval, nzval) as a tuple
        .def_property_readonly("kkt_system",
            [](Workspace& w) {
                SMat& kkt = w.kkt_system;
                kkt.makeCompressed();
                Eigen::VectorXi colptr(kkt.outerSize() + 1);
                Eigen::VectorXi rowval(kkt.nonZeros());
                Vec nzval(kkt.nonZeros());
                std::copy(kkt.outerIndexPtr(), kkt.outerIndexPtr() + kkt.outerSize() + 1, colptr.data());
                std::copy(kkt.innerIndexPtr(), kkt.innerIndexPtr() + kkt.nonZeros(),      rowval.data());
                std::copy(kkt.valuePtr(),      kkt.valuePtr()      + kkt.nonZeros(),      nzval.data());
                return py::make_tuple((int)kkt.rows(), (int)kkt.cols(), colptr, rowval, nzval);
            })
        // QDLDL factorization data
        .def_property_readonly("D",
            [](const Workspace& w) { return w.D; })
        .def_property_readonly("amd_perm_vec",
            [](const Workspace& w) { return w.amd_perm_vec; })
        .def_property_readonly("amd_iperm_vec",
            [](const Workspace& w) { return w.amd_iperm_vec; })
        .def_property_readonly("scaling",
            [](const Workspace& w) { return w.scaling; });

    py::class_<Solver>(m, "Solver")
        .def(py::init<>())
        // Problem setup: takes a Problem (built from dense numpy arrays or scipy
        // CSC matrices via the Problem constructors, which validate dimensions).
        .def("set_problem", [](Solver& s, Problem& problem, Solver::Options& options) {
                 s.set_problem(problem, options);
             },
             py::arg("problem"), py::arg("options"))
        .def("ruiz_equilibration", [](Solver& s, int niter) {
                 s.ruiz_equilibration(niter);
                 return s.get_workspace().scaling;
             },
             py::arg("niter"))
        .def("get_problem", [](Solver& s) {
                 s.get_problem();  // raises if no problem is set
                 return std::const_pointer_cast<Problem>(s.prob);
             })
        .def("get_options", [](Solver& s) -> Solver::Options& { return s.options; },
             py::return_value_policy::reference_internal)
        // Main solve
        .def("solve", [](Solver& s) { return s.solve(); })
        .def("convergence", &Solver::convergence)
        // Workspace / filter access
        .def("get_workspace", [](Solver& s) {
                 s.get_workspace();  // raises if no problem is set
                 return s.workspace;
             })
        .def("get_filter", [](Solver& s) { return s.filter; })
        // Retraction maps
        .def("retract", &Solver::retract,
             py::arg("s"), py::arg("relax_param"))
        .def("retract_deriv", &Solver::retract_deriv,
             py::arg("s"), py::arg("relax_param"))
        .def("retract_second_deriv", &Solver::retract_second_deriv,
             py::arg("s"), py::arg("relax_param"))
        .def("retract_drelax", &Solver::retract_drelax,
             py::arg("s"), py::arg("relax_param"))
        .def("retract_deriv_drelax", &Solver::retract_deriv_drelax,
             py::arg("s"), py::arg("relax_param"))
        // KKT updates
        .def("update_residuals", &Solver::update_residuals)
        .def("update_KKT_residual", &Solver::update_KKT_residual,
             py::arg("relax_param"), py::arg("penalty_param"))
        .def("update_KKT_system", &Solver::update_KKT_system,
             py::arg("relax_param"), py::arg("penalty_param"))
        .def("update_KKT_ineq", [](Solver& s, const Vec& s_ineq, double relax_param) {
                 s.update_KKT_ineq(s_ineq, relax_param);
             },
             py::arg("s_ineq"), py::arg("relax_param"))
        .def("update_KKT_comp", [](Solver& s, const Vec& s_comp, const Vec& m_comp_L, const Vec& m_comp_R, double relax_param) {
                 s.update_KKT_comp(s_comp, m_comp_L, m_comp_R, relax_param);
             },
             py::arg("s_comp"), py::arg("m_comp_L"), py::arg("m_comp_R"), py::arg("relax_param"))
        .def("update_KKT_penalty", &Solver::update_KKT_penalty,
             py::arg("penalty_param"))
        .def("update_KKT_primal_regularizer", &Solver::update_KKT_primal_regularizer,
             py::arg("reg"))
        .def("update_dKKT_residual_drelax", &Solver::update_dKKT_residual_drelax,
             py::arg("relax_param"))
        // Factorization and solve
        .def("initialize_kkt_sparsity", &Solver::initialize_kkt_sparsity)
        .def("compute_amd_ordering",    &Solver::compute_amd_ordering)
        .def("analytical_factorization",&Solver::analytical_factorization)
        .def("numerical_factorization", &Solver::numerical_factorization)
        .def("check_inertia",           &Solver::check_inertia)
        .def("backsolve", [](Solver& s, const Vec& b) {
                 Vec x;
                 s.backsolve(b, x);
                 return x;
             }, py::arg("b"))
        // Linesearch
        .def("filter_linesearch", &Solver::filter_linesearch,
             py::arg("relax_param"), py::arg("penalty_param"), py::arg("max_iters"))
        .def("entry_from_solution", [](const Solver& s, double relax_param, double penalty_param) {
                 Filter::Entry e = s.entry_from_solution(relax_param, penalty_param);
                 return py::make_tuple(e.feas, e.merit);
             },
             py::arg("relax_param"), py::arg("penalty_param"))
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
        .def_property_readonly("m_comp_L_inds",
            [](const Solver& s) { return s.m_comp_L_inds; })
        .def_property_readonly("m_comp_R_inds",
            [](const Solver& s) { return s.m_comp_R_inds; })
        .def_property_readonly("z_z_inds",
            [](const Solver& s) { return s.z_z_inds; })
        .def_property_readonly("s_ineq_s_ineq_inds",
            [](const Solver& s) { return s.s_ineq_s_ineq_inds; })
        .def_property_readonly("s_ineq_m_ineq_inds",
            [](const Solver& s) { return s.s_ineq_m_ineq_inds; })
        .def_property_readonly("s_comp_s_comp_inds",
            [](const Solver& s) { return s.s_comp_s_comp_inds; })
        .def_property_readonly("s_comp_m_comp_L_inds",
            [](const Solver& s) { return s.s_comp_m_comp_L_inds; })
        .def_property_readonly("s_comp_m_comp_R_inds",
            [](const Solver& s) { return s.s_comp_m_comp_R_inds; });
}
