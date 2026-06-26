#include "workspace.h"
#include "kkt_system.h"
#include <algorithm>

namespace {

template <typename Indices>
double* slice_ptr(Vec& x, const Indices& inds) {
    return x.data() + (inds.size() ? inds[0] : 0);
}

}  // namespace

Workspace::Workspace(const Problem& prob)
    : Workspace(prob, KKTSystem::KKTIndices(prob)) {}

Workspace::Workspace(const Problem& prob, const KKTSystem::KKTIndices& inds) : solution(inds.n_vars),
    z(slice_ptr(solution, inds.z), inds.z.size()),
    s_ineq(slice_ptr(solution, inds.s_ineq), inds.s_ineq.size()),
    s_comp(slice_ptr(solution, inds.s_comp), inds.s_comp.size()),
    m_eq(slice_ptr(solution, inds.m_eq), inds.m_eq.size()),
    m_ineq(slice_ptr(solution, inds.m_ineq), inds.m_ineq.size()),
    m_comp_L(slice_ptr(solution, inds.m_comp_L), inds.m_comp_L.size()),
    m_comp_R(slice_ptr(solution, inds.m_comp_R), inds.m_comp_R.size()),
    m_eq_est(prob.n_eq),
    m_ineq_est(prob.n_ineq),
    m_comp_L_est(prob.n_comp),
    m_comp_R_est(prob.n_comp),
    residual_eq(prob.n_eq),
    residual_ineq(prob.n_ineq),
    residual_comp_L(prob.n_comp),
    residual_comp_R(prob.n_comp),
    ineq_retract(prob.n_ineq),
    comp_retract(prob.n_comp),
    relax_param(0.0),
    penalty_param(0.0),
    newton_step(solution.size()),
    relax_correction_step(solution.size())
{
    solution.setZero();
    m_eq_est.setZero();
    m_ineq_est.setZero();
    m_comp_L_est.setZero();
    m_comp_R_est.setZero();

    residual_eq   = prob.J_eq * z + prob.b_eq;
    residual_ineq = prob.J_ineq * z + prob.b_ineq;
    residual_comp_L = prob.L * z + prob.l;
    residual_comp_R = prob.R * z + prob.r;

    ineq_retract.setZero();
    comp_retract.setZero();
    newton_step.setZero();
    relax_correction_step.setZero();
}
