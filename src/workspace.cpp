#include "workspace.h"
#include <algorithm>

Workspace::Workspace(const Problem& prob) : solution(prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq + prob.n_ineq + 2 * prob.n_comp),
    z(solution.data(), prob.nz),
    s_ineq(solution.data() + prob.nz, prob.n_ineq),
    s_comp(solution.data() + prob.nz + prob.n_ineq, prob.n_comp),
    m_eq(solution.data() + prob.nz + prob.n_ineq + prob.n_comp, prob.n_eq),
    m_ineq(solution.data() + prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq, prob.n_ineq),
    m_comp_L(solution.data() + prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq + prob.n_ineq, prob.n_comp),
    m_comp_R(solution.data() + prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq + prob.n_ineq + prob.n_comp, prob.n_comp),
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
    // Initialize solution to 0
    solution.setZero();
    m_eq_est.setZero();
    m_ineq_est.setZero();
    m_comp_L_est.setZero();
    m_comp_R_est.setZero();

    // Initialize workspace constraint residuals
    residual_eq   = prob.J_eq * z + prob.c_eq;
    residual_ineq = prob.J_ineq * z + prob.c_ineq;
    residual_comp_L = prob.L * z + prob.l;
    residual_comp_R = prob.R * z + prob.r;

    // Initialize steps to 0
    ineq_retract.setZero();
    comp_retract.setZero();
    newton_step.setZero();
    relax_correction_step.setZero();
}
