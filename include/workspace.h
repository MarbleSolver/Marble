#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "relaxation_map.h"
#include <cstring>

/**
 * Contains all elements computed during each solver iteration, including current solution estimates, KKT residual terms and hessians, and Newton step
 * 
 * The workspace is initialized in Solver::set_problem() after the problem dimensions are known, and is updated during each iteration of the solver, minimizing allocations,
 * though more work needs to be done to fully leverage this.
 */
class Workspace {
public:
    Vec solution;               /** Stacked solution vector [z; s_ineq; s_comp; m_eq; m_ineq; m_comp] */

    // Views into the solution vector
    Eigen::Map<Vec> z;          /** Primal variables */
    Eigen::Map<Vec> s_ineq;     /** Inequality slacks in the b_kappa domain */
    Eigen::Map<Vec> s_comp;     /** Complementarity slacks in the b_kappa domain */
    Eigen::Map<Vec> m_eq;       /** Equality multipliers */
    Eigen::Map<Vec> m_ineq;     /** Inequality multipliers */
    Eigen::Map<Vec> m_comp_L;   /** Left complementarity multipliers */
    Eigen::Map<Vec> m_comp_R;   /** Right complementarity multipliers */

    // Current multiplier estimates for AL
    Vec m_eq_est;               /** Equality multiplier estimates for AL */
    Vec m_ineq_est;             /** Inequality multiplier estimates for AL */
    Vec m_comp_L_est;           /** Left complementarity multiplier estimates for AL */
    Vec m_comp_R_est;           /** Right complementarity multiplier estimates for AL */

    // Constraint evaluations
    Vec residual_eq;        /** Equality constraint residuals */
    Vec residual_ineq;      /** Inequality constraint residuals */
    Vec residual_comp_L;    /** Left complementarity residuals, L*z + l */
    Vec residual_comp_R;    /** Right complementarity residuals, R*z + r */

    // Relaxed slack (b_kappa) values at the current slacks, cached for assembling
    // the KKT residual and Jacobian.
    RelaxedSlackValues ineq_retract;   /** b_kappa values at s_ineq */
    RelaxedSlackValues comp_retract;   /** b_kappa values at s_comp */

    // Relaxation and penalty parameters
    double relax_param;    /** Relaxation parameter kappa for b_kappa */
    double penalty_param;  /** Penalty parameter for the augmented Lagrangian */

    double newton_step_size; /** Step size for the current Newton step */
    Vec newton_step; /** Newton step for the KKT system (possibly computed with regularization) */
    Vec relax_correction_step; /** Correction step to account for changes in the relaxation parameter across iters, computed from the KKT residual gradient */

    // Empty constructor
    Workspace(const Problem& prob);
};
