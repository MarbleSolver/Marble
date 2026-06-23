#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "problem.h"
#include "relaxation_map.h"
#include <cstring>

/**
 * Mutable solver state allocated from a problem's dimensions
 */
class Workspace {
public:
    Vec solution; // Stacked vector [z; s_ineq; s_comp; m_eq; m_ineq; m_comp_L; m_comp_R]

    Eigen::Map<Vec> z;        // Primal variables
    Eigen::Map<Vec> s_ineq;   // Inequality slacks
    Eigen::Map<Vec> s_comp;   // Complementarity slacks
    Eigen::Map<Vec> m_eq;     // Equality multipliers
    Eigen::Map<Vec> m_ineq;   // Inequality multipliers
    Eigen::Map<Vec> m_comp_L; // Left complementarity multipliers
    Eigen::Map<Vec> m_comp_R; // Right complementarity multipliers

    Vec m_eq_est;     // Equality multiplier estimate
    Vec m_ineq_est;   // Inequality multiplier estimate
    Vec m_comp_L_est; // Left complementarity multiplier estimate
    Vec m_comp_R_est; // Right complementarity multiplier estimate

    Vec residual_eq;     // Equality constraint residuals
    Vec residual_ineq;   // Inequality constraint residuals
    Vec residual_comp_L; // Left complementarity residuals
    Vec residual_comp_R; // Right complementarity residuals

    RelaxedSlackValues ineq_retract; // Relaxation map values at s_ineq
    RelaxedSlackValues comp_retract; // Relaxation map values at s_comp

    double relax_param;   // Relaxation parameter
    double penalty_param; // Augmented Lagrangian penalty parameter

    double newton_step_size;    // Step length currently applied to newton_step
    Vec newton_step;            // Current Newton direction
    Vec relax_correction_step;  // First-order correction for relaxation changes

    /**
     * Allocate workspace vectors and views for a problem
     *
     * @param prob Problem whose dimensions determine the workspace layout
     */
    Workspace(const Problem& prob);
};
