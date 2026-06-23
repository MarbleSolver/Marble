#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>

#include "problem.h"
#include "workspace.h"
#include "ldlt_system.h"

class MarbleKKTSystem {
    private:
        struct KktVariableIndices {
            const Eigen::VectorXi z;
            const Eigen::VectorXi v;
            const Eigen::VectorXi sigma;
            const Eigen::VectorXi m_eq;
            const Eigen::VectorXi m_ineq;
            const Eigen::VectorXi m_comp_L;
            const Eigen::VectorXi m_comp_R;

            /**
             * Build logical variable index ranges from problem dimensions
             *
             * @param prob Problem whose dimensions define the KKT layout
             */
            explicit KktVariableIndices(const std::shared_ptr<const Problem>& prob);
        };

        struct ResidualEquationIndices {
            const Eigen::VectorXi& z_stat;
            const Eigen::VectorXi& v_stat;
            const Eigen::VectorXi& sigma_stat;
            const Eigen::VectorXi& pd_eq;
            const Eigen::VectorXi& pd_ineq;
            const Eigen::VectorXi& pd_comp_L;
            const Eigen::VectorXi& pd_comp_R;
        };

    public:
        const int n_primals, n_duals, n_vars;

        // The problem that this KKT system corresponds to
        std::shared_ptr<const Problem> prob;

        // Workspace for shared values
        std::shared_ptr<const Workspace> workspace;

        double primal_regularizer = 0.0; // Current regularization value for inertia correction

        struct KktBlocks {
            LdltSystem::Block* hessian             = nullptr;  // (z, z)
            LdltSystem::Block* J_eq_T              = nullptr;  // (z, m_eq)
            LdltSystem::Block* J_ineq_T            = nullptr;  // (z, m_ineq)
            LdltSystem::Block* L_T                 = nullptr;  // (z, m_comp_L)
            LdltSystem::Block* R_T                 = nullptr;  // (z, m_comp_R)

            LdltSystem::Block* v_stat_v            = nullptr;  // (v, v)
            LdltSystem::Block* v_stat_m_ineq       = nullptr;  // (v, m_ineq)
            LdltSystem::Block* sigma_stat_sigma    = nullptr;  // (sigma, sigma)
            LdltSystem::Block* sigma_stat_m_comp_L = nullptr;  // (sigma, m_comp_L)
            LdltSystem::Block* sigma_stat_m_comp_R = nullptr;  // (sigma, m_comp_R)

            LdltSystem::Block* pd_eq               = nullptr;  // (m_eq, m_eq)
            LdltSystem::Block* pd_ineq             = nullptr;  // (m_ineq, m_ineq)
            LdltSystem::Block* pd_comp_L           = nullptr;  // (m_comp_L, m_comp_L)
            LdltSystem::Block* pd_comp_R           = nullptr;  // (m_comp_R, m_comp_R)

            LdltSystem::Block* primal_reg          = nullptr;  // diag(z, v, sigma)
        };

        KktVariableIndices variable_inds;
        ResidualEquationIndices residual_inds;
        KktBlocks blocks;

        /**
         * Create a KKT system from problem dimensions and workspace storage
         *
         * @param prob Problem data
         * @param workspace Solver workspace
         */
        MarbleKKTSystem(std::shared_ptr<const Problem> prob,
                        std::shared_ptr<const Workspace> workspace);

        /**
         * Declare the fixed sparsity pattern and block handles
         */
        void initialize_sparsity();
        
        /**
         * Compute Ruiz equilibration scaling from problem data
         *
         * @param niter Number of Ruiz iterations
         * @return Logical-indexed scaling vector
         */
        Vec ruiz_equilibration(int niter) const;

        /**
         * Build the LDLT system and seed constant KKT blocks
         *
         * @param s Logical-indexed Ruiz scaling vector
         */
        void build(const Vec& s);

        /**
         * Recompute the KKT residual from current workspace values
         */
        void update_residual();

        /**
         * Recompute all iteration-dependent KKT matrix entries
         */
        void update_kkt_system();

        /**
         * Set the primal diagonal regularizer
         *
         * @param regularizer New regularizer value
         */
        void update_primal_regularizer(double regularizer);

        /**
         * Recompute derivative of the residual with respect to relaxation
         */
        void update_residual_relax_grad();

        /**
         * Factorize the current KKT matrix values
         *
         * @return True on successful numeric factorization
         */
        bool numerical_factorization() { return kkt.factorize(); }

        /**
         * Check whether the last factorization has the expected saddle-point inertia
         *
         * @return True when the factorization has n_primals positive and n_duals negative pivots
         */
        bool check_inertia() const { return kkt.check_inertia(n_primals, n_duals, 1e-10); }
        
        /**
         * Compute an inertia correction regularizer
         *
         * @return Regularizer value
         */
        double inertia_correction();

        /**
         * Solve the factored KKT system in logical coordinates
         *
         * @param step Output solution vector
         * @param rhs Right-hand side vector
         */
        void compute_step(Vec& step, const Vec& rhs) { kkt.solve(step, rhs); }

        LdltSystem kkt;
        Vec residual;
        Vec grad_residual_relax_param;

    private:
        /**
         * Update z stationarity residual entries
         */
        void update_z_stationarity();

        /**
         * Update inequality slack stationarity residual entries
         */
        void update_ineq_slack_stationarity();

        /**
         * Update complementarity slack stationarity residual entries
         */
        void update_comp_slack_stationarity();

        /**
         * Update dual feasibility residual entries
         */
        void update_dual_feasibility();

        /**
         * Update inequality-dependent KKT blocks
         */
        void update_ineq_blocks();

        /**
         * Update complementarity-dependent KKT blocks
         */
        void update_comp_blocks();

        /**
         * Update penalty-dependent KKT blocks
         */
        void update_penalty_blocks();
};
