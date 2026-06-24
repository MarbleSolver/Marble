#pragma once
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>

#include "problem.h"
#include "ldlt_system.h"

class Workspace;

class KKTSystem {
    public:
        /**
         * Shared slices for the stacked KKT vector
         */
        struct KKTIndices {
            using Range = decltype(Eigen::seqN(Eigen::Index(0), Eigen::Index(0)));

            const Range z;
            const Range s_ineq;
            const Range s_comp;
            const Range m_eq;
            const Range m_ineq;
            const Range m_comp_L;
            const Range m_comp_R;

            const int n_primals;
            const int n_duals;
            const int n_vars;

            /**
             * Build contiguous slices from problem dimensions
             *
             * @param prob Problem whose dimensions define the KKT layout
             */
            explicit KKTIndices(const Problem& prob)
                : z       (Eigen::seqN(Eigen::Index(0), Eigen::Index(prob.nz))),
                  s_ineq  (Eigen::seqN(Eigen::Index(prob.nz), Eigen::Index(prob.n_ineq))),
                  s_comp  (Eigen::seqN(Eigen::Index(prob.nz + prob.n_ineq), Eigen::Index(prob.n_comp))),
                  m_eq    (Eigen::seqN(Eigen::Index(prob.nz + prob.n_ineq + prob.n_comp), Eigen::Index(prob.n_eq))),
                  m_ineq  (Eigen::seqN(Eigen::Index(prob.nz + prob.n_ineq + prob.n_comp + prob.n_eq), Eigen::Index(prob.n_ineq))),
                  m_comp_L(Eigen::seqN(Eigen::Index(prob.nz + 2 * prob.n_ineq + prob.n_comp + prob.n_eq), Eigen::Index(prob.n_comp))),
                  m_comp_R(Eigen::seqN(Eigen::Index(prob.nz + 2 * prob.n_ineq + 2 * prob.n_comp + prob.n_eq), Eigen::Index(prob.n_comp))),
                  n_primals(prob.nz + prob.n_ineq + prob.n_comp),
                  n_duals(prob.n_eq + prob.n_ineq + 2 * prob.n_comp),
                  n_vars(n_primals + n_duals) {}
        };

        const int n_primals, n_duals, n_vars;

        // The problem that this KKT system corresponds to
        std::shared_ptr<const Problem> prob;

        // Workspace for shared values
        std::shared_ptr<const Workspace> workspace;

        double primal_regularizer = 0.0; // Current regularization value for inertia correction

        // Accessors into the LDLT mat blocks
        struct KKTBlocks {
            LdltSystem::SparseBlock* hessian             = nullptr;  // (z, z)
            LdltSystem::SparseBlock* J_eq_T              = nullptr;  // (z, m_eq)
            LdltSystem::SparseBlock* J_ineq_T            = nullptr;  // (z, m_ineq)
            LdltSystem::SparseBlock* L_T                 = nullptr;  // (z, m_comp_L)
            LdltSystem::SparseBlock* R_T                 = nullptr;  // (z, m_comp_R)

            LdltSystem::SparseBlock* s_ineq_stat_s_ineq  = nullptr;  // (s_ineq, s_ineq)
            LdltSystem::SparseBlock* s_ineq_stat_m_ineq  = nullptr;  // (s_ineq, m_ineq)
            LdltSystem::SparseBlock* s_comp_stat_s_comp  = nullptr;  // (s_comp, s_comp)
            LdltSystem::SparseBlock* s_comp_stat_m_comp_L = nullptr; // (s_comp, m_comp_L)
            LdltSystem::SparseBlock* s_comp_stat_m_comp_R = nullptr; // (s_comp, m_comp_R)

            LdltSystem::SparseBlock* m_eq_m_eq            = nullptr;  // (m_eq, m_eq)
            LdltSystem::SparseBlock* m_ineq_m_ineq        = nullptr;  // (m_ineq, m_ineq)
            LdltSystem::SparseBlock* m_comp_L_m_comp_L    = nullptr;  // (m_comp_L, m_comp_L)
            LdltSystem::SparseBlock* m_comp_R_m_comp_R    = nullptr;  // (m_comp_R, m_comp_R)

            LdltSystem::SparseBlock* primal_reg           = nullptr;  // diag(z, s_ineq, s_comp)
        };

        KKTIndices inds;
        KKTBlocks blocks;

        /**
         * Create a KKT system from problem dimensions and workspace storage
         *
         * @param prob Problem data
         * @param workspace Solver workspace
         */
        KKTSystem(std::shared_ptr<const Problem> prob,
                        std::shared_ptr<const Workspace> workspace);

        /**
         * Declare the fixed sparsity pattern and block handles
         */
        void initialize_sparsity();
        
        /**
         * Compute Ruiz equilibration scaling from problem data
         *
         * @param niter Number of Ruiz iterations
         * @return Problem-indexed scaling vector
         */
        Vec ruiz_equilibration(int niter) const;

        /**
         * Build the LDLT system and seed constant KKT blocks
         *
         * @param s Problem-indexed Ruiz scaling vector
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
         * Solve the factored KKT system in problem coordinates
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
