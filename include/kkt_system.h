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
        // Implementation detail: the residual equations of this saddle-point
        // system are indexed identically to the variables, so the two index
        // sets share a single backing store. Only the public instances below
        // (with their respective field names) are part of the interface.

        // Variable indices for the KKT system columns. Laid out once from the
        // problem dimensions and immutable thereafter:
        // [ z | v | sigma | m_eq | m_ineq | m_comp_L | m_comp_R ].
        struct KktVariableIndices {
            const Eigen::VectorXi z;
            const Eigen::VectorXi v;
            const Eigen::VectorXi sigma;
            const Eigen::VectorXi m_eq;
            const Eigen::VectorXi m_ineq;
            const Eigen::VectorXi m_comp_L;
            const Eigen::VectorXi m_comp_R;

            explicit KktVariableIndices(const std::shared_ptr<const Problem>& prob);
        };

        // Indices into the KKT residual — references aliasing the variable
        // indices above (same storage, residual-equation names). Bound once in
        // the constructor; read-only.
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

        // Handles to the KKT blocks rewritten across iters. Constant
        // problem-data blocks are seeded once in build(); the rest change each
        // step. Pointers are owned by `kkt_system`; null when the block is empty.
        struct KktBlocks {
            // Constant problem-data blocks (upper triangle), set once.
            LdltSystem::Block* hessian             = nullptr;  // (z, z)
            LdltSystem::Block* J_eq_T              = nullptr;  // (z, m_eq)
            LdltSystem::Block* J_ineq_T            = nullptr;  // (z, m_ineq)
            LdltSystem::Block* L_T                 = nullptr;  // (z, m_comp_L)
            LdltSystem::Block* R_T                 = nullptr;  // (z, m_comp_R)

            // Mutable diagonal coupling blocks, rewritten each iteration.
            LdltSystem::Block* v_stat_v            = nullptr;  // (v, v)
            LdltSystem::Block* v_stat_m_ineq       = nullptr;  // (v, m_ineq)
            LdltSystem::Block* sigma_stat_sigma    = nullptr;  // (sigma, sigma)
            LdltSystem::Block* sigma_stat_m_comp_L = nullptr;  // (sigma, m_comp_L)
            LdltSystem::Block* sigma_stat_m_comp_R = nullptr;  // (sigma, m_comp_R)

            // Primal-dual (augmented-Lagrangian) diagonal regularizers.
            LdltSystem::Block* pd_eq               = nullptr;  // (m_eq, m_eq)
            LdltSystem::Block* pd_ineq             = nullptr;  // (m_ineq, m_ineq)
            LdltSystem::Block* pd_comp_L           = nullptr;  // (m_comp_L, m_comp_L)
            LdltSystem::Block* pd_comp_R           = nullptr;  // (m_comp_R, m_comp_R)

            // Full primal diagonal (z, v, sigma) where the inertia-correction
            // regularizer is written. Aliases the hessian / v_stat_v /
            // sigma_stat_sigma diagonals (same storage slots) and guarantees
            // every primal diagonal entry exists in the pattern (QDLDL needs it).
            LdltSystem::Block* primal_reg          = nullptr;  // diag(z, v, sigma)
        };

        // Declaration order matters: variable_inds must precede residual_inds,
        // whose references bind to it (both bound in the constructor).
        KktVariableIndices variable_inds;
        ResidualEquationIndices residual_inds;
        KktBlocks blocks;

        // Create KKT system based on problem dimensions and sparsity pattern
        MarbleKKTSystem(std::shared_ptr<const Problem> prob,
                        std::shared_ptr<const Workspace> workspace);

        // Declare the fixed sparsity pattern: variable/residual indices and the
        // block handles. Must run before build().
        void initialize_sparsity();
        
        // Compute the Ruiz equilibration scaling from the problem data
        Vec ruiz_equilibration(int niter) const;

        // Finalize AMD ordering / Ruiz scaling and seed the constant blocks.
        // `s` is the logical-indexed Ruiz scaling diagonal (length n_vars).
        void build(const Vec& s);

        // Recompute the KKT residual from the current workspace iterates and
        // cached retraction values. Writes residual.
        void update_residual();

        // Recompute the iteration-dependent KKT system (Jacobian) entries from
        // the current workspace values. Writes the mutable blocks in place.
        void update_kkt_system();

        // Set the primal-diagonal inertia-correction regularizer to `regularizer`
        // (replaces, not accumulates, the previously written value).
        void update_primal_regularizer(double regularizer);

        // Recompute d(residual)/d(relaxation parameter) from the cached
        // retraction kappa-derivatives. Writes grad_residual_relax_param;
        // used to first-order correct the iterate when the relaxation changes.
        void update_residual_relax_grad();

        // Numeric LDL^T factorization of the current KKT matrix values
        bool numerical_factorization() { return kkt.factorize(); }

        // True if the last numerical_factorization() produced the correct inertia
        // for this saddle-point system: n_primals positive and n_duals negative
        // pivots in the LDL^T diagonal.
        bool check_inertia() const { return kkt.check_inertia(n_primals, n_duals, 1e-10); }
        
        double inertia_correction();

        // Solve K * step = rhs. `rhs` is in logical ordering and unscaled; `step`
        // is returned in the same logical/unscaled space. Requires a prior
        // numerical_factorization(). `rhs` and `step` may be the same vector.
        void compute_step(Vec& step, const Vec& rhs) { kkt.solve(step, rhs); }

        // Sparse KKT system matrix (permuted + scaled, upper-triangular CSC).
        // Pinned (non-movable): handles in `blocks` point into it.
        LdltSystem kkt;
        Vec residual;
        Vec grad_residual_relax_param;

    private:
        // Residual sub-updates (write segments of residual).
        void update_z_stationarity();
        void update_ineq_slack_stationarity();
        void update_comp_slack_stationarity();
        void update_dual_feasibility();

        // KKT-system sub-updates (write the mutable blocks).
        void update_ineq_blocks();
        void update_comp_blocks();
        void update_penalty_blocks();
};
