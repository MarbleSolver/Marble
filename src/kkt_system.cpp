#include <kkt_system.h>
#include <utils.h>

namespace {

using Coords = LdltSystem::Coords;

/**
 * Map sparse matrix nonzeros to logical block coordinates
 *
 * @param B Sparse matrix whose pattern is mapped
 * @param rows Logical row indices
 * @param cols Logical column indices
 * @param upper_only If true, keep only entries with row <= col
 * @return Logical coordinates in traversal order
 */
Coords block_coords(const SMat& B, const Eigen::VectorXi& rows,
                    const Eigen::VectorXi& cols, bool upper_only) {
    assert(B.rows() == rows.size() && B.cols() == cols.size() &&
           "block_coords: B dimensions must match rows/cols");
    Coords c;
    c.reserve(B.nonZeros());
    for (int k = 0; k < B.outerSize(); ++k)
        for (SMat::InnerIterator it(B, k); it; ++it) {
            int i = rows[it.row()];
            int j = cols[it.col()];
            if (upper_only && i > j) continue;
            c.emplace_back(i, j);
        }
    return c;
}

/**
 * Map transpose nonzeros to logical block coordinates
 *
 * @param B Sparse matrix whose transpose pattern is mapped
 * @param rows Logical row indices for columns of B
 * @param cols Logical column indices for rows of B
 * @return Logical coordinates in traversal order
 */
Coords block_coords_T(const SMat& B, const Eigen::VectorXi& rows,
                      const Eigen::VectorXi& cols) {
    assert(B.cols() == rows.size() && B.rows() == cols.size() &&
           "block_coords_T: B^T dimensions must match rows/cols");
    Coords c;
    c.reserve(B.nonZeros());
    for (int k = 0; k < B.outerSize(); ++k)
        for (SMat::InnerIterator it(B, k); it; ++it)
            c.emplace_back(rows[it.col()], cols[it.row()]);
    return c;
}

/**
 * Build paired coordinates (rows[k], cols[k])
 *
 * @param rows Logical row indices
 * @param cols Logical column indices
 * @return Paired logical coordinates
 */
Coords paired_coords(const Eigen::VectorXi& rows, const Eigen::VectorXi& cols) {
    Coords c;
    c.reserve(rows.size());
    for (int k = 0; k < rows.size(); ++k)
        c.emplace_back(rows[k], cols[k]);
    return c;
}

/**
 * Extract sparse values in the same order as block_coords()
 *
 * @param B Sparse matrix whose values are extracted
 * @param upper_only If true, keep only entries with row <= col
 * @return Values in declaration order
 */
Vec block_values(const SMat& B, bool upper_only) {
    std::vector<double> v;
    v.reserve(B.nonZeros());
    for (int k = 0; k < B.outerSize(); ++k)
        for (SMat::InnerIterator it(B, k); it; ++it) {
            if (upper_only && it.row() > it.col()) continue;
            v.push_back(it.value());
        }
    return Eigen::Map<Vec>(v.data(), static_cast<Eigen::Index>(v.size()));
}

/**
 * Extract sparse values in the same order as block_coords_T()
 *
 * @param B Sparse matrix whose transpose values are extracted
 * @return Values in declaration order
 */
Vec block_values_T(const SMat& B) {
    Vec v(B.nonZeros());
    Eigen::Index n = 0;
    for (int k = 0; k < B.outerSize(); ++k)
        for (SMat::InnerIterator it(B, k); it; ++it)
            v[n++] = it.value();
    return v;
}

}  // namespace

MarbleKKTSystem::KktVariableIndices::KktVariableIndices(const std::shared_ptr<const Problem>& p)
    : z       (safe_linspaced(p->nz,     0)),
      v       (safe_linspaced(p->n_ineq, p->nz)),
      sigma   (safe_linspaced(p->n_comp, p->nz + p->n_ineq)),
      m_eq    (safe_linspaced(p->n_eq,   p->nz + p->n_ineq + p->n_comp)),
      m_ineq  (safe_linspaced(p->n_ineq, p->nz + p->n_ineq + p->n_comp + p->n_eq)),
      m_comp_L(safe_linspaced(p->n_comp, p->nz + 2 * p->n_ineq + p->n_comp + p->n_eq)),
      m_comp_R(safe_linspaced(p->n_comp, p->nz + 2 * p->n_ineq + 2 * p->n_comp + p->n_eq)) {}

MarbleKKTSystem::MarbleKKTSystem(std::shared_ptr<const Problem> problem,
                                 std::shared_ptr<const Workspace> ws)
    : n_primals(problem->nz + problem->n_ineq + problem->n_comp),  // primals + slacks
      n_duals(problem->n_eq + problem->n_ineq + 2 * problem->n_comp),
      n_vars(n_primals + n_duals),
      prob(problem),
      workspace(std::move(ws)),
      variable_inds(problem),
      residual_inds{variable_inds.z, variable_inds.v, variable_inds.sigma,
                    variable_inds.m_eq, variable_inds.m_ineq,
                    variable_inds.m_comp_L, variable_inds.m_comp_R},
      kkt(n_vars) {
    initialize_sparsity();
}

void MarbleKKTSystem::initialize_sparsity() {
    if (prob->nz > 0)
        blocks.hessian = &kkt.add_entries(
            block_coords(prob->cost_hessian, residual_inds.z_stat, variable_inds.z, /*upper_only=*/true));

    if (prob->n_eq > 0)
        blocks.J_eq_T = &kkt.add_entries(
            block_coords_T(prob->J_eq, residual_inds.z_stat, variable_inds.m_eq));
    if (prob->n_ineq > 0)
        blocks.J_ineq_T = &kkt.add_entries(
            block_coords_T(prob->J_ineq, residual_inds.z_stat, variable_inds.m_ineq));
    if (prob->n_comp > 0) {
        blocks.L_T = &kkt.add_entries(
            block_coords_T(prob->L, residual_inds.z_stat, variable_inds.m_comp_L));
        blocks.R_T = &kkt.add_entries(
            block_coords_T(prob->R, residual_inds.z_stat, variable_inds.m_comp_R));
    }

    // Mutable diagonal blocks
    if (prob->n_ineq > 0) {
        blocks.v_stat_v      = &kkt.add_diagonal(variable_inds.v);
        blocks.v_stat_m_ineq = &kkt.add_entries(
            paired_coords(variable_inds.v, variable_inds.m_ineq));
    }
    if (prob->n_comp > 0) {
        blocks.sigma_stat_sigma    = &kkt.add_diagonal(variable_inds.sigma);
        blocks.sigma_stat_m_comp_L = &kkt.add_entries(
            paired_coords(variable_inds.sigma, variable_inds.m_comp_L));
        blocks.sigma_stat_m_comp_R = &kkt.add_entries(
            paired_coords(variable_inds.sigma, variable_inds.m_comp_R));
    }

    if (prob->n_eq   > 0) blocks.pd_eq   = &kkt.add_diagonal(variable_inds.m_eq);
    if (prob->n_ineq > 0) blocks.pd_ineq = &kkt.add_diagonal(variable_inds.m_ineq);
    if (prob->n_comp > 0) {
        blocks.pd_comp_L = &kkt.add_diagonal(variable_inds.m_comp_L);
        blocks.pd_comp_R = &kkt.add_diagonal(variable_inds.m_comp_R);
    }

    if (n_primals > 0)
        blocks.primal_reg = &kkt.add_diagonal(safe_linspaced(n_primals, 0));
}

Vec MarbleKKTSystem::ruiz_equilibration(int niter) const {
    const int nz = prob->nz, n_eq = prob->n_eq, n_ineq = prob->n_ineq, n_comp = prob->n_comp;

    SMat H = prob->cost_hessian, J_eq = prob->J_eq, J_ineq = prob->J_ineq,
         L = prob->L, R = prob->R;

    Vec d = Vec::Ones(nz), e_eq = Vec::Ones(n_eq), e_ineq = Vec::Ones(n_ineq),
        e_comp_L = Vec::Ones(n_comp), e_comp_R = Vec::Ones(n_comp);

    auto col_absmax = [](Vec& out, const SMat& M) {     // out[col] = max |M(:,col)|
        for (int c = 0; c < M.outerSize(); ++c)
            for (SMat::InnerIterator it(M, c); it; ++it)
                out[c] = std::max(out[c], std::abs(it.value()));
    };
    auto row_absmax = [](Vec& out, const SMat& M) {     // out[row] = max |M(row,:)|
        for (int c = 0; c < M.outerSize(); ++c)
            for (SMat::InnerIterator it(M, c); it; ++it)
                out[it.row()] = std::max(out[it.row()], std::abs(it.value()));
    };
    auto rescale = [](SMat& M, const Vec& rows, const Vec& cols) {  // M <- diag(rows) M diag(cols)
        for (int c = 0; c < M.outerSize(); ++c)
            for (SMat::InnerIterator it(M, c); it; ++it)
                it.valueRef() *= rows[it.row()] * cols[c];
    };
    auto clamp_rsqrt = [](Vec v) {
        return v.cwiseMax(1e-4).cwiseMin(1e4).array().rsqrt().matrix().eval();
    };

    for (int iter = 0; iter < niter; ++iter) {
        Vec d_t = Vec::Zero(nz);                                       // column norms
        col_absmax(d_t, H); col_absmax(d_t, J_eq); col_absmax(d_t, J_ineq);
        col_absmax(d_t, L); col_absmax(d_t, R);

        Vec eq_t = Vec::Zero(n_eq);   row_absmax(eq_t, J_eq);          // row norms
        Vec in_t = Vec::Zero(n_ineq); row_absmax(in_t, J_ineq);
        Vec cL_t = Vec::Zero(n_comp); row_absmax(cL_t, L);
        Vec cR_t = Vec::Zero(n_comp); row_absmax(cR_t, R);

        d_t = clamp_rsqrt(d_t);   eq_t = clamp_rsqrt(eq_t); in_t = clamp_rsqrt(in_t);
        cL_t = clamp_rsqrt(cL_t); cR_t = clamp_rsqrt(cR_t);

        rescale(H, d_t, d_t);   rescale(J_eq, eq_t, d_t); rescale(J_ineq, in_t, d_t);
        rescale(L, cL_t, d_t);  rescale(R, cR_t, d_t);

        d = d.cwiseProduct(d_t);          e_eq = e_eq.cwiseProduct(eq_t);
        e_ineq = e_ineq.cwiseProduct(in_t);
        e_comp_L = e_comp_L.cwiseProduct(cL_t); e_comp_R = e_comp_R.cwiseProduct(cR_t);
    }

    Vec s = Vec::Ones(n_vars);
    s(variable_inds.z)        = d;
    s(variable_inds.m_eq)     = e_eq;
    s(variable_inds.m_ineq)   = e_ineq;
    s(variable_inds.m_comp_L) = e_comp_L;
    s(variable_inds.m_comp_R) = e_comp_R;
    return s;
}

void MarbleKKTSystem::build(const Vec& s) {
    kkt.build(s);

    if (blocks.hessian)
        blocks.hessian->set(block_values(prob->cost_hessian, true));
    if (blocks.J_eq_T)
        blocks.J_eq_T->set(block_values_T(prob->J_eq));
    if (blocks.J_ineq_T)
        blocks.J_ineq_T->set(block_values_T(prob->J_ineq));
    if (blocks.L_T)
        blocks.L_T->set(block_values_T(prob->L));
    if (blocks.R_T)
        blocks.R_T->set(block_values_T(prob->R));

    residual = Vec::Zero(n_vars);
    grad_residual_relax_param = Vec::Zero(n_vars);
}

void MarbleKKTSystem::update_residual() {
    update_z_stationarity();
    update_ineq_slack_stationarity();
    update_comp_slack_stationarity();
    update_dual_feasibility();
}

void MarbleKKTSystem::update_z_stationarity() {
    // Hz + g + J_eq' m_eq + J_ineq' m_ineq + L' m_comp_L + R' m_comp_R = 0
    residual(residual_inds.z_stat) =
        prob->cost_hessian * workspace->z + prob->cost_gradient +
        prob->J_eq.transpose() * workspace->m_eq +
        prob->J_ineq.transpose() * workspace->m_ineq +
        prob->L.transpose() * workspace->m_comp_L +
        prob->R.transpose() * workspace->m_comp_R;
}

void MarbleKKTSystem::update_ineq_slack_stationarity() {
    const RelaxedSlackValues& ineq = workspace->ineq_retract;

    // -b'(s_ineq) * (m_ineq + b_kappa(-s_ineq)) = 0
    residual(residual_inds.v_stat) =
        -ineq.b_prime.cwiseProduct(workspace->m_ineq + ineq.b_neg);
}

void MarbleKKTSystem::update_comp_slack_stationarity() {
    const RelaxedSlackValues& comp = workspace->comp_retract;

    // -b'(s_comp) * m_comp_L + b'(-s_comp) * m_comp_R = 0
    residual(residual_inds.sigma_stat) =
        -comp.b_prime.cwiseProduct(workspace->m_comp_L) +
        comp.b_neg_prime.cwiseProduct(workspace->m_comp_R);
}

void MarbleKKTSystem::update_dual_feasibility() {
    const RelaxedSlackValues& ineq = workspace->ineq_retract;
    const RelaxedSlackValues& comp = workspace->comp_retract;
    const double inv_penalty = 1.0 / workspace->penalty_param;

    // J_eq * z + c_eq - 1/rho * (m_eq - m_eq_est) = 0
    residual(residual_inds.pd_eq) =
        workspace->residual_eq - inv_penalty * (workspace->m_eq - workspace->m_eq_est);

    // J_ineq * z + c_ineq - b(s_ineq) - 1/rho * (m_ineq - m_ineq_est) = 0
    residual(residual_inds.pd_ineq) =
        workspace->residual_ineq - ineq.b -
        inv_penalty * (workspace->m_ineq - workspace->m_ineq_est);

    // L * z + c_comp_L - b(s_comp) - 1/rho * (m_comp_L - m_comp_L_est) = 0
    residual(residual_inds.pd_comp_L) =
        workspace->residual_comp_L - comp.b -
        inv_penalty * (workspace->m_comp_L - workspace->m_comp_L_est);

    // R * z + c_comp_R - b(-s_comp) - 1/rho * (m_comp_R - m_comp_R_est) = 0
    residual(residual_inds.pd_comp_R) =
        workspace->residual_comp_R - comp.b_neg -
        inv_penalty * (workspace->m_comp_R - workspace->m_comp_R_est);
}

void MarbleKKTSystem::update_kkt_system() {
    update_primal_regularizer(0.0);
    update_ineq_blocks();
    update_comp_blocks();
    update_penalty_blocks();
}

void MarbleKKTSystem::update_ineq_blocks() {
    if (prob->n_ineq == 0) return;
    
    const RelaxedSlackValues& ineq = workspace->ineq_retract;

    blocks.v_stat_v->set(
        ineq.b_prime.cwiseProduct(ineq.b_neg_prime));
    blocks.v_stat_m_ineq->set(-ineq.b_prime);
}

void MarbleKKTSystem::update_comp_blocks() {
    if (prob->n_comp == 0) return;
    
    const RelaxedSlackValues& comp = workspace->comp_retract;
    
    blocks.sigma_stat_sigma->set(
        -comp.b_double_prime.cwiseProduct(workspace->m_comp_L + workspace->m_comp_R));
    blocks.sigma_stat_m_comp_L->set(-comp.b_prime);
    blocks.sigma_stat_m_comp_R->set(comp.b_neg_prime);
}

void MarbleKKTSystem::update_penalty_blocks() {
    if (prob->n_eq == 0 && prob->n_ineq == 0 && prob->n_comp == 0) return;

    const double neg_inv_penalty = -1.0 / workspace->penalty_param;
    if (blocks.pd_eq)     blocks.pd_eq->set(neg_inv_penalty);
    if (blocks.pd_ineq)   blocks.pd_ineq->set(neg_inv_penalty);
    if (blocks.pd_comp_L) blocks.pd_comp_L->set(neg_inv_penalty);
    if (blocks.pd_comp_R) blocks.pd_comp_R->set(neg_inv_penalty);
}

void MarbleKKTSystem::update_residual_relax_grad() {
    grad_residual_relax_param.setZero();

    const RelaxedSlackValues& ineq = workspace->ineq_retract;
    const RelaxedSlackValues& comp = workspace->comp_retract;

    grad_residual_relax_param(residual_inds.v_stat) =
        -ineq.d_b_prime_d_kappa.cwiseProduct(workspace->m_ineq + ineq.b_neg) -
        ineq.b_prime.cwiseProduct(ineq.d_b_neg_d_kappa);

    grad_residual_relax_param(residual_inds.sigma_stat) =
        -comp.d_b_prime_d_kappa.cwiseProduct(workspace->m_comp_L) +
        comp.d_b_neg_prime_d_kappa.cwiseProduct(workspace->m_comp_R);

    grad_residual_relax_param(residual_inds.pd_ineq) = -ineq.d_b_d_kappa;

    grad_residual_relax_param(residual_inds.pd_comp_L) = -comp.d_b_d_kappa;
    grad_residual_relax_param(residual_inds.pd_comp_R) = -comp.d_b_neg_d_kappa;
}

void MarbleKKTSystem::update_primal_regularizer(double regularizer) {
    if (!blocks.primal_reg) return;

    blocks.primal_reg->add(regularizer - primal_regularizer);
    primal_regularizer = regularizer;
}
