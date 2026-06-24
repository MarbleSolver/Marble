#pragma once

#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "qdldl.h"
#include "utils.h"

/**
 * Fixed-pattern symmetric system represented internally as P^T S K S P for QDLDL
 */
class LdltSystem {
public:
    using SMat    = ::SMat;
    using Scalar  = SMat::Scalar;
    using Index   = SMat::StorageIndex;
    using Vec     = ::Vec;
    using IVec    = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<Scalar, Index>;
    using Coords  = std::vector<std::pair<Index, Index>>;

    /**
     * Handle to a fixed set of problem-space matrix entries
     */
    class SparseBlock {
    public:
        /**
         * Construct an empty block handle
         */
        SparseBlock() = default;

        /**
         * Overwrite block entries with problem-space nonzero values
         *
         * @param nzval Nonzero values in declaration order
         */
        void set(const Eigen::Ref<const Vec>& nzval) const {
            assert(owner && owner->is_built && "use a SparseBlock only after build()");
            owner->nonzeros()(internal_nz_indices) = nzval.cwiseProduct(scale);
        }

        /**
         * Overwrite every block entry with one problem-space scalar
         *
         * @param c Value to write to each entry
         */
        void set(Scalar c) const {
            assert(owner && owner->is_built && "use a SparseBlock only after build()");
            owner->nonzeros()(internal_nz_indices) = c * scale;
        }

        /**
         * Add problem-space nonzero values to block entries
         *
         * @param nzval Nonzero values in declaration order
         */
        void add(const Eigen::Ref<const Vec>& nzval) const {
            assert(owner && owner->is_built && "use a SparseBlock only after build()");
            owner->nonzeros()(internal_nz_indices) += nzval.cwiseProduct(scale);
        }

        /**
         * Add one problem-space scalar to every block entry
         *
         * @param c Value to add to each entry
         */
        void add(Scalar c) const {
            assert(owner && owner->is_built && "use a SparseBlock only after build()");
            owner->nonzeros()(internal_nz_indices) += c * scale;
        }

        /**
         * Return the number of entries in this block
         *
         * @return Entry count
         */
        Index size() const {
            return static_cast<Index>(problem_coords.size()
                                      ? problem_coords.size()
                                      : internal_nz_indices.size());
        }

    private:
        friend class LdltSystem;
        LdltSystem* owner = nullptr;  // matrix that owns this block
        Coords problem_coords;        // problem-space entries, kept until build()
        IVec internal_nz_indices;     // positions into mat.valuePtr() after build()
        Vec scale;                    // cached scaling_vec[row] * scaling_vec[col]
    };

    /**
     * Construct an empty fixed-pattern system
     *
     * @param n Matrix dimension
     */
    LdltSystem(Index n) : dim(n) {}

    LdltSystem(const LdltSystem&)            = delete;
    LdltSystem& operator=(const LdltSystem&) = delete;
    LdltSystem(LdltSystem&&)                 = delete;
    LdltSystem& operator=(LdltSystem&&)      = delete;

    /**
     * Declare arbitrary problem-space matrix entries
     *
     * @param coords Problem-space row and column coordinates
     * @return Stable block handle, usable after build()
     */
    SparseBlock& add_block(const Coords& coords) {
        assert(!is_built && "declare all blocks before build()");
        sparse_blocks.push_back(std::make_unique<SparseBlock>());
        SparseBlock& block = *sparse_blocks.back();
        block.owner = this;
        block.problem_coords = coords;
        return block;
    }

    /**
     * Declare diagonal entries for problem-space indices
     *
     * @param problem_idx Problem-space indices whose diagonal entries are declared
     * @return Stable block handle, usable after build()
     */
    template <typename Indices>
    SparseBlock& add_block_diagonal(const Indices& problem_idx) {
        Coords coords;
        coords.reserve(problem_idx.size());
        for (Index k = 0; k < problem_idx.size(); ++k) {
            coords.emplace_back(problem_idx[k], problem_idx[k]);
        }
        return add_block(coords);
    }

    /**
     * Finalize sparsity, ordering, scaling, and symbolic factorization
     *
     * @param scaling Problem-space Ruiz scaling vector
     */
    void build(const Vec& scaling) {
        assert(!is_built && "build() once");
        assert(scaling.size() == dim && "Ruiz scaling vector must have length n");
        scaling_vec = scaling;
        internal_rhs.resize(dim);

        // Merge declared problem-space blocks into a symmetric upper pattern
        std::vector<Triplet> problem_pattern;
        for (const auto& block_ptr : sparse_blocks) {
            for (const auto& coord : block_ptr->problem_coords) {
                Index i = coord.first;
                Index j = coord.second;
                if (i > j) std::swap(i, j);
                problem_pattern.emplace_back(i, j, Scalar(0));
            }
        }
        SMat problem_upper(dim, dim);
        problem_upper.setFromTriplets(problem_pattern.begin(), problem_pattern.end());
        problem_upper.makeCompressed();

        // AMD orders the problem-space symmetric pattern; perm maps internal slots to problem indices
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm_matrix;
        Eigen::AMDOrdering<Index> amd;
        amd(problem_upper.selfadjointView<Eigen::Upper>(), perm_matrix);
        perm_ = perm_matrix.indices();
        iperm_.resize(dim);

        for (Index i = 0; i < dim; ++i)
            iperm_[perm_[i]] = i;

        SMat internal_full(dim, dim);
        internal_full = problem_upper.selfadjointView<Eigen::Upper>().twistedBy(perm_matrix.inverse());
        mat = internal_full.triangularView<Eigen::Upper>();
        mat.makeCompressed();

        // Resolve each sparse block to value slots and scaling factors
        is_built = true;
        for (auto& block_ptr : sparse_blocks) bind_block(*block_ptr);

        // QDLDL reuses this symbolic data for every numeric refactorization
        symbolic_factorize();
    }

    /**
     * Check whether build() has completed
     *
     * @return True after build()
     */
    bool built() const { return is_built; }

    /**
     * Numerically factorize the current internal matrix values
     *
     * @param count Optional unused counter pointer
     * @return True when QDLDL succeeds
     */
    bool factorize() {
        const QDLDL_int status = QDLDL_factor(
            dim, Ap.data(), Ai.data(), mat.valuePtr(),
            Lp.data(), Li.data(), Lx.data(), D.data(), Dinv.data(),
            Lnz.data(), etree.data(), bwork.data(), iwork.data(), fwork.data());
        return status != -1;
    }

    /**
     * Solve K x = b in problem coordinates using the current factorization
     *
     * @param x Output solution vector
     * @param b Problem-space right-hand side
     */
    void solve(Vec& x, const Vec& b) {
        assert(b.size() == dim && "rhs must have length n");
        assert(internal_rhs.size() == dim && "internal_rhs must have length n");
        assert(x.size() == dim && "solution must have length n");

        problem_to_internal(b, internal_rhs);
        QDLDL_solve(dim, Lp.data(), Li.data(), Lx.data(), Dinv.data(),
                    internal_rhs.data());
        internal_to_problem(internal_rhs, x);
    }

    /**
     * Check the signs of the current LDLT diagonal pivots
     *
     * @param n_pos Expected number of positive pivots
     * @param n_neg Expected number of negative pivots
     * @param atol Absolute tolerance for zero pivots
     * @return True when the pivot counts match
     */
    bool check_inertia(Index n_pos, Index n_neg, double atol=1e-10) const {
        return (D.array() > atol).count() == n_pos &&
               (D.array() < -atol).count() == n_neg;
    }

    /**
     * Access the internal upper-triangular matrix
     *
     * @return Mutable internal matrix
     */
    SMat& matrix() { return mat; }

    /**
     * Access the internal upper-triangular matrix
     *
     * @return Internal matrix
     */
    const SMat& matrix() const { return mat; }

    /**
     * Access the internal-to-problem permutation
     *
     * @return Permutation indices
     */
    const IVec& perm() const { return perm_; }

    /**
     * Access the problem-to-internal inverse permutation
     *
     * @return Inverse permutation indices
     */
    const IVec& iperm() const { return iperm_; }

    /**
     * Access the problem-space scaling vector
     *
     * @return Scaling vector
     */
    const Vec& scaling() const { return scaling_vec; }

    /**
     * Convert an internal vector to problem coordinates
     *
     * @param x_internal Internal-coordinate vector
     * @param x_problem Output problem-coordinate vector
     */
    void internal_to_problem(const Vec& x_internal, Vec& x_problem) const {
        x_problem = scaling_vec.cwiseProduct(x_internal(iperm_));
    }

    /**
     * Convert a problem vector to internal coordinates
     *
     * @param x_problem Problem-coordinate vector
     * @param x_internal Output internal-coordinate vector
     */
    void problem_to_internal(const Vec& x_problem, Vec& x_internal) const {
        x_internal = (x_problem.cwiseProduct(scaling_vec))(perm_);
    }

private:
    friend class SparseBlock;

    Index dim;
    bool is_built = false;
    SMat mat;
    IVec perm_;
    IVec iperm_;
    Vec scaling_vec;
    std::vector<std::unique_ptr<SparseBlock>> sparse_blocks;

    /**
     * Map the internal nonzero value array
     *
     * @return Mutable vector view over mat values
     */
    Eigen::Map<Vec> nonzeros() {
        return Eigen::Map<Vec>(mat.valuePtr(), mat.nonZeros());
    }

    /**
     * Find the value index of an upper-triangular internal entry
     *
     * @param a Internal row index
     * @param b Internal column index
     * @return Index into mat.valuePtr()
     */
    Index data_index(Index a, Index b) const {
        const Index* outer = mat.outerIndexPtr();
        const Index* inner = mat.innerIndexPtr();
        const Index* lo = inner + outer[b];
        const Index* hi = inner + outer[b + 1];
        const Index* p = std::lower_bound(lo, hi, a);
        assert(p != hi && *p == a && "problem entry is not in the internal pattern");
        return static_cast<Index>(p - inner);
    }

    /**
     * Find the value index of a problem-space matrix entry
     *
     * @param i Problem-space row index
     * @param j Problem-space column index
     * @return Index into mat.valuePtr()
     */
    Index data_index_problem(Index i, Index j) const {
        Index a = iperm_[i];
        Index b = iperm_[j];
        if (a > b) std::swap(a, b);
        return data_index(a, b);
    }

    /**
     * Bind a declared block to internal value slots and scaling factors
     *
     * @param block Sparse block to bind
     */
    void bind_block(SparseBlock& block) {
        const Index m = static_cast<Index>(block.problem_coords.size());
        block.internal_nz_indices.resize(m);
        block.scale.resize(m);
        for (Index k = 0; k < m; ++k) {
            const Index i = block.problem_coords[k].first;
            const Index j = block.problem_coords[k].second;
            block.internal_nz_indices[k] = data_index_problem(i, j);
            block.scale[k] = scaling_vec[i] * scaling_vec[j];
        }
        block.problem_coords.clear();
        block.problem_coords.shrink_to_fit();
    }

    /**
     * Build QDLDL symbolic factorization data for the fixed structure
     */
    void symbolic_factorize() {
        const QDLDL_int nnz = static_cast<QDLDL_int>(mat.nonZeros());
        Ap.resize(dim + 1);
        Ai.resize(nnz);
        for (Index k = 0; k <= dim; ++k) Ap[k] = mat.outerIndexPtr()[k];
        for (QDLDL_int k = 0; k < nnz; ++k) Ai[k] = mat.innerIndexPtr()[k];

        etree.resize(dim);
        Lnz.resize(dim);
        iwork.resize(3 * dim);
        bwork.resize(dim);
        fwork.resize(dim);
        Lp.resize(dim + 1);
        D.resize(dim);
        Dinv.resize(dim);

        sumLnz = QDLDL_etree(dim, Ap.data(), Ai.data(),
                             iwork.data(), Lnz.data(), etree.data());
        assert(sumLnz >= 0 && "LdltSystem: not a valid upper-triangular CSC");
        Li.resize(sumLnz);
        Lx.resize(sumLnz);
    }

    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> Ap, Ai;
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> etree, Lnz, iwork, Lp, Li;
    std::vector<QDLDL_bool> bwork;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> fwork, Lx, D, Dinv;
    QDLDL_int sumLnz = 0;
    Vec internal_rhs;
};
