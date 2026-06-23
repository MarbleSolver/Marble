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
 * Fixed-pattern symmetric system stored as P^T S K S P for QDLDL
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
     * Handle to a fixed set of logical matrix entries
     */
    class Block {
    public:
        /**
         * Construct an empty block handle
         */
        Block() = default;

        /**
         * Overwrite block entries with logical unscaled values
         *
         * @param v Values in declaration order
         */
        void set(const Eigen::Ref<const Vec>& v) const {
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) = v.cwiseProduct(scale_);
        }

        /**
         * Overwrite every block entry with one logical unscaled scalar
         *
         * @param c Value to write to each entry
         */
        void set(Scalar c) const {
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) = c * scale_;
        }

        /**
         * Add logical unscaled values to block entries
         *
         * @param v Values in declaration order
         */
        void add(const Eigen::Ref<const Vec>& v) const {
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) += v.cwiseProduct(scale_);
        }

        /**
         * Add one logical unscaled scalar to every block entry
         *
         * @param c Value to add to each entry
         */
        void add(Scalar c) const {
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) += c * scale_;
        }

        /**
         * Return the number of entries in this block
         *
         * @return Entry count
         */
        Index size() const {
            return static_cast<Index>(coords_.size() ? coords_.size() : idx_.size());
        }

    private:
        friend class LdltSystem;
        LdltSystem* parent_ = nullptr;
        Coords coords_;
        IVec idx_;
        Vec scale_;
    };

    /**
     * Construct an empty fixed-pattern system
     *
     * @param n Matrix dimension
     */
    LdltSystem(Index n) : n_(n) {}

    LdltSystem(const LdltSystem&)            = delete;
    LdltSystem& operator=(const LdltSystem&) = delete;
    LdltSystem(LdltSystem&&)                 = delete;
    LdltSystem& operator=(LdltSystem&&)      = delete;

    /**
     * Declare arbitrary logical matrix entries
     *
     * @param coords Logical row and column coordinates
     * @return Stable block handle, usable after build()
     */
    Block& add_entries(const Coords& coords) {
        assert(!built_ && "declare all blocks before build()");
        blocks_.push_back(std::make_unique<Block>());
        Block& b = *blocks_.back();
        b.parent_ = this;
        b.coords_ = coords;
        return b;
    }

    /**
     * Declare diagonal entries for logical indices
     *
     * @param logical_idx Logical indices whose diagonal entries are declared
     * @return Stable block handle, usable after build()
     */
    Block& add_diagonal(const IVec& logical_idx) {
        Coords c;
        c.reserve(logical_idx.size());
        for (Index k = 0; k < logical_idx.size(); ++k) {
            c.emplace_back(logical_idx[k], logical_idx[k]);
        }
        return add_entries(c);
    }

    /**
     * Finalize sparsity, AMD ordering, scaling, and symbolic factorization
     *
     * @param s Logical-indexed scaling vector
     */
    void build(const Vec& s) {
        assert(!built_ && "build() once");
        assert(s.size() == n_ && "Ruiz scaling vector must have length n");
        s_ = s;
        stored_rhs_.resize(n_);

        std::vector<Triplet> pat;
        for (const auto& bp : blocks_) {
            for (const auto& c : bp->coords_) {
                Index i = c.first;
                Index j = c.second;
                if (i > j) std::swap(i, j);
                pat.emplace_back(i, j, Scalar(0));
            }
        }
        SMat Ku(n_, n_);
        Ku.setFromTriplets(pat.begin(), pat.end());
        Ku.makeCompressed();

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm;
        Eigen::AMDOrdering<Index> amd;
        amd(Ku.selfadjointView<Eigen::Upper>(), perm);
        perm_ = perm.indices();
        iperm_.resize(n_);
        for (Index i = 0; i < n_; ++i) iperm_[perm_[i]] = i;

        std::vector<Triplet> st;
        st.reserve(pat.size());
        for (const auto& t : pat) {
            Index a = iperm_[t.row()];
            Index b = iperm_[t.col()];
            if (a > b) std::swap(a, b);
            st.emplace_back(a, b, Scalar(0));
        }
        K_ = SMat(n_, n_);
        K_.setFromTriplets(st.begin(), st.end());
        K_.makeCompressed();

        built_ = true;
        for (auto& bp : blocks_) resolve(*bp);
        symbolic_factorize();
    }

    /**
     * Check whether build() has completed
     *
     * @return True after build()
     */
    bool built() const { return built_; }

    /**
     * Numerically factorize the current stored matrix values
     *
     * @param count Optional unused counter pointer
     * @return True when QDLDL succeeds
     */
    bool factorize(int *count=nullptr) {
        (void)count;
        const QDLDL_int status = QDLDL_factor(
            n_, Ap_.data(), Ai_.data(), K_.valuePtr(),
            Lp_.data(), Li_.data(), Lx_.data(), D_.data(), Dinv_.data(),
            Lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(), fwork_.data());
        return status != -1;
    }

    /**
     * Solve K x = b in logical coordinates using the current factorization
     *
     * @param x Output solution vector
     * @param b Logical unscaled right-hand side
     */
    void solve(Vec& x, const Vec& b) {
        assert(b.size() == n_ && "rhs must have length n");
        assert(stored_rhs_.size() == n_ && "stored_rhs_ must have length n");
        assert(x.size() == n_ && "solution must have length n");

        logical_to_stored(b, stored_rhs_);
        QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(),
                    stored_rhs_.data());
        stored_to_logical(stored_rhs_, x);
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
        return (D_.array() > atol).count() == n_pos &&
               (D_.array() < -atol).count() == n_neg;
    }

    /**
     * Access the stored upper-triangular matrix
     *
     * @return Mutable stored matrix
     */
    SMat& matrix() { return K_; }

    /**
     * Access the stored upper-triangular matrix
     *
     * @return Stored matrix
     */
    const SMat& matrix() const { return K_; }

    /**
     * Access the stored-to-logical permutation
     *
     * @return Permutation indices
     */
    const IVec& perm() const { return perm_; }

    /**
     * Access the logical-to-stored inverse permutation
     *
     * @return Inverse permutation indices
     */
    const IVec& iperm() const { return iperm_; }

    /**
     * Access the logical-indexed scaling vector
     *
     * @return Scaling vector
     */
    const Vec& scaling() const { return s_; }

    /**
     * Convert a stored vector to logical coordinates
     *
     * @param x_stored Stored-coordinate vector
     * @param x_logical Output logical-coordinate vector
     */
    void stored_to_logical(const Vec& x_stored, Vec& x_logical) const {
        x_logical = s_.cwiseProduct(x_stored(iperm_));
    }

    /**
     * Convert a logical vector to stored coordinates
     *
     * @param x_logical Logical-coordinate vector
     * @param x_stored Output stored-coordinate vector
     */
    void logical_to_stored(const Vec& x_logical, Vec& x_stored) const {
        x_stored = (x_logical.cwiseProduct(s_))(perm_);
    }

private:
    friend class Block;

    Index n_;
    bool built_ = false;
    SMat K_;
    IVec perm_;
    IVec iperm_;
    Vec s_;
    std::vector<std::unique_ptr<Block>> blocks_;

    /**
     * Map the stored nonzero value array
     *
     * @return Mutable vector view over K_ values
     */
    Eigen::Map<Vec> nz() {
        return Eigen::Map<Vec>(K_.valuePtr(), K_.nonZeros());
    }

    /**
     * Find the stored value index of an upper-triangular stored entry
     *
     * @param a Stored row index
     * @param b Stored column index
     * @return Index into K_.valuePtr()
     */
    Index data_index(Index a, Index b) const {
        const Index* outer = K_.outerIndexPtr();
        const Index* inner = K_.innerIndexPtr();
        const Index* lo = inner + outer[b];
        const Index* hi = inner + outer[b + 1];
        const Index* p = std::lower_bound(lo, hi, a);
        assert(p != hi && *p == a && "logical entry is not in the stored pattern");
        return static_cast<Index>(p - inner);
    }

    /**
     * Find the stored value index of a logical matrix entry
     *
     * @param i Logical row index
     * @param j Logical column index
     * @return Index into K_.valuePtr()
     */
    Index data_index_logical(Index i, Index j) const {
        Index a = iperm_[i];
        Index b = iperm_[j];
        if (a > b) std::swap(a, b);
        return data_index(a, b);
    }

    /**
     * Resolve a declared block into stored indices and scaling factors
     *
     * @param b Block to resolve
     */
    void resolve(Block& b) {
        const Index m = static_cast<Index>(b.coords_.size());
        b.idx_.resize(m);
        b.scale_.resize(m);
        for (Index k = 0; k < m; ++k) {
            const Index i = b.coords_[k].first;
            const Index j = b.coords_[k].second;
            b.idx_[k] = data_index_logical(i, j);
            b.scale_[k] = s_[i] * s_[j];
        }
        b.coords_.clear();
        b.coords_.shrink_to_fit();
    }

    /**
     * Build QDLDL symbolic factorization data for the fixed structure
     */
    void symbolic_factorize() {
        const QDLDL_int nnz = static_cast<QDLDL_int>(K_.nonZeros());
        Ap_.resize(n_ + 1);
        Ai_.resize(nnz);
        for (Index k = 0; k <= n_; ++k) Ap_[k] = K_.outerIndexPtr()[k];
        for (QDLDL_int k = 0; k < nnz; ++k) Ai_[k] = K_.innerIndexPtr()[k];

        etree_.resize(n_);
        Lnz_.resize(n_);
        iwork_.resize(3 * n_);
        bwork_.resize(n_);
        fwork_.resize(n_);
        Lp_.resize(n_ + 1);
        D_.resize(n_);
        Dinv_.resize(n_);

        sum_Lnz_ = QDLDL_etree(n_, Ap_.data(), Ai_.data(),
                               iwork_.data(), Lnz_.data(), etree_.data());
        assert(sum_Lnz_ >= 0 && "LdltSystem: not a valid upper-triangular CSC");
        Li_.resize(sum_Lnz_);
        Lx_.resize(sum_Lnz_);
    }

    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> Ap_, Ai_;
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> etree_, Lnz_, iwork_, Lp_, Li_;
    std::vector<QDLDL_bool> bwork_;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> fwork_, Lx_, D_, Dinv_;
    QDLDL_int sum_Lnz_ = 0;
    Vec stored_rhs_;
};
