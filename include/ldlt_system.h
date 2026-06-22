#pragma once
//
// KKTSystem — a symmetric saddle-point matrix with a *fixed* sparsity pattern,
// stored in the form actually handed to the factorization:
//
//        K_stored = Pᵀ · S · K · S · P     (upper-triangular CSC)
//
// where S = diag(s) is a FIXED Ruiz equilibration (computed once, externally,
// and supplied to build()) and P is an AMD fill-reducing permutation (P =
// perm.indices(), same convention as Solver::compute_amd_ordering on GitHub).
// Callers think entirely in the *logical* system (natural variable ordering,
// unscaled); see perm_/iperm_ below for the exact index relationships.
//
// Two-phase usage:
//
//   1. DECLARE.  Ask for the Block handles you'll mutate, in their logical
//      locations (a diagonal, a Jacobian block, an arbitrary index set). The
//      union of every declared block's coordinates IS the sparsity pattern —
//      you state each location exactly once.
//
//          KKTSystem K(n);
//          auto& H   = K.add_entries(hessian_coords);
//          auto& Jeq = K.add_entries(jeq_coords);
//          auto& dvv = K.add_diagonal(v_idx);
//
//   2. BUILD, then USE.  build() resolves the pattern, AMD ordering, storage
//      slots, and cached scaling. The handles you already hold become live;
//      every write is then an O(#entries) vectorized scatter — no searching,
//      no scaling math.
//
//          K.build(s);
//          H.set(...);  dvv.set(...);   // same handles, now usable
//
// Requires Eigen >= 3.4 (IndexedView assignment).
//
#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>
#include <vector>

#include "qdldl.h"
#include "utils.h"   // shared Mat/Vec/SMat — single source of truth for the sparse type

class LdltSystem {
public:
    // Derive everything from the shared SMat (defined once in utils.h) so the
    // problem Jacobians, the coordinates/values written via Block, and the stored
    // matrix are all the exact same type — not just coincidentally equal. (QDLDL's
    // long long indices are an internal detail, handled in symbolic_factorize.)
    using SMat    = ::SMat;                 // Eigen::SparseMatrix<double, ColMajor, int>
    using Scalar  = SMat::Scalar;           // double
    using Index   = SMat::StorageIndex;     // int
    using Vec     = ::Vec;                  // Eigen::VectorXd
    using IVec    = Eigen::Matrix<Index, Eigen::Dynamic, 1>;
    using Triplet = Eigen::Triplet<Scalar, Index>;
    using Coords  = std::vector<std::pair<Index, Index>>;

    // ---------------------------------------------------------------------
    //  Block — a handle to a fixed set of logical entries you rewrite each
    //  iteration. Declared before build(), usable after. Values passed in are
    //  LOGICAL/UNSCALED and in declaration order.
    // ---------------------------------------------------------------------
    class Block {
    public:
        Block() = default;

        void set(const Eigen::Ref<const Vec>& v) const {        // overwrite
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) = v.cwiseProduct(scale_);
        }
        void set(Scalar c) const {
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) = c * scale_;
        }
        void add(const Eigen::Ref<const Vec>& v) const {        // accumulate
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) += v.cwiseProduct(scale_);
        }
        void add(Scalar c) const {
            assert(parent_ && parent_->built_ && "use a Block only after build()");
            parent_->nz()(idx_) += c * scale_;
        }

        Index size() const { return static_cast<Index>(coords_.size()
                                            ? coords_.size() : idx_.size()); }

    private:
        friend class LdltSystem;
        LdltSystem* parent_ = nullptr;
        Coords coords_;     // logical entries; kept until build() resolves them
        IVec   idx_;        // positions into K_stored.valuePtr()  (after build)
        Vec    scale_;      // cached s[row]·s[col]                (after build)
    };

    LdltSystem(Index n) : n_(n) {}

    // Handles hold a back-pointer, so the object is pinned.
    LdltSystem(const LdltSystem&)            = delete;
    LdltSystem& operator=(const LdltSystem&) = delete;
    LdltSystem(LdltSystem&&)                 = delete;
    LdltSystem& operator=(LdltSystem&&)      = delete;

    // ---- phase 1: declare blocks (also declares the pattern) -------------
    // General: arbitrary logical (i,j) entries, in the order you'll supply
    // values. Returns a stable handle; do NOT use it until build().
    Block& add_entries(const Coords& coords) {
        assert(!built_ && "declare all blocks before build()");
        blocks_.push_back(std::make_unique<Block>());
        Block& b = *blocks_.back();
        b.parent_ = this;
        b.coords_ = coords;
        return b;
    }

    // Convenience: the diagonal entries (i,i) for the given logical indices.
    Block& add_diagonal(const IVec& logical_idx) {
        Coords c; c.reserve(logical_idx.size());
        for (Index k = 0; k < logical_idx.size(); ++k)
            c.emplace_back(logical_idx[k], logical_idx[k]);
        return add_entries(c);
    }

    // ---- phase 2: finalize pattern, ordering, scaling -------------------
    // `s` is the logical-indexed Ruiz scaling diagonal (length n), computed
    // once externally. After this, every declared Block handle is live.
    void build(const Vec& s) {
        assert(!built_ && "build() once");
        assert(s.size() == n_ && "Ruiz scaling vector must have length n");
        s_ = s;
        stored_rhs_.resize(n_);

        // Pattern = union of all declared blocks, canonicalized to the upper
        // triangle in LOGICAL ordering. AMD's general overload orders on the
        // pattern of A + Aᵀ, so the upper-only matrix yields the correct
        // symmetric ordering — no full matrix needed.
        std::vector<Triplet> pat;
        for (const auto& bp : blocks_)
            for (const auto& c : bp->coords_) {
                Index i = c.first, j = c.second;
                if (i > j) std::swap(i, j);
                pat.emplace_back(i, j, Scalar(0));
            }
        SMat Ku(n_, n_);
        Ku.setFromTriplets(pat.begin(), pat.end());   // sums dups, keeps zeros
        Ku.makeCompressed();

        // AMD fill-reducing ordering on the symmetric pattern. perm_ is AMD's
        // permutation exactly as returned (stored -> logical); iperm_ is its
        // inverse (logical -> stored). This mirrors Solver::compute_amd_ordering
        // on GitHub:
        //     amd_perm_vec  = amd_perm.indices()       // == perm_
        //     amd_iperm_vec[amd_perm_vec[i]] = i       // == iperm_
        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> perm;
        Eigen::AMDOrdering<Index> amd;
        amd(Ku.selfadjointView<Eigen::Upper>(), perm);
        perm_ = perm.indices();
        iperm_.resize(n_);
        for (Index i = 0; i < n_; ++i) iperm_[perm_[i]] = i;

        // Materialize the stored upper-triangular, permuted pattern (zeros).
        std::vector<Triplet> st;
        st.reserve(pat.size());
        for (const auto& t : pat) {
            Index a = iperm_[t.row()], b = iperm_[t.col()];
            if (a > b) std::swap(a, b);
            st.emplace_back(a, b, Scalar(0));
        }
        K_ = SMat(n_, n_);
        K_.setFromTriplets(st.begin(), st.end());
        K_.makeCompressed();

        // Resolve each handle: storage slot + cached scaling, then drop coords.
        built_ = true;
        for (auto& bp : blocks_) resolve(*bp);

        // Symbolic factorization: the structure is now fixed.
        symbolic_factorize();
    }

    bool built() const { return built_; }

    // ---- factorization & solve (QDLDL) ----------------------------------
    // Numeric LDLᵀ factorization of the current stored matrix values. The
    // symbolic phase ran once in build(); call this after each value update.
    // Returns false on a zero pivot (singular).
    bool factorize(int *count=nullptr) {
        const QDLDL_int status = QDLDL_factor(
            n_, Ap_.data(), Ai_.data(), K_.valuePtr(),
            Lp_.data(), Li_.data(), Lx_.data(), D_.data(), Dinv_.data(),
            Lnz_.data(), etree_.data(), bwork_.data(), iwork_.data(), fwork_.data());
        return status != -1;
    }

    // Solve K x = b. `b` is logical/unscaled; `x` is returned logical/unscaled.
    // Requires a prior factorize(). `x` and `b` may alias (decoupled via scratch).
    void solve(Vec& x, const Vec& b) {
        assert(b.size() == n_ && "rhs must have length n");
        assert(stored_rhs_.size() == n_ && "stored_rhs_ must have length n");
        assert(x.size() == n_ && "solution must have length n");
        
        logical_to_stored(b, stored_rhs_);   // stored_rhs_ = Pᵀ S b

        QDLDL_solve(n_, Lp_.data(), Li_.data(), Lx_.data(), Dinv_.data(),
                    stored_rhs_.data());

        stored_to_logical(stored_rhs_, x);  // x = S P stored_solution
    }

    // Inertia test on the LDLᵀ pivots: exactly n_pos positive and n_neg negative
    // (zero pivots count as neither). The partition is caller-supplied since this
    // matrix does not know the primal/dual split.
    bool check_inertia(Index n_pos, Index n_neg, double atol=1e-10) const {
        return (D_.array() > atol).count() == n_pos &&
               (D_.array() < -atol).count() == n_neg;
    }

    // ---- the matrix to factorize, and solve bookkeeping -----------------
    SMat&       matrix()       { return K_; }   // upper-triangular CSC for QDLDL
    const SMat& matrix() const { return K_; }
    const IVec&  perm()    const { return perm_; }    // stored slot -> logical
    const IVec&  iperm()   const { return iperm_; }   // logical -> stored slot
    const Vec&   scaling() const { return s_; }       // logical-indexed Ruiz s

    void stored_to_logical(const Vec& x_stored, Vec& x_logical) const {
        x_logical = s_.cwiseProduct(x_stored(iperm_));
    }

    void logical_to_stored(const Vec& x_logical, Vec& x_stored) const {
        x_stored = (x_logical.cwiseProduct(s_))(perm_);
    }

    // // Map a stored solution back:  x_logical = S · P · x_stored, i.e.
    // // x_logical[j] = s[j] * x_stored[iperm_[j]].
    // void from_stored(const Vec& x_stored, Vec& x_logical) const {
    //     x_logical.resize(n_);
    //     for (Index i = 0; i < n_; ++i)
    //         x_logical[i] = s_[i] * x_stored[iperm_[i]];
    // }

private:
    friend class Block;

    Index n_;
    bool  built_ = false;
    SMat K_;        // stored: P^T S K S P, upper-triangular, compressed

    // AMD permutation, stored both ways (perm_ and iperm_ are inverses):
    //     stored[i]  = logical[perm_[i]]      (perm_  : stored  -> logical)
    //     logical[j] = stored[iperm_[j]]      (iperm_ : logical -> stored)
    //
    // i.e. logical variable r lives at stored slot iperm_[r], and stored slot i
    // holds logical variable perm_[i]. Equivalently, with P = perm.indices():
    // the stored matrix is Pᵀ K P, a logical entry (r,c) lands at stored
    // (iperm_[r], iperm_[c])
    IVec  perm_;     // stored slot i -> logical index perm_[i]
    IVec  iperm_;    // logical index j -> stored slot iperm_[j]
    Vec   s_;        // logical-indexed Ruiz scaling
    std::vector<std::unique_ptr<Block>> blocks_;   // own handles -> stable refs

    // Writable view over the stored value array (pattern is fixed -> stable).
    Eigen::Map<Vec> nz() {
        return Eigen::Map<Vec>(K_.valuePtr(), K_.nonZeros());
    }

    // data-array index of a stored upper entry (a <= b): binary-search col b.
    Index data_index(Index a, Index b) const {
        const Index* outer = K_.outerIndexPtr();
        const Index* inner = K_.innerIndexPtr();
        const Index* lo = inner + outer[b];
        const Index* hi = inner + outer[b + 1];
        const Index* p  = std::lower_bound(lo, hi, a);
        assert(p != hi && *p == a && "logical entry is not in the stored pattern");
        return static_cast<Index>(p - inner);
    }

    Index data_index_logical(Index i, Index j) const {
        Index a = iperm_[i], b = iperm_[j];
        if (a > b) std::swap(a, b);          // canonicalize into upper triangle
        return data_index(a, b);
    }

    void resolve(Block& b) {
        const Index m = static_cast<Index>(b.coords_.size());
        b.idx_.resize(m);
        b.scale_.resize(m);
        for (Index k = 0; k < m; ++k) {
            const Index i = b.coords_[k].first, j = b.coords_[k].second;
            b.idx_[k]   = data_index_logical(i, j);
            b.scale_[k] = s_[i] * s_[j];
        }
        b.coords_.clear();
        b.coords_.shrink_to_fit();
    }

    // Build the elimination tree and size the QDLDL workspace. QDLDL_int is
    // long long while K_ is int-indexed, so we keep long long copies of the
    // (fixed) CSC structure; only values change afterwards.
    void symbolic_factorize() {
        const QDLDL_int nnz = static_cast<QDLDL_int>(K_.nonZeros());
        Ap_.resize(n_ + 1);
        Ai_.resize(nnz);
        for (Index k = 0; k <= n_; ++k)     Ap_[k] = K_.outerIndexPtr()[k];
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

    // QDLDL factorization state (long long indices per QDLDL_int)
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> Ap_, Ai_;   // CSC structure copy
    Eigen::Matrix<QDLDL_int, Eigen::Dynamic, 1> etree_, Lnz_, iwork_, Lp_, Li_;
    std::vector<QDLDL_bool> bwork_;
    Eigen::Matrix<QDLDL_float, Eigen::Dynamic, 1> fwork_, Lx_, D_, Dinv_;
    QDLDL_int sum_Lnz_ = 0;

    // Reusable storage for scratch space in solve()
    Vec stored_rhs_;
};
