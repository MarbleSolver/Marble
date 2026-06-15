from __future__ import annotations

import warnings

import numpy as np

from . import _core
from ._core import Filter, FilterEntry, Problem, SolveResult, SolverOptions, Workspace


def _scipy_sparse():
    """Return the ``scipy.sparse`` module, or ``None`` if SciPy is unavailable."""
    try:
        import scipy.sparse as sp
    except ImportError:
        return None
    return sp


def _is_sparse(M) -> bool:
    sp = _scipy_sparse()
    return sp is not None and sp.issparse(M)


def _vec(v) -> np.ndarray:
    """Coerce to a contiguous 1-D float64 array."""
    return np.ascontiguousarray(np.asarray(v, dtype=np.float64)).reshape(-1)


def _dense(A, ncols: int) -> np.ndarray:
    """Coerce a matrix to a dense 2-D float64 array, promoting a 1-D row."""
    if _is_sparse(A):
        return np.ascontiguousarray(A.toarray(), dtype=np.float64)
    a = np.asarray(A, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(0, ncols) if a.size == 0 else a.reshape(1, -1)
    return np.ascontiguousarray(a)


def _csc(A, ncols: int):
    """Coerce any matrix to a scipy CSC float64 matrix."""
    sp = _scipy_sparse()
    if _is_sparse(A):
        return A.tocsc().astype(np.float64, copy=False)
    return sp.csc_matrix(_dense(A, ncols))


class Solver(_core.Solver):
    """Solver for ``min 1/2 x'Q x + q'x + c0`` subject to

    ``J_eq x + c_eq == 0``, ``J_ineq x + c_ineq >= 0`` and the complementarity
    ``0 <= L x + l  ⟂  R x + r >= 0``.
    """

    def setup(self, Q, q, c0=0.0, *, J_eq=None, c_eq=None, J_ineq=None, c_ineq=None,
              L=None, l=None, R=None, r=None, **options):
        """Set up the problem from matrices and vectors.

        Only ``Q`` and ``q`` are required; any omitted constraint block defaults
        to an empty block of the right shape. Solver options (``max_iters``,
        ``convergence_kkt_norm``, ...) are passed as keyword arguments; unknown
        names are ignored with a warning.
        """
        opts = SolverOptions()
        for name, value in options.items():
            if hasattr(opts, name):
                setattr(opts, name, value)
            else:
                warnings.warn(f"Unknown solver option {name!r}; ignoring")

        n = _vec(q).shape[0]
        J_eq   = np.zeros((0, n)) if J_eq   is None else J_eq
        c_eq   = np.zeros(0)      if c_eq   is None else c_eq
        J_ineq = np.zeros((0, n)) if J_ineq is None else J_ineq
        c_ineq = np.zeros(0)      if c_ineq is None else c_ineq
        L = np.zeros((0, n)) if L is None else L
        l = np.zeros(0)      if l is None else l
        R = np.zeros((0, n)) if R is None else R
        r = np.zeros(0)      if r is None else r

        # If any block is sparse build the whole problem from CSC, else dense
        blocks = (Q, J_eq, J_ineq, L, R)
        if any(_is_sparse(b) for b in blocks):
            if _scipy_sparse() is None:
                raise ImportError("SciPy is required to set up a sparse problem")
            conv = _csc
        else:
            conv = _dense
        problem = Problem(
            conv(Q, n),      _vec(q), float(c0),
            conv(J_eq, n),   _vec(c_eq),
            conv(J_ineq, n), _vec(c_ineq),
            conv(L, n),      _vec(l),
            conv(R, n),      _vec(r),
        )
        self.set_problem(problem, opts)
        return self

    @property
    def options(self) -> SolverOptions:
        """The solver's current options."""
        return self.get_options()

    @property
    def problem(self) -> Problem:
        """The problem currently set on the solver."""
        return self.get_problem()
