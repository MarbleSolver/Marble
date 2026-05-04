using CxxWrap
using LinearAlgebra
using SparseArrays

# ---------------------------------------------------------------------------
# Load the module
# ---------------------------------------------------------------------------
module RCQP
    using CxxWrap
    @wrapmodule(() -> joinpath(@__DIR__, "../build/librcqp_julia"))
    function __init__()
        @initcxx
    end
end

# ---------------------------------------------------------------------------
# Helper: reconstruct a Julia SparseMatrixCSC from the kkt_system tuple
# ---------------------------------------------------------------------------
function get_kkt(solver)
    ws = RCQP.get_workspace(solver)
    nr, nc, colptr, rowval, nzval = RCQP.kkt_system(ws)
    return SparseMatrixCSC(nr, nc, colptr .+ 1, rowval .+ 1, copy(nzval))
end

# ---------------------------------------------------------------------------
# Test 1 – Problem construction and dimension accessors
# ---------------------------------------------------------------------------
println("Test 1: Problem construction")

nz = 2
H      = Matrix(1.0 * I(nz))   # Identity Hessian
g      = [1.0, 2.0]             # Linear cost
J_empty = zeros(0, nz)
c_empty = zeros(0)

prob = RCQP.Problem(H, g, 0.0, J_empty, c_empty, J_empty, c_empty, J_empty, c_empty)
@assert RCQP.nz(prob)     == 2
@assert RCQP.n_eq(prob)   == 0
@assert RCQP.n_ineq(prob) == 0
@assert RCQP.n_comp(prob) == 0
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 2 – SolverOptions getters and setters
# ---------------------------------------------------------------------------
println("Test 2: SolverOptions")

opts = RCQP.SolverOptions()
RCQP.max_iters!(opts, 500)
RCQP.convergence_kkt_norm!(opts, 1e-6)
RCQP.output_dir!(opts, "/dev/null")
@assert RCQP.max_iters(opts)            == 500
@assert RCQP.convergence_kkt_norm(opts) ≈  1e-6
@assert RCQP.output_dir(opts)           == "/dev/null"
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 3 – FilterEntry construction and accessors
# ---------------------------------------------------------------------------
println("Test 3: FilterEntry")

entry = RCQP.FilterEntry(0.5, 1.2)
@assert RCQP.feas(entry)  ≈ 0.5
@assert RCQP.merit(entry) ≈ 1.2
RCQP.feas!(entry, 0.1)
@assert RCQP.feas(entry)  ≈ 0.1
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 4 – Filter update and acceptability
# ---------------------------------------------------------------------------
println("Test 4: Filter")

filt = RCQP.Filter()
@assert RCQP.size(filt) == 0
RCQP.update(filt, RCQP.FilterEntry(1.0, 1.0))
@assert RCQP.size(filt) == 1
# entries() returns a flat [feas, merit, ...] vector
flat = RCQP.entries(filt)
@assert flat[1] ≈ 1.0 && flat[2] ≈ 1.0
# A candidate strictly better on both axes should be acceptable
@assert RCQP.acceptable(filt, RCQP.FilterEntry(0.5, 0.5))
RCQP.clear(filt)
@assert RCQP.size(filt) == 0
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 5 – Retraction maps
# ---------------------------------------------------------------------------
println("Test 5: Retraction maps")

solver = RCQP.Solver(opts)
RCQP.set_problem(solver, prob, ones(nz))

s = [0.1, 0.2]
r   = RCQP.retract(solver, s, 0.1)
dr  = RCQP.retract_deriv(solver, s, 0.1)
ddr = RCQP.retract_second_deriv(solver, s, 0.1)
@assert length(r)   == nz
@assert length(dr)  == nz
@assert length(ddr) == nz
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 6 – Solve and check solution
# Unconstrained QP:  min  0.5 * x' * I * x + [1,2]' * x
# Optimal solution:  x* = [-1, -2]
# ---------------------------------------------------------------------------
println("Test 6: Solve unconstrained QP")

converged = RCQP.solve(solver, opts)
@assert converged

ws = RCQP.get_workspace(solver)
z  = copy(RCQP.z(ws))
println("  Solution z  = $z")
println("  Expected    = [-1.0, -2.0]")
@assert norm(z - [-1.0, -2.0], Inf) < 1e-3
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 7 – Workspace residuals
# ---------------------------------------------------------------------------
println("Test 7: Workspace residuals")

@assert norm(RCQP.kkt_residual(ws),  Inf) < 1e-5
@assert norm(RCQP.residual_eq(ws),   Inf) < 1e-8
@assert norm(RCQP.residual_ineq(ws), Inf) < 1e-8
@assert norm(RCQP.residual_comp(ws), Inf) < 1e-8
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 8 – KKT index vectors are non-empty and correctly sized
# ---------------------------------------------------------------------------
println("Test 8: KKT index vectors")

@assert length(RCQP.z_inds(solver))    == nz
@assert length(RCQP.m_eq_inds(solver)) == 0   # no equality constraints
println("  n_primals = $(RCQP.n_primals(solver))")
println("  n_duals   = $(RCQP.n_duals(solver))")
println("  n_vars    = $(RCQP.n_vars(solver))")
println("  PASSED")

# ---------------------------------------------------------------------------
# Test 9 – KKT system sparse matrix reconstruction
# ---------------------------------------------------------------------------
println("Test 9: KKT system sparse matrix")

kkt = get_kkt(solver)
@assert size(kkt, 1) == size(kkt, 2)
@assert size(kkt, 1) == RCQP.n_vars(solver)
println("  KKT size: $(size(kkt))")
println("  PASSED")

# ---------------------------------------------------------------------------
println("\nAll tests passed!")
