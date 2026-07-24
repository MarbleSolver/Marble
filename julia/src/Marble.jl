module Marble
    using CxxWrap
    using SparseArrays
    using NLPModels, JuMP, NLPModelsJuMP
    using LinearAlgebra
    using AmplNLReader
    using Preferences

    const _libmarble_julia = @load_preference("libmarble_julia_path",
        joinpath(@__DIR__, "..", "..", "build", "lib", "libmarble_julia"))

    @wrapmodule(() -> _libmarble_julia)

    function __init__()
        @initcxx
    end

    function setup!(solver::Marble.Solver, Q::AbstractMatrix, q::AbstractVector, c0::Real=0.0;
                    J_eq=nothing, b_eq=nothing, J_ineq=nothing, b_ineq=nothing,
                    L=nothing, l=nothing, R=nothing, r=nothing, kwargs...)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)

        n = length(q)
        J_eq   = isnothing(J_eq)   ? zeros(0, n) : J_eq
        b_eq   = isnothing(b_eq)   ? zeros(0)    : b_eq
        J_ineq = isnothing(J_ineq) ? zeros(0, n) : J_ineq
        b_ineq = isnothing(b_ineq) ? zeros(0)    : b_ineq
        L = isnothing(L) ? zeros(0, n) : L
        l = isnothing(l) ? zeros(0)    : l
        R = isnothing(R) ? zeros(0, n) : R
        r = isnothing(r) ? zeros(0)    : r

        prob = Problem(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        Marble.set_problem!(solver, prob, opts)
        return nothing
    end

    # Extra constructor for a Marble.Problem that handles type conversion, either to sparse
    # matrices based by their data or to dense FLoat64 matrices depending on the input. In
    # the sparse case every block is converted to a SparseMatrixCSC
    function Problem(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
        blocks = (Q, J_eq, J_ineq, L, R)
        fvec(v) = collect(Float64, v)
        if any(b -> b isa AbstractSparseMatrix, blocks)
            Qs, Es, Is, Ls, Rs = sparse(Q), sparse(J_eq), sparse(J_ineq), sparse(L), sparse(R)
            return Marble.Problem(size(Qs, 2),
                Qs.colptr, Qs.rowval, Qs.nzval, fvec(q), Float64(c0),
                size(Es, 1), Es.colptr, Es.rowval, Es.nzval, fvec(b_eq),
                size(Is, 1), Is.colptr, Is.rowval, Is.nzval, fvec(b_ineq),
                size(Ls, 1), Ls.colptr, Ls.rowval, Ls.nzval, fvec(l),
                             Rs.colptr, Rs.rowval, Rs.nzval, fvec(r))
        else
            fmat(M) = Matrix{Float64}(M)
            return Marble.Problem(
                fmat(Q), fvec(q), Float64(c0),
                fmat(J_eq), fvec(b_eq), fmat(J_ineq), fvec(b_ineq),
                fmat(L), fvec(l), fmat(R), fvec(r))
        end
    end

    const OPTIONS = [
        :convergence_kkt_norm, :convergence_eq_violation, :convergence_ineq_violation,
        :convergence_comp_violation, :outer_step_kkt_norm,
        :penalty_initial, :penalty_max, :penalty_scaling,
        :relaxation_initial, :relaxation_min, :relaxation_scaling,
        :max_iters, :max_iters_linesearch,
        :gamma_objective, :gamma_constraint, :ruiz_iterations,
        :inertia_warmstart, :comp_init_random, :comp_init_seed,
        :verbosity, :retraction_type, :clamp_hessian, :warmstart
    ]

    function update_settings!(solver::Marble.Solver; kwargs...)
        opts = Marble.options(solver)
        for (opt_name, opt_val) in kwargs
            try
                opt_setter = getfield(Marble, Symbol(opt_name, :!))
                opt_setter(opts, opt_val)
            catch err
                if err isa UndefVarError || err isa MethodError
                    @warn "Option $(opt_name) not found in Marble.SolverOptions. Skipping."
                else
                    rethrow(err)
                end
            end
        end
    end

    function Base.show(io::IO, opts::Marble.SolverOptions)
        title = " Marble Options "
        
        max_key_len = maximum(length, string.(Marble.OPTIONS))
        total_width = max(max_key_len + 25, length(title) + 10)
        pad_left = (total_width - length(title)) ÷ 2
        
        printstyled(io, "─"^total_width * "\n", color=:light_black)
        printstyled(io, " "^pad_left * title * "\n", bold=true, color=:cyan)
        printstyled(io, "─"^total_width * "\n", color=:light_black)
        for opt in Marble.OPTIONS
            opt_val = getfield(Marble, opt)(opts)
            key_str = rpad(string(opt), max_key_len)
            print(io, "  ", key_str, " ")
            printstyled(io, "│ ", color=:light_black)
            println(io, opt_val)
        end
        printstyled(io, "─"^total_width * "\n", color=:light_black)
    end

    ## Helper for converting Eigen::SparseMatrixCSC to Julia SparseMatrixCSC
    function sparse_from_eigen(nrows, ncols, colptr::Vector, rowval::Vector, nzval::Vector{Float64})
        return SparseMatrixCSC(nrows, ncols, colptr .+ 1, rowval .+ 1, nzval)
    end

    ## Accessors for solver members
    Base.getproperty(solver::Marble.Solver, sym::Symbol) = solver_property(solver, Val(sym))
    solver_property(solver::Marble.Solver, ::Val{:options}) = CxxWrap.dereference_argument(Marble.options(solver))
    solver_property(solver::Marble.Solver, ::Val{:problem}) = CxxWrap.dereference_argument(Marble.get_problem(solver))
    solver_property(solver::Marble.Solver, ::Val{:workspace}) = CxxWrap.dereference_argument(Marble.get_workspace(solver))
    solver_property(solver::Marble.Solver, ::Val{:z_inds}) = Marble.z_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{:s_ineq_inds}) = Marble.s_ineq_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{:s_comp_inds}) = Marble.s_comp_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{:m_eq_inds}) = Marble.m_eq_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{:m_ineq_inds}) = Marble.m_ineq_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{:m_comp_L_inds}) = Marble.m_comp_L_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{:m_comp_R_inds}) = Marble.m_comp_R_inds(solver) .+ 1
    solver_property(solver::Marble.Solver, ::Val{sym}) where sym = getfield(solver, sym)

    Base.propertynames(solver::Marble.Solver, private::Bool=false) = 
        (:options, :problem, :z_inds, :s_ineq_inds, :s_comp_inds, :m_eq_inds, :m_ineq_inds,
         :m_comp_L_inds, :m_comp_R_inds, fieldnames(typeof(solver))...)

    ## Accessors for problem members
    Base.getproperty(problem::Marble.Problem, sym::Symbol) = problem_property(problem, Val(sym))
    problem_property(problem::Marble.Problem, ::Val{:nz}) = Marble.nz(problem)
    problem_property(problem::Marble.Problem, ::Val{:n_eq}) = Marble.n_eq(problem)
    problem_property(problem::Marble.Problem, ::Val{:n_ineq}) = Marble.n_ineq(problem)
    problem_property(problem::Marble.Problem, ::Val{:n_comp}) = Marble.n_comp(problem)
    problem_property(problem::Marble.Problem, ::Val{:cost_hessian}) = sparse_from_eigen(Marble.cost_hessian(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:cost_gradient}) = Marble.cost_gradient(problem)
    problem_property(problem::Marble.Problem, ::Val{:cost_const}) = Marble.cost_const(problem)
    problem_property(problem::Marble.Problem, ::Val{:J_eq}) = sparse_from_eigen(Marble.J_eq(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:c_eq}) = Marble.c_eq(problem)
    problem_property(problem::Marble.Problem, ::Val{:J_ineq}) = sparse_from_eigen(Marble.J_ineq(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:c_ineq}) = Marble.c_ineq(problem)
    problem_property(problem::Marble.Problem, ::Val{:L_comp}) = sparse_from_eigen(Marble.L_comp(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:l_comp}) = Marble.l_comp(problem)
    problem_property(problem::Marble.Problem, ::Val{:R_comp}) = sparse_from_eigen(Marble.R_comp(problem)...)
    problem_property(problem::Marble.Problem, ::Val{:r_comp}) = Marble.r_comp(problem)
    problem_property(problem::Marble.Problem, ::Val{sym}) where sym = getfield(problem, sym)

    Base.propertynames(problem::Marble.Problem, private::Bool=false) = 
        (:nz, :n_eq, :n_ineq, :n_comp, :cost_hessian, :cost_gradient, :cost_const, :J_eq, :c_eq,
         :J_ineq, :c_ineq, :L_comp, :l_comp, :R_comp, :r_comp, fieldnames(typeof(problem))...)

    ## Accessors for workspace members
    Base.getproperty(workspace::Marble.Workspace, sym::Symbol) = workspace_property(workspace, Val(sym))
    workspace_property(workspace::Marble.Workspace, ::Val{:solution}) = Marble.solution(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:z}) = Marble.z(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:s_ineq}) = Marble.s_ineq(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:s_comp}) = Marble.s_comp(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_eq}) = Marble.m_eq(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_ineq}) = Marble.m_ineq(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_comp_L}) = Marble.m_comp_L(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_comp_R}) = Marble.m_comp_R(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_eq_est}) = Marble.m_eq_est(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_ineq_est}) = Marble.m_ineq_est(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_comp_L_est}) = Marble.m_comp_L_est(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:m_comp_R_est}) = Marble.m_comp_R_est(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:kkt_residual}) = Marble.kkt_residual(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:dkkt_residual_drelax}) = Marble.dkkt_residual_drelax(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:kkt_system}) = sparse_from_eigen(Marble.kkt_system(workspace)...)
    workspace_property(workspace::Marble.Workspace, ::Val{:amd_perm_vec}) = Marble.amd_perm_vec(workspace) .+ 1
    workspace_property(workspace::Marble.Workspace, ::Val{:amd_iperm_vec}) = Marble.amd_iperm_vec(workspace) .+ 1
    workspace_property(workspace::Marble.Workspace, ::Val{:scaling}) = Marble.scaling(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:relax_param}) = Marble.relax_param(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{:penalty_param}) = Marble.penalty_param(workspace)
    workspace_property(workspace::Marble.Workspace, ::Val{sym}) where sym = getfield(workspace, sym)

    Base.propertynames(workspace::Marble.Workspace, private::Bool=false) = 
        (:kkt_residual, :dkkt_residual_drelax, :kkt_system, :solution, :z, :s_ineq, :s_comp, :m_eq, :m_ineq, :m_comp_L, :m_comp_R, :amd_perm_vec, :amd_iperm_vec, :scaling,
         :m_eq_est, :m_ineq_est, :m_comp_L_est, :m_comp_R_est, :relax_param, :penalty_param, fieldnames(typeof(workspace))...)

    ## Accessors for result members
    Base.getproperty(result::Marble.SolveResult, sym::Symbol) = result_property(result, Val(sym))
    result_property(result::Marble.SolveResult, ::Val{:converged}) = Marble.converged(result)
    result_property(result::Marble.SolveResult, ::Val{:iterations}) = Marble.iterations(result)
    result_property(result::Marble.SolveResult, ::Val{:iterations_outer}) = Marble.iterations_outer(result)
    result_property(result::Marble.SolveResult, ::Val{:iterations_inner}) = Marble.iterations_inner(result)
    result_property(result::Marble.SolveResult, ::Val{:factorizations}) = Marble.factorizations(result)
    result_property(result::Marble.SolveResult, ::Val{:z}) = Marble.z(result)
    result_property(result::Marble.SolveResult, ::Val{:s_ineq}) = Marble.s_ineq(result)
    result_property(result::Marble.SolveResult, ::Val{:s_comp}) = Marble.s_comp(result)
    result_property(result::Marble.SolveResult, ::Val{:m_eq}) = Marble.m_eq(result)
    result_property(result::Marble.SolveResult, ::Val{:m_ineq}) = Marble.m_ineq(result)
    result_property(result::Marble.SolveResult, ::Val{:m_comp_L}) = Marble.m_comp_L(result)
    result_property(result::Marble.SolveResult, ::Val{:m_comp_R}) = Marble.m_comp_R(result)
    result_property(result::Marble.SolveResult, ::Val{:setup_time_s}) = Marble.setup_time_s(result)
    result_property(result::Marble.SolveResult, ::Val{:solve_time_s}) = Marble.solve_time_s(result)
    result_property(result::Marble.SolveResult, ::Val{sym}) where sym = getfield(result, sym)

    Base.propertynames(result::Marble.SolveResult, private::Bool=false) =
        (:converged, :iterations, :iterations_outer, :iterations_inner, :factorizations, :z, :s_ineq, :s_comp, :m_eq, :m_ineq, :m_comp_L, :m_comp_R,
         :setup_time_s, :solve_time_s, fieldnames(typeof(result))...)

    # Helper functions for evaluating objectives and residuals
    obj(solver::Marble.Solver, z::Vector{Float64}) = Marble.obj(Marble.get_problem(solver), z)
    obj(data::NamedTuple, z::Vector{Float64}) = Marble.obj(Problem(data...), z)

    residual_eq(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_eq(Marble.get_problem(solver), z)
    residual_eq(data::NamedTuple, z::Vector{Float64}) = Marble.residual_eq(Problem(data...), z)

    residual_ineq(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_ineq(Marble.get_problem(solver), z)
    residual_ineq(data::NamedTuple, z::Vector{Float64}) = Marble.residual_ineq(Problem(data...), z)

    residual_comp(solver::Marble.Solver, z::Vector{Float64}) = Marble.residual_comp(Marble.get_problem(solver), z)
    residual_comp(data::NamedTuple, z::Vector{Float64}) = Marble.residual_comp(Problem(data...), z)

    # Helper function for getting KKT system 
    function get_kkt_system(workspace::Marble.Workspace)
        kkt_mat = workspace.kkt_system # upper triangular
        kkt_mat = kkt_mat + kkt_mat' - Diagonal(diag(kkt_mat)) # symmetric
        scale_mat = spdiagm(1 ./workspace.scaling)
        kkt_mat = scale_mat*kkt_mat[workspace.amd_iperm_vec, workspace.amd_iperm_vec]*scale_mat
        return kkt_mat
    end

    # Problem construction with JuMP: MPCC, complementarity indexing, and
    # the conversion of an MPCC into the matrix/vector data Marble consumes.
    include("jump_mpcc.jl")

    export MPCC, complementarity_indices
    export reformulate_sos1, var_inds
    export mpcc_to_marble, get_kkt_system

    # Build a Marble problem from an MPCC (defined here since it depends on
    # both MPCC/mpcc_to_marble above and the build_problem helper).
    function setup!(solver::Marble.Solver, mpcc::MPCC; kwargs...)
        Marble.update_settings!(solver; kwargs...)
        opts = Marble.options(solver)
        data = mpcc_to_marble(mpcc)
        prob = Problem(data.Q, data.q, data.c0,
                       data.J_eq, data.b_eq, data.J_ineq, data.b_ineq, data.L, data.l, data.R, data.r)
        Marble.set_problem!(solver, prob, opts)
        return nothing
    end
end
