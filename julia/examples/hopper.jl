using Pkg; Pkg.activate(@__DIR__)
using Revise
using JuMP, Marble, NLPModelsJuMP, LinearAlgebra
import MeshCat as mc
import ForwardDiff as FD
using GeometryBasics, CoordinateTransformations, Colors
import Rotations: RotX

# Visualization utility function
function init_vis(vis = mc.Visualizer())
    foot = HyperSphere(Point(0.0, 0.0, 0.0), 0.05)
    head = HyperSphere(Point(0.0, 0.0, 0.0), 0.1)
    body = Cylinder(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, 1.0), 0.025)
    mc.setobject!(vis[:foot], foot, mc.MeshLambertMaterial(color = mc.RGBA(1, 0, 0, 1.0)))
    mc.setobject!(vis[:head], head, mc.MeshLambertMaterial(color = mc.RGBA(1, 0, 0, 1.0)))
    mc.setobject!(vis["body"], body, mc.MeshLambertMaterial(color = mc.RGBA(1,0,0,1.0)))
    step_geom = HyperRectangle(Vec(0, -50, -50.0), Vec(1.0, 100.0, 50))
    mc.setobject!(vis["floor"], step_geom, mc.MeshLambertMaterial(color=RGBA(0, 0.5, 1, 0.7)))
    step_geom = HyperRectangle(Vec(-0.5, 0.5, 0.0), Vec(1.0, 0.25, 0.1))
    mc.setobject!(vis["step1"], step_geom, mc.MeshLambertMaterial(color=RGBA(0, 0.5, 1, 0.7)))
    step_geom = HyperRectangle(Vec(-0.5, 0.75, 0.0), Vec(1.0, 0.25, 0.2))
    mc.setobject!(vis["step2"], step_geom, mc.MeshLambertMaterial(color=RGBA(0, 0.5, 1, 0.7)))
    step_geom = HyperRectangle(Vec(-0.5, 1.0, 0.0), Vec(1.0, 1.0, 0.3))
    mc.setobject!(vis["step3"], step_geom, mc.MeshLambertMaterial(color=RGBA(0, 0.5, 1, 0.7)))
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "orthographic", true)
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "left", -2)
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "right", 2)
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "top", 2)
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "bottom", -2)
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "zoom", 40.0)
    mc.setprop!(vis["/Cameras/default/rotated/<object>"], "enableRotate", false)
    mc.settransform!(vis["/Cameras/default"], Translation(50, 0, 0))
    mc.setprop!(vis["/Grid"], "visible", false)
    mc.setprop!(vis["/Axes"], "visible", false)
end

# Animate utility function
function animate(vis, X, dt)
    anim = mc.Animation(vis, fps = convert(Int, floor(1.0 / dt)))
    for (t,x) in enumerate(X)
        h = 0.25
        r = 0.05
        foot_transform = mc.Translation([0; x[1]; x[2] + r])
        head_transform =  mc.Translation([0; x[3]; x[4] + h + r])
        ps = [0; x[1]; x[2] + r]
        v = [0; x[4] + h + r - (x[2] + r); x[3] - x[1]]
        ang = atan(v[3], v[2])
        body_transform = Translation(ps) ∘ LinearMap(RotX(-ang)) ∘ LinearMap(diagm([1.0, 1.0, norm(v)]))
        mc.atframe(anim, t) do 
            mc.settransform!(vis[:foot], foot_transform)
            mc.settransform!(vis[:head], head_transform)
            mc.settransform!(vis[:body], body_transform)
        end
    end
    mc.setanimation!(vis, anim)
end

# Planar 2-D hopper problem, two equal mass point masses 
# Configuration is stacked position [x; y] of each mass (foot, then head) 
# State is [configuration; velocity], 8 dimensions
# Controls are internal x and y forces (correspond to prismatic joint and linearized revolute)

problem, x, u, f, comp, comp_type = let
    nq, nx, nu, nd = 4, 8, 2, 2 # nd is num dimensions (2 = planar)
    pos_xi, pos_zi = 1, 2 # Indexing for coordinates
    g = 9.81
    grav_comp = [0; -g]
    N, dt, μ = 2, 0.05, 0.1
    Q, Qf = diagm([1e1; 0; 1e1; 0; 1e-1*ones(4)]), diagm([1e3; 0; 1e3; 1e3; 1e-2*ones(4)])
    R, Rd = diagm([1e-3; 1e-3])/90/90,diagm([1e0; 1e0])/90
    speed = 0.04
    x_goal = [[[speed*k; 0.0; speed*k; 0.0; zeros(4)] for k in 0:N-1]..., [speed*N + 0.3; 0.0; speed*N + 0.3; 0.0; zeros(4)]]

    problem = JuMP.Model()

    # Init state, controls, and forces variables
    @variable(problem, x[1:N+1, 1:nx])
    @variable(problem, u[1:N, 1:nu])
    @variable(problem, f[1:N, 1:nd])
    @variable(problem, s_fric[1:N, 1:2])
    @variable(problem, s_step[1:N, 1:12])
    q_foot, q_head = x[:, 1:nd], x[:, nd .+ (1:nd)]
    v_foot, v_head = x[:, 2*nd .+ (1:nd)], x[:, 3*nd .+ (1:nd)]

    # Objective
    @objective(problem, Min, 
        0.5*sum((x[k, :] - x_goal[k])'*Q*(x[k, :] - x_goal[k]) + u[k, :]'*R*u[k, :] for k = 1:N) +    # Tracking
        0.5*(x[end, :] - x_goal[end])'*Qf*(x[end, :] - x_goal[end]) +                                                           # terminal
        0.5*sum((q_foot[k+1, pos_zi] - q_head[k+1, pos_zi])^2*1e1 for k = 1:N) +                    # prismatic joint length
        0.5*sum((u[k + 1, :] - u[k, :])'*Rd*(u[k + 1, :] - u[k, :])*10 for k in 1:N-1) +
        0.5*sum((s_fric[k, :])'*s_fric[k, :]/20/20 for k in 1:N)
        )

    @constraint(problem, x[1, :] .== zeros(8)) # Initial condition

    s_fric = s_fric./20 # Rescale for conditioning

    # Backwards euler dynamics
    u_scale, f_scale = 90, 30
    @constraint(problem, [k=1:N], q_foot[k+1, :] .== q_foot[k, :] + v_foot[k+1, :]*dt);
    @constraint(problem, [k=1:N], q_head[k+1, :] .== q_head[k, :] + v_head[k+1, :]*dt);
    @constraint(problem, [k=1:N], 
                        v_foot[k+1, :]/dt .== v_foot[k, :]/dt + (-[0; g] + f[k, :]*f_scale + (u[k, :]*u_scale + grav_comp)));
    @constraint(problem, [k=1:N], 
                        v_head[k+1, :]/dt .== v_head[k, :]/dt + (-[0; g] - (u[k, :]*u_scale + grav_comp)));

    # steps
    step1_start, step_height1 = 0.5, 0.1
    step_start2, step_height2 = 0.75, 0.1
    step_start3, step_height3 = 1.0, 0.1
    step_end = 2.0
    h1, h1_l1, h1_l2, h2, h2_l1, h2_l2, h3, h3_l1, h3_l2, h4, h4_l1, h4_l2 = [s_step[:, i] for i in 1:12]

    # step stationarity conditions
    @constraint(problem, [k=1:N], q_foot[k+1, pos_xi] - step1_start .== (h1_l1[k, :] - h1_l2[k, :]))
    @constraint(problem, [k=1:N], q_foot[k+1, pos_xi] - step_start2 .== (h2_l1[k, :] - h2_l2[k, :]))
    @constraint(problem, [k=1:N], q_foot[k+1, pos_xi] - step_start3 .== (h3_l1[k, :] - h3_l2[k, :]))
    @constraint(problem, [k=1:N], q_foot[k+1, pos_xi] - step_end .== (h4_l1[k, :] - h4_l2[k, :]))

    # step complementarities
    @constraint(problem, step1_ub_comp1[k=1:N], h1_l1[k] ≥ 0)
    @constraint(problem, step1_ub_comp2[k=1:N], 1 ≥ h1[k])
    @constraint(problem, step1_lb_comp1[k=1:N], h1_l2[k] ≥ 0)
    @constraint(problem, step1_lb_comp2[k=1:N], h1[k] ≥ -1)
    
    @constraint(problem, step2_ub_comp1[k=1:N], h2_l1[k] ≥ 0)
    @constraint(problem, step2_ub_comp2[k=1:N], 1 ≥ h2[k])
    @constraint(problem, step2_lb_comp1[k=1:N], h2_l2[k] ≥ 0)
    @constraint(problem, step2_lb_comp2[k=1:N], h2[k] ≥ -1)
    
    @constraint(problem, step3_ub_comp1[k=1:N], h3_l1[k] ≥ 0)
    @constraint(problem, step3_ub_comp2[k=1:N], 1 ≥ h3[k])
    @constraint(problem, step3_lb_comp1[k=1:N], h3_l2[k] ≥ 0)
    @constraint(problem, step3_lb_comp2[k=1:N], h3[k] ≥ -1)
    
    @constraint(problem, step4_ub_comp1[k=1:N], h4_l1[k] ≥ 0)
    @constraint(problem, step4_ub_comp2[k=1:N], 1 ≥ h4[k])
    @constraint(problem, step4_lb_comp1[k=1:N], h4_l2[k] ≥ 0)
    @constraint(problem, step4_lb_comp2[k=1:N], h4[k] ≥ -1)

    # Signed distance function
    height = q_foot[2:end, pos_zi] - 0.5*(h1 .+ 1)*step_height1 - 0.5*(h2 .+ 1)*step_height2 - 0.5*(h3 .+ 1)*step_height3 + 0.5*(h4 .+ 1)*0.3
    @constraint(problem, [k=1:N], height[k] >= 0)

    # Friction force (TODO: motion)
    @constraint(problem, [k=1:N], q_foot[k+1, pos_xi] - q_foot[k, pos_xi] .== s_fric[k, 1] - s_fric[k, 2])
    @constraint(problem, [k=1:N], μ*f[k, pos_zi] .≥ f[k, pos_xi])
    @constraint(problem, [k=1:N], f[k, pos_xi] .≥ -μ*f[k, pos_zi])
    @constraint(problem, s_fric .≥ 0)

    # Complementarity between normal force and combined position + velocity magnitude
    @constraint(problem, contact_normal_f_comp[k=1:N], f[k, pos_zi] >= 0)
    @constraint(problem, contact_normal_x_comp[k=1:N], height[k] + s_fric[k, 1] + s_fric[k, 2] ≥ 0)

    # Collect complementarities (MUST BE DONE AFTER PROBLEM CONSTRUCTION)
    normal_comp = con_con_complementarities(problem, contact_normal_x_comp, contact_normal_f_comp)
    box_comp = [con_con_complementarities(problem, step1_ub_comp1, step1_ub_comp2)...,
                con_con_complementarities(problem, step1_lb_comp1, step1_lb_comp2)...,
                con_con_complementarities(problem, step2_ub_comp1, step2_ub_comp2)...,
                con_con_complementarities(problem, step2_lb_comp1, step2_lb_comp2)...,
                con_con_complementarities(problem, step3_ub_comp1, step3_ub_comp2)...,
                con_con_complementarities(problem, step3_lb_comp1, step3_lb_comp2)...,
                con_con_complementarities(problem, step4_ub_comp1, step4_ub_comp2)...,
                con_con_complementarities(problem, step4_lb_comp1, step4_lb_comp2)...]
    comp = [normal_comp; box_comp]
    comp_type = [fill((:con, :con), N); fill((:con, :con), 8*N)]

    problem, x, u, f, comp, comp_type
end; 

solver = Marble.Solver();
Marble.setup!(solver, problem, first.(comp), last.(comp), comp_type; 
            verbosity = 1, max_iters = 10000);
prob = solver.problem

# TEMP: test for problem alignment vs hopper_testing.jld2 
using JLD2, SparseArrays
@load joinpath(@__DIR__, "hopper_testing.jld2") H g J_eq J_ineq J_comp c_eq c_ineq c_comp x_inds u_inds f_inds s_inds box_s_inds

# Permutation matrix
x_reorg = [2; 4; 1; 3; 6; 8; 5; 7]
perm = spzeros(length(g), length(prob.cost_gradient))
[perm[CartesianIndex.(i1, i2[x_reorg])] .= 1.0 for (i1, i2) in zip(x_inds, [var_inds(problem)[:x][k+1, :] for k in 1:N])]
[perm[CartesianIndex.(i1, i2)] .= 1.0 for (i1, i2) in zip(u_inds, [var_inds(problem)[:u][k, :] for k in 1:N])]
[perm[CartesianIndex.(i1, i2)] .= 1.0 for (i1, i2) in zip(f_inds, [var_inds(problem)[:f][k, :] for k in 1:N])]
[perm[CartesianIndex.(i1, i2)] .= 1.0 for (i1, i2) in zip(s_inds, [var_inds(problem)[:s_fric][k, :] for k in 1:N])]
[perm[CartesianIndex.(i1, i2)] .= 1.0 for (i1, i2) in zip(box_s_inds, [var_inds(problem)[:s_step][k, :] for k in 1:N])]

@assert norm(perm*prob.cost_gradient - g, Inf) < eps()
@assert norm(perm*prob.cost_hessian*perm' - H, Inf) < eps()

# Test procrustes
pro_A = J_eq
pro_B = prob.J_eq[9:end, :]*perm'
pro_svd = svd(Matrix(pro_B*pro_A'), alg=LinearAlgebra.QRIteration())
pro_R = sparse(pro_svd.U*pro_svd.V')
norm(pro_A - pro_R'*pro_B, Inf)

[pro_R[k, :] = [i == argmax(abs.(pro_R[k, :])) for i in 1:size(pro_R, 2)] for k=1:size(pro_R, 1) if 1 - maximum(abs.(pro_R[k, :])) < 1e-3]

results = Marble.solve!(solver);

z = Marble.z(results);

X = [z[var_inds(problem)[:x]][k, :] for k in 1:N + 1];
U = [z[var_inds(problem)[:u]][k, :].*90 for k in 1:N];
F = [z[var_inds(problem)[:f]][k, :].*30 for k in 1:N];
S = [z[var_inds(problem)[:s_fric]][k, :].*20 for k in 1:N];
init_vis(vis);
animate(vis, X, dt)
