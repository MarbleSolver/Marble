import h5py
import numpy as np
import rcqp
import json
import logging
import os


def load_problem(path):
    with h5py.File(path, "r") as f:
        return {k: np.asarray(f[k]).T for k in f.keys()}


def solve_macmpec_lcqp(name):
    data = load_problem(os.path.join(os.path.dirname(__file__), f"../../examples/data/macmpec/h5/{name}.h5"))
    logger.info(f"{data['n_eq']} equality constraints, {data['n_ineq']} inequality constraints, {data['n_comp']} complementarity constraints")
    
    H,      q,      f0     = data["H"],      data["q"],      float(data["f0"].item())
    J_eq,   b_eq           = data["J_eq"],   data["b_eq"]
    J_ineq, b_ineq         = data["J_ineq"], data["b_ineq"]
    J_comp, b_comp         = data["J_comp"], data["b_comp"]

    obj = lambda z: 0.5 * z.T @ H @ z + q.T @ z + f0

    solver  = rcqp.Solver()
    problem = rcqp.Problem(H, q, f0,
                           J_eq,   b_eq,
                           J_ineq, b_ineq,
                           J_comp, b_comp)
    solver.set_problem(problem)
    assert solver.solve(rcqp.SolverOptions()), "Failed to solve problem"

    logger.info(f"\tObjective value: {obj(solver.get_workspace().z)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    json_path = os.path.join(os.path.dirname(__file__), "../../examples/data/macmpec/macmpec_lcqp.json")
    with open(json_path, "r") as f:
        json_data = json.load(f)

    for key in sorted(json_data.keys()):
        logger.info(f"Solving MacMPEC problem: {key}")
        solve_macmpec_lcqp(key)
