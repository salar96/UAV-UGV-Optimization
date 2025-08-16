import pyomo.environ as pyo
import numpy as np
import time


def SolveMIP(s_data, e_data, num_agents, num_nodes, T, dim, threshold, y_init=None, gap=0.05):

    # === Convert start & end to dicts ===
    if isinstance(s_data, np.ndarray):
        s_dict = {
            (a, d): float(s_data[a, 0, d])
            for a in range(num_agents)
            for d in range(dim)
        }
    else:
        s_dict = s_data

    if isinstance(e_data, np.ndarray):
        e_dict = {
            (a, d): float(e_data[a, 0, d])
            for a in range(num_agents)
            for d in range(dim)
        }
    else:
        e_dict = e_data

    # === Initial node locations ===
    if y_init is not None:
        assert y_init.shape == (1, num_nodes, dim), "y_init must have shape (1, num_nodes, dim)"
        y_init_dict = {(j, d): float(y_init[0, j, d]) for j in range(num_nodes) for d in range(dim)}
    else:
        y_init_dict = {(j, d): 0.5 for j in range(num_nodes) for d in range(dim)}

    # === Model ===
    model = pyo.ConcreteModel()
    model.A = pyo.RangeSet(0, num_agents - 1)
    model.N = pyo.RangeSet(0, num_nodes - 1)  # real shared nodes
    model.J = pyo.RangeSet(0, num_nodes)      # includes virtual end index = num_nodes
    model.T = pyo.RangeSet(1, T)

    # === Params for start and end locations ===
    model.s = pyo.Param(model.A, range(dim), initialize=s_dict, mutable=False)
    model.e = pyo.Param(model.A, range(dim), initialize=e_dict, mutable=False)

    # === Variables ===
    model.y = pyo.Var(
        model.N, range(dim),
        domain=pyo.Reals,
        bounds=(0.05, 0.95),
        initialize=y_init_dict,
    )
    model.eta = pyo.Var(model.A, model.T, model.J, domain=pyo.Binary)

    # Motion difference variables
    model.xdiff = pyo.Var(model.A, model.T, range(dim), domain=pyo.Reals)
    # Euclidean distance per step
    model.dist = pyo.Var(model.A, model.T, domain=pyo.NonNegativeReals)

    # === Constraints ===
    # One choice per timestep
    def one_choice(m, a, t):
        return sum(m.eta[a, t, j] for j in m.J) == 1
    model.one_choice = pyo.Constraint(model.A, model.T, rule=one_choice)

    # No revisit of real nodes
    model.no_revisit = pyo.Constraint(
        model.A, model.N,
        rule=lambda m, a, j: sum(m.eta[a, t, j] for t in m.T) <= 1
    )

    # End stickiness
    model.end_stickiness = pyo.ConstraintList()
    for a in range(num_agents):
        for t in range(1, T):
            model.end_stickiness.add(model.eta[a, t, num_nodes] <= model.eta[a, t + 1, num_nodes])

    # Symmetry-breaking
    model.sym_idx = pyo.RangeSet(0, num_nodes - 2)
    model.sym_x = pyo.Constraint(model.sym_idx, rule=lambda m, j: m.y[j, 0] <= m.y[j + 1, 0])
    model.sym_lex = pyo.Constraint(
        model.sym_idx,
        rule=lambda m, j: m.y[j, 1] <= m.y[j + 1, 1] + (m.y[j + 1, 0] - m.y[j, 0])
    )

    # === Coordinate helper ===
    def coord(m, a, j, d):
        return m.y[j, d] if j < num_nodes else m.e[a, d]

    # Delta definition
    def delta_def(m, a, t, d):
        pos_t = sum(m.eta[a, t, j] * coord(m, a, j, d) for j in m.J)
        if t == 1:
            pos_tm1 = m.s[a, d]
        else:
            pos_tm1 = sum(m.eta[a, t - 1, j] * coord(m, a, j, d) for j in m.J)
        return m.xdiff[a, t, d] == pos_t - pos_tm1
    model.delta_def = pyo.Constraint(model.A, model.T, range(dim), rule=delta_def)

    # SOC: dist >= sqrt(x^2 + y^2)
    def soc_rule(m, a, t):
        return m.dist[a, t] ** 2 >= m.xdiff[a, t, 0] ** 2 + m.xdiff[a, t, 1] ** 2
    model.soc = pyo.Constraint(model.A, model.T, rule=soc_rule)

    # Threshold constraint: each step must be <= threshold
    def max_step_rule(m, a, t):
        return m.dist[a, t] <= threshold
    model.max_step = pyo.Constraint(model.A, model.T, rule=max_step_rule)

    # === Objective: minimize average distance ===
    def total_cost(m):
        return sum(m.dist[a, t] for a in m.A for t in m.T) / num_agents
    model.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)

    # === Solve ===
    start_time = time.time()
    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    solver.options["MIPGap"] = gap
    results = solver.solve(model, tee=True)
    elapsed_time = time.time() - start_time

    # === Extract solution ===
    best_y = np.array([[pyo.value(model.y[j, d]) for d in range(dim)] for j in model.N])
    best_eta = np.array(
        [
            [pyo.value(model.eta[a, t, j]) for j in model.J]
            for a in model.A
            for t in model.T
        ]
    ).reshape(num_agents, T, num_nodes + 1)
    best_cost = pyo.value(model.obj)

    return best_y, best_eta, best_cost, elapsed_time


if __name__ == "__main__":
    # Example usage
    # === Parameters ===
    num_nodes = 4
    num_agents = 2
    dim = 2
    T = num_nodes + 1  # total time steps (1…T)

    # === Start and end locations (data) ===
    s_data = {
        (0, 0): 0.0,
        (0, 1): 0.0,
        (1, 0): 1.0,
        (1, 1): 0.0,
    }
    e_data = {
        (0, 0): 1.0,
        (0, 1): 1.0,
        (1, 0): 0.0,
        (1, 1): 1.0,
    }

    SolveMIP(s_data, e_data, num_agents, num_nodes, T, dim, threshold=10)
