import numpy as np
from time import time
import matplotlib.pyplot as plt
import utils

def aco(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    threshold,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    num_iters=100,
    num_ants=100,
    alpha=1.0,  # pheromone importance
    beta=1.0,  # heuristic importance
    rho=0.1,  # pheromone evaporation
    Q=1.0,  # pheromone deposit weight
    verbose=False,
):
    T = num_nodes + 1

    # === ACO Structures ===
    dim_y = num_nodes * dim
    pheromone_eta = np.ones((num_agents, T, num_nodes + 1))
    heuristic_eta = np.ones_like(pheromone_eta)

    # === Helper Functions ===
    def decode_eta(eta_int):
        eta_int = np.clip(eta_int, 0, num_nodes)
        eta_int[:, -1] = num_nodes
        return eta_int.astype(int)

    def compute_total_cost(y_flat, eta_int):
        y = y_flat.reshape(num_nodes, dim)
        eta_int = decode_eta(eta_int)
        total_cost = 0.0
        penalty = 0.0
        for a in range(num_agents):
            prev = s[a, 0]
            for t in range(T):
                j = eta_int[a, t]
                curr = y[j] if j < num_nodes else e[a, 0]
                dist = np.sum((curr - prev) ** 2) ** 0.5
                total_cost += dist
                total_cost += utils.penalty(dist - threshold)
                prev = curr
            if eta_int[a, -1] != num_nodes: 
                penalty += 1000.0
        return total_cost + penalty

    start_time = time()
    # === ACO Main Loop ===
    best_cost = np.inf
    best_y = None
    best_eta = None

    for it in range(num_iters):
        if Y_init is not None and it == 0:
            ant_ys = np.repeat(Y_init.reshape(1, dim_y), num_ants, axis=0)
        else:
            ant_ys = np.random.uniform(YMIN, YMAX, size=(num_ants, dim_y))
        ant_etas = np.zeros((num_ants, num_agents, T), dtype=int)
        costs = np.zeros(num_ants)

        for k in range(num_ants):
            eta_k = np.zeros((num_agents, T), dtype=int)
            for a in range(num_agents):
                for t in range(T):
                    if t == T - 1:
                        eta_k[a, t] = num_nodes
                    else:
                        probs = (
                            pheromone_eta[a, t, :num_nodes] ** alpha
                            * heuristic_eta[a, t, :num_nodes] ** beta
                        )
                        probs = probs / np.sum(probs)
                        eta_k[a, t] = np.random.choice(num_nodes, p=probs)
            ant_etas[k] = eta_k
            cost = compute_total_cost(ant_ys[k], eta_k)
            costs[k] = cost

            if cost < best_cost:
                best_cost = cost
                best_y = ant_ys[k].copy()
                best_eta = eta_k.copy()

        # === Pheromone Update ===
        pheromone_eta *= 1 - rho
        for k in range(num_ants):
            eta_k = decode_eta(ant_etas[k])
            for a in range(num_agents):
                for t in range(T):
                    j = eta_k[a, t]
                    if j < num_nodes:
                        pheromone_eta[a, t, j] += Q / costs[k]
        if verbose:
            if it % 50 == 0 or it == num_iters - 1:
                print(f"Iter {it:3d} | Best Cost: {best_cost:.4f}")
    elapsed_time = time() - start_time
    if verbose:
        print(f"Best Cost: {best_cost:.4f}, Time: {elapsed_time:.2f}s")
    return best_y, best_eta, best_cost, elapsed_time


def plot_aco(s, e, num_nodes, num_agents, dim, best_y, best_eta):
    y = best_y.reshape(num_nodes, dim)
    eta = best_eta
    plt.figure(figsize=(6, 6))
    colors = ["r", "b", "g", "m", "c"]
    for a in range(num_agents):
        path = [s[a, 0]]
        for t in range(num_nodes):
            j = eta[a, t]
            path.append(y[j])
        path.append(e[a, 0])
        path = np.array(path)
        plt.plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            color=colors[a % len(colors)],
            label=f"Agent {a}",
        )
        plt.scatter(
            s[a, 0, 0], s[a, 0, 1], color=colors[a % len(colors)], marker="s", s=100
        )
        plt.scatter(
            e[a, 0, 0], e[a, 0, 1], color=colors[a % len(colors)], marker="*", s=100
        )
    plt.title("ACO Solution Paths")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_aco(s, e, num_nodes, num_agents, dim, best_y, best_eta):
    y = best_y.reshape(num_nodes, dim)
    print("Optimal node locations:")
    for i in range(num_nodes):
        print(f"Node {i}: {y[i]}")
    print()
    for a in range(num_agents):
        route = []
        for t in range(num_nodes):
            j = best_eta[a, t]
            route.append(f"node {j}")
        route_str = " -> ".join(route)
        print(f"Agent {a}: {route_str}")


if __name__ == "__main__":
    num_nodes = 3
    num_agents = 2
    dim = 2

    s = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )  # shape (num_agents, 1, dim)
    e = np.array(
        [
            [[1.0, 1.0]],
            [[0.0, 1.0]],
        ]
    )  # shape (num_agents, 1, dim)

    
    # Y_init = None
    Y_init = np.array(
        [[[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]]
    )  # shape (1, num_nodes, dim)

    best_y, best_eta, best_cost, elapsed_time = aco(
        s,
        e,
        num_nodes,
        num_agents,
        dim,
        threshold=0.5,
        Y_init=Y_init,
        YMIN=0.0,
        YMAX=1.0,
        num_iters=200,
        num_ants=50,
        alpha=1.0,
        beta=1.0,
        rho=1.0,
        Q=5.0,
        verbose=True,
    )
    print_aco(s, e, num_nodes, num_agents, dim, best_y, best_eta)
    plot_aco(s, e, num_nodes, num_agents, dim, best_y, best_eta)
