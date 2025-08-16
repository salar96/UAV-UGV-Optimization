import numpy as np
import time
import utils

# === Simulated Annealing ===


def sa(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    threshold,
    Y_init=None,
    YMIN=0,
    YMAX=1.0,
    iters=1000,
    T_start=1.0,
    T_end=1e-3,
    alpha=0.995,
    verbose=False,
):
    T = num_nodes + 1

    def initialize_solution():
        if Y_init is not None:
            y = Y_init.copy().reshape(num_nodes, dim)
        else:
            y = np.random.uniform(YMIN, YMAX, size=(num_nodes, dim))
        eta = np.zeros((num_agents, T, num_nodes + 1), dtype=int)
        for a in range(num_agents):
            for t in range(T - 1):
                j = np.random.randint(0, num_nodes)
                eta[a, t, j] = 1
            eta[a, T - 1, num_nodes] = 1  # final step to end
        return y, eta

    def evaluate_cost(y, eta):
        total = 0
        for a in range(num_agents):
            prev = s[a, 0]
            for t in range(T):
                j = np.argmax(eta[a, t])
                curr = y[j] if j < num_nodes else e[a, 0]
                dist = np.sum((curr - prev) ** 2) ** 0.5
                total += dist
                total += utils.penalty(dist - threshold)
                prev = curr
        return total

    def perturb_solution(y, eta, y_sigma=0.1, eta_prob=0.05):
        new_y = y + np.random.normal(0, y_sigma, size=y.shape)
        new_y = np.clip(new_y, YMIN, YMAX)
        new_eta = eta.copy()
        for a in range(num_agents):
            for t in range(T - 1):  # exclude final step
                if np.random.rand() < eta_prob:
                    new_eta[a, t] = 0
                    j = np.random.randint(0, num_nodes)
                    new_eta[a, t, j] = 1
        return new_y, new_eta

    start_time = time.time()
    y, eta = initialize_solution()
    best_y, best_eta = y.copy(), eta.copy()
    best_cost = evaluate_cost(y, eta)
    cost_history = [best_cost]

    T_curr = T_start
    for i in range(iters):
        y_new, eta_new = perturb_solution(y, eta)
        cost_new = evaluate_cost(y_new, eta_new)
        delta = cost_new - best_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / T_curr):
            y, eta = y_new, eta_new
            if cost_new < best_cost:
                best_cost = cost_new
                best_y = y_new.copy()
                best_eta = eta_new.copy()
        cost_history.append(best_cost)
        T_curr *= alpha
        if verbose and (i % max(1, iters // 10) == 0 or i == iters - 1):
            print(
                f"Iteration {i+1}/{iters}, Best Cost: {best_cost:.4f}, Temperature: {T_curr:.4f}"
            )

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Elapsed time: {elapsed_time:.2f} seconds Best Cost: {best_cost:.4f}")
    return best_y, best_eta, best_cost, elapsed_time


def print_sa(s, e, best_y, best_eta):
    print("Final solution:")
    print("Node positions (y):")
    for i, pos in enumerate(best_y):
        print(f"  Node {i}: {pos}")
    print("Agent paths (eta):")
    for a in range(best_eta.shape[0]):
        path = []
        for t in range(best_eta.shape[1]):
            j = np.argmax(best_eta[a, t])
            if j < best_y.shape[0]:
                path.append(f"Node {j}")
            else:
                path.append("End")
        print(f"  Agent {a}: {' -> '.join(path)}")
    print("Start positions (s):")
    for a in range(s.shape[0]):
        print(f"  Agent {a}: {s[a, 0]}")
    print("End positions (e):")
    for a in range(e.shape[0]):
        print(f"  Agent {a}: {e[a, 0]}")


def plot_sa(s, e, best_y, best_eta):
    import matplotlib.pyplot as plt

    colors = ["r", "b", "g", "m", "c", "y", "k"]
    plt.figure(figsize=(6, 6))
    # Plot start and end positions
    for a in range(s.shape[0]):
        plt.scatter(
            *s[a, 0], marker="o", color=colors[a % len(colors)], label=f"Start {a}"
        )
        plt.scatter(
            *e[a, 0], marker="X", color=colors[a % len(colors)], label=f"End {a}"
        )
    # Plot node positions
    for i, pos in enumerate(best_y):
        plt.scatter(*pos, marker="s", color="k", label=f"Node {i}" if i == 0 else None)
    # Plot agent paths
    for a in range(best_eta.shape[0]):
        path = [s[a, 0]]
        for t in range(best_eta.shape[1]):
            j = np.argmax(best_eta[a, t])
            if j < best_y.shape[0]:
                path.append(best_y[j])
            else:
                path.append(e[a, 0])
        path = np.array(path)
        plt.plot(
            path[:, 0],
            path[:, 1],
            color=colors[a % len(colors)],
            label=f"Agent {a} path",
        )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Simulated Annealing Solution")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    num_agents = 20
    num_nodes = 4
    dim = 2
    s = np.random.rand(num_agents, 1, dim)  # Example start positions
    e = np.random.rand(num_agents, 1, dim)  # Example end positions
    best_y, best_eta, best_cost, elapsed_time = sa(
        s,
        e,
        num_nodes,
        num_agents,
        dim,
        threshold=10,
        Y_init=None,
        YMIN=0,
        YMAX=1.0,
        iters=1000,
        verbose=True,
    )
    print_sa(s, e, best_y, best_eta)
    plot_sa(s, e, best_y, best_eta)
