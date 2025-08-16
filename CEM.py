import numpy as np
from time import time
import utils
import matplotlib.pyplot as plt


def cem(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    threshold,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    n_iter=100,
    pop_size=100,
    elite_frac=0.2,
    verbose=False,
):
    T = num_nodes + 1

    def evaluate_solution(y, eta):
        total_cost = 0.0
        for a in range(num_agents):
            prev = s[a, 0]
            for t in range(T):
                j = np.argmax(eta[a, t])  # one-hot decoding
                curr = y[j] if j < num_nodes else e[a, 0]
                dist = np.sum((curr - prev) ** 2) ** 0.5
                total_cost += dist
                total_cost += utils.penalty(dist - threshold)
                prev = curr
        return total_cost

    
    start_time = time()
    elite_size = int(pop_size * elite_frac)

    
    if Y_init is not None:
        y_mean = Y_init[0].copy()
    else:
        y_mean = np.random.uniform(YMIN, YMAX, (num_nodes, dim))
    y_std = np.ones((num_nodes, dim)) * 0.5

    eta_logits = np.random.randn(num_agents, T, num_nodes + 1)

    best_cost = np.inf
    best_y = None
    best_eta = None

    for iteration in range(n_iter):
        y_samples = np.random.normal(y_mean, y_std, (pop_size, num_nodes, dim))
        eta_samples = np.zeros((pop_size, num_agents, T, num_nodes + 1), dtype=int)
        costs = np.zeros(pop_size)

        for k in range(pop_size):
            eta = np.zeros((num_agents, T, num_nodes + 1), dtype=int)
            for a in range(num_agents):
                for t in range(T):
                    probs = np.exp(eta_logits[a, t])  # softmax
                    probs /= np.sum(probs)
                    j = np.random.choice(num_nodes + 1, p=probs)
                    eta[a, t, j] = 1
            # Force final step to go to end location
            eta[:, -1, :] = 0
            eta[:, -1, num_nodes] = 1
            cost = evaluate_solution(y_samples[k], eta)
            eta_samples[k] = eta
            costs[k] = cost

            if cost < best_cost:
                best_cost = cost
                best_y = y_samples[k]
                best_eta = eta.copy()

        elite_idxs = np.argsort(costs)[:elite_size]
        y_mean = np.mean(y_samples[elite_idxs], axis=0)
        y_std = np.std(y_samples[elite_idxs], axis=0) + 1e-6

        # Update logits by computing average one-hot vectors of elite samples
        for a in range(num_agents):
            for t in range(T):
                elite_one_hots = eta_samples[elite_idxs, a, t, :]
                avg = np.mean(elite_one_hots, axis=0)
                eta_logits[a, t] = np.log(avg + 1e-6)
        if verbose and (
            iteration % max(1, n_iter // 10) == 0 or iteration == n_iter - 1
        ):
            print(f"Iteration {iteration+1}/{n_iter}, Best Cost: {best_cost:.4f}")

    elapsed_time = time() - start_time
    return best_y, best_eta, best_cost, elapsed_time


def print_cem(s, e, best_y, best_eta, best_cost, elapsed_time):
    print("CEM Solution:")
    print(f"Best Cost: {best_cost:.4f}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print("Node positions (y):")
    print(best_y)
    print("Agent paths (indices):")
    num_agents, T, _ = best_eta.shape
    for a in range(num_agents):
        path = []
        for t in range(T):
            j = np.argmax(best_eta[a, t])
            if j < best_y.shape[0]:
                path.append(f"Node {j}")
            else:
                path.append("End")
        print(f"Agent {a}: {' -> '.join(path)}")


def plot_cem(s, e, best_y, best_eta):
    num_agents, T, _ = best_eta.shape
    plt.figure(figsize=(6, 6))
    # Plot nodes
    plt.scatter(best_y[:, 0], best_y[:, 1], c="blue", label="Nodes")
    # Plot start and end
    for a in range(num_agents):
        plt.scatter(
            s[a, 0, 0],
            s[a, 0, 1],
            c="green",
            marker="s",
            label=f"Start {a}" if a == 0 else None,
        )
        plt.scatter(
            e[a, 0, 0],
            e[a, 0, 1],
            c="red",
            marker="*",
            label=f"End {a}" if a == 0 else None,
        )
    # Plot paths
    for a in range(num_agents):
        path = [s[a, 0]]
        for t in range(T):
            j = np.argmax(best_eta[a, t])
            if j < best_y.shape[0]:
                path.append(best_y[j])
            else:
                path.append(e[a, 0])
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], label=f"Agent {a} Path")
    plt.legend()
    plt.title("CEM Solution Paths")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    num_nodes = 3
    num_agents = 2
    dim = 2
    s = np.array(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    e = np.array(
        [
            [[1.0, 1.0]],
            [[0.0, 1.0]],
        ]
    )
    # Optional initial guess for node positions
    Y_init = np.array([[[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]])

    best_y, best_eta, best_cost, elapsed_time = cem(
        s,
        e,
        num_nodes,
        num_agents,
        dim,
        threshold=10,
        Y_init=Y_init,
        YMIN=0,
        YMAX=1.0,
        n_iter=100,
        pop_size=100,
        elite_frac=0.2,
        verbose=True,
    )
    print_cem(s, e, best_y, best_eta, best_cost, elapsed_time)
    plot_cem(s, e, best_y, best_eta)
