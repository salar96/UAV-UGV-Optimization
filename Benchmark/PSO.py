import numpy as np
import time
import utils

def pso(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    threshold,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    num_particles=100,
    num_iters=300,
    w=0.5, # inertia weight
    c1=1.5, # cognitive (individual) weight
    c2=1.5, # social (group) weight
    verbose=False,
):
    def decode_eta(eta_int, num_nodes):
        eta_int = np.clip(eta_int, 0, num_nodes)
        eta_int[:, -1] = num_nodes
        return eta_int.astype(int)

    def compute_total_cost(s, e, y_flat, eta_int, T, num_agents, num_nodes, dim):
        y = y_flat.reshape(num_nodes, dim)
        eta_int = decode_eta(eta_int, num_nodes)
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
            if eta_int[a, -1] != num_nodes:  # num_nodes encodes depot in your decode
                penalty += 1000.0

        return total_cost + penalty

    T = num_nodes + 1
    start_time = time.time()
    dim_y = num_nodes * dim
    dim_eta = (num_agents, T)

    if Y_init is not None:
        Y_init_flat = Y_init.reshape(-1)
        particles_y = np.tile(Y_init_flat, (num_particles, 1))
        particles_y += np.random.uniform(-0.05, 0.05, size=particles_y.shape)
    else:
        particles_y = np.random.uniform(YMIN, YMAX, size=(num_particles, dim_y))
    velocities_y = np.zeros_like(particles_y)

    particles_eta = np.random.randint(0, num_nodes, size=(num_particles, *dim_eta))
    velocities_eta = np.zeros_like(particles_eta, dtype=float)

    personal_best_y = particles_y.copy()
    personal_best_eta = particles_eta.copy()
    personal_best_cost = np.array(
        [
            compute_total_cost(s, e, y, eta, T, num_agents, num_nodes, dim)
            for y, eta in zip(particles_y, particles_eta)
        ]
    )

    global_best_idx = np.argmin(personal_best_cost)
    global_best_y = personal_best_y[global_best_idx].copy()
    global_best_eta = personal_best_eta[global_best_idx].copy()
    global_best_cost = personal_best_cost[global_best_idx]

    for it in range(num_iters):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities_y[i] = (
                w * velocities_y[i]
                + c1 * r1 * (personal_best_y[i] - particles_y[i])
                + c2 * r2 * (global_best_y - particles_y[i])
            )
            particles_y[i] += velocities_y[i]
            particles_y[i] = np.clip(particles_y[i], YMIN, YMAX)

            r1_eta, r2_eta = np.random.rand(), np.random.rand()
            velocities_eta[i] = (
                w * velocities_eta[i]
                + c1 * r1_eta * (personal_best_eta[i] - particles_eta[i])
                + c2 * r2_eta * (global_best_eta - particles_eta[i])
            )
            particles_eta[i] = np.round(
                particles_eta[i].astype(float) + velocities_eta[i]
            ).astype(int)
            particles_eta[i] = np.clip(particles_eta[i], 0, num_nodes)

            cost = compute_total_cost(
                s, e, particles_y[i], particles_eta[i], T, num_agents, num_nodes, dim
            )
            if cost < personal_best_cost[i]:
                personal_best_cost[i] = cost
                personal_best_y[i] = particles_y[i].copy()
                personal_best_eta[i] = particles_eta[i].copy()
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_y = particles_y[i].copy()
                    global_best_eta = particles_eta[i].copy()
        if verbose:
            if it % 100 == 0 or it == num_iters - 1:
                mean_cost = personal_best_cost.mean()
                print(f"Iter {it:3d} | Best Cost: {global_best_cost:.4f} | Mean Cost: {mean_cost:.4f}")

    elapsed_time = time.time() - start_time
    y_final = global_best_y.reshape(num_nodes, dim)
    eta_final = decode_eta(global_best_eta, num_nodes)
    return y_final, eta_final, global_best_cost, elapsed_time


def pso_print(y, eta, cost, elapsed_time, s, e):
    print("=== PSO Results ===")
    print(f"Best Cost: {cost:.4f}")
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    print("Node positions (y):")
    print(y)
    print("Agent paths (eta):")
    print(eta)
    num_agents = s.shape[0]
    num_nodes = y.shape[0]
    for a in range(num_agents):
        path_str = []
        for t in range(eta.shape[1]):
            j = eta[a, t]
            if j < num_nodes:
                path_str.append(f"Node {j}")
            else:
                path_str.append("End")
        print(f"Agent {a} path:")
        print(" -> ".join(path_str))


def pso_plot(y, eta, s, e):
    import matplotlib.pyplot as plt

    colors = ["r", "b", "g", "m", "c", "y", "k"]
    plt.figure(figsize=(6, 6))
    plt.scatter(y[:, 0], y[:, 1], c="k", marker="o", label="Nodes")
    num_agents = s.shape[0]
    num_nodes = y.shape[0]
    for a in range(num_agents):
        plt.scatter(
            s[a, 0, 0],
            s[a, 0, 1],
            c=colors[a % len(colors)],
            marker="s",
            label=f"Start {a}",
        )
        plt.scatter(
            e[a, 0, 0],
            e[a, 0, 1],
            c=colors[a % len(colors)],
            marker="*",
            label=f"End {a}",
        )
    for a in range(num_agents):
        path = [s[a, 0]]
        for t in range(eta.shape[1]):
            j = eta[a, t]
            if j < num_nodes:
                path.append(y[j])
            else:
                path.append(e[a, 0])
        path = np.array(path)
        plt.plot(
            path[:, 0], path[:, 1], c=colors[a % len(colors)], label=f"Agent {a} path"
        )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("PSO Agent Paths")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    num_agents = 2
    num_nodes = 3
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
    Y_init = np.array(
        [
            [
                [0.3, 0.3],
                [0.7, 0.2],
                [0.5, 0.8],
            ]
        ]
    )
    y_final, eta_final, global_best_cost, elapsed_time = pso(
        s, e, num_nodes, num_agents, dim, threshold=10, Y_init=Y_init, verbose=True
    )
    pso_print(y_final, eta_final, global_best_cost, elapsed_time, s, e)
    pso_plot(y_final, eta_final, s, e)
