import numpy as np
import random
import time
import datetime
from matplotlib.image import imread
import utils

def ga(
    s,
    e,
    num_nodes,
    num_agents,
    dim,
    threshold,
    Y_init=None,
    YMIN=0.0,
    YMAX=1.0,
    pop_size=100,
    generations=50,
    mutation_rate=0.1,
    verbose=False,
):
    start_time = time.time()
    # === Chromosome Encoding ===
    # [y1_x, y1_y, ..., yM_x, yM_y, η values flattened]
    T = num_nodes + 1

    def random_policy():
        eta = np.zeros((num_agents, T, num_nodes + 1), dtype=int)
        for a in range(num_agents):
            for t in range(T):
                if t == T - 1:
                    eta[a, t, num_nodes] = 1  # force end node
                else:
                    j = np.random.randint(0, num_nodes)
                    eta[a, t, j] = 1
        return eta

    def encode(y, eta):
        return np.concatenate([y.flatten(), eta.flatten()])

    def decode(chrom):
        y_flat = chrom[: num_nodes * dim]
        eta_flat = chrom[num_nodes * dim :]
        y = y_flat.reshape(num_nodes, dim)
        eta = eta_flat.reshape(num_agents, T, num_nodes + 1)
        return y, eta

    # === Cost Function ===
    def cost_function(chrom):
        y, eta = decode(chrom)
        total = 0
        for a in range(num_agents):
            prev = s[a, 0]
            for t in range(T):
                selected_j = np.argmax(eta[a, t])
                if selected_j < num_nodes:
                    curr = y[selected_j]
                else:
                    curr = e[a, 0]
                dist = np.sum((curr - prev) ** 2) ** 0.5
                total += dist
                total += utils.penalty(dist - threshold)
                prev = curr
        return total

    # === Initial Population ===
    population = []
    for _ in range(pop_size):
        if Y_init is not None:
            y_init = Y_init[0]
        else:
            y_init = np.random.uniform(YMIN, YMAX, (num_nodes, dim))
        eta_init = random_policy()
        chrom = encode(y_init, eta_init)
        population.append(chrom)

    # === GA Main Loop ===
    for gen in range(generations):
        fitness = np.array([cost_function(ind) for ind in population])
        sorted_idx = np.argsort(fitness)
        population = [population[i] for i in sorted_idx]

        # tournament size
        elites = population[:5]  # keep top 5

        # Crossover (uniform)
        new_population = elites.copy()
        while len(new_population) < pop_size:
            p1, p2 = random.sample(elites, 2)
            mask = np.random.rand(len(p1)) < 0.5 # crossover
            child = np.where(mask, p1, p2)

            # Mutation
            for i in range(num_nodes * dim):
                if np.random.rand() < mutation_rate:
                    child[i] = np.random.uniform(YMIN, YMAX)

            # Mutation in η space (single-point switch)
            eta_start = num_nodes * dim
            for a in range(num_agents):
                for t in range(T):
                    if np.random.rand() < mutation_rate:
                        base = eta_start + (a * T + t) * (num_nodes + 1)
                        new_j = (
                            num_nodes if t == T - 1 else np.random.randint(0, num_nodes)
                        )
                        child[base : base + num_nodes + 1] = 0
                        child[base + new_j] = 1
            new_population.append(child)

        population = new_population

        if verbose and (gen % 50 == 0 or gen == generations - 1):
            print(
                f"Generation {gen+1}/{generations}, Best Cost: {fitness[sorted_idx[0]]:.4f}"
            )

    # === Best Solution ===
    best_y, best_eta = decode(population[0])
    best_cost = cost_function(population[0])
    elapsed_time = time.time() - start_time
    if verbose:
        print(f"Finished in {elapsed_time:.2f}s. Best Cost: {best_cost:.4f}")
    return best_y, best_eta, best_cost, elapsed_time


def ga_print(s, e, best_y, best_eta, best_cost, elapsed_time):
    num_agents = best_eta.shape[0]
    T = best_eta.shape[1]
    num_nodes = best_y.shape[0]
    print("GA Solution:")
    print(f"Best Cost: {best_cost:.4f}")
    print(f"Elapsed Time: {elapsed_time:.2f}s")
    print("Node positions (y):")
    for i, pos in enumerate(best_y):
        print(f"  Node {i}: {pos}")
    for a in range(num_agents):
        path = []
        prev = s[a, 0]
        print(f"Agent {a} path:")
        for t in range(T):
            selected_j = np.argmax(best_eta[a, t])
            if selected_j < num_nodes:
                curr = best_y[selected_j]
                path.append(f"Node {selected_j}")
            else:
                curr = e[a, 0]
                path.append("End")
            prev = curr
        print("  -> ".join(path))


def ga_plot(s, e, best_y, best_eta):
    import matplotlib.pyplot as plt
    # map_img = imread("Benchmark/map.jpg")
    # img_extent = [-0.1, 1.1, -0.1, 1.1]
    plt.figure(figsize=(6, 6))

    # plt.imshow(map_img, extent=img_extent, origin='lower', alpha=0.4)
    num_agents = best_eta.shape[0]
    T = best_eta.shape[1]
    num_nodes = best_y.shape[0]
    dim = best_y.shape[1]
    colors = ["#5B33F9", "#DE4141", "g", "m", "c", "y", "k"]
    
    # Plot nodes
    plt.scatter(best_y[:, 0], best_y[:, 1], c="k", marker="^", s=200, label="Nodes",zorder = 5)
    for i, pos in enumerate(best_y):
        plt.text(pos[0]+0.01, pos[1]+0.01, f"$Y_{i+1}$", fontsize=30, color="k")
    # Plot start/end
    for a in range(s.shape[0]):
        start_coord = s[a, 0]
        end_coord = e[a, 0]
        plt.scatter(
            start_coord[0],
            start_coord[1],
            c=colors[a % len(colors)],
            marker="s",
            s=200,
            label=f"Start {a}",
            
        )
        plt.text(
            start_coord[0]-0.02,
            start_coord[1] +0.05,
            f"$S_{a+1}$",
            fontsize=30,
            color=colors[a % len(colors)],
        )
        plt.scatter(
            end_coord[0],
            end_coord[1],
            c=colors[a % len(colors)],
            marker="*",
            s=200,
            label=f"End {a}",
        )
        plt.text(
            end_coord[0]-0.04,
            end_coord[1]-0.1,
            f"$\Delta_{a+1}$",
            fontsize=30,
            color=colors[a % len(colors)],
            zorder=5
        )
    # Plot agent paths
    for a in range(num_agents):
        path = [s[a, 0]]
        for t in range(T):
            selected_j = np.argmax(best_eta[a, t])
            if selected_j < num_nodes:
                path.append(best_y[selected_j])
            else:
                path.append(e[a, 0])
        path = np.array(path)
        plt.plot(
            path[:, 0],
            path[:, 1],
            "-",
            c=colors[a % len(colors)],
            label=f"Agent {a} path",
            alpha=0.5,
            linewidth=3,
        )
    # plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.title("GA Solution Paths")
    # plt.grid(True)
    plt.axis('off')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"ga_solution_paths_{timestamp}.png", bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Define s and e as numpy arrays of shape (num_agents, 1, dim)
    np.random.seed(150)  # For reproducibility
    
    num_nodes = 4
    num_agents = 10
    dim = 2
    drones = [
    ((10.0, 5.0), (45.0, 50.0), 0.7),
    ((3.0, 40.0), (50.0, 10.0), 0.5),
    ((20.0, 15.0), (35.0, 35.0), 0.6),
    ((5.0, 30.0), (25.0, 5.0), 0.4),
    ((40.0, 45.0), (10.0, 10.0), 0.8),
    ((30.0, 20.0), (5.0, 35.0), 0.6),
    ((15.0, 10.0), (40.0, 40.0), 0.4),
    ((35.0, 5.0), (10.0, 45.0), 0.5),
    ((25.0, 40.0), (20.0, 10.0), 0.7),
    ((45.0, 15.0), (5.0, 20.0), 0.3)
    ]

    # Extract start and end coordinates
    s = np.array([ [list(start)] for start, _, _ in drones ]) / 50
    e = np.array([ [list(end)] for _, end, _ in drones ]) / 50

    Y_init = np.array([[[25, 25], [25, 25], [25, 25], [25, 25]]]) / 50
    best_y, best_eta, best_cost, elapsed_time = ga(
        s,
        e,
        num_nodes,
        num_agents,
        dim,
        threshold = 0.5,
        Y_init=Y_init,
        verbose=True,
        pop_size=100,
        generations=1000,
        mutation_rate=0.1,
    )
    ga_print(s, e, best_y, best_eta, best_cost, elapsed_time)
    ga_plot(s, e, best_y, best_eta)
