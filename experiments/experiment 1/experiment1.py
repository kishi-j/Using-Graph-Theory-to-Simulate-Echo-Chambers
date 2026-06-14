import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/exp1_frames", exist_ok=True)


def generate_graph():
    sizes = [60, 60, 60]
    p_in = 0.15
    p_out = 0.005

    probs = [[p_in, p_out, p_out],
             [p_out, p_in, p_out],
             [p_out, p_out, p_in]]

    return nx.stochastic_block_model(sizes, probs, seed=42)


def initialize_opinions(G):
    rng = np.random.default_rng(42)
    return {n: rng.uniform(-1, 1) for n in G.nodes()}


def compute_modularity(G):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    return nx.algorithms.community.modularity(G, communities)


def compute_assortativity(G, opinions):
    edges = list(G.edges())
    if len(edges) == 0:
        return 0

    x = np.array([opinions[u] for u, v in edges])
    y = np.array([opinions[v] for u, v in edges])

    if np.std(x) == 0 or np.std(y) == 0:
        return 0

    return np.corrcoef(x, y)[0, 1]


def run_experiment1(steps=150, save_every=15):

    G = generate_graph()
    opinions = initialize_opinions(G)

    rng = np.random.default_rng(0)

    pos = nx.spring_layout(G, seed=42)

    Q_values, r_values, t_values = [], [], []

    fig, ax = plt.subplots(figsize=(6, 6))

    def update(t):
        nonlocal pos

        ax.clear()

        Q = compute_modularity(G)
        r = compute_assortativity(G, opinions)

        Q_values.append(Q)
        r_values.append(r)
        t_values.append(t)

        print(f"Step {t:03d} | Q = {Q:.4f} | r = {r:.4f}")

        if t % save_every == 0 or t == 149:
            plt.figure(figsize=(6, 6))

            nx.draw(
                G,
                pos,
                node_color=[opinions[n] for n in G.nodes()],
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                node_size=50,
                edge_color="lightgray",
                width=0.4
            )

            plt.title(f"Experiment 1 | Step {t} | Q = {Q:.3f} | r = {r:.3f}")
            plt.axis("off")

            plt.savefig(f"results/exp1_frames/frame_{t:04d}.png",
                        bbox_inches="tight",
                        dpi=150)
            plt.close()

        pos = nx.spring_layout(G, pos=pos, iterations=3, seed=t)

        nx.draw(
            G,
            pos,
            node_color=[opinions[n] for n in G.nodes()],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            node_size=50,
            edge_color="lightgray",
            width=0.4,
            ax=ax
        )

        ax.set_title(f"Experiment 1 | Step {t} | Q = {Q:.3f} | r = {r:.3f}")

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)

    ani.save("results/exp1_static_moving.gif", writer="pillow", fps=10)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(t_values, Q_values, label="Q(t)")
    plt.plot(t_values, r_values, label="r(t)")

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Experiment 1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/exp1_Q_r.png", dpi=200)
    plt.show()

    print("Experiment 1 complete")


run_experiment1()