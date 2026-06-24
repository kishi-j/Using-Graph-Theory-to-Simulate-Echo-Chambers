import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/exp2_frames", exist_ok=True)


def generate_graph():
    sizes = [60, 60, 60]
    p_in = 0.15
    p_out = 0.005

    probs = [
        [p_in, p_out, p_out],
        [p_out, p_in, p_out],
        [p_out, p_out, p_in],
    ]

    return nx.stochastic_block_model(sizes, probs, seed=42)


def initialize_opinions(G):
    rng = np.random.default_rng(42)
    return {n: rng.uniform(-1, 1) for n in G.nodes()}


def update_opinions(G, opinions, alpha=0.5, epsilon=0.25):
    new_opinions = {}

    for u in G.nodes():
        neighbors = [
            v for v in G.neighbors(u)
            if abs(opinions[u] - opinions[v]) < epsilon
        ]

        if len(neighbors) == 0:
            new_opinions[u] = opinions[u]
        else:
            new_opinions[u] = (1 - alpha) * opinions[u] + alpha * np.mean(
                [opinions[v] for v in neighbors]
            )

    return new_opinions


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


def run_experiment2(steps=150, save_every=15, alpha=0.5, epsilon=0.25):

    G = generate_graph()
    opinions = initialize_opinions(G)

    pos = nx.spring_layout(G, seed=42)

    Q_values, r_values, t_values = [], [], []

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("black")

    def draw_graph(ax, pos, Q, r, t):
        ax.clear()
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="white",
            width=0.6,
            alpha=0.7
        )

        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=[opinions[n] for n in G.nodes()],
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            node_size=50
        )

        ax.set_title(
            f"Step {t} | Q={Q:.3f} | r={r:.3f}",
            color="white"
        )

        ax.set_axis_off()

    def update(t):
        nonlocal pos, opinions

        Q = compute_modularity(G)
        r = compute_assortativity(G, opinions)

        Q_values.append(Q)
        r_values.append(r)
        t_values.append(t)

        opinions = update_opinions(G, opinions, alpha, epsilon)

        pos = nx.spring_layout(G, pos=pos, iterations=3, seed=t)

        draw_graph(ax, pos, Q, r, t)

        if t % save_every == 0 or t == steps - 1:
            fig_save, ax_save = plt.subplots(figsize=(6, 6))
            fig_save.patch.set_facecolor("black")
            ax_save.set_facecolor("black")

            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax_save,
                edge_color="white",
                width=0.6,
                alpha=0.7
            )

            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax_save,
                node_color=[opinions[n] for n in G.nodes()],
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                node_size=50
            )

            ax_save.set_title(
                f"Step {t} | Q={Q:.3f} | r={r:.3f}",
                color="white"
            )

            ax_save.set_axis_off()

            plt.savefig(
                f"results/exp2_frames/frame_{t:04d}.png",
                dpi=150,
                facecolor="black",
                bbox_inches="tight"
            )

            plt.close(fig_save)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)

    ani.save(
        "results/exp2_static_moving.gif",
        writer="pillow",
        fps=10,
        savefig_kwargs={"facecolor": "black"}
    )

    plt.close()

    # FINAL METRICS PLOT (IDENTICAL STYLE TO EXP 1)
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    fig2.patch.set_facecolor("black")
    ax2.set_facecolor("black")

    ax2.plot(t_values, Q_values, label="Q(t)")
    ax2.plot(t_values, r_values, label="r(t)")

    ax2.set_xlabel("Time step", color="white")
    ax2.set_ylabel("Value", color="white")
    ax2.set_title("Experiment 2", color="white")

    ax2.tick_params(colors="white")

    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("results/exp2_Q_r.png", dpi=200, facecolor="black")
    plt.show()

    print("Experiment 2 complete")


run_experiment2()