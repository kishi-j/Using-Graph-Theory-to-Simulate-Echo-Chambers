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

    probs = [
        [p_in, p_out, p_out],
        [p_out, p_in, p_out],
        [p_out, p_out, p_in],
    ]

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

    pos = nx.spring_layout(G, seed=42)

    Q_values, r_values, t_values = [], [], []

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("black")

    def draw_graph(ax, pos, Q, r, t):
        ax.clear()
        ax.set_facecolor("black")
        fig.patch.set_facecolor("black")

        # EDGES FIRST (behind nodes)
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edge_color="white",
            width=0.6,
            alpha=0.7
        )

        # NODES SECOND (on top of edges)
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
        nonlocal pos

        Q = compute_modularity(G)
        r = compute_assortativity(G, opinions)

        Q_values.append(Q)
        r_values.append(r)
        t_values.append(t)

        print(f"Step {t:03d} | Q={Q:.4f} | r={r:.4f}")

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
                f"results/exp1_frames/frame_{t:04d}.png",
                dpi=150,
                facecolor="black",
                bbox_inches="tight"
            )
            plt.close(fig_save)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)

    ani.save(
        "results/exp1_static_moving.gif",
        writer="pillow",
        fps=10,
        savefig_kwargs={"facecolor": "black"}
    )

    plt.close()

    print("Experiment 1 complete")


run_experiment1()