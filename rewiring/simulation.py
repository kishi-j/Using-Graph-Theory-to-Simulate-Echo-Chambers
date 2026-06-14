import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/gifs", exist_ok=True)

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
    centers = [-0.7, 0.0, 0.7]
    opinions = {}
    for n in G.nodes():
        cluster = G.nodes[n]["block"]
        opinions[n] = np.clip(centers[cluster] + rng.normal(0, 0.3), -1, 1)
    return opinions

def update_bounded(G, opinions, epsilon=0.25, alpha=0.05):
    new_opinions = {}
    for u in G.nodes():
        neighbors = [v for v in G.neighbors(u) if abs(opinions[u] - opinions[v]) < epsilon]
        if not neighbors:
            new_opinions[u] = opinions[u]
            continue
        avg = np.mean([opinions[v] for v in neighbors])
        new_opinions[u] = (1 - alpha) * opinions[u] + alpha * avg
    return new_opinions

# Edge rewiring — drop dissimilar edges, gain similar ones
# rewire_frac: fraction of edges to consider rewiring each step
# similarity_threshold: opinion difference above which an edge may be dropped

def rewire_edges(G, opinions, rewire_frac=0.05, similarity_threshold=0.3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    edges = list(G.edges())
    n_rewire = max(1, int(len(edges) * rewire_frac))

    # Pick random edges to consider dropping
    candidates = rng.choice(len(edges), size=min(n_rewire, len(edges)), replace=False)

    for idx in candidates:
        u, v = edges[idx]
        # Drop edge if opinions are too different
        if abs(opinions[u] - opinions[v]) > similarity_threshold:
            G.remove_edge(u, v)
            # u tries to rewire to a random node with similar opinion
            all_nodes = list(G.nodes())
            rng.shuffle(all_nodes)
            for w in all_nodes:
                if w != u and not G.has_edge(u, w) and abs(opinions[u] - opinions[w]) < similarity_threshold:
                    G.add_edge(u, w)
                    break


def run_simulation(steps=200):
    G = generate_graph()
    opinions = initialize_opinions(G)
    rng = np.random.default_rng(0)

    pos = {n: rng.uniform(0, 1, 2) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(6, 6))

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Opinion")

    def update(frame):
        nonlocal opinions, pos
        ax.clear()
        ax.set_title(f"Echo Chamber Formation — Step {frame}")

        # Update opinions first
        opinions = update_bounded(G, opinions)

        # Then rewire edges based on new opinions
        rewire_edges(G, opinions, rewire_frac=0.05, similarity_threshold=0.3, rng=rng)

        # Gradually pull connected nodes together
        pos_array = nx.spring_layout(G, pos=pos, iterations=2, weight=None, seed=frame)
        pos.update(pos_array)

        colors = [opinions[n] for n in G.nodes()]
        nx.draw(G, pos,
                node_color=colors,
                cmap="coolwarm",
                vmin=-1, vmax=1,
                node_size=50,
                edge_color="lightgray",
                width=0.4,
                ax=ax)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)
    ani.save("results/gifs/echo_chamber_formation.gif", writer="pillow", fps=10)
    plt.close()
    print("Saved to results/gifs/echo_chamber_formation.gif")

run_simulation()