import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/gifs", exist_ok=True)

# initial clusters with sbm model
def generate_graph():
    sizes = [60, 60, 60]
    p_in = 0.15
    p_out = 0.005

    probs = [[p_in, p_out, p_out],
             [p_out, p_in, p_out],
             [p_out, p_out, p_in]]

    return nx.stochastic_block_model(sizes, probs, seed=42)


# inital opinions are clustered
def initialize_opinions(G):
    rng = np.random.default_rng(42)
    centers = [-0.7, 0.0, 0.7]

    opinions = {}
    for n in G.nodes():
        block = G.nodes[n]["block"]
        opinions[n] = np.clip(centers[block] + rng.normal(0, 0.2), -1, 1)

    return opinions

# averaging
def update_averaging(G, opinions, alpha=0.3):
    new_opinions = {}

    for u in G.nodes():
        neighbors = list(G.neighbors(u))

        if not neighbors:
            new_opinions[u] = opinions[u]
            continue

        avg = np.mean([opinions[v] for v in neighbors])

        # move toward full neighbourhood average
        new_opinions[u] = (1 - alpha) * opinions[u] + alpha * avg

    return new_opinions

# modularity function
def compute_modularity(G):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    return nx.algorithms.community.modularity(G, communities)

# full simulation
def run_simulation(steps=150):
    G = generate_graph()
    opinions = initialize_opinions(G)
    rng = np.random.default_rng(0)

    # layout starts random but reveals SBM structure
    pos = {n: rng.uniform(0, 1, 2) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(6, 6))

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Opinion")

    def update(frame):
        nonlocal opinions, pos
        ax.clear()

        opinions = update_averaging(G, opinions)

        # compute modularity
        Q = compute_modularity(G)

        ax.set_title(f"Averaging — Step {frame} | Q = {Q:.3f}")

        # layout evolution (reveals clusters initially)
        pos = nx.spring_layout(G, pos=pos, iterations=3, seed=frame)

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
    ani.save("results/gifs/sbm_averaging.gif", writer="pillow", fps=10)
    plt.close()

    print("saved averaging gif")

run_simulation()