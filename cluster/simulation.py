import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/gifs", exist_ok=True)

# -------------------------------
# Graph generator
# -------------------------------
def generate_graph():
    sizes = [60, 60, 60]
    p_in = 0.15
    p_out = 0.005
    probs = [[p_in, p_out, p_out],
             [p_out, p_in, p_out],
             [p_out, p_out, p_in]]
    return nx.stochastic_block_model(sizes, probs, seed=42)

# -------------------------------
# Biased-random initial opinions
# each cluster leans in a direction but has wide enough spread to look mixed
# -------------------------------
def initialize_opinions(G):
    rng = np.random.default_rng(42)
    centers = [-0.7, 0.0, 0.7]
    opinions = {}
    for n in G.nodes():
        cluster = G.nodes[n]["block"]
        opinions[n] = np.clip(centers[cluster] + rng.normal(0, 0.3), -1, 1)
    return opinions

# -------------------------------
# Bounded confidence update — no noise so clusters can actually stabilise
# -------------------------------
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

# -------------------------------
# Simulation + GIF
# -------------------------------
def run_simulation(steps=150):
    G = generate_graph()
    opinions = initialize_opinions(G)
    rng = np.random.default_rng(0)

    # Start from random positions so physical clustering is also visible
    pos = {n: rng.uniform(0, 1, 2) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(6, 6))

    # Colorbar created once outside update loop
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Opinion")

    def update(frame):
        nonlocal opinions, pos
        ax.clear()
        ax.set_title(f"Echo Chamber Formation — Step {frame}")

        opinions = update_bounded(G, opinions)

        # Fewer iterations per frame for speed
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