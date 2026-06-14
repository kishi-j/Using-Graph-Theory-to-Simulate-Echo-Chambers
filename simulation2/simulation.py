import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/gifs", exist_ok=True)

# -------------------------------
# Graph generator (same for all)
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
# Opinion initializers
# -------------------------------
def initialize_random(G):
    rng = np.random.default_rng(42)
    return {n: rng.uniform(-1, 1) for n in G.nodes()}

def initialize_three_ranges(G):
    rng = np.random.default_rng(42)
    opinions = {}
    nodes = list(G.nodes())
    n = len(nodes)
    for i, node in enumerate(nodes):
        if i < n//3:
            opinions[node] = rng.uniform(-0.9, -0.3)
        elif i < 2*n//3:
            opinions[node] = rng.uniform(-0.1, 0.1)
        else:
            opinions[node] = rng.uniform(0.3, 0.9)
    return opinions

# -------------------------------
# Update rules
# -------------------------------
def update_averaging(G, opinions, alpha=0.2):
    new_op = {}
    for u in G.nodes():
        neighbors = list(G.neighbors(u))
        if not neighbors:
            new_op[u] = opinions[u]
            continue
        avg = np.mean([opinions[v] for v in neighbors])
        new_op[u] = (1 - alpha) * opinions[u] + alpha * avg
    return new_op

def update_bounded(G, opinions, epsilon=0.25, alpha=0.3):
    new_op = {}
    for u in G.nodes():
        neighbors = [v for v in G.neighbors(u) if abs(opinions[u]-opinions[v])<epsilon]
        if not neighbors:
            new_op[u] = opinions[u]
            continue
        avg = np.mean([opinions[v] for v in neighbors])
        new_op[u] = (1 - alpha)*opinions[u] + alpha*avg
    return new_op

# -------------------------------
# Run a simulation and save GIF
# -------------------------------
def run_simulation_gif(update_fn, initialize_fn, name, steps=80):
    G = generate_graph()
    opinions = initialize_fn(G)
    pos = nx.spring_layout(G, seed=0)

    fig, ax = plt.subplots(figsize=(6,6))

    def update(frame):
        nonlocal opinions, pos
        ax.clear()
        ax.set_title(f"{name} — Step {frame}")
        opinions = update_fn(G, opinions)
        pos = nx.spring_layout(G, pos=pos, iterations=5, weight=None)
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
    ani.save(f"results/gifs/{name}.gif", writer="pillow", fps=10)
    plt.close()

# -------------------------------
# Run all three GIFs
# -------------------------------
run_simulation_gif(update_averaging, initialize_random, "averaging_out")
run_simulation_gif(update_bounded, initialize_three_ranges, "bounded_clusters")
run_simulation_gif(update_bounded, initialize_three_ranges, "cluster_formation")