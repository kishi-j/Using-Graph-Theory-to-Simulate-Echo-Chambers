import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from numpy.linalg import eigvalsh
import os

# ---------------------------------------------------------------------
# Helper: ensure results folders exist
# ---------------------------------------------------------------------
os.makedirs("results/gifs", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# ---------------------------------------------------------------------
# Generate a graph with 3 clear communities
# ---------------------------------------------------------------------
def generate_graph():
    sizes = [60, 60, 60]   # 3 communities of 60 nodes
    p_in = 0.10            # probability of edges inside a community
    p_out = 0.01           # probability of edges across communities

    probs = [[p_in, p_out, p_out],
             [p_out, p_in, p_out],
             [p_out, p_out, p_in]]

    G = nx.stochastic_block_model(sizes, probs, seed=42)

    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    return G

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def lambda2(G):
    L = nx.normalized_laplacian_matrix(G, weight="weight").astype(float)
    vals = eigvalsh(L.toarray())
    return sorted(vals)[1]  # second-smallest eigenvalue

def modularity(G, labels):

    comms = {}
    for n, c in labels.items():
        comms.setdefault(c, []).append(n)
    communities = [set(v) for v in comms.values()]
    return nx.algorithms.community.quality.modularity(G, communities, weight="weight")

def assortativity(G, labels):
    nx.set_node_attributes(G, labels, "comm")
    return nx.attribute_assortativity_coefficient(G, "comm")

def intra_inter_ratio(G, labels):
    intra, inter = 0, 0
    for u, v, w in G.edges(data="weight"):
        if labels[u] == labels[v]:
            intra += w
        else:
            inter += w
    return intra / max(inter, 1e-9)

# ---------------------------------------------------------------------
# Update rules (feed-ranking algorithms)
# ---------------------------------------------------------------------
def apply_similarity(G, labels, strength=0.05):
    for u, v in G.edges():
        if labels[u] == labels[v]:
            G[u][v]["weight"] *= (1 + strength)
        else:
            G[u][v]["weight"] *= (1 - strength/2)

def apply_engagement(G, labels, strength=0.05):
    rng = np.random.default_rng(42)
    for u, v in G.edges():
        engagement = rng.uniform(0, 1)
        G[u][v]["weight"] *= (1 + strength * engagement)

def apply_popularity(G, labels, strength=0.05):
    deg = dict(G.degree(weight="weight"))
    maxd = max(deg.values())
    for u, v in G.edges():
        boost = strength * (deg[u] + deg[v]) / (2 * maxd)
        G[u][v]["weight"] *= (1 + boost)

# ---------------------------------------------------------------------
# Make GIF for any update rule
# ---------------------------------------------------------------------
def run_animation(update_fn, name, steps=40):
    G = generate_graph()

    # initial communities
    comms = nx.algorithms.community.greedy_modularity_communities(G)
    labels = {n:i for i,c in enumerate(comms) for n in c}

    pos = nx.spring_layout(G, seed=0, weight="weight")
    fig, ax = plt.subplots(figsize=(6, 6))

    # metrics storage
    metrics = {"step":[], "modularity":[], "lambda2":[],
               "assortativity":[], "ratio":[]}

    def update(frame):
        nonlocal pos   # allow modification of outer variable

        ax.clear()
        ax.set_title(f"{name} — Step {frame}", fontsize=14)

        update_fn(G, labels)

        pos = nx.spring_layout(G, pos=pos, iterations=5, weight="weight")

        colors = [labels[n] for n in G.nodes()]
        nx.draw(G, pos, node_color=colors, cmap="tab10",
                node_size=40, edge_color="lightgray", width=0.4, ax=ax)

        # metrics
        Q = modularity(G, labels)
        lam2 = lambda2(G)
        a = assortativity(G, labels)
        r = intra_inter_ratio(G, labels)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=200)
    ani.save(f"results/gifs/{name}.gif", writer="pillow", fps=5)

    # save metrics
    df = pd.DataFrame(metrics)
    df.to_csv(f"results/metrics/{name}.csv", index=False)

    # plot metrics
    plt.figure(figsize=(10,6))
    for col in ["modularity", "lambda2", "assortativity", "ratio"]:
        plt.plot(df["step"], df[col], label=col)
    plt.legend()
    plt.title(f"Metrics Over Time — {name}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}_metrics.png")
    plt.close()

# ---------------------------------------------------------------------
# Run all algorithms
# ---------------------------------------------------------------------
run_animation(lambda G,labels: apply_similarity(G, labels), "similarity")
run_animation(lambda G,labels: apply_engagement(G, labels), "engagement")
run_animation(lambda G,labels: apply_popularity(G, labels), "popularity")