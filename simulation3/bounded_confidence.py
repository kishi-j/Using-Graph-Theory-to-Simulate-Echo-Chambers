import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

os.makedirs("results/gifs", exist_ok=True)

# generate random graph with no clusters to start with
def generate_graph():
    n = 180 #number of nodes
    p = 0.03 # probability of connection 
    return nx.erdos_renyi_graph(n, p, seed=42) # erdos–renyi model
 
# each node randomly assigned -1 to 1
def initialize_opinions(G):
    rng = np.random.default_rng(42)
    return {n: rng.uniform(-1, 1) for n in G.nodes()}

# bounded confidence model
def update_bounded(G, opinions, epsilon=0.2, alpha=0.1):
    new_opinions = {}
    for u in G.nodes():
        # neighbours if similar opinion (dif less than eps)
        neighbors = [v for v in G.neighbors(u)
                     if abs(opinions[u] - opinions[v]) < epsilon]

        # no change if no new neighbours
        if not neighbors:
            new_opinions[u] = opinions[u]
            continue
        
        # calculate avg of accepted neightbours and shift towards it 
        # alpha controls how much shift
        avg = np.mean([opinions[v] for v in neighbors])
        new_opinions[u] = (1 - alpha) * opinions[u] + alpha * avg

    return new_opinions

# homophilic rewiring
def rewire_edges(G, opinions, rewire_frac=0.08, similarity_threshold=0.25, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # only modify a max of 8% of edges 
    edges = list(G.edges())
    n_rewire = max(1, int(len(edges) * rewire_frac))

    # choose random edges
    candidates = rng.choice(len(edges), size=min(n_rewire, len(edges)), replace=False)

    for idx in candidates:
        u, v = edges[idx]

        # remove dissimilar edges
        if abs(opinions[u] - opinions[v]) > similarity_threshold:
            G.remove_edge(u, v)

            # rewire to similar node (w random search order)
            nodes = list(G.nodes())
            rng.shuffle(nodes)

            # reconnect to a similar edge
            for w in nodes:
                # avoids self loops and duplicate edges 
                if w != u and not G.has_edge(u, w):
                    if abs(opinions[u] - opinions[w]) < similarity_threshold:
                        G.add_edge(u, w)
                        break

# modularity function
def compute_modularity(G):
    communities = list(nx.algorithms.community.greedy_modularity_communities(G))
    return nx.algorithms.community.modularity(G, communities)

# whole simulation loop
def run_simulation(steps=150):
    G = generate_graph()
    opinions = initialize_opinions(G)
    rng = np.random.default_rng(0)

    # random initial positions
    pos = {n: rng.uniform(0, 1, 2) for n in G.nodes()}

    fig, ax = plt.subplots(figsize=(6, 6))

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, label="Opinion")

    def update(frame):
        nonlocal opinions, pos
        ax.clear()

        # opinion update
        opinions = update_bounded(G, opinions)

        # network rewiring
        rewire_edges(G, opinions, rng=rng)

        # compute modularity after rewiring
        Q = compute_modularity(G)

        ax.set_title(f"Bounded Confidence — Step {frame} | Q = {Q:.3f}")

        # layout evolves slowly (reveals clusters)
        pos = nx.spring_layout(G, pos=pos, iterations=3, seed=frame)

        colors = [opinions[n] for n in G.nodes()]

        nx.draw(G, pos,
                node_color=colors,
                cmap="coolwarm",
                vmin=-1, vmax=1,
                node_size=40,
                edge_color="lightgray",
                width=0.3,
                ax=ax)

    ani = animation.FuncAnimation(fig, update, frames=steps, interval=100)
    ani.save("results/gifs/emergent_clusters.gif", writer="pillow", fps=10)
    plt.close()

    print("Saved to results/gifs/emergent_clusters.gif")

run_simulation()