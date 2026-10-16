"""
=========================================================
Graph Clustering with Precomputed Shortest-Path Distances
=========================================================

This example clusters network nodes with :class:`~clarans.FastCLARANS` using
all-pairs shortest-path distances on a connected graph.

A synthetic relaxed caveman network with five dense communities and inter-cluster
rewiring is converted into a pairwise shortest-path distance matrix via SciPy,
then partitioned with ``metric='precomputed'``. The verbose output showcases the
local search convergence dynamics across restarts, and the resulting medoids
represent the central hub nodes (routing centers) for each community.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from clarans import FastCLARANS

# 1. Generate a relaxed caveman graph: 5 dense communities of 35 nodes each
# Rewiring probability p=0.12 creates noisy cross-community bridge edges
k = 5
G = nx.relaxed_caveman_graph(l=k, k=35, p=0.12, seed=42)

# Extract the giant connected component
largest_cc = max(nx.connected_components(G), key=len)
G_giant = G.subgraph(largest_cc).copy()
nodes = list(G_giant.nodes())
n_nodes = len(nodes)
print(f"Number of nodes in Giant Component: {n_nodes}")
print(f"Total graph edges: {G_giant.number_of_edges()}")

# 2. Compute the all-pairs shortest-path distance matrix
adj_matrix = nx.to_scipy_sparse_array(G_giant, weight=None)
dist_matrix = shortest_path(adj_matrix, directed=False, unweighted=True)
dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

# 3. Cluster network nodes using FastCLARANS with precomputed graph distances
# verbose=2 details each successful medoid swap and cost reduction
model = FastCLARANS(
    n_clusters=k,
    num_local=3,
    init="random",
    metric="precomputed",
    random_state=42,
    verbose=2,
)
model.fit(dist_matrix)

# Identify medoid hub nodes representing each community
medoid_nodes = [nodes[idx] for idx in model.medoid_indices_]
print(f"\nMedoid Hubs (community representatives): {medoid_nodes}")
print(f"Total shortest-path inertia: {model.inertia_:.1f}")
print(f"Total swaps performed: {model.n_swaps_}")

# 4. Visualize the graph partition and highlight medoid hubs
pos = nx.spring_layout(G_giant, seed=42, k=0.22, iterations=80)

fig, ax = plt.subplots(figsize=(9.0, 7.5))
palette = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e6ab02"]
node_colors = [palette[model.labels_[i]] for i in range(n_nodes)]

# Draw cross-community and internal edges
nx.draw_networkx_edges(
    G_giant,
    pos,
    ax=ax,
    alpha=0.15,
    edge_color="#7f8c8d",
    width=0.8,
)

# Draw graph member nodes colored by community
nx.draw_networkx_nodes(
    G_giant,
    pos,
    ax=ax,
    node_color=node_colors,
    node_size=60,
    alpha=0.85,
)

# Highlight central medoid hubs with gold stars
nx.draw_networkx_nodes(
    G_giant,
    pos,
    ax=ax,
    nodelist=medoid_nodes,
    node_color="gold",
    node_size=280,
    node_shape="*",
    linewidths=1.5,
    edgecolors="black",
    label="Community Medoid Hubs",
)

ax.set_title(
    f"Graph Community Detection with FastCLARANS (k={k}, N={n_nodes})\n"
    f"Precomputed Shortest-Path Inertia = {model.inertia_:.1f} (Total Swaps: {model.n_swaps_})",
    fontsize=11,
    pad=10,
)
ax.legend(
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    fontsize=9.0,
)
ax.axis("off")
plt.tight_layout()
plt.show()
