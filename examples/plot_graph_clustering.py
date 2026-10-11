"""
=========================================================
Graph Clustering with Precomputed Shortest-Path Distances
=========================================================

This example clusters network nodes with :class:`~clarans.CLARANS` using
shortest-path distances on a connected graph.

A synthetic caveman graph with four dense communities is converted into
a pairwise shortest-path distance matrix via SciPy, then partitioned with
`metric='precomputed'`. The resulting medoids represent central hub nodes
for each community.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from clarans import CLARANS


# Generate a connected caveman graph with 4 dense cliques of 25 nodes each
G = nx.connected_caveman_graph(l=4, k=25)

# Keep the giant connected component
largest_cc = max(nx.connected_components(G), key=len)
G_giant = G.subgraph(largest_cc).copy()
nodes = list(G_giant.nodes())
n_nodes = len(nodes)
print(f"Number of nodes in Giant Component: {n_nodes}")

# Compute the all-pairs shortest-path distance matrix
adj_matrix = nx.to_scipy_sparse_array(G_giant, weight=None)
dist_matrix = shortest_path(adj_matrix, directed=False, unweighted=True)
dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

# Cluster the graph using precomputed graph distances
k = 4
model = CLARANS(
    n_clusters=k,
    num_local=3,
    metric="precomputed",
    random_state=42,
    verbose=2,
)
model.fit(dist_matrix)

# Identify medoid hub nodes representing each community
medoid_nodes = [nodes[idx] for idx in model.medoid_indices_]
print(f"\nMedoid Hubs (community representatives): {medoid_nodes}")
print(f"Total shortest-path inertia: {model.inertia_:.1f}")

# Visualize the graph partition and highlight medoid hubs
pos = nx.spring_layout(G_giant, seed=42)

plt.figure(figsize=(10, 8))

# Draw member nodes colored by cluster assignment
nx.draw_networkx_nodes(
    G_giant,
    pos,
    node_color=model.labels_,
    cmap="tab10",
    node_size=90,
    alpha=0.85,
)

# Draw edges
nx.draw_networkx_edges(G_giant, pos, alpha=0.25, edge_color="gray")

# Highlight medoid hubs with red stars
nx.draw_networkx_nodes(
    G_giant,
    pos,
    nodelist=medoid_nodes,
    node_color="red",
    node_size=280,
    node_shape="*",
    label="Medoid Hubs",
)

plt.title(
    f"Graph Clustering with scikit-clarans (k={k})\n"
    f"Red stars indicate community medoids (hub nodes)",
    fontsize=12,
    pad=12,
)
plt.legend(loc="upper right", fontsize=10)
plt.axis("off")
plt.tight_layout()
plt.show()
