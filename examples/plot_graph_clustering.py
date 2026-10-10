"""
=====================================================
Graph Clustering with scikit-clarans (Neo4j Workflow)
=====================================================

Tái hiện bài toán phân cụm đồ thị lớn (K-Medoids trên mạng lưới)
tương tự bài blog của Neo4j.

This example reproduces graph clustering with CLARANS on network data
(inspired by Neo4j's graph clustering tutorial). Shortest-path distances
are computed via SciPy, and CLARANS partitions the network using
`metric='precomputed'`.
"""

# Authors: Ngọc Thiện Nguyễn <thiennguyen03001@gmail.com>
# License: MIT

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from clarans import CLARANS

# ==============================================================================
# Bước 1: Tạo/Load đồ thị mẫu (Mô phỏng 4 cụm cộng đồng Caveman Graph)
# ==============================================================================
# Đồ thị gồm 4 cụm (hang), mỗi cụm có 25 nút kết nối dày đặc với nhau
G = nx.connected_caveman_graph(l=4, k=25)

# Lọc thành phần liên thông lớn nhất (Giant Component) đúng như blog Neo4j
largest_cc = max(nx.connected_components(G), key=len)
G_giant = G.subgraph(largest_cc).copy()
nodes = list(G_giant.nodes())
n_nodes = len(nodes)
print(f"Number of nodes in Giant Component: {n_nodes}")

# ==============================================================================
# Bước 2: Tính ma trận khoảng cách đường đi ngắn nhất (Shortest Path Distance)
# ==============================================================================
# Trích xuất ma trận kề thưa từ NetworkX
adj_matrix = nx.to_scipy_sparse_array(G_giant, weight=None)

# Tính khoảng cách đường đi ngắn nhất giữa các đỉnh (dùng C-kernel của SciPy cực nhanh)
dist_matrix = shortest_path(adj_matrix, directed=False, unweighted=True)
dist_matrix = np.asarray(dist_matrix, dtype=np.float64)

# ==============================================================================
# Bước 3: Phân cụm đồ thị bằng CLARANS (scikit-clarans)
# ==============================================================================
k = 4
model = CLARANS(
    n_clusters=k,
    num_local=3,
    metric="precomputed",  # Chỉ định ma trận khoảng cách đã tính trước
    random_state=42,
    verbose=2,  # Bật log chi tiết từng restart
)
model.fit(dist_matrix)

# Trích xuất các Medoid Hubs (nút trung tâm đại diện cho từng cụm)
medoid_nodes = [nodes[idx] for idx in model.medoid_indices_]
print(f"\nMedoid Hubs (community representatives): {medoid_nodes}")
print(f"Total shortest-path inertia: {model.inertia_:.1f}")

# ==============================================================================
# Bước 4: Trực quan hóa đồ thị và đánh dấu Medoids
# ==============================================================================
pos = nx.spring_layout(G_giant, seed=42)

plt.figure(figsize=(10, 8))

# 1. Vẽ các nút thành viên phân màu theo cụm
nx.draw_networkx_nodes(
    G_giant,
    pos,
    node_color=model.labels_,
    cmap="tab10",
    node_size=90,
    alpha=0.85,
)

# 2. Vẽ các cạnh nối mờ
nx.draw_networkx_edges(G_giant, pos, alpha=0.25, edge_color="gray")

# 3. Đánh dấu nổi bật các Medoid Hubs bằng ngôi sao đỏ
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
