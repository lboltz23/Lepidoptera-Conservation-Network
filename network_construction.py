import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances

# -----------------------------
# Settings (match your earlier run)
# -----------------------------
K_NEIGHBORS = 10
SYMMETRY = "union"   # "union" or "mutual"
SEED = 42

IN_PATH = "trait_incidence_matrix_final_network.csv"

OUT_EDGES = "final_network_edges.csv"
OUT_GRAPHML = "final_network.graphml"
OUT_NODES = "final_network_nodes.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(IN_PATH)

target_cols = [c for c in ["gb_threatened_binary", "ie_threatened_binary"] if c in df.columns]
id_cols = [c for c in df.columns if df[c].dtype == object]

# Choose a node ID column (best effort)
# If you have something like "species" or "species_name", it will be in id_cols
if len(id_cols) == 0:
    raise ValueError("No identifier (object) columns found. Please include a species name column.")
node_id_col = id_cols[0]  # use first object column as node id
node_ids = df[node_id_col].astype(str).tolist()

feature_cols = [c for c in df.columns if c not in id_cols + target_cols]
X = df[feature_cols].fillna(0).astype(int).values

n = X.shape[0]
print("Species:", n, "Features:", X.shape[1])

# -----------------------------
# Jaccard similarity
# -----------------------------
dist = pairwise_distances(X, metric="jaccard")
sim = 1.0 - dist
np.fill_diagonal(sim, 0.0)

# -----------------------------
# Build kNN neighbor sets
# -----------------------------
neighbors = []
for i in range(n):
    nbrs = np.argsort(sim[i])[::-1][:K_NEIGHBORS]
    nbrs = [j for j in nbrs if sim[i, j] > 0]
    neighbors.append(set(nbrs))

# Decide which edges to keep (union or mutual)
edges = []
for i in range(n):
    for j in neighbors[i]:
        if SYMMETRY == "mutual":
            if i in neighbors[j]:
                edges.append((i, j, float(sim[i, j])))
        else:  # union
            edges.append((i, j, float(sim[i, j])))

# Deduplicate undirected edges, keep max weight
edge_dict = {}
for i, j, w in edges:
    a, b = (i, j) if i < j else (j, i)
    if (a, b) not in edge_dict or w > edge_dict[(a, b)]:
        edge_dict[(a, b)] = w

# -----------------------------
# Build NetworkX graph
# -----------------------------
G = nx.Graph()
for idx, sid in enumerate(node_ids):
    G.add_node(sid)

# Add edges with weights
for (a, b), w in edge_dict.items():
    G.add_edge(node_ids[a], node_ids[b], weight=w)

print("Edges:", G.number_of_edges())

# Optional: attach targets as node attributes (for later analysis/plotting)
for c in target_cols:
    vals = df[c].tolist()
    nx.set_node_attributes(G, {node_ids[i]: vals[i] for i in range(n)}, name=c)

# -----------------------------
# Save outputs
# -----------------------------
# Edge list
edge_rows = [
    {"source": u, "target": v, "weight": d["weight"]}
    for u, v, d in G.edges(data=True)
]
pd.DataFrame(edge_rows).to_csv(OUT_EDGES, index=False)

# Node list
node_out = df[id_cols + target_cols].copy()
node_out.to_csv(OUT_NODES, index=False)

# GraphML (Gephi/Cytoscape)
nx.write_graphml(G, OUT_GRAPHML)

print("Wrote:")
print(" ", OUT_EDGES)
print(" ", OUT_NODES)
print(" ", OUT_GRAPHML)
