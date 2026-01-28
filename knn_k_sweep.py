import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score

# -----------------------------
# Settings
# -----------------------------
SEED = 42
np.random.seed(SEED)

IN_PATH = "trait_incidence_matrix_final_network.csv"

# k values to test (edit if desired)
K_VALUES = [5, 8, 10, 12, 15, 20, 25]

# Bootstrap settings
B_BOOT = 200
SUBSAMPLE_P = 0.85

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(IN_PATH)

target_cols = [c for c in ["gb_threatened_binary", "ie_threatened_binary"] if c in df.columns]
id_cols = [c for c in df.columns if df[c].dtype == object]
feature_cols = [c for c in df.columns if c not in id_cols + target_cols]

X = df[feature_cols].fillna(0).astype(int).values
n = X.shape[0]

print("Species:", n, "Features:", X.shape[1])

# -----------------------------
# Helpers
# -----------------------------
def jaccard_similarity(Xbin: np.ndarray) -> np.ndarray:
    dist = pairwise_distances(Xbin, metric="jaccard")
    sim = 1.0 - dist
    np.fill_diagonal(sim, 0.0)
    return sim

def build_union_knn_graph(sim: np.ndarray, k: int) -> nx.Graph:
    """
    Union-kNN: add edges from each node to its top-k neighbors.
    Undirected graph; if edge duplicated, keep max weight.
    """
    n = sim.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        nbrs = np.argsort(sim[i])[::-1][:k]
        for j in nbrs:
            w = float(sim[i, j])
            if w <= 0 or i == j:
                continue
            if G.has_edge(i, j):
                if w > G[i][j]["weight"]:
                    G[i][j]["weight"] = w
            else:
                G.add_edge(i, j, weight=w)
    return G

def louvain_labels(G: nx.Graph) -> np.ndarray:
    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=SEED)
    labels = np.empty(G.number_of_nodes(), dtype=int)
    for cid, comm in enumerate(comms):
        for node in comm:
            labels[node] = cid
    return labels

def graph_diagnostics(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    comps = list(nx.connected_components(G))
    n_comp = len(comps)
    giant = max((len(c) for c in comps), default=0)
    giant_frac = giant / n if n > 0 else 0
    avg_deg = (2 * m / n) if n > 0 else 0
    return {
        "n_edges": m,
        "n_components": n_comp,
        "giant_component_frac": giant_frac,
        "avg_degree": avg_deg,
    }

def bootstrap_ari(sim_full: np.ndarray, k: int, B: int, subsample_p: float):
    # baseline labels on full graph
    G0 = build_union_knn_graph(sim_full, k)
    base_labels = louvain_labels(G0)

    n = sim_full.shape[0]
    m = int(subsample_p * n)
    ari_vals = []

    for b in range(B):
        idx = np.random.choice(n, size=m, replace=False)
        sim_sub = sim_full[np.ix_(idx, idx)]
        G = build_union_knn_graph(sim_sub, k)
        labs = louvain_labels(G)
        ari_vals.append(adjusted_rand_score(base_labels[idx], labs))

    return base_labels, np.array(ari_vals)

# -----------------------------
# Precompute similarity once (big speedup)
# -----------------------------
print("Computing full Jaccard similarity matrix...")
SIM = jaccard_similarity(X)

# -----------------------------
# Sweep over k
# -----------------------------
summary_rows = []
long_rows = []

for k in K_VALUES:
    print(f"\n=== k = {k} ===")

    # Graph diagnostics on full graph
    G_full = build_union_knn_graph(SIM, k)
    diag = graph_diagnostics(G_full)

    # Bootstrap stability
    _, ari = bootstrap_ari(SIM, k, B_BOOT, SUBSAMPLE_P)

    row = {
        "k": k,
        **diag,
        "bootstrap_mean_ari": float(ari.mean()),
        "bootstrap_sd_ari": float(ari.std()),
        "bootstrap_min_ari": float(ari.min()),
        "bootstrap_max_ari": float(ari.max()),
    }
    summary_rows.append(row)

    for b, val in enumerate(ari, start=1):
        long_rows.append({"k": k, "bootstrap": b, "ari": float(val)})

    print(
        f"edges={diag['n_edges']}, comps={diag['n_components']}, "
        f"giant={diag['giant_component_frac']:.3f}, "
        f"ARI_mean={ari.mean():.4f} sd={ari.std():.4f}"
    )

# Save results
summary = pd.DataFrame(summary_rows).sort_values("k")
summary.to_csv("knn_k_sweep_summary.csv", index=False)
pd.DataFrame(long_rows).to_csv("knn_k_sweep_bootstrap_ari_long.csv", index=False)

print("\nWrote:")
print("  knn_k_sweep_summary.csv")
print("  knn_k_sweep_bootstrap_ari_long.csv")

# -----------------------------
# Simple recommendation (plateau + connectivity)
# -----------------------------
# A: smallest k with giant component >= 0.95
# B: among those, pick smallest k where mean ARI is within 0.01 of the best mean ARI
best_mean = summary["bootstrap_mean_ari"].max()
cand = summary[summary["giant_component_frac"] >= 0.95].copy()
if len(cand) > 0:
    cand["within_001_of_best"] = (best_mean - cand["bootstrap_mean_ari"]) <= 0.01
    cand2 = cand[cand["within_001_of_best"]]
    if len(cand2) > 0:
        k_star = int(cand2.sort_values("k").iloc[0]["k"])
    else:
        k_star = int(cand.sort_values("k").iloc[0]["k"])
    print(f"\nSuggested k (connectivity + ARI plateau heuristic): {k_star}")
else:
    print("\nNo k achieved giant_component_frac >= 0.95; consider increasing k grid.")
