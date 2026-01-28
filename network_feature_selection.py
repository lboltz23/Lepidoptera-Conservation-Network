import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances
from collections import defaultdict

# -----------------------------
# Fixed parameters (set a priori)
# -----------------------------
K_NEIGHBORS = 10       # k for kNN graph
B_BOOT = 200           # bootstraps
SUBSAMPLE_P = 0.85     # fraction of species per bootstrap replicate
SEED = 42

np.random.seed(SEED)

# -----------------------------
# File paths
# -----------------------------
X_PATH = "trait_incidence_matrix_step1_filtered.csv"
MAP_PATH = "trait_family_mapping_refined_v3_no_extinct.csv"

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(X_PATH)
mapping = pd.read_csv(MAP_PATH)

target_cols = {"gb_threatened_binary", "ie_threatened_binary"}
id_cols = [c for c in df.columns if df[c].dtype == object]
feature_cols = [c for c in df.columns if c not in id_cols and c not in target_cols]

X = df[feature_cols].fillna(0).astype(int)
n = X.shape[0]

# family -> columns
family_to_cols = defaultdict(list)
for _, r in mapping.iterrows():
    col = r["column_name"]
    fam = r["trait_family"]
    if col in feature_cols:
        family_to_cols[fam].append(col)

families = sorted(family_to_cols.keys())

print(f"Species: {n}")
print(f"Features: {X.shape[1]}")
print(f"Families: {len(families)}")

# -----------------------------
# Helpers
# -----------------------------
def build_knn_graph_from_similarity(sim: np.ndarray, k: int) -> nx.Graph:
    """
    Build an undirected weighted kNN graph from a similarity matrix.
    Edges are added for top-k neighbors per node (ties broken by argsort order).
    """
    np.fill_diagonal(sim, 0.0)
    G = nx.Graph()
    G.add_nodes_from(range(sim.shape[0]))

    for i in range(sim.shape[0]):
        # top-k similarity neighbors
        nbrs = np.argsort(sim[i])[::-1][:k]
        for j in nbrs:
            w = sim[i, j]
            if w > 0:
                # undirected: keep max weight if edge repeated
                if G.has_edge(i, j):
                    if w > G[i][j]["weight"]:
                        G[i][j]["weight"] = float(w)
                else:
                    G.add_edge(i, j, weight=float(w))
    return G

def cluster_louvain(G: nx.Graph) -> np.ndarray:
    """
    Louvain clustering using NetworkX built-in (no external deps).
    Returns membership labels aligned to node order 0..n-1.
    """
    # Requires networkx >= 2.8-ish; for newer versions it's in this location:
    communities = nx.algorithms.community.louvain_communities(
        G, weight="weight", seed=SEED
    )
    labels = np.empty(G.number_of_nodes(), dtype=int)
    for cid, comm in enumerate(communities):
        for node in comm:
            labels[node] = cid
    return labels

def jaccard_similarity_matrix(Xbin: np.ndarray) -> np.ndarray:
    """
    Compute Jaccard similarity from a binary matrix.
    Uses sklearn pairwise_distances(metric='jaccard') which returns distance.
    similarity = 1 - distance
    """
    dist = pairwise_distances(Xbin, metric="jaccard")
    sim = 1.0 - dist
    return sim

def bootstrap_stability(X_df: pd.DataFrame, k: int, B: int, subsample_p: float):
    """
    Baseline clustering on full set; then bootstrap subsamples compared to baseline
    using Adjusted Rand Index (ARI).
    Returns: baseline_labels, ari_list
    """
    Xbin = X_df.values.astype(int)
    sim_full = jaccard_similarity_matrix(Xbin)
    G0 = build_knn_graph_from_similarity(sim_full, k)
    base_labels = cluster_louvain(G0)

    m = int(subsample_p * X_df.shape[0])
    ari_vals = []

    for b in range(B):
        idx = np.random.choice(X_df.shape[0], size=m, replace=False)
        Xsub = X_df.iloc[idx].values.astype(int)

        sim_sub = jaccard_similarity_matrix(Xsub)
        G = build_knn_graph_from_similarity(sim_sub, k)
        labs = cluster_louvain(G)

        ari = adjusted_rand_score(base_labels[idx], labs)
        ari_vals.append(ari)

    return base_labels, np.array(ari_vals)

# -----------------------------
# Baseline
# -----------------------------
print("\nRunning baseline bootstrap...")
base_labels, baseline_ari = bootstrap_stability(X, K_NEIGHBORS, B_BOOT, SUBSAMPLE_P)
print(f"Baseline ARI mean={baseline_ari.mean():.4f} sd={baseline_ari.std():.4f}")

pd.DataFrame({"bootstrap": np.arange(1, B_BOOT + 1), "ari": baseline_ari}).to_csv(
    "baseline_bootstrap_ari.csv", index=False
)

# -----------------------------
# Family ablation
# -----------------------------
print("\nRunning family ablations...")
summary_rows = []
long_rows = []

baseline_mean = baseline_ari.mean()
baseline_sd = baseline_ari.std()

for fam in families:
    drop_cols = family_to_cols[fam]
    X_ablate = X.drop(columns=drop_cols, errors="ignore")

    _, ari = bootstrap_stability(X_ablate, K_NEIGHBORS, B_BOOT, SUBSAMPLE_P)

    row = {
        "trait_family": fam,
        "n_cols_removed": len(drop_cols),
        "baseline_mean_ari": baseline_mean,
        "ablated_mean_ari": ari.mean(),
        "delta_mean_ari": ari.mean() - baseline_mean,
        "baseline_sd_ari": baseline_sd,
        "ablated_sd_ari": ari.std(),
    }
    summary_rows.append(row)

    for b, val in enumerate(ari, start=1):
        long_rows.append({"trait_family": fam, "bootstrap": b, "ari": val})

    print(f"  {fam:25s} removed={len(drop_cols):3d}  mean_ARI={ari.mean():.4f}  delta={ari.mean()-baseline_mean:+.4f}")

res_df = pd.DataFrame(summary_rows).sort_values("delta_mean_ari")
res_df.to_csv("network_feature_ablation_results.csv", index=False)
pd.DataFrame(long_rows).to_csv("ablation_bootstrap_ari_long.csv", index=False)

print("\nSaved:")
print("  network_feature_ablation_results.csv")
print("  baseline_bootstrap_ari.csv")
print("  ablation_bootstrap_ari_long.csv")

# Optional: simple retention heuristic
THRESH_DELTA = -0.02
retain = res_df[res_df["delta_mean_ari"] <= THRESH_DELTA]["trait_family"].tolist()
drop = res_df[res_df["delta_mean_ari"] > THRESH_DELTA]["trait_family"].tolist()

print(f"\nHeuristic retention (delta_mean_ari <= {THRESH_DELTA}):")
print("Retain:", ", ".join(retain) if retain else "(none)")
print("Drop:", ", ".join(drop) if drop else "(none)")
