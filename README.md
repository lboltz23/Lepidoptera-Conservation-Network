# Network Construction and Stability Analysis

This repository contains Python scripts for constructing a k-nearest neighbor (kNN) similarity network from binary trait data, evaluating clustering stability, and assessing feature importance via trait-family ablation. The code is intended for research and exploratory network analysis.

---

## Repository Structure

* `network_construction.py` – Build and export the final kNN similarity network
* `knn_k_sweep.py` – Sweep over k values to assess connectivity and clustering stability
* `network_feature_selection.py` – Trait-family ablation to evaluate feature importance

---

## Scripts

### `network_construction.py`

Constructs a weighted, undirected kNN network using Jaccard similarity.

**Outputs**

* `final_network_edges.csv`
* `final_network_nodes.csv`
* `final_network.graphml` (for Gephi / Cytoscape)

**Key parameters**

* `K_NEIGHBORS`: number of neighbors per node
* `SYMMETRY`: `union` or `mutual` kNN rule

---

### `knn_k_sweep.py`

Evaluates how different values of k affect network structure and clustering stability.

**What it does**

* Builds kNN graphs for multiple k values
* Computes graph diagnostics (edges, components, giant component)
* Uses Louvain clustering with bootstrap ARI for stability assessment

**Outputs**

* `knn_k_sweep_summary.csv`
* `knn_k_sweep_bootstrap_ari_long.csv`

The script prints a recommended k based on connectivity and ARI plateau heuristics.

---

### `network_feature_selection.py`

Performs trait-family ablation to assess the contribution of feature groups to clustering stability.

**What it does**

* Computes a baseline clustering stability
* Iteratively removes trait families
* Compares bootstrap ARI against the baseline

**Outputs**

* `baseline_bootstrap_ari.csv`
* `network_feature_ablation_results.csv`
* `ablation_bootstrap_ari_long.csv`

Negative changes in mean ARI indicate trait families important for stable clustering.

---

## Input Data

Expected inputs include:

* One row per species (or entity)
* Binary trait columns (0/1)
* At least one identifier column (string/object type)
* Optional conservation status columns (e.g. `gb_threatened_binary`, `ie_threatened_binary`)

For feature ablation, a trait-to-family mapping CSV is required.

---

## Dependencies

* Python 3.8+
* pandas
* numpy
* networkx
* scikit-learn

Install with:

```bash
pip install pandas numpy networkx scikit-learn
```

---

## Recommended Workflow

1. Run `knn_k_sweep.py` to select an appropriate k
2. Build the final network using `network_construction.py`
3. Assess feature importance with `network_feature_selection.py`

---

## Additional Data Files

In addition to the scripts, this repository includes several CSV files that were useful for analyzing Lepidoptera traits and conservation status. These files are provided to support transparency, reproducibility, and further exploration of the data.

These CSVs may include:

* Trait incidence matrices used as input to network construction
* Trait-to-family mapping files used for feature ablation
* Conservation status labels or derived binary indicators
* Intermediate or summary CSVs generated during analysis

The exact contents of these files may vary, but all are related to understanding how trait information and conservation status interact within the network framework.

---

## Notes

* Random seeds are fixed for reproducibility
* Louvain clustering uses NetworkX’s built-in implementation
* Designed for large, sparse binary trait matrices
