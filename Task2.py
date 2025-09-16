# task02_clustering.py
# I304 – Assessment 1 (Task 02: K-Means Clustering – Credit Card dataset)
# Run: python task02_clustering.py
#
# This version ensures ALL figures are cluster scatter plots (no elbow/metric line charts).
# We still compute metrics (silhouette, CH, DBI) for reporting (printed + CSV), but only
# save PCA 2D scatter plots colored by cluster for each k.

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# =========================
# Config
# =========================
CSV_PATH = "data/CC_GENERAL.csv"   # <- adjust if needed
ID_GUESS_TOKENS = ("id", "cust", "customer")  # guess an ID-like column

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# Load
# =========================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find {CSV_PATH}. Place the credit card CSV in ./data and update CSV_PATH if needed."
    )

df_raw = pd.read_csv(CSV_PATH)
print("Loaded shape:", df_raw.shape)
print("Columns:", list(df_raw.columns))

# =========================
# Identify ID column (if any) & numeric-only feature set
# =========================
df = df_raw.copy()

# Try common ID variants
id_col = None
for c in df.columns:
    cl = str(c).lower()
    if any(tok in cl for tok in ID_GUESS_TOKENS):
        id_col = c
        break

# Secondary heuristic: object columns with nearly-unique values
if id_col is None:
    obj_like = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_like:
        if df[c].nunique(dropna=True) >= int(0.95 * len(df)):
            id_col = c
            break

print(f"Detected ID column: {id_col}" if id_col else "No ID column detected (that’s fine).")

# Keep numeric columns only
num_df = df.select_dtypes(include=[np.number]).copy()
if num_df.empty:
    raise RuntimeError("No numeric columns found for clustering.")

# Drop zero-variance columns
nunique = num_df.nunique()
degenerate = nunique[nunique <= 1].index.tolist()
if degenerate:
    print("Dropping zero-variance columns:", degenerate)
    num_df = num_df.drop(columns=degenerate)

# =========================
# Impute & Scale
# =========================
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imputed = imputer.fit_transform(num_df)
X = scaler.fit_transform(X_imputed)

print("Final feature matrix shape:", X.shape)

# =========================
# PCA (fit once for consistent projections across k)
# =========================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

# Helper for cluster scatter (PCA 2D)
def plot_cluster_scatter(X2d, labels, title, outpath):
    plt.figure(figsize=(7, 6))
    # One chart, clusters as colours (default palette)
    for c in np.unique(labels):
        mask = labels == c
        plt.scatter(X2d[mask, 0], X2d[mask, 1], s=14, alpha=0.8, label=f"Cluster {c}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# =========================
# K sweep (k = 3 .. 15)  — ALL FIGURES ARE CLUSTER SCATTERS
# =========================
ks = list(range(3, 16))
metrics_rows = []

best_k = None
best_sil = -np.inf

for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels_k = km.fit_predict(X)

    # Metrics (printed + saved as CSV later, but NOT plotted as lines)
    try:
        sil = silhouette_score(X, labels_k)
    except Exception:
        sil = np.nan
    try:
        ch = calinski_harabasz_score(X, labels_k)
    except Exception:
        ch = np.nan
    try:
        dbi = davies_bouldin_score(X, labels_k)
    except Exception:
        dbi = np.nan

    metrics_rows.append(
        {
            "k": k,
            "inertia": km.inertia_,
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": dbi,
        }
    )

    # Cluster scatter figure for this k (PCA 2D)
    fig_path = f"figures/task02_kmeans_k{k:02d}_pca_clusters.png"
    plot_cluster_scatter(X_pca, labels_k, f"KMeans clusters (k={k}) – PCA 2D", fig_path)
    print(f"[FIG] Saved PCA cluster plot for k={k} -> {fig_path}")

    # Track best by silhouette
    if not np.isnan(sil) and sil > best_sil:
        best_sil = sil
        best_k = k

# Save metrics table
metrics_df = pd.DataFrame(metrics_rows).sort_values("k").reset_index(drop=True)
metrics_path = "outputs/task02_kmeans_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"[CSV] Metrics saved -> {metrics_path}")

# Fallback if silhouette was all NaN
if best_k is None:
    print("[WARN] All silhouette scores NaN; falling back to min DBI.")
    best_k = int(metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "k"])

print(f"\nChosen best k: {best_k} (by silhouette where available)")

# =========================
# Final fit, labels, and highlighted cluster figure for best_k
# =========================
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = kmeans.fit_predict(X)

df_clusters = df.copy()
df_clusters["cluster"] = labels

# Save per-cluster means for original numeric features (imputed, unscaled)
cluster_summary = (
    pd.DataFrame(X_imputed, columns=num_df.columns)
    .assign(cluster=labels)
    .groupby("cluster")
    .mean()
    .sort_index()
)
summary_path = "outputs/task02_cluster_feature_means.csv"
cluster_summary.to_csv(summary_path, index=True)
print(f"[CSV] Cluster feature means saved -> {summary_path}")

# Highlighted PCA scatter for best_k
best_fig_path = f"figures/task02_kmeans_bestk{best_k:02d}_pca_clusters.png"
plot_cluster_scatter(X_pca, labels, f"KMeans clusters (k={best_k}) – PCA 2D [Best k]", best_fig_path)
print(f"[FIG] Saved BEST-K PCA cluster plot -> {best_fig_path}")

# =========================
# Export labeled data (optional, helps marking)
# =========================
out_df = df_clusters.copy()
cols_to_keep = [id_col] if id_col else []
cols_to_keep += ["cluster"]
assign_path = "outputs/task02_cluster_assignments.csv"
out_df[cols_to_keep].to_csv(assign_path, index=False)
print(f"[CSV] Cluster assignments saved -> {assign_path}")

print("\nDone. All figures are cluster scatter plots (see figures/).")
