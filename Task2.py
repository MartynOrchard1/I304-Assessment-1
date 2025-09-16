# task02_clustering.py
# I304 – Assessment 1 (Task 02: K-Means Clustering – Credit Card dataset)
# Run: python task02_clustering.py
#
# All output figures are CLUSTER SCATTERS.
# For each k (3..15) we save:
#   - PCA 2D cluster scatter
#   - Original feature-pair cluster scatter (configurable; auto-picks top-variance if not found)
#
# Also saves:
#   - outputs/task02_kmeans_metrics.csv
#   - outputs/task02_cluster_feature_means.csv
#   - outputs/task02_cluster_assignments.csv

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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# =========================
# Config
# =========================
CSV_PATH = "data/CC_GENERAL.csv"        # <- adjust if needed
ID_GUESS_TOKENS = ("id", "cust", "customer")

# Optional: choose a pair of ORIGINAL numeric columns to plot per-k.
# If either is missing, the script will auto-pick the 2 highest-variance numeric features.
FEATURE_PAIR = ("BALANCE", "PURCHASES")  # e.g. ("BALANCE","PURCHASES") or None to always auto-pick

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =========================
# Load
# =========================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Put the CSV in ./data or update CSV_PATH.")

df_raw = pd.read_csv(CSV_PATH)
print("Loaded shape:", df_raw.shape)
print("Columns:", list(df_raw.columns))

# =========================
# Identify ID column (if present) & keep numeric features
# =========================
df = df_raw.copy()

id_col = None
for c in df.columns:
    cl = str(c).lower()
    if any(tok in cl for tok in ID_GUESS_TOKENS):
        id_col = c
        break

if id_col is None:
    obj_like = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_like:
        if df[c].nunique(dropna=True) >= int(0.95 * len(df)):
            id_col = c
            break

print(f"Detected ID column: {id_col}" if id_col else "No ID column detected.")

# Numeric-only for KMeans
num_df = df.select_dtypes(include=[np.number]).copy()
if num_df.empty:
    raise RuntimeError("No numeric columns found for clustering.")

# Drop zero-variance columns
degenerate = num_df.columns[num_df.nunique() <= 1].tolist()
if degenerate:
    print("Dropping zero-variance columns:", degenerate)
    num_df = num_df.drop(columns=degenerate)

# =========================
# Impute & Scale
# =========================
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imputed = imputer.fit_transform(num_df)  # original scale (imputed) for orig-feature plots & summaries
X = scaler.fit_transform(X_imputed)        # scaled for KMeans

print("Final feature matrix shape:", X.shape)

# Keep a DataFrame of imputed numeric features for plotting by original columns
num_imputed_df = pd.DataFrame(X_imputed, columns=num_df.columns).reset_index(drop=True)

# =========================
# Decide which original pair to plot
# =========================
def pick_feature_pair(df_num, preferred_pair):
    cols = list(df_num.columns)
    if preferred_pair:
        a, b = preferred_pair
        if a in cols and b in cols:
            return a, b
    # Auto-pick by variance if not found
    var = df_num.var(numeric_only=True).sort_values(ascending=False)
    if len(var) < 2:
        raise RuntimeError("Need at least two numeric columns to plot an original-feature scatter.")
    return var.index[0], var.index[1]

feat_x, feat_y = pick_feature_pair(num_imputed_df, FEATURE_PAIR)
print(f"Original-feature plot pair: ({feat_x}, {feat_y})")

# =========================
# PCA (fit once for consistent projections across k)
# =========================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

def plot_cluster_scatter_pca(X2d, labels, title, outpath):
    plt.figure(figsize=(7, 6))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(X2d[m, 0], X2d[m, 1], s=14, alpha=0.8, label=f"Cluster {c}")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_cluster_scatter_orig(df_num_imputed, labels, fx, fy, title, outpath):
    x = df_num_imputed[fx].values
    y = df_num_imputed[fy].values
    plt.figure(figsize=(7, 6))
    for c in np.unique(labels):
        m = labels == c
        plt.scatter(x[m], y[m], s=14, alpha=0.8, label=f"Cluster {c}")
    plt.title(title)
    plt.xlabel(fx)
    plt.ylabel(fy)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# =========================
# K sweep (k = 3 .. 15) — save ONLY cluster plots (PCA & orig-pair) for each k
# =========================
ks = list(range(3, 16))
metrics_rows = []

best_k = None
best_sil = -np.inf

for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels_k = km.fit_predict(X)

    # Metrics (for table/report, not plotted)
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
        {"k": k, "inertia": km.inertia_, "silhouette": sil, "calinski_harabasz": ch, "davies_bouldin": dbi}
    )

    # 1) PCA cluster plot
    pca_path = f"figures/task02_kmeans_k{k:02d}_pca_clusters.png"
    plot_cluster_scatter_pca(X_pca, labels_k, f"KMeans clusters (k={k}) – PCA 2D", pca_path)
    print(f"[FIG] {pca_path}")

    # 2) Original-feature pair cluster plot
    orig_path = f"figures/task02_kmeans_k{k:02d}_orig_{feat_x}_{feat_y}_clusters.png"
    plot_cluster_scatter_orig(num_imputed_df, labels_k, feat_x, feat_y,
                              f"KMeans clusters (k={k}) – {feat_x} vs {feat_y}", orig_path)
    print(f"[FIG] {orig_path}")

    # Track best k by silhouette
    if not np.isnan(sil) and sil > best_sil:
        best_sil = sil
        best_k = k

# Save metrics CSV
metrics_df = pd.DataFrame(metrics_rows).sort_values("k").reset_index(drop=True)
metrics_path = "outputs/task02_kmeans_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"[CSV] Metrics saved -> {metrics_path}")

# Fallback if all silhouettes NaN
if best_k is None:
    print("[WARN] All silhouette scores NaN; falling back to min DBI.")
    best_k = int(metrics_df.loc[metrics_df["davies_bouldin"].idxmin(), "k"])

print(f"\nChosen best k: {best_k}")

# =========================
# Final fit and highlighted cluster plots for best_k
# =========================
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = kmeans.fit_predict(X)

df_clusters = df.copy()
df_clusters["cluster"] = labels

# Per-cluster means (on imputed numeric features)
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

# Highlighted PCA plot
best_pca_path = f"figures/task02_kmeans_bestk{best_k:02d}_pca_clusters.png"
plot_cluster_scatter_pca(X_pca, labels, f"KMeans clusters (k={best_k}) – PCA 2D [Best k]", best_pca_path)
print(f"[FIG] {best_pca_path}")

# Highlighted original-feature pair plot
best_orig_path = f"figures/task02_kmeans_bestk{best_k:02d}_orig_{feat_x}_{feat_y}_clusters.png"
plot_cluster_scatter_orig(num_imputed_df, labels, feat_x, feat_y,
                          f"KMeans clusters (k={best_k}) – {feat_x} vs {feat_y} [Best k]", best_orig_path)
print(f"[FIG] {best_orig_path}")

# Export labeled data (ID + cluster if ID found)
assign_path = "outputs/task02_cluster_assignments.csv"
cols_to_keep = [id_col] if id_col else []
cols_to_keep += ["cluster"]
df_clusters[cols_to_keep].to_csv(assign_path, index=False)
print(f"[CSV] Cluster assignments saved -> {assign_path}")

print("\nDone. Check the figures/ folder — each k has TWO cluster plots (PCA and original-feature pair).")
