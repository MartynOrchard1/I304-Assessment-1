# task02_clustering.py
# I304 – Assessment 1 (Task 02: K-Means Clustering – Credit Card dataset)
# Run: python task02_clustering.py

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
CSV_PATH = "data/CC_GENERAL.csv"   # <- change if your filename differs
ID_GUESS_TOKENS = ("id", "cust", "customer")  # used to auto-detect an ID column

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
# Identify ID column (if present) & keep only numeric features
# =========================
df = df_raw.copy()

# Try common ID variants first
id_col = None
for c in df.columns:
    cl = str(c).lower()
    if any(tok in cl for tok in ID_GUESS_TOKENS):
        id_col = c
        break

# Secondary heuristic: object dtype columns with unique count ~ nrows often are IDs
if id_col is None:
    obj_like = [c for c in df.columns if df[c].dtype == "object"]
    for c in obj_like:
        if df[c].nunique(dropna=True) >= int(0.95 * len(df)):
            id_col = c
            break

if id_col:
    print(f"Detected ID column: {id_col}")
else:
    print("No ID column detected (that’s fine).")

# Keep numeric columns only (KMeans requires numeric)
num_df = df.select_dtypes(include=[np.number]).copy()
if num_df.empty:
    raise RuntimeError("No numeric columns found for clustering.")

# Drop obvious degenerate columns (zero variance)
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
# K sweep (k = 3 .. 15)
# =========================
ks = list(range(3, 16))
inertias = []
sil_scores = []
ch_scores = []
dbi_scores = []

for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X)

    inertias.append(km.inertia_)
    # Some metrics can error on pathological data; guard them
    try:
        sil_scores.append(silhouette_score(X, labels))
    except Exception:
        sil_scores.append(np.nan)
    try:
        ch_scores.append(calinski_harabasz_score(X, labels))
    except Exception:
        ch_scores.append(np.nan)
    try:
        dbi_scores.append(davies_bouldin_score(X, labels))
    except Exception:
        dbi_scores.append(np.nan)

# =========================
# Plots: Elbow + Metrics
# =========================
plt.figure(figsize=(7, 5))
plt.plot(ks, inertias, marker="o")
plt.title("KMeans: Elbow (Inertia vs k)")
plt.xlabel("k (clusters)")
plt.ylabel("Inertia (sum of squared distances)")
plt.tight_layout()
plt.savefig("figures/task02_elbow_inertia.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(ks, sil_scores, marker="o")
plt.title("KMeans: Silhouette Score vs k")
plt.xlabel("k (clusters)")
plt.ylabel("Silhouette (higher is better)")
plt.tight_layout()
plt.savefig("figures/task02_silhouette.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(ks, ch_scores, marker="o")
plt.title("KMeans: Calinski–Harabasz vs k")
plt.xlabel("k (clusters)")
plt.ylabel("CH Score (higher is better)")
plt.tight_layout()
plt.savefig("figures/task02_calinski_harabasz.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(ks, dbi_scores, marker="o")
plt.title("KMeans: Davies–Bouldin vs k")
plt.xlabel("k (clusters)")
plt.ylabel("DBI (lower is better)")
plt.tight_layout()
plt.savefig("figures/task02_davies_bouldin.png", dpi=150)
plt.show()

# =========================
# Pick best k (max silhouette by default)
# =========================
sil_array = np.array(sil_scores, dtype=float)
if np.all(np.isnan(sil_array)):
    # fallback: choose k with minimum DBI
    print("[WARN] All silhouette scores are NaN; falling back to min DBI.")
    dbi_array = np.array(dbi_scores, dtype=float)
    best_k = ks[int(np.nanargmin(dbi_array))]
else:
    best_k = ks[int(np.nanargmax(sil_array))]

print(f"\nChosen best k (by silhouette): {best_k}")

# =========================
# Final fit and labeling
# =========================
kmeans = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = kmeans.fit_predict(X)

df_clusters = df.copy()
df_clusters["cluster"] = labels

# Save a tidy summary of cluster means (on original numeric features)
cluster_summary = (
    pd.DataFrame(X_imputed, columns=num_df.columns)
    .assign(cluster=labels)
    .groupby("cluster")
    .mean()
    .sort_index()
)
cluster_summary.to_csv("outputs/task02_cluster_feature_means.csv", index=True)
print("Cluster feature means saved to outputs/task02_cluster_feature_means.csv")

# =========================
# PCA (2D) for visualization
# =========================
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 6))
for c in range(best_k):
    mask = labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=12, alpha=0.7, label=f"Cluster {c}")
plt.title(f"KMeans Clusters (k={best_k}) – PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.tight_layout()
plt.savefig("figures/task02_pca_clusters.png", dpi=150)
plt.show()

# =========================
# Export labeled data (optional, helpful for marking)
# =========================
out_df = df_clusters.copy()
# Keep ID if found, plus cluster
cols_to_keep = [id_col] if id_col else []
cols_to_keep += ["cluster"]
out_df[cols_to_keep].to_csv("outputs/task02_cluster_assignments.csv", index=False)
print("Cluster assignments saved to outputs/task02_cluster_assignments.csv")

print("\nDone. See figures/ for charts and outputs/ for CSV summaries.")
