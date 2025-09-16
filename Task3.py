# Part C – Clustering on heart.csv
# --------------------------------------------------------------
# Run with:
#   pip install pandas numpy matplotlib scikit-learn
#
# Tasks:
# - Read heart.csv
# - Use first 13 columns as features (ignore 'num' classification column)
# - Apply clustering:
#     1) KMeans with n_clusters = 2, 3, 4
#     2) MeanShift (lets it pick cluster count)
# - Visualize clusters with scatter plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, MeanShift

# ======== CONFIG ========
CSV_PATH = "heart.csv"           # same folder
FEATURE_FOR_PLOTS = "trestbps"   # choose feature for x-axis
FEATURE_FOR_PLOTS_Y = "thalch"   # choose feature for y-axis
# ========================

def main():
    # 1) Load data
    df = pd.read_csv(CSV_PATH)

    # Use first 13 columns as features (assignment spec)
    X = df.iloc[:, :13].dropna().reset_index(drop=True)

    # Identify categorical vs numeric
    categorical_cols = [c for c in X.columns if X[c].dtype == "O"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Preprocessing: scale numeric + one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    X_proc = preprocessor.fit_transform(X)

    print("[INFO] Data prepared for clustering")
    print("  Samples:", X_proc.shape[0], "Features after encoding:", X_proc.shape[1])

    # Helper: plot scatter for 2 chosen features coloured by cluster labels
    def scatter_clusters(x_feat, y_feat, labels, title):
        x_vals = X[x_feat].astype(float).values
        y_vals = X[y_feat].astype(float).values
        plt.figure()
        plt.scatter(x_vals, y_vals, c=labels)
        plt.xlabel(x_feat)
        plt.ylabel(y_feat)
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()

    # -------------------------
    # 1) KMeans (k=2,3,4)
    # -------------------------
    for k in [2, 3, 4]:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_proc)

        print(f"\n[KMeans] k={k}")
        print("Cluster sizes:", np.bincount(labels))

        scatter_clusters(FEATURE_FOR_PLOTS, FEATURE_FOR_PLOTS_Y,
                         labels, f"KMeans Clusters (k={k})")

    # -------------------------
    # 2) MeanShift
    # -------------------------
    ms = MeanShift()
    labels_ms = ms.fit_predict(X_proc)

    print("\n[MeanShift]")
    print("Estimated clusters:", len(np.unique(labels_ms)))
    print("Cluster sizes:", np.bincount(labels_ms))

    scatter_clusters(FEATURE_FOR_PLOTS, FEATURE_FOR_PLOTS_Y,
                     labels_ms, "MeanShift Clusters")

if __name__ == "__main__":
    main()
