# Task 4 – Principal Component Analysis (PCA) with Wine dataset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


# Action 1: Dataset summary
"""
Wine dataset (UCI):
- 178 samples
- 13 chemical analysis features (alcohol, malic acid, ash, magnesium, flavanoids, etc.)
- Target: 3 classes of wine (class 1, 2, 3)
Link: https://archive.ics.uci.edu/ml/machine-learning-databases/wine/
"""

# Action 2: Load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# According to UCI, first column is class label, remaining 13 are features
columns = [
    "class", "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline"
]

df = pd.read_csv(url, header=None, names=columns)
X = df.drop(columns=["class"])
y = df["class"]

# Action 3: PCA (n=2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("[INFO] PCA explained variance ratios:", pca.explained_variance_ratio_)
print("[INFO] Total variance explained by 2 components:", sum(pca.explained_variance_ratio_))

# Action 4: Logistic Regression on PCA features
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

logreg = LogisticRegression(max_iter=1000, multi_class="auto")
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"[RESULT] Logistic Regression accuracy (on 2 PCA components): {acc:.4f}")

# Confusion matrix
os.makedirs("figures", exist_ok=True)
disp = ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["Class 1","Class 2","Class 3"])
plt.title("Task 4 – Logistic Regression Confusion Matrix (PCA features)")
plt.tight_layout()
plt.savefig("figures/task04_logreg_confusion_matrix.png", dpi=150)
plt.show()

# Action 5: Scatter plot
plt.figure(figsize=(7,6))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="viridis", edgecolor="k", s=60)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Task 4 – PCA Projection (Wine Dataset)")
plt.legend(handles=scatter.legend_elements()[0], labels=["Class 1","Class 2","Class 3"])
plt.tight_layout()
plt.savefig("figures/task04_pca_scatter.png", dpi=150)
plt.show()


# Side-by-side scatter (True vs Predicted classes)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# True labels
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k", s=60)
axes[0].set_title("PCA Projection – True Classes")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# Predicted labels (on the full PCA dataset, not just test set)
y_pred_all = logreg.predict(X_pca)
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_all, cmap="viridis", edgecolor="k", s=60)
axes[1].set_title("PCA Projection – Predicted Classes (LogReg)")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

# Shared legend
fig.legend(handles=scatter1.legend_elements()[0], labels=["Class 1","Class 2","Class 3"], loc="lower center", ncol=3)
plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig("figures/task04_pca_true_vs_predicted.png", dpi=150)
plt.show()
