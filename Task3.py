# Task 3 – Breast Cancer Wisconsin (Original) — Classification

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.metrics import classification_report


# Setup
os.makedirs("figures", exist_ok=True)
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/breast-cancer-wisconsin.data"
)
columns = [
    "id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
    "bland_chromatin", "normal_nucleoli", "mitoses", "class"
]

# -------------------------------------------------------------------------------

# Load & clean data
df = pd.read_csv(url, names=columns)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="raise")

# Map classes: {2 -> 0 (Benign), 4 -> 1 (Malignant)}
df["class"] = df["class"].map({2: 0, 4: 1}).astype(int)
X = df.drop(columns=["id", "class"])
y = df["class"]


# Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.27, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, solver="liblinear")
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {acc_logreg:.4f}")
print("\n[LogReg] Classification report:\n",
      classification_report(y_test, y_pred_logreg, target_names=["Benign","Malignant"]))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"KNN Accuracy: {acc_knn:.4f}")
print("\n[KNN] Classification report:\n",
      classification_report(y_test, y_pred_knn, target_names=["Benign","Malignant"]))

print(f"\nLogistic Regression vs KNN: {acc_logreg:.4f} vs {acc_knn:.4f}")

# Graphs
# Confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg, labels=[0, 1])
disp_logreg = ConfusionMatrixDisplay(confusion_matrix=cm_logreg,
                                     display_labels=["Benign", "Malignant"])
disp_logreg.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/task3_logreg_confusion_matrix.png", dpi=150)
plt.show()

# Confusion matrix for KNN
cm_knn = confusion_matrix(y_test, y_pred_knn, labels=[0, 1])
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn,
                                  display_labels=["Benign", "Malignant"])
disp_knn.plot()
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig("figures/task3_knn_confusion_matrix.png", dpi=150)
plt.show()

# Bar plot comparing accuracies
plt.figure()
plt.bar(["Logistic Regression", "KNN"], [acc_logreg, acc_knn])
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig("figures/task3_accuracy_comparison.png", dpi=150)
plt.show()

# Precision, Recall, F1 comparison
from sklearn.metrics import precision_score, recall_score, f1_score

# Logistic Regression metrics
prec_log = precision_score(y_test, y_pred_logreg)
rec_log = recall_score(y_test, y_pred_logreg)
f1_log = f1_score(y_test, y_pred_logreg)

# KNN metrics
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

# Organize into a DataFrame for easy plotting
import pandas as pd
metrics_df = pd.DataFrame({
    "Logistic Regression": [prec_log, rec_log, f1_log],
    "KNN": [prec_knn, rec_knn, f1_knn]
}, index=["Precision", "Recall", "F1-score"])

print("\nMetrics comparison (LogReg vs KNN):")
print(metrics_df)

# Bar plot
metrics_df.plot(kind="bar")
plt.title("Precision, Recall, and F1-score Comparison")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)  # zoom in to highlight differences
plt.tight_layout()
plt.savefig("figures/task3_precision_recall_f1_comparison.png", dpi=150)
plt.show()

