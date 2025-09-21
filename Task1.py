# I304 – Assessment 1 (Task 01: Linear Regression with Yahoo Finance)

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import yfinance as yf

# Config (edit if you like)
TICKER  = "AAPL"   # e.g., "MSFT", "TSLA", "GOOG", "SPY"
PERIOD  = "5y"     # "1y" | "5y" | "10y" | "max"
INTERVAL = "1d"    # "1d" | "1wk" | "1mo"
os.makedirs("figures", exist_ok=True)

# Helpers
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x is not None]).strip()
            for tup in df.columns.to_flat_index()
        ]
    df.columns = [str(c) for c in df.columns]
    if pd.Index(df.columns).duplicated().any():
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    return df

def resolve_col(df: pd.DataFrame, target: str) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    t = target.lower()
    if t in lower_map:
        return lower_map[t]

    # try contains token (e.g., 'close' in 'AAPL_Close' or 'Close_AAPL')
    candidates = [c for c in df.columns if t in c.lower()]
    if candidates:
        # prefer shortest name (usually the clean one)
        candidates.sort(key=len)
        return candidates[0]

    raise RuntimeError(f"Required column '{target}' not found. Columns: {list(df.columns)}")

def ensure_series(obj) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.squeeze(axis=1)
    return pd.Series(obj)

# 1) Download & sanitize
print(f"Downloading {TICKER} history ({PERIOD}, {INTERVAL}) …")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=False)

if df is None or df.empty:
    raise RuntimeError("No data returned. Try a different TICKER/PERIOD/INTERVAL.")

df = flatten_columns(df)

# Explanations (printed for your report)
explanations = {
    "Open": "Price at market open for the day",
    "High": "Highest price during the day",
    "Low": "Lowest price during the day",
    "Close": "Price at market close for the day (target y)",
    "Adj Close": "Close adjusted for splits/dividends",
    "Volume": "Number of shares traded during the day",
}
print("\nColumns:", list(df.columns)[:12], ("… + more" if len(df.columns) > 12 else ""))
print("Column explanations (generic):")
for k, v in explanations.items():
    print(f" - {k}: {v}")

# Resolve canonical columns robustly
OPEN_COL   = resolve_col(df, "Open")
HIGH_COL   = resolve_col(df, "High")
LOW_COL    = resolve_col(df, "Low")
CLOSE_COL  = resolve_col(df, "Close")
VOLUME_COL = resolve_col(df, "Volume")


# 2) Feature engineering
df = df.dropna().copy()

# Moving averages of Close (avoid Adj Close to prevent leakage)
df["MA5"]  = pd.to_numeric(df[CLOSE_COL], errors="coerce").rolling(5).mean()
df["MA10"] = pd.to_numeric(df[CLOSE_COL], errors="coerce").rolling(10).mean()

df = df.dropna().copy()  # drop initial NaNs from rolling

# Final features (6 total)
feature_cols_resolved = [OPEN_COL, HIGH_COL, LOW_COL, VOLUME_COL, "MA5", "MA10"]

# Build X (numeric DataFrame) and y (Series)
X = df.loc[:, feature_cols_resolved].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df.loc[:, CLOSE_COL], errors="coerce")

mask = X.notna().all(axis=1) & y.notna()
X = X.loc[mask]
y = y.loc[mask]

print("\nUsing features:", feature_cols_resolved, "| Target:", CLOSE_COL)
print("X shape:", X.shape, "y shape:", y.shape)

# 3) Plot: Open vs Close
plt.figure(figsize=(7, 5))
plt.scatter(df.loc[mask, OPEN_COL], y, s=10, alpha=0.6)
plt.title(f"{TICKER}: Open vs Close")
plt.xlabel("Open")
plt.ylabel("Close")
plt.tight_layout()
plt.savefig(f"figures/task01_{TICKER}_open_vs_close.png", dpi=150)
plt.show()



# 4) Correlations vs Close
corr_matrix = df.loc[mask, feature_cols_resolved + [CLOSE_COL]].corr(numeric_only=True)

# Force to Series to avoid pandas version quirks
corrs = ensure_series(corr_matrix.loc[:, CLOSE_COL]).drop(labels=[CLOSE_COL], errors="ignore")

# Drop Volume_AAPL if present
corrs = corrs.drop(labels=["Volume_AAPL"], errors="ignore")

# Sort remaing features
corrs = corrs.sort_values(ascending=False)

print("\nCorrelation with Close:")
print(corrs)

plt.figure(figsize=(7, 4))
ensure_series(corrs).plot(kind="bar")
plt.title(f"{TICKER}: Feature correlation with Close")
plt.ylabel("Pearson r")
plt.tight_layout()
plt.savefig(f"figures/task01_{TICKER}_feature_corrs.png", dpi=150)
plt.show()

# 5) Train/Test & Fit (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, shuffle=True
)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Feature names as seen by the model (sklearn >= 1.0)
feature_names = (
    list(getattr(linreg, "feature_names_in_", X_train.columns))
    if hasattr(linreg, "feature_names_in_") else list(X_train.columns)
)

coefs_arr = np.asarray(linreg.coef_).ravel()
if coefs_arr.shape[0] != len(feature_names):
    print(f"[WARN] coef size {coefs_arr.shape[0]} != features {len(feature_names)}; aligning.")
    feature_names = feature_names[:coefs_arr.shape[0]]

intercept = float(linreg.intercept_)
coeffs = pd.Series(coefs_arr, index=feature_names)

y_pred_train = linreg.predict(X_train)
y_pred_test  = linreg.predict(X_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)

print("\n=== Linear Regression Results (fit_intercept=True) ===")
print(f"Intercept: {intercept:.6f}")
print("Coefficients:")
for name, val in coeffs.items():
    print(f"  {name:>10}: {val:.6f}")
print(f"R^2 (train): {r2_train:.4f}")
print(f"R^2 (test) : {r2_test:.4f}")

# Predicted vs Actual (Test)
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_test, s=10, alpha=0.6)
plt.title(f"{TICKER}: Predicted vs Actual (Test Set)")
plt.xlabel("Actual Close")
plt.ylabel("Predicted Close")
plt.tight_layout()
plt.savefig(f"figures/task01_{TICKER}_pred_vs_actual_test.png", dpi=150)
plt.show()

# 6) Predict 3 imaginary rows
quantiles = X.quantile([0.10, 0.50, 0.90])
imaginary = pd.DataFrame(
    [quantiles.loc[0.10], quantiles.loc[0.50], quantiles.loc[0.90]],
    index=["low_p10", "median_p50", "high_p90"]
)
imaginary_predictions = linreg.predict(imaginary)

print("\nImaginary feature rows and predicted Close:")
for idx, pred in zip(imaginary.index, imaginary_predictions):
    print(f"  {idx:>10}: predicted Close = {pred:.2f}")
print("\nImaginary rows used:\n", imaginary)



# 7) Refit with fit_intercept=False on FULL X
linreg_no_intercept = LinearRegression(fit_intercept=False)
linreg_no_intercept.fit(X, y)
coefs_no_int = pd.Series(np.asarray(linreg_no_intercept.coef_).ravel(), index=feature_names)

print("\n=== Linear Regression (fit_intercept=False) on full X ===")
for name, val in coefs_no_int.items():
    print(f"  {name:>10}: {val:.6f}")

print("\nDone. Figures saved in ./figures/")
