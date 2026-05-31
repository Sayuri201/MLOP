import os
import sys
import json
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

METRICS_PATH = params["evaluate"]["metrics_path"]
BATCH_SIZE   = params["model"]["batch_size"]
EPOCHS       = params["model"]["epochs"]
TEST_SIZE    = params["data"]["test_size"]

# ── Check required files ──────────────────────────────────────────────────────
required = [
    "artifacts/X_test_cnn.npy",
    "artifacts/y_test.npy",
    "artifacts/training_history.json",
    "artifacts/feature_columns.json",
    "models/model.keras",
    "test/test.csv",
]
for path in required:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

# ── Custom R² — must match model.py ──────────────────────────────────────────
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# ── Load model and test data ──────────────────────────────────────────────────
print("Loading model and test data...")
model      = tf.keras.models.load_model("models/model.keras",
                                        custom_objects={"r2_metric": r2_metric})
X_test_cnn = np.load("artifacts/X_test_cnn.npy")
y_test     = np.load("artifacts/y_test.npy")
print(f"X_test_cnn: {X_test_cnn.shape}  |  y_test: {y_test.shape}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

# ── Evaluate ──────────────────────────────────────────────────────────────────
score = model.evaluate(X_test_cnn, y_test, verbose=0)
preds = model.predict(X_test_cnn, verbose=0).flatten()

mse_val  = float(mean_squared_error(y_test, preds))
rmse_val = float(np.sqrt(mse_val))
mae_val  = float(mean_absolute_error(y_test, preds))
r2_val   = float(r2_score(y_test, preds))

print(f"\nTest Results:")
print(f"  MSE:  {mse_val:.4f}")
print(f"  RMSE: {rmse_val:.4f}")
print(f"  MAE:  {mae_val:.4f}")
print(f"  R²:   {r2_val:.4f}")

# ── Plot: predictions vs actual ───────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.4, s=10)
lim = [y_test.min(), y_test.max()]
plt.plot(lim, lim, "r--", lw=2, label="Perfect prediction")
plt.xlabel("True Values (y)")
plt.ylabel("Predicted Values")
plt.title("Predictions vs True Values")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("predictions_vs_actual.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/predictions_vs_actual.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: predictions_vs_actual.png")

# ── Plot: residuals ───────────────────────────────────────────────────────────
residuals = y_test - preds
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].axvline(x=0, color="r", linestyle="--", lw=2)
axes[0].set_title("Residuals Distribution")
axes[0].set_xlabel("Residual"); axes[0].set_ylabel("Count")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(preds, residuals, alpha=0.3, s=8)
axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
axes[1].set_title("Residuals vs Predicted")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("residuals_analysis.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/residuals_analysis.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: residuals_analysis.png")

# ── metrics.json (read by DVC) ────────────────────────────────────────────────
metrics_out = {
    "timestamp":   datetime.now().isoformat(),
    "model_type":  "1D CNN Regression",
    "test_size":   TEST_SIZE,
    "batch_size":  BATCH_SIZE,
    "final_epoch": len(history_dict["loss"]),
    "metrics": {
        "mse":  mse_val,
        "rmse": rmse_val,
        "mae":  mae_val,
        "r2":   r2_val,
    },
    "training_history": {
        "final_train_loss": float(history_dict["loss"][-1]),
        "final_val_loss":   float(history_dict["val_loss"][-1]),
    }
}
if "r2_metric" in history_dict:
    metrics_out["training_history"]["final_train_r2"] = float(history_dict["r2_metric"][-1])
    metrics_out["training_history"]["final_val_r2"]   = float(history_dict["val_r2_metric"][-1])

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"Saved: {METRICS_PATH}")

# ── metrics.txt (human readable) ─────────────────────────────────────────────
with open("metrics.txt", "w") as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 50 + "\n")
    f.write(f"MSE:  {mse_val:.4f}\n")
    f.write(f"RMSE: {rmse_val:.4f}\n")
    f.write(f"MAE:  {mae_val:.4f}\n")
    f.write(f"R2:   {r2_val:.4f}\n")
    f.write("=" * 50 + "\n\n")
    f.write("TRAINING HISTORY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Final Train Loss: {history_dict['loss'][-1]:.4f}\n")
    f.write(f"Final Val Loss:   {history_dict['val_loss'][-1]:.4f}\n")
    if "r2_metric" in history_dict:
        f.write(f"Final Train R2:   {history_dict['r2_metric'][-1]:.4f}\n")
        f.write(f"Final Val R2:     {history_dict['val_r2_metric'][-1]:.4f}\n")
print("Saved: metrics.txt")

# ── Submission CSV (Kaggle-style — test.csv has no y) ─────────────────────────
with open("artifacts/feature_columns.json") as f:
    feature_columns = json.load(f)

if os.path.exists("artifacts/preprocessing/freq_maps.json"):
    with open("artifacts/preprocessing/freq_maps.json") as f:
        freq_maps = json.load(f)
else:
    freq_maps = {}

if os.path.exists("artifacts/preprocessing/scaler.pkl"):
    import joblib
    scaler = joblib.load("artifacts/preprocessing/scaler.pkl")
else:
    scaler = None

dtest     = pd.read_csv("test/test.csv")
test_data = dtest.copy()
if "ID" in test_data.columns:
    test_ids  = test_data["ID"]
    test_data = test_data.drop("ID", axis=1)
else:
    test_ids = pd.Series(range(len(test_data)))

# Apply frequency encoding to categorical columns
cat_cols = [c for c in test_data.columns if test_data[c].dtype == "O"]
for col in cat_cols:
    if col in freq_maps:
        test_data[f"{col}_freq"] = test_data[col].map(freq_maps[col]).fillna(0)
    else:
        test_data[f"{col}_freq"] = 0
test_data = test_data.drop(cat_cols, axis=1)

# Align to training feature columns
for col in feature_columns:
    if col not in test_data.columns:
        test_data[col] = 0.0
test_data = test_data[feature_columns]
test_data = test_data.apply(pd.to_numeric, errors="coerce").fillna(0)

# Scale and reshape
if scaler is not None:
    X_sub = scaler.transform(test_data.values)
else:
    X_sub = test_data.values
X_sub = X_sub.reshape(X_sub.shape[0], X_sub.shape[1], 1)

preds_sub = model.predict(X_sub, verbose=0).flatten()
submission = pd.DataFrame({"ID": test_ids, "y": preds_sub})
submission.to_csv("submission.csv", index=False)
submission.to_csv("artifacts/submission.csv", index=False)
print(f"Saved: submission.csv ({len(submission)} predictions)")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
