import os
import sys
import json
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── Load params ───────────────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

METRICS_PATH = params["evaluate"]["metrics_path"]
BATCH_SIZE   = params["model"]["batch_size"]
EPOCHS       = params["model"]["epochs"]
TEST_SIZE    = params["data"]["test_size"]

# ── Check required files ──────────────────────────────────────────────────────
required = [
    "artifacts/data/X_test.npy",
    "artifacts/data/y_test.npy",
    "artifacts/training_history.json",
    "artifacts/feature_columns.json",
    "artifacts/preprocessing/encoder.pkl",
    "models/model.keras",
    "test/test.csv",
]
for path in required:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

# ── Load model, encoder, and test data ───────────────────────────────────────
print("Loading model and test data...")
model      = tf.keras.models.load_model("models/model.keras")
X_test_cnn = np.load("artifacts/data/X_test.npy")
y_test     = np.load("artifacts/data/y_test.npy")
le         = joblib.load("artifacts/preprocessing/encoder.pkl")
class_names = list(le.classes_)

print(f"X_test_cnn: {X_test_cnn.shape}  |  y_test: {y_test.shape}")
print(f"Classes: {class_names}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

# ── Predictions ───────────────────────────────────────────────────────────────
y_pred_proba = model.predict(X_test_cnn, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

# ── Classification metrics ────────────────────────────────────────────────────
acc      = float(accuracy_score(y_test, y_pred))
f1_macro = float(f1_score(y_test, y_pred, average="macro",  zero_division=0))
f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

print(f"\nTest Results:")
print(f"  Accuracy  : {acc:.4f}")
print(f"  F1 Macro  : {f1_macro:.4f}")
print(f"  F1 Weighted: {f1_weighted:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names,
                            labels=list(range(len(class_names))), zero_division=0))

os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports",   exist_ok=True)

# ── Confusion matrix plot ─────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names, ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion Matrix (Accuracy={acc:.3f})")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: confusion_matrix.png")

# ── Training history plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history_dict["loss"],     label="Train Loss")
axes[0].plot(history_dict["val_loss"], label="Val Loss")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history_dict["accuracy"],     label="Train Accuracy")
axes[1].plot(history_dict["val_accuracy"], label="Val Accuracy")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/training_history.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: training_history.png")

# ── Per-class F1 chart ────────────────────────────────────────────────────────
f1_per_class = f1_score(y_test, y_pred, average=None,
                        labels=list(range(len(class_names))), zero_division=0)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(class_names, f1_per_class, color="steelblue", edgecolor="black")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Class")
ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score")
plt.xticks(rotation=45, ha="right")
for bar, val in zip(bars, f1_per_class):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("per_class_f1.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/per_class_f1.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: per_class_f1.png")

# ── metrics.json (read by DVC) ────────────────────────────────────────────────
metrics_out = {
    "timestamp":    datetime.now().isoformat(),
    "model_type":   "1D CNN Classifier",
    "task":         "obesity level classification",
    "test_size":    TEST_SIZE,
    "final_epoch":  len(history_dict["loss"]),
    "metrics": {
        "accuracy":      acc,
        "f1_macro":      f1_macro,
        "f1_weighted":   f1_weighted,
        "f1_per_class":  {cls: float(f1) for cls, f1 in zip(class_names, f1_per_class)},
    },
    "training_history": {
        "final_train_loss":     float(history_dict["loss"][-1]),
        "final_val_loss":       float(history_dict["val_loss"][-1]),
        "final_train_accuracy": float(history_dict["accuracy"][-1]),
        "final_val_accuracy":   float(history_dict["val_accuracy"][-1]),
    }
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"Saved: {METRICS_PATH}")

# ── Save full evaluation metrics for monitoring ───────────────────────────────
with open("artifacts/metrics/evaluation_metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)

# ── metrics.txt (human readable) ─────────────────────────────────────────────
report_str = classification_report(y_test, y_pred, target_names=class_names,
                                   labels=list(range(len(class_names))), zero_division=0)
with open("metrics.txt", "w") as f:
    f.write("=" * 55 + "\n")
    f.write("OBESITY CLASSIFICATION — MODEL METRICS\n")
    f.write("=" * 55 + "\n")
    f.write(f"Accuracy    : {acc:.4f}\n")
    f.write(f"F1 Macro    : {f1_macro:.4f}\n")
    f.write(f"F1 Weighted : {f1_weighted:.4f}\n")
    f.write("=" * 55 + "\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 55 + "\n")
    f.write(report_str)
    f.write("\nTRAINING HISTORY\n")
    f.write("=" * 55 + "\n")
    f.write(f"Final Train Loss    : {history_dict['loss'][-1]:.4f}\n")
    f.write(f"Final Val Loss      : {history_dict['val_loss'][-1]:.4f}\n")
    f.write(f"Final Train Accuracy: {history_dict['accuracy'][-1]:.4f}\n")
    f.write(f"Final Val Accuracy  : {history_dict['val_accuracy'][-1]:.4f}\n")
print("Saved: metrics.txt")

# ── Submission CSV (predictions on test set) ──────────────────────────────────
with open("artifacts/feature_columns.json") as f:
    feature_columns = json.load(f)

freq_maps = {}
if os.path.exists("artifacts/preprocessing/freq_maps.json"):
    with open("artifacts/preprocessing/freq_maps.json") as f:
        freq_maps = json.load(f)

scaler = None
if os.path.exists("artifacts/preprocessing/scaler.pkl"):
    scaler = joblib.load("artifacts/preprocessing/scaler.pkl")

dtest     = pd.read_csv("test/test.csv")
test_data = dtest.copy()

# Save IDs if present
if "ID" in test_data.columns:
    test_ids  = test_data["ID"]
    test_data = test_data.drop("ID", axis=1)
else:
    test_ids = pd.Series(range(len(test_data)))

# Drop target if present
if "y" in test_data.columns:
    test_data = test_data.drop("y", axis=1)

# Apply frequency encoding
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
X_sub = scaler.transform(test_data.values) if scaler else test_data.values
X_sub = X_sub.reshape(X_sub.shape[0], X_sub.shape[1], 1)

preds_sub    = model.predict(X_sub, verbose=0)
preds_labels = le.inverse_transform(np.argmax(preds_sub, axis=1))

submission = pd.DataFrame({"ID": test_ids, "predicted_class": preds_labels})
submission.to_csv("submission.csv", index=False)
submission.to_csv("artifacts/submission.csv", index=False)
print(f"Saved: submission.csv ({len(submission)} predictions)")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
