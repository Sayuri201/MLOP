import os
import sys
import json
from io import StringIO
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from dvclive import Live

# ── Load hyperparameters ──────────────────────────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS        = params["model"]["epochs"]
BATCH_SIZE    = params["model"]["batch_size"]
LEARNING_RATE = params["model"]["learning_rate"]
SEED          = params["data"]["random_seed"]
TEST_SIZE     = params["data"]["test_size"]
CNN_FILTERS   = params["model"]["cnn_filters"]
KERNEL_SIZE   = params["model"]["kernel_size"]
POOL_SIZE     = params["model"]["pool_size"]
DENSE_1       = params["model"]["dense_units_1"]
DENSE_2       = params["model"]["dense_units_2"]
DENSE_3       = params["model"]["dense_units_3"]
DROPOUT_1     = params["model"]["dropout_1"]
DROPOUT_2     = params["model"]["dropout_2"]
ES_PATIENCE   = params["callbacks"]["early_stopping_patience"]
LR_PATIENCE   = params["callbacks"]["reduce_lr_patience"]
LR_FACTOR     = params["callbacks"]["reduce_lr_factor"]
LR_MIN        = params["callbacks"]["reduce_lr_min_lr"]

# ── Directories ───────────────────────────────────────────────────────────────
for d in ["artifacts", "artifacts/preprocessing", "artifacts/data",
          "artifacts/metrics", "artifacts/metadata", "models", "logs"]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("STARTING CNN REGRESSION MODEL — TRAINING")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
for path in ["train/train.csv", "test/test.csv"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        sys.exit(1)

print("\nLoading data...")
data  = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train shape: {data.shape}  |  Test shape: {dtest.shape}")

# ── Drop ID column ────────────────────────────────────────────────────────────
if "ID" in data.columns:
    data = data.drop("ID", axis=1)

if "y" not in data.columns:
    print("ERROR: 'y' column not found in train.csv")
    sys.exit(1)

# ── Categorical columns: X0, X1, X2, X3, X4, X5, X6, X8 ─────────────────────
# Use frequency encoding so CNN can learn from them numerically.
CAT_COLS = [c for c in data.columns if data[c].dtype == "O" and c not in ["ID", "y"]]
print(f"\nCategorical columns ({len(CAT_COLS)}): {CAT_COLS}")

freq_maps = {}
for col in CAT_COLS:
    freq = data[col].value_counts().to_dict()
    freq_maps[col] = freq
    data[f"{col}_freq"] = data[col].map(freq)

data = data.drop(CAT_COLS, axis=1)

# Save frequency maps for use in preprocess_new_data.py and evaluate.py
with open("artifacts/preprocessing/freq_maps.json", "w") as f:
    json.dump(freq_maps, f, indent=2)
print("Saved: artifacts/preprocessing/freq_maps.json")

# ── Constant columns ──────────────────────────────────────────────────────────
constant_cols = [c for c in data.columns if data[c].nunique() <= 1]
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant columns")
    data = data.drop(constant_cols, axis=1)

# Save dropped constant cols so evaluate/monitor can replicate
with open("artifacts/preprocessing/constant_cols.json", "w") as f:
    json.dump(constant_cols, f)

# ── Features & target ─────────────────────────────────────────────────────────
X = data.drop("y", axis=1).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean()).fillna(0).values
y = data["y"].values
print(f"\nX shape: {X.shape}  |  y shape: {y.shape}")
print(f"Target — mean: {y.mean():.2f}  std: {y.std():.2f}  min: {y.min():.2f}  max: {y.max():.2f}")

# Save feature columns
feature_columns = list(data.drop("y", axis=1).columns)
with open("artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)
with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)
print(f"Saved feature_columns.json ({len(feature_columns)} features)")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ── StandardScaler ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")
print("Saved: artifacts/preprocessing/scaler.pkl")

# ── Save data splits ──────────────────────────────────────────────────────────
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

np.save("artifacts/X_test_cnn.npy",  X_test_cnn)
np.save("artifacts/y_test.npy",      y_test)
np.save("artifacts/data/X_train_cnn.npy", X_train_cnn)
np.save("artifacts/data/X_test_cnn.npy",  X_test_cnn)
np.save("artifacts/data/y_train.npy", y_train)
np.save("artifacts/data/y_test.npy",  y_test)
print("Saved CNN-shaped arrays to artifacts/")

# ── Custom R² metric ──────────────────────────────────────────────────────────
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# ── Build 1D CNN ──────────────────────────────────────────────────────────────
tf.random.set_seed(SEED)
n_features = X_train_cnn.shape[1]

model = Sequential([
    Conv1D(CNN_FILTERS[0], kernel_size=KERNEL_SIZE, activation="relu",
           input_shape=(n_features, 1), padding="same"),
    MaxPooling1D(pool_size=POOL_SIZE),

    Conv1D(CNN_FILTERS[1], kernel_size=KERNEL_SIZE, activation="relu", padding="same"),
    MaxPooling1D(pool_size=POOL_SIZE),

    Conv1D(CNN_FILTERS[2], kernel_size=KERNEL_SIZE, activation="relu", padding="same"),
    MaxPooling1D(pool_size=POOL_SIZE),

    Flatten(),
    Dense(DENSE_1, activation="relu"),
    Dropout(DROPOUT_1),
    Dense(DENSE_2, activation="relu"),
    Dropout(DROPOUT_2),
    Dense(DENSE_3, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["mae", r2_metric]
)
model.summary()

stream = StringIO()
model.summary(print_fn=lambda x: stream.write(x + "\n"))
with open("model_summary.txt", "w", encoding="utf-8") as f:
    f.write(stream.getvalue())

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR,
                      patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1)
]

# ── Train with DVCLive tracking ───────────────────────────────────────────────
print("\nTraining model...")
with Live(dir="dvclive", report="html") as live:
    live.log_param("epochs",      EPOCHS)
    live.log_param("batch_size",  BATCH_SIZE)
    live.log_param("lr",          LEARNING_RATE)
    live.log_param("cnn_filters", str(CNN_FILTERS))
    live.log_param("n_features",  n_features)

    history = model.fit(
        X_train_cnn, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test_cnn, y_test),
        callbacks=callbacks,
        verbose=1
    )

    for i in range(len(history.history["loss"])):
        live.log_metric("train_loss", history.history["loss"][i])
        live.log_metric("val_loss",   history.history["val_loss"][i])
        live.log_metric("train_mae",  history.history["mae"][i])
        live.log_metric("val_mae",    history.history["val_mae"][i])
        if "r2_metric" in history.history:
            live.log_metric("train_r2", history.history["r2_metric"][i])
            live.log_metric("val_r2",   history.history["val_r2_metric"][i])
        live.next_step()

print("Training completed!")

# ── Save model ────────────────────────────────────────────────────────────────
model.save("models/model.keras")
print("Saved: models/model.keras")

# ── Training plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss (MSE) over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE")
axes[0].legend(); axes[0].grid(True)

if "r2_metric" in history.history:
    axes[1].plot(history.history["r2_metric"],     label="Train R²")
    axes[1].plot(history.history["val_r2_metric"], label="Val R²")
    axes[1].set_title("R² Score over Epochs")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("R²")
    axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("model_results.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/model_results.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: model_results.png")

# ── Save training history ─────────────────────────────────────────────────────
history_dict = {
    "loss":     [float(v) for v in history.history["loss"]],
    "val_loss": [float(v) for v in history.history["val_loss"]],
    "mae":      [float(v) for v in history.history["mae"]],
    "val_mae":  [float(v) for v in history.history["val_mae"]],
}
if "r2_metric" in history.history:
    history_dict["r2_metric"]     = [float(v) for v in history.history["r2_metric"]]
    history_dict["val_r2_metric"] = [float(v) for v in history.history["val_r2_metric"]]

with open("artifacts/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)
with open("artifacts/metrics/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

# ── Test metrics ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_cnn, verbose=0).flatten()
test_metrics = {
    "mse":       float(mean_squared_error(y_test, y_pred)),
    "rmse":      float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "mae":       float(mean_absolute_error(y_test, y_pred)),
    "r2":        float(r2_score(y_test, y_pred)),
    "timestamp": datetime.now().isoformat()
}
with open("artifacts/metrics/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"Test — MSE: {test_metrics['mse']:.4f}  RMSE: {test_metrics['rmse']:.4f}  "
      f"MAE: {test_metrics['mae']:.4f}  R²: {test_metrics['r2']:.4f}")

# ── Model metadata ────────────────────────────────────────────────────────────
model_info = {
    "model_type":           "1D CNN Regression",
    "n_features":           n_features,
    "n_train_samples":      int(X_train.shape[0]),
    "n_test_samples":       int(X_test.shape[0]),
    "categorical_cols":     CAT_COLS,
    "constant_cols_dropped": len(constant_cols),
    "training_completed":   datetime.now().isoformat(),
    "hyperparameters": {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "cnn_filters": CNN_FILTERS
    },
    "test_performance": test_metrics
}
with open("artifacts/metadata/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().isoformat())

# Data info for monitor.py
data_info = {
    "train_samples":  int(X_train.shape[0]),
    "test_samples":   int(X_test.shape[0]),
    "n_features":     int(X.shape[1]),
    "categorical_cols": CAT_COLS,
    "target_mean":    float(y.mean()),
    "target_std":     float(y.std()),
    "target_min":     float(y.min()),
    "target_max":     float(y.max()),
}
with open("data_info.json", "w") as f:
    json.dump(data_info, f, indent=2)
with open("artifacts/metadata/data_info.json", "w") as f:
    json.dump(data_info, f, indent=2)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Next: run  python src/evaluate.py")
