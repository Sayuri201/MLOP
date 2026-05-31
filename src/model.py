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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
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
DENSE_1       = params["model"]["dense_units_1"]
DENSE_2       = params["model"]["dense_units_2"]
DENSE_3       = params["model"]["dense_units_3"]
DENSE_4       = params["model"]["dense_units_4"]
DROPOUT_1     = params["model"]["dropout_1"]
DROPOUT_2     = params["model"]["dropout_2"]
NUM_CLASSES   = params["model"]["num_classes"]
ES_PATIENCE   = params["callbacks"]["early_stopping_patience"]
LR_PATIENCE   = params["callbacks"]["reduce_lr_patience"]
LR_FACTOR     = params["callbacks"]["reduce_lr_factor"]
LR_MIN        = params["callbacks"]["reduce_lr_min_lr"]

# ── Directories ───────────────────────────────────────────────────────────────
for d in ["artifacts", "artifacts/preprocessing", "artifacts/data",
          "artifacts/metrics", "artifacts/metadata", "artifacts/models",
          "models", "logs", "reports"]:
    os.makedirs(d, exist_ok=True)

print("=" * 60)
print("OBESITY CLASSIFICATION — MLP TRAINING")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────
for path in ["train/train.csv", "test/test.csv"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        print("Run:  python data_setup.py  first.")
        sys.exit(1)

print("\nLoading data...")
data  = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train shape: {data.shape}  |  Test shape: {dtest.shape}")

if "y" not in data.columns:
    print("ERROR: 'y' column not found in train.csv")
    sys.exit(1)

# ── BMI feature (Weight / Height²) ───────────────────────────────────────────
if "Weight" in data.columns and "Height" in data.columns:
    data["BMI"] = data["Weight"] / (data["Height"] ** 2)
    print("Added feature: BMI = Weight / Height²")

# ── Categorical columns: frequency-encode ────────────────────────────────────
CAT_COLS = [c for c in data.columns if data[c].dtype == "O" and c != "y"]
print(f"\nCategorical columns ({len(CAT_COLS)}): {CAT_COLS}")

freq_maps = {}
for col in CAT_COLS:
    freq = data[col].value_counts().to_dict()
    freq_maps[col] = freq
    data[f"{col}_freq"] = data[col].map(freq)

data = data.drop(CAT_COLS, axis=1)

with open("artifacts/preprocessing/freq_maps.json", "w") as f:
    json.dump(freq_maps, f, indent=2)
print("Saved: artifacts/preprocessing/freq_maps.json")

# ── Drop constant columns ─────────────────────────────────────────────────────
constant_cols = [c for c in data.columns if c != "y" and data[c].nunique() <= 1]
if constant_cols:
    print(f"Dropping {len(constant_cols)} constant columns")
    data = data.drop(constant_cols, axis=1)

with open("artifacts/preprocessing/constant_cols.json", "w") as f:
    json.dump(constant_cols, f)

# ── Features & target ─────────────────────────────────────────────────────────
X = data.drop("y", axis=1).apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean()).fillna(0).values
y_raw = data["y"].values

le = LabelEncoder()
y  = le.fit_transform(y_raw)
NUM_CLASSES = len(le.classes_)

print(f"\nClasses ({NUM_CLASSES}): {list(le.classes_)}")
print(f"X shape: {X.shape}  |  y shape: {y.shape}")

joblib.dump(le, "artifacts/preprocessing/encoder.pkl")
print("Saved: artifacts/preprocessing/encoder.pkl")

feature_columns = list(data.drop("y", axis=1).columns)
with open("artifacts/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)
with open("artifacts/preprocessing/feature_columns.json", "w") as f:
    json.dump(feature_columns, f, indent=2)
with open("artifacts/preprocessing/class_names.json", "w") as f:
    json.dump(list(le.classes_), f, indent=2)
print(f"Saved feature_columns.json ({len(feature_columns)} features)")

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ── StandardScaler ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, "artifacts/preprocessing/scaler.pkl")
print("Saved: artifacts/preprocessing/scaler.pkl")

# ── Save arrays (2D — no reshape needed for MLP) ──────────────────────────────
np.save("artifacts/data/X_train.npy", X_train_scaled)
np.save("artifacts/data/X_test.npy",  X_test_scaled)
np.save("artifacts/data/y_train.npy", y_train)
np.save("artifacts/data/y_test.npy",  y_test)
print("Saved arrays to artifacts/data/")

# ── Build MLP model ───────────────────────────────────────────────────────────
tf.random.set_seed(SEED)
n_features = X_train_scaled.shape[1]

model = Sequential([
    Dense(DENSE_1, activation="relu", input_shape=(n_features,)),
    BatchNormalization(),
    Dropout(DROPOUT_1),

    Dense(DENSE_2, activation="relu"),
    BatchNormalization(),
    Dropout(DROPOUT_2),

    Dense(DENSE_3, activation="relu"),
    Dropout(DROPOUT_2),

    Dense(DENSE_4, activation="relu"),

    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
)
model.summary()

stream = StringIO()
model.summary(print_fn=lambda x: stream.write(x + "\n"))
with open("model_summary.txt", "w", encoding="utf-8") as f:
    f.write(stream.getvalue())
print("Saved: model_summary.txt")

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=LR_FACTOR,
                      patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1)
]

# ── Class weights (handle imbalance) ─────────────────────────────────────────
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights_arr))
print(f"\nClass weights: { {le.classes_[i]: round(w, 2) for i, w in class_weight_dict.items()} }")

# ── Train with DVCLive tracking ───────────────────────────────────────────────
print("\nTraining model...")
with Live(dir="dvclive", report="html") as live:
    live.log_param("epochs",      EPOCHS)
    live.log_param("batch_size",  BATCH_SIZE)
    live.log_param("lr",          LEARNING_RATE)
    live.log_param("num_classes", NUM_CLASSES)
    live.log_param("n_features",  n_features)
    live.log_param("model_type",  "MLP")

    history = model.fit(
        X_train_scaled, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    for i in range(len(history.history["loss"])):
        live.log_metric("train_loss",     history.history["loss"][i])
        live.log_metric("val_loss",       history.history["val_loss"][i])
        live.log_metric("train_accuracy", history.history["accuracy"][i])
        live.log_metric("val_accuracy",   history.history["val_accuracy"][i])
        live.next_step()

print("Training completed!")

# ── Save model ────────────────────────────────────────────────────────────────
model.save("models/model.keras")
model.save("artifacts/models/model.keras")
print("Saved: models/model.keras")

# ── Training plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss over Epochs")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[1].set_title("Accuracy over Epochs")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("model_results.png", dpi=200, bbox_inches="tight")
plt.savefig("artifacts/model_results.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: model_results.png")

# ── Save training history ─────────────────────────────────────────────────────
history_dict = {
    "loss":         [float(v) for v in history.history["loss"]],
    "val_loss":     [float(v) for v in history.history["val_loss"]],
    "accuracy":     [float(v) for v in history.history["accuracy"]],
    "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
}
with open("artifacts/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)
with open("artifacts/metrics/training_history.json", "w") as f:
    json.dump(history_dict, f, indent=2)

# ── Quick test metrics ────────────────────────────────────────────────────────
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)
test_acc     = float(accuracy_score(y_test, y_pred))

print(f"\nTest Accuracy: {test_acc:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_,
                             labels=list(range(NUM_CLASSES)), zero_division=0))

test_metrics = {
    "accuracy":  test_acc,
    "timestamp": datetime.now().isoformat()
}
with open("artifacts/metrics/test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

# ── Model metadata ────────────────────────────────────────────────────────────
model_info = {
    "model_type":            "MLP Classifier",
    "task":                  "obesity level classification",
    "num_classes":           NUM_CLASSES,
    "class_names":           list(le.classes_),
    "n_features":            n_features,
    "n_train_samples":       int(X_train.shape[0]),
    "n_test_samples":        int(X_test.shape[0]),
    "categorical_cols":      CAT_COLS,
    "constant_cols_dropped": len(constant_cols),
    "training_completed":    datetime.now().isoformat(),
    "hyperparameters": {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "dense_units": [DENSE_1, DENSE_2, DENSE_3, DENSE_4]
    },
    "test_performance": test_metrics
}
with open("artifacts/metadata/model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)
with open("artifacts/metadata/last_retrain.txt", "w") as f:
    f.write(datetime.now().isoformat())

data_info = {
    "train_samples":    int(X_train.shape[0]),
    "test_samples":     int(X_test.shape[0]),
    "n_features":       int(X.shape[1]),
    "num_classes":      NUM_CLASSES,
    "class_names":      list(le.classes_),
    "categorical_cols": CAT_COLS,
}
with open("data_info.json", "w") as f:
    json.dump(data_info, f, indent=2)
with open("artifacts/metadata/data_info.json", "w") as f:
    json.dump(data_info, f, indent=2)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("Next: run  python src/evaluate.py")
