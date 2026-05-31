"""
Preprocess new_data.csv (must contain a 'y' column with obesity class labels)
and merge it into train/train.csv so the next dvc repro retrains on the
expanded dataset.

Usage:
  python src/preprocess_new_data.py
"""
import os
import sys
import json
import shutil
import pandas as pd
from datetime import datetime

print("=" * 60)
print("PREPROCESSING NEW DATA — OBESITY DATASET")
print("=" * 60)

NEW_DATA_PATH = "data/new_data.csv"

# ── Check new data ────────────────────────────────────────────────────────────
if not os.path.exists(NEW_DATA_PATH):
    print(f"No new data found at {NEW_DATA_PATH}")
    sys.exit(0)

df_new = pd.read_csv(NEW_DATA_PATH)
print(f"Loaded new data: {df_new.shape}")

if "y" not in df_new.columns:
    print("ERROR: new_data.csv must have a 'y' column (obesity class label).")
    sys.exit(1)

print(f"New data class distribution:\n{df_new['y'].value_counts()}")

# ── Load existing training data ───────────────────────────────────────────────
if not os.path.exists("train/train.csv"):
    print("ERROR: train/train.csv not found.")
    sys.exit(1)

df_train = pd.read_csv("train/train.csv")
print(f"\nExisting training data: {df_train.shape}")

# ── Validate columns match ────────────────────────────────────────────────────
train_cols = set(df_train.columns)
new_cols   = set(df_new.columns)
missing    = train_cols - new_cols
extra      = new_cols - train_cols

if missing:
    print(f"New data missing {len(missing)} columns — filling with 0: {list(missing)[:5]}")
    for col in missing:
        df_new[col] = 0

if extra:
    print(f"New data has {len(extra)} extra columns — dropping: {list(extra)[:5]}")
    df_new = df_new.drop(list(extra), axis=1)

# Align column order to train
df_new = df_new[df_train.columns]

# ── Combine ───────────────────────────────────────────────────────────────────
df_combined = pd.concat([df_train, df_new], ignore_index=True)
df_combined = df_combined.dropna(subset=["y"])
print(f"Combined dataset: {df_combined.shape}")
print(f"Combined class distribution:\n{df_combined['y'].value_counts()}")

# ── Backup original training data ─────────────────────────────────────────────
os.makedirs("train", exist_ok=True)
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = f"train/train_backup_{timestamp}.csv"
shutil.copy("train/train.csv", backup_path)
print(f"\nBackup saved: {backup_path}")

# ── Save combined dataset ─────────────────────────────────────────────────────
df_combined.to_csv("train/train.csv", index=False)
print(f"Updated train/train.csv: {len(df_train)} + {len(df_new)} = {len(df_combined)} rows")

# ── Archive processed new data ────────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
archive_path = f"data/processed/new_data_{timestamp}.csv"
shutil.move(NEW_DATA_PATH, archive_path)
print(f"Archived: {archive_path}")

# ── Save preprocessing log ────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_entry = {
    "timestamp":             datetime.now().isoformat(),
    "new_rows_added":        len(df_new),
    "total_training_rows":   len(df_combined),
    "archive_path":          archive_path,
    "new_class_distribution": df_new["y"].value_counts().to_dict(),
}

log_path = "logs/preprocessing.log"
logs = []
if os.path.exists(log_path):
    try:
        with open(log_path) as f:
            logs = json.load(f)
    except Exception:
        logs = []
logs.append(log_entry)
with open(log_path, "w") as f:
    json.dump(logs, f, indent=2)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"  New rows added   : {len(df_new)}")
print(f"  Total train rows : {len(df_combined)}")
print("\nNext step: dvc repro  (or GitHub Actions will do this automatically)")
