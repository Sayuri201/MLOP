"""
One-time data setup script.
Run from the project root:
    python data_setup.py

Reads cleaned_obesity.csv (place it in the project root),
splits 80/20 into train/train.csv and test/test.csv,
and creates data/new_data.csv with a small batch of samples
to test the retraining pipeline.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

OBESITY_CSV = "cleaned_obesity.csv"

if not os.path.exists(OBESITY_CSV):
    print(f"ERROR: {OBESITY_CSV} not found in project root.")
    print("Please copy cleaned_obesity.csv to:", os.path.abspath("."))
    sys.exit(1)

print("Loading cleaned_obesity.csv ...")
df = pd.read_csv(OBESITY_CSV)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Rename target column to 'y' (pipeline convention)
if "NObeyesdad" not in df.columns:
    print("ERROR: 'NObeyesdad' column not found.")
    sys.exit(1)

df = df.rename(columns={"NObeyesdad": "y"})
print(f"\nClass distribution:\n{df['y'].value_counts()}")

# ── Train / test split ─────────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["y"]
)
print(f"\nTrain: {train_df.shape}  |  Test: {test_df.shape}")

# ── Create new_data.csv (simulate incoming labeled data for retraining) ────────
# Take 30 stratified samples — these will NOT overlap with test set
new_data_df = train_df.groupby("y", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 4), random_state=7)
).reset_index(drop=True)
print(f"New data sample: {new_data_df.shape}")

# ── Save files ─────────────────────────────────────────────────────────────────
os.makedirs("train",         exist_ok=True)
os.makedirs("test",          exist_ok=True)
os.makedirs("data",          exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

train_df.to_csv("train/train.csv", index=False)
test_df.to_csv("test/test.csv",   index=False)
new_data_df.to_csv("data/new_data.csv", index=False)

print("\n" + "=" * 55)
print("DATA SETUP COMPLETE")
print("=" * 55)
print(f"  train/train.csv   : {len(train_df)} rows")
print(f"  test/test.csv     : {len(test_df)} rows")
print(f"  data/new_data.csv : {len(new_data_df)} rows")
print("\nNext steps:")
print("  1. Run:  dvc init  (if not already done)")
print("  2. Run:  dvc add train/train.csv test/test.csv")
print("  3. Run:  dvc repro")
