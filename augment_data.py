"""
Augments the 488-row cleaned_obesity.csv into a balanced ~2000-row dataset
using SMOTE (Synthetic Minority Oversampling Technique), then saves it as
cleaned_obesity_full.csv and re-runs the train/test/new_data split.

Run once from the project root:
    pip install imbalanced-learn
    python augment_data.py
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("DATA AUGMENTATION WITH SMOTE")
print("=" * 60)

# ── Check source file ─────────────────────────────────────────────────────────
SRC = "cleaned_obesity.csv"
if not os.path.exists(SRC):
    print(f"ERROR: {SRC} not found. Make sure it is in the project root.")
    sys.exit(1)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("ERROR: imbalanced-learn not installed.")
    print("Run:  pip install imbalanced-learn")
    sys.exit(1)

df = pd.read_csv(SRC)
print(f"Original shape  : {df.shape}")
print(f"Target column   : NObeyesdad")
print(f"\nOriginal class distribution:\n{df['NObeyesdad'].value_counts()}")

# ── Encode all columns for SMOTE (needs numeric input) ────────────────────────
df_enc  = df.copy()
encoders = {}

cat_cols = df_enc.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "NObeyesdad"]

for col in cat_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(df_enc["NObeyesdad"])
X = df_enc.drop("NObeyesdad", axis=1).values

# ── Apply SMOTE ───────────────────────────────────────────────────────────────
# Target: ~300 samples per class (balanced)
n_classes     = len(np.unique(y))
target_count  = 300

smote = SMOTE(
    sampling_strategy={
        i: target_count for i in range(n_classes)
        if np.sum(y == i) < target_count
    },
    random_state=42,
    k_neighbors=min(5, min(np.bincount(y)) - 1)
)

X_res, y_res = smote.fit_resample(X, y)
print(f"\nAugmented shape : {X_res.shape}")

# ── Rebuild DataFrame ─────────────────────────────────────────────────────────
feature_cols = df.drop("NObeyesdad", axis=1).columns.tolist()
df_aug = pd.DataFrame(X_res, columns=feature_cols)

# Decode categorical columns back to strings (original values)
for col in cat_cols:
    le = encoders[col]
    df_aug[col] = le.inverse_transform(df_aug[col].round().astype(int).clip(
        0, len(le.classes_) - 1
    ))

# Decode target
df_aug["NObeyesdad"] = le_target.inverse_transform(y_res)

print(f"\nAugmented class distribution:\n{df_aug['NObeyesdad'].value_counts()}")

# ── Save augmented CSV ────────────────────────────────────────────────────────
OUT = "cleaned_obesity_full.csv"
df_aug.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}  ({len(df_aug)} rows)")

# ── Re-run train/test/new_data split on the full dataset ─────────────────────
print("\nRe-splitting into train/test/new_data ...")

df_aug = df_aug.rename(columns={"NObeyesdad": "y"})

train_df, test_df = train_test_split(
    df_aug, test_size=0.2, random_state=42, stratify=df_aug["y"]
)

new_data_df = train_df.groupby("y", group_keys=False).apply(
    lambda x: x.sample(min(len(x), 4), random_state=7),
    include_groups=False
).reset_index(drop=True)

# Re-attach y column (groupby without include_groups drops it)
new_data_df = train_df.groupby("y").apply(
    lambda x: x.sample(min(len(x), 4), random_state=7)
).reset_index(drop=True)

os.makedirs("train", exist_ok=True)
os.makedirs("test",  exist_ok=True)
os.makedirs("data",  exist_ok=True)

train_df.to_csv("train/train.csv",     index=False)
test_df.to_csv("test/test.csv",        index=False)
new_data_df.to_csv("data/new_data.csv", index=False)

print("\n" + "=" * 60)
print("AUGMENTATION COMPLETE")
print("=" * 60)
print(f"  train/train.csv   : {len(train_df)} rows")
print(f"  test/test.csv     : {len(test_df)} rows")
print(f"  data/new_data.csv : {len(new_data_df)} rows")
print("\nNext step:  dvc repro")
