"""
Dataset Preparation Script for DDoS Attack Detection.
Cleans raw labelled dataset, resolves conflicting label combinations,
removes duplicate records, and generates processed and train/test splits.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Check for original dataset
input_path = PROJECT_DIR / "data" / "original" / "Flooding_Based_DDoS_Multi-class_Dataset.csv"
if not input_path.exists():
    input_path = PROJECT_DIR / "data" / "Flooding_Based_DDoS_Multi-class_Dataset.csv"

# Processed paths
output_processed_dir = PROJECT_DIR / "data" / "processed"
output_processed_dir.mkdir(parents=True, exist_ok=True)
output_path = output_processed_dir / "ddos_cleaned.csv"

# External splits
external_dir = PROJECT_DIR / "data" / "external"
external_dir.mkdir(parents=True, exist_ok=True)
train_path = external_dir / "train.csv"
test_path = external_dir / "test.csv"

print(f"Reading raw dataset from: {input_path}")
df = pd.read_csv(input_path, low_memory=False)
orig_rows = len(df)

label_col = "Label"
feature_cols = [col for col in df.columns if col != label_col]

# Remove feature combinations assigned conflicting labels
label_counts = df.groupby(feature_cols, dropna=False)[label_col].nunique()
conflicting_features = label_counts[label_counts > 1].index

feature_index = pd.MultiIndex.from_frame(df[feature_cols])
conflict_mask = feature_index.isin(conflicting_features)
df = df.loc[~conflict_mask].copy()

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# Save cleaned dataset
df.to_csv(output_path, index=False)
# Also maintain root data/ddos_cleaned.csv for backward compatibility
df.to_csv(PROJECT_DIR / "data" / "ddos_cleaned.csv", index=False)

# Create stratified train (80%) / test (20%) splits
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df[label_col],
)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n" + "=" * 55)
print("DATASET PREPARATION & SPLIT COMPLETE")
print("=" * 55)
print(f"Original rows       : {orig_rows}")
print(f"Cleaned total rows  : {len(df)}")
print(f"Feature columns     : {len(feature_cols)}")
print(f"Train set rows      : {len(train_df)} -> {train_path}")
print(f"Test set rows       : {len(test_df)} -> {test_path}")
print("\nClass distribution in cleaned dataset:")
print(df[label_col].value_counts().to_string())
print(f"\nSaved cleaned dataset to: {output_path}")
