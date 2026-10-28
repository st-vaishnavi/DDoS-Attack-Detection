
from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "Flooding_Based_DDoS_Multi-class_Dataset.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at:\n{DATA_PATH}\n"
        "Check that the CSV is inside the project's data folder."
    )

# Load dataset
df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 60)
print("DDOS DATASET QUALITY CHECK")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

print("\n--- COLUMN NAMES AND DATA TYPES ---")
print(df.dtypes.to_string())

print("\n--- FIRST 5 ROWS ---")
print(df.head().to_string())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).to_string()
      if missing.any() else "No missing values.")

print("\n--- EXACT DUPLICATE ROWS ---")
duplicate_count = int(df.duplicated().sum())
print(f"Duplicate rows: {duplicate_count:,} ({duplicate_count / len(df) * 100:.2f}%)")

# Identify the likely target/label column
label_candidates = [
    col for col in df.columns
    if col.strip().lower() in {"label", "class", "attack", "attack_type", "category"}
]

print("\n--- POSSIBLE LABEL COLUMNS ---")
print(label_candidates if label_candidates else "No standard label column detected.")

if len(label_candidates) == 1:
    label_col = label_candidates[0]

    print(f"\n--- CLASS DISTRIBUTION: {label_col} ---")
    counts = df[label_col].value_counts(dropna=False)
    print(counts.to_string())

    print("\n--- CLASS PERCENTAGES ---")
    print((df[label_col].value_counts(normalize=True, dropna=False) * 100)
          .round(2).astype(str).add("%").to_string())

    print("\n--- IDENTICAL FEATURES WITH DIFFERENT LABELS ---")
    feature_cols = [col for col in df.columns if col != label_col]
    conflicting = (
        df.groupby(feature_cols, dropna=False)[label_col]
        .nunique(dropna=False)
    )
    conflict_groups = int((conflicting > 1).sum())
    print(f"Feature groups associated with multiple labels: {conflict_groups:,}")
else:
    print("\nPlease inspect the column names above to identify the correct label column.")

print("\n--- CHECK COMPLETE ---")
print("No rows have been deleted or modified.")
