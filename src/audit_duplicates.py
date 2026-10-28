
from pathlib import Path
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data" / "Flooding_Based_DDoS_Multi-class_Dataset.csv"

df = pd.read_csv(file_path, low_memory=False)
label_col = "Label"
feature_cols = [c for c in df.columns if c != label_col]

print("=" * 55)
print("DUPLICATE AUDIT")
print("=" * 55)

# 1. Exact duplicate rows, including the label
print("\n1. Exact duplicate rows:")
print("Rows repeated after their first occurrence:", int(df.duplicated().sum()))

# 2. Repeated feature rows, ignoring the label
feature_duplicate_mask = df.duplicated(subset=feature_cols, keep=False)
print("\n2. Rows sharing identical features, ignoring the label:")
print("Rows in repeated feature groups:", int(feature_duplicate_mask.sum()))
print("Unique feature rows:", int((~df.duplicated(subset=feature_cols)).sum()))

# 3. Distribution of repeated exact rows across labels
repeated_mask = df.duplicated(keep="first")
print("\n3. Repeated exact rows by label:")
print(df.loc[repeated_mask, label_col].value_counts().to_string())

# 4. Feature combinations assigned to multiple labels
groups = (
    df.groupby(feature_cols, dropna=False)[label_col]
    .agg(row_count="size", label_count="nunique",
         labels=lambda s: ", ".join(sorted(s.unique().astype(str))))
    .reset_index()
)

conflicts = groups[groups["label_count"] > 1]

print("\n4. Feature groups assigned multiple labels:")
print("Conflicting feature groups:", len(conflicts))

if not conflicts.empty:
    print("\nConflicting groups and their labels:")
    print(conflicts[["row_count", "label_count", "labels"]].to_string(index=False))

# 5. Number of distinct feature combinations per class
print("\n5. Distinct feature combinations per label:")
print(
    df.groupby(label_col)[feature_cols]
    .apply(lambda group: len(group.drop_duplicates()))
    .sort_values(ascending=False)
    .to_string()
)

print("\nAudit complete. No data has been modified.")
