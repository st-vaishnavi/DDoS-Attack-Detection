
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "ddos_cleaned.csv"
if not DATA_PATH.exists():
    DATA_PATH = PROJECT_DIR / "data" / "ddos_cleaned.csv"

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
TARGET = "Label"

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# Keep the same class proportions in training and testing.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced",
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

results = []
best_f1 = -1
best_model = None
best_name = None

for name, estimator in models.items():
    print(f"\n{'=' * 60}")
    print(f"Training: {name}")
    print("=" * 60)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
    )
    print(report)

    labels = sorted(y.unique())
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.csv")

    (OUTPUT_DIR / f"{name.lower().replace(' ', '_')}_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    results.append({
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    })

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        best_model = pipeline
        best_name = name

# Save comparison and selected model.
comparison = pd.DataFrame(results).sort_values(
    "macro_f1",
    ascending=False,
)
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

joblib.dump(best_model, MODEL_DIR / "ddos_best_model.joblib")

metadata = {
    "model_name": best_name,
    "target_column": TARGET,
    "feature_columns": list(X.columns),
    "classes": sorted(y.unique().tolist()),
    "selection_metric": "macro_f1",
    "random_state": 42,
    "test_size": 0.20,
}

(MODEL_DIR / "model_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(comparison.to_string(index=False))
print(f"\nSelected model: {best_name}")
print(f"Saved model: {MODEL_DIR / 'ddos_best_model.joblib'}")
print(f"Reports saved in: {OUTPUT_DIR}")
