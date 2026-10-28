"""
Model Evaluation Module for DDoS Attack Detection.
Evaluates accuracy, precision, recall, F1-score, generates and plots the
confusion matrix, and saves classification reports to outputs/.
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Project path resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils import load_model, load_metadata, get_outputs_dir, get_data_dir


def evaluate(
    test_csv_path: Path = None,
    output_dir: Path = None,
    save_plots: bool = True,
):
    """
    Run comprehensive evaluation of the saved model on the test dataset.
    """
    if output_dir is None:
        output_dir = get_outputs_dir()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    if test_csv_path is None:
        test_csv_path = get_data_dir() / "external" / "test.csv"
        if not test_csv_path.exists():
            test_csv_path = get_data_dir() / "processed" / "ddos_cleaned.csv"
            if not test_csv_path.exists():
                test_csv_path = get_data_dir() / "ddos_cleaned.csv"

    test_csv_path = Path(test_csv_path)
    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found at: {test_csv_path}")

    print(f"Loading test dataset from: {test_csv_path}")
    df_test = pd.read_csv(test_csv_path)

    model = load_model()
    metadata = load_metadata()

    target_col = metadata.get("target_column", "Label")
    feature_cols = metadata.get("feature_columns", [])
    classes = metadata.get("classes", sorted(df_test[target_col].unique().tolist()))

    X_test = df_test[feature_cols]
    y_test = df_test[target_col].astype(str)

    print(f"Evaluating {len(df_test)} test samples across {len(classes)} classes...")
    y_pred = model.predict(X_test)

    # Metrics computation
    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print("\n" + "=" * 65)
    print("DDoS DETECTION MODEL EVALUATION REPORT")
    print("=" * 65)
    print(f"Model Name        : {metadata.get('model_name', 'Random Forest')}")
    print(f"Accuracy          : {acc * 100:.2f}%")
    print(f"Macro Precision   : {prec_macro * 100:.2f}%")
    print(f"Macro Recall      : {rec_macro * 100:.2f}%")
    print(f"Macro F1-Score    : {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score : {f1_weighted * 100:.2f}%")
    print("=" * 65)
    print("\nDetailed Per-Class Classification Report:\n")
    print(report_str)

    # Save classification report
    report_file = output_dir / "classification_report.txt"
    report_file.write_text(
        f"DDoS Detection Evaluation Report\n{'=' * 35}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro Precision: {prec_macro:.4f}\n"
        f"Macro Recall: {rec_macro:.4f}\n"
        f"Macro F1: {f1_macro:.4f}\n"
        f"Weighted F1: {f1_weighted:.4f}\n\n"
        f"{report_str}",
        encoding="utf-8",
    )
    print(f"Saved classification report to: {report_file}")

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    if save_plots:
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            cbar=True,
        )
        plt.title(f"Confusion Matrix - {metadata.get('model_name', 'DDoS Classifier')}", fontsize=14, pad=12)
        plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
        plt.ylabel("Actual Class", fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_png_path = output_dir / "confusion_matrix.png"
        plt.savefig(cm_png_path, dpi=300)
        plt.close()
        print(f"Saved confusion matrix plot to: {cm_png_path}")

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    evaluate()
