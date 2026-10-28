"""
Prediction Module for DDoS Attack Detection.
Loads the trained Random Forest model and metadata, performs inference on
validated flow feature vectors, maps class probabilities, and generates
security alerts with timestamps and window identifiers.
"""

from pathlib import Path
import argparse
import sys
from datetime import datetime
from typing import Dict, List, Union, Tuple, Optional
import pandas as pd
import numpy as np

# Adjust sys.path so it runs standalone or as a package
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils import load_model, load_metadata, log_alert, get_outputs_dir
from src.feature_extractor import EXPECTED_FEATURES, validate_feature_dataframe


class DDoSClassifier:
    """
    Wrapper for loading the best trained DDoS model, validating input features,
    generating predictions with class probabilities, and logging security alerts.
    """

    def __init__(self, model_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        self.model = load_model(model_path)
        self.metadata = load_metadata(metadata_path)
        self.feature_columns = self.metadata.get("feature_columns", EXPECTED_FEATURES)
        self.classes = list(getattr(self.model, "classes_", self.metadata.get("classes", [])))

    def validate_inputs(self, df: pd.DataFrame, strict_protocol_sum: bool = False) -> Tuple[bool, List[str], pd.DataFrame]:
        """
        Validate input features against expected columns, types, and domain bounds.
        """
        return validate_feature_dataframe(df, strict_protocol_sum=strict_protocol_sum)

    def prepare_features(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """
        Validate and align features to the exact 25 columns and order expected by the model.
        When strict=True, rejects inputs with missing columns or unconvertible values.
        When strict=False, fills missing feature columns with 0.0 for backward compatibility.
        """
        if df.empty:
            return pd.DataFrame(columns=self.feature_columns)

        if strict:
            is_valid, issues, validated = self.validate_inputs(df)
            if not is_valid:
                raise ValueError(f"Input feature validation failed: {'; '.join(issues)}")
            return validated[self.feature_columns]

        prepared = pd.DataFrame()
        for col in self.feature_columns:
            if col in df.columns:
                prepared[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            else:
                prepared[col] = 0.0
        return prepared[self.feature_columns]

    def predict(self, features_input: Union[pd.DataFrame, dict, list], strict: bool = False) -> np.ndarray:
        """
        Predict traffic class labels for provided feature rows.
        Rejects invalid inputs if strict=True.
        """
        df = self._to_dataframe(features_input)
        if df.empty:
            return np.array([])

        X = self.prepare_features(df, strict=strict)
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, features_input: Union[pd.DataFrame, dict, list], strict: bool = False) -> Optional[np.ndarray]:
        """
        Predict class probabilities for provided feature rows.
        Returns shape (n_samples, n_classes) aligned with self.classes.
        """
        df = self._to_dataframe(features_input)
        if df.empty:
            return None

        X = self.prepare_features(df, strict=strict)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def evaluate_traffic_and_alert(
        self,
        features_df: pd.DataFrame,
        log_to_alerts_file: bool = True,
        timestamp_series: Optional[pd.Series] = None,
        save_predictions_file: bool = True,
        min_confidence: float = 0.80,
        persistence_threshold: int = 2,
    ) -> pd.DataFrame:
        """
        Run inference on validated traffic flow features, attach timestamps and window identifiers,
        compute class probabilities, implement window persistence rules to filter transient spikes,
        and log verified security alerts.
        """
        if features_df is None or features_df.empty:
            print("Warning: Empty feature set passed to evaluate_traffic_and_alert.")
            return pd.DataFrame()

        # Validate inputs and reject invalid feature sets
        is_valid, issues, validated_X = self.validate_inputs(features_df, strict_protocol_sum=False)
        if not is_valid:
            print(f"Input validation warning: {issues}")

        predictions = self.predict(validated_X)
        probas = self.predict_proba(validated_X)

        results = features_df.copy()

        # Assign window identifiers
        if "window_id" not in results.columns:
            results.insert(0, "window_id", [f"win_{i:04d}" for i in range(len(results))])

        # Attach timestamps (convert epoch to datetime string if available)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timestamps = []
        for i in range(len(results)):
            if timestamp_series is not None and i < len(timestamp_series):
                timestamps.append(str(timestamp_series.iloc[i]))
            elif "window_start_time" in results.columns and pd.notna(results["window_start_time"].iloc[i]):
                try:
                    ts_val = float(results["window_start_time"].iloc[i])
                    timestamps.append(datetime.fromtimestamp(ts_val).strftime("%Y-%m-%d %H:%M:%S"))
                except (ValueError, OSError):
                    timestamps.append(now_str)
            else:
                timestamps.append(now_str)
        results["timestamp"] = timestamps

        results["predicted_label"] = predictions

        # Add overall confidence and per-class probabilities
        if probas is not None:
            results["confidence"] = np.max(probas, axis=1)
            for c_idx, class_name in enumerate(self.classes):
                results[f"prob_{class_name}"] = np.round(probas[:, c_idx], 4)
        else:
            results["confidence"] = 1.0

        # Phase 6: Window Persistence & Confidence Threshold Rules
        # Distinguish raw predictions from confirmed security alerts
        alert_statuses = []
        consecutive_counts = []
        current_attack: Optional[str] = None
        current_streak = 0

        for pred in predictions:
            if pred != "Normal":
                if pred == current_attack:
                    current_streak += 1
                else:
                    current_attack = pred
                    current_streak = 1
            else:
                current_attack = None
                current_streak = 0

            consecutive_counts.append(current_streak)

            if pred == "Normal":
                alert_statuses.append("BENIGN")
            elif current_streak >= persistence_threshold:
                alert_statuses.append("CONFIRMED_ATTACK")
            else:
                alert_statuses.append("UNCONFIRMED_SPIKE")

        results["consecutive_windows"] = consecutive_counts
        results["threat_status"] = alert_statuses

        alert_records = []
        for idx, pred in enumerate(predictions):
            max_prob = float(results["confidence"].iloc[idx])
            win_id = str(results["window_id"].iloc[idx])
            ts = str(results["timestamp"].iloc[idx])
            status = results["threat_status"].iloc[idx]
            consec = int(results["consecutive_windows"].iloc[idx])

            if pred != "Normal":
                # Evidence extraction
                feat_row = results.iloc[idx]
                evidence_items = []
                if pred == "SYN Flooding":
                    evidence_items.append(f"SYN_Ratio={feat_row.get('Avg_syn_flag', 0):.2f}")
                    evidence_items.append(f"Payload={feat_row.get('Avg_payload', 0):.1f}B")
                elif pred == "UDP Flooding":
                    evidence_items.append(f"UDP_Ratio={feat_row.get('Avg_UDP_pkt', 0):.2f}")
                    evidence_items.append(f"PktLen={feat_row.get('Avg_pkts_lenght', 0):.1f}B")
                elif pred == "ACK Flooding":
                    evidence_items.append(f"ACK_Ratio={feat_row.get('Avg_ack_flag', 0):.2f}")
                elif pred == "HTTP Flooding":
                    evidence_items.append(f"TCP_Ratio={feat_row.get('Avg_TCP_pkt', 0):.2f}")
                    evidence_items.append(f"Payload={feat_row.get('Avg_payload', 0):.1f}B")

                evidence_str = ", ".join(evidence_items) if evidence_items else "Volumetric Flow Anomaly"

                # Assign severity based on persistence and confidence
                if status == "CONFIRMED_ATTACK":
                    severity = "CRITICAL" if max_prob >= 0.90 else "HIGH"
                else:
                    severity = "MEDIUM" if max_prob >= min_confidence else "LOW"

                alert_records.append({
                    "timestamp": ts,
                    "window_id": win_id,
                    "attack_type": pred,
                    "status": status,
                    "severity": severity,
                    "confidence_pct": round(max_prob * 100, 2),
                    "consecutive_windows": consec,
                    "evidence": evidence_str,
                    "mitigation_mode": "MONITOR_ONLY_NO_AUTO_BLOCK",
                    "details": f"[{status}] {pred} detected ({max_prob * 100:.1f}% confidence, window {consec}). Evidence: {evidence_str}",
                })

        alerts_df = pd.DataFrame(alert_records)

        # Log alerts to outputs/alerts.csv
        if log_to_alerts_file and not alerts_df.empty:
            alerts_path = get_outputs_dir() / "alerts.csv"
            alerts_path.parent.mkdir(parents=True, exist_ok=True)
            if alerts_path.exists():
                try:
                    existing = pd.read_csv(alerts_path)
                    combined = pd.concat([existing, alerts_df], ignore_index=True).drop_duplicates(subset=["timestamp", "window_id", "attack_type"])
                    combined.to_csv(alerts_path, index=False)
                except Exception:
                    alerts_df.to_csv(alerts_path, index=False)
            else:
                alerts_df.to_csv(alerts_path, index=False)

        # Save annotated predictions to outputs/live_predictions.csv
        if save_predictions_file and not results.empty:
            preds_path = get_outputs_dir() / "live_predictions.csv"
            preds_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(preds_path, index=False)

        return results

    def _to_dataframe(self, features_input: Union[pd.DataFrame, dict, list]) -> pd.DataFrame:
        """Convert input data structure into a standardized pandas DataFrame."""
        if isinstance(features_input, dict):
            return pd.DataFrame([features_input])
        elif isinstance(features_input, list):
            return pd.DataFrame(features_input)
        elif isinstance(features_input, pd.DataFrame):
            return features_input
        else:
            raise TypeError("features_input must be a DataFrame, dict, or list of dicts")


def main():
    parser = argparse.ArgumentParser(description="Predict DDoS Attack class on validated flow feature CSV")
    parser.add_argument(
        "--input",
        type=str,
        default=str(get_outputs_dir() / "live_model_features_provisional.csv"),
        help="Path to CSV containing flow features",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(get_outputs_dir() / "live_predictions.csv"),
        help="Path to save predictions CSV",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} feature records from {input_path}")

    classifier = DDoSClassifier()
    results = classifier.evaluate_traffic_and_alert(df, log_to_alerts_file=True, save_predictions_file=True)

    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(results["predicted_label"].value_counts().to_string())

    if args.output:
        out_path = Path(args.output)
        results.to_csv(out_path, index=False)
        print(f"\nSaved annotated predictions with window IDs to: {out_path}")

    alerts_path = get_outputs_dir() / "alerts.csv"
    if alerts_path.exists():
        alerts = pd.read_csv(alerts_path)
        print(f"Total alerts recorded in {alerts_path}: {len(alerts)}")


if __name__ == "__main__":
    main()
