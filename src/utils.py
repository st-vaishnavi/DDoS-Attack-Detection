"""
Utilities for DDoS Attack Detection System.
Provides path resolution, model/metadata loading, metric formatting,
plotting, and alert logging.
"""

from pathlib import Path
import json
import logging
import joblib
import pandas as pd
import numpy as np

# Configure module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return data directory path."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Return models directory path."""
    return get_project_root() / "models"


def get_outputs_dir() -> Path:
    """Return outputs directory path."""
    return get_project_root() / "outputs"


def load_model(model_path: Path = None):
    """
    Load the trained joblib model.
    Falls back to models/ddos_best_model.joblib.
    """
    if model_path is None:
        model_path = get_models_dir() / "ddos_best_model.joblib"
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    logger.info(f"Loading model from: {model_path}")
    return joblib.load(model_path)


def load_metadata(metadata_path: Path = None) -> dict:
    """
    Load the model metadata containing feature names and class labels.
    """
    if metadata_path is None:
        metadata_path = get_models_dir() / "model_metadata.json"
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Model metadata not found at: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_metadata(metadata: dict, metadata_path: Path = None) -> None:
    """Save metadata dictionary to JSON."""
    if metadata_path is None:
        metadata_path = get_models_dir() / "model_metadata.json"
    metadata_path = Path(metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved metadata to: {metadata_path}")


def log_alert(
    timestamp: str,
    attack_type: str,
    confidence: float,
    details: str = "",
    severity: str = "HIGH",
    alerts_path: Path = None,
) -> pd.DataFrame:
    """
    Log an alert record to outputs/alerts.csv.
    """
    if alerts_path is None:
        alerts_path = get_outputs_dir() / "alerts.csv"
    alerts_path = Path(alerts_path)
    alerts_path.parent.mkdir(parents=True, exist_ok=True)

    new_row = {
        "timestamp": timestamp,
        "attack_type": attack_type,
        "severity": severity,
        "confidence_pct": round(confidence * 100, 2),
        "details": details,
    }

    if alerts_path.exists():
        try:
            df = pd.read_csv(alerts_path)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        except Exception:
            df = pd.DataFrame([new_row])
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(alerts_path, index=False)
    return df
