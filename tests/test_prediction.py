"""
Unit tests for src/predict.py.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predict import DDoSClassifier
from src.feature_extractor import EXPECTED_FEATURES


@pytest.fixture
def classifier():
    return DDoSClassifier()


def test_classifier_initialization(classifier):
    assert classifier.model is not None
    assert len(classifier.feature_columns) == 25
    assert len(classifier.classes) > 0
    assert "Normal" in classifier.classes


def test_prepare_features(classifier):
    # Test with empty df
    empty_df = pd.DataFrame()
    prep = classifier.prepare_features(empty_df)
    assert prep.shape == (0, 25)
    assert list(prep.columns) == classifier.feature_columns

    # Test with partial columns
    partial_df = pd.DataFrame({"Avg_syn_flag": [1.0], "extra_col": [999]})
    prep = classifier.prepare_features(partial_df)
    assert prep.shape == (1, 25)
    assert prep.iloc[0]["Avg_syn_flag"] == 1.0
    assert prep.iloc[0]["Avg_urg_flag"] == 0.0
    assert "extra_col" not in prep.columns


def test_predict_and_proba(classifier):
    # Create a synthetic normal-looking feature vector
    normal_vector = {col: 0.0 for col in classifier.feature_columns}
    normal_vector["Avg_TCP_pkt"] = 1.0
    normal_vector["Avg_pkts_lenght"] = 64.0

    pred = classifier.predict(normal_vector)
    assert len(pred) == 1
    assert pred[0] in classifier.classes

    probas = classifier.predict_proba(normal_vector)
    assert probas is not None
    assert probas.shape == (1, len(classifier.classes))
    assert np.isclose(np.sum(probas[0]), 1.0)


def test_evaluate_traffic_and_alert(classifier, tmp_path):
    # Create 2 sample records
    records = [
        {col: 0.0 for col in classifier.feature_columns},
        {col: 0.0 for col in classifier.feature_columns},
    ]
    # Set one record with high SYN flag to simulate SYN flooding
    records[1]["Avg_syn_flag"] = 1.0
    records[1]["Avg_TCP_pkt"] = 1.0

    df = pd.DataFrame(records)
    alerts_file = tmp_path / "test_alerts.csv"

    # Evaluate traffic
    results = classifier.evaluate_traffic_and_alert(
        df,
        log_to_alerts_file=False,
    )
    assert "predicted_label" in results.columns
    assert "confidence" in results.columns
    assert "window_id" in results.columns
    assert "timestamp" in results.columns
    assert "prob_Normal" in results.columns
    assert len(results) == 2


def test_strict_input_rejection(classifier):
    # Missing required columns should raise ValueError when strict=True
    incomplete_df = pd.DataFrame([{"Avg_syn_flag": 1.0}])
    with pytest.raises(ValueError) as excinfo:
        classifier.prepare_features(incomplete_df, strict=True)
    assert "Input feature validation failed" in str(excinfo.value)

