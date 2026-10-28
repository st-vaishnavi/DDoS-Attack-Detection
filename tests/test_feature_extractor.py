"""
Unit tests for src/feature_extractor.py.
"""

import math
import pytest
import pandas as pd
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.feature_extractor import (
    EXPECTED_FEATURES,
    WINDOW_SIZE,
    to_float,
    parse_flag,
    parse_packet,
    compute_window_features,
    extract_flow_features_from_packets,
)


def test_to_float_and_parse_flag():
    assert to_float("123.45") == 123.45
    assert to_float("", default=0.0) == 0.0
    assert to_float(None, default=1.0) == 1.0
    assert to_float("invalid", default=-1.0) == -1.0

    assert parse_flag("True") == 1.0
    assert parse_flag("1") == 1.0
    assert parse_flag("true") == 1.0
    assert parse_flag("False") == 0.0
    assert parse_flag("0") == 0.0
    assert parse_flag(None) == 0.0


def test_parse_packet():
    raw = {
        "frame.time_epoch": "1792947105.213",
        "frame.len": "1500",
        "frame.protocols": "eth:ip:tcp",
        "tcp.flags.syn": "True",
        "tcp.flags.ack": "False",
        "tcp.flags.fin": "False",
        "tcp.flags.push": "False",
        "tcp.flags.reset": "False",
        "tcp.flags.urg": "False",
        "tcp.len": "1440",
    }
    parsed = parse_packet(raw)
    assert parsed is not None
    assert parsed["time"] == 1792947105.213
    assert parsed["length"] == 1500.0
    assert parsed["syn"] == 1.0
    assert parsed["ack"] == 0.0
    assert parsed["tcp"] == 1.0
    assert parsed["udp"] == 0.0
    assert parsed["tcp_len"] == 1440.0


def test_parse_packet_rejects_non_ip():
    # Non-transport packets (e.g. ARP or pure layer 2) should be rejected
    arp_pkt = {
        "frame.time_epoch": "100.0",
        "frame.len": "42",
        "frame.protocols": "eth:arp",
    }
    assert parse_packet(arp_pkt) is None


def test_flag_priority_chain():
    # PSH+ACK packet: PSH has priority over ACK
    psh_ack_pkt = {
        "frame.time_epoch": "100.0",
        "frame.len": "120",
        "frame.protocols": "eth:ip:tcp",
        "tcp.flags.psh": "True",
        "tcp.flags.ack": "True",
        "tcp.len": "60",
    }
    parsed = parse_packet(psh_ack_pkt)
    assert parsed["psh"] == 1.0
    assert parsed["ack"] == 0.0  # ACK suppressed by higher priority PSH


def test_compute_window_features():
    # Construct a window of 10 identical TCP packets
    packets = []
    for i in range(10):
        packets.append({
            "time": 100.0 + i * 0.1,
            "length": 60.0,
            "protocols": {"tcp"},
            "tcp": 1.0,
            "udp": 0.0,
            "icmp": 0.0,
            "dns": 0.0,
            "fin": 0.0,
            "syn": 1.0 if i == 0 else 0.0,
            "rst": 0.0,
            "psh": 0.0,
            "ack": 1.0 if i > 0 else 0.0,
            "urg": 0.0,
            "tcp_len": 20.0,  # <= 32 bytes (small payload)
        })

    feats = compute_window_features(packets, window_size=10)
    assert len(feats) == len(EXPECTED_FEATURES)
    for col in EXPECTED_FEATURES:
        assert col in feats

    # 1 SYN packet out of 10 -> Avg_syn_flag = 0.1
    assert pytest.approx(feats["Avg_syn_flag"]) == 0.1
    # 9 ACK packets out of 10 -> Avg_ack_flag = 0.9
    assert pytest.approx(feats["Avg_ack_flag"]) == 0.9
    # All 10 are small payload (20 <= 32) -> Avg_small_payload_pkt = 1.0
    assert pytest.approx(feats["Avg_small_payload_pkt"]) == 1.0
    assert pytest.approx(feats["Duration_window_flow"]) == 0.9
    # Duration / 10 = 0.09
    assert pytest.approx(feats["Avg_delta_time"]) == 0.09
    assert pytest.approx(feats["Avg_pkts_lenght"]) == 60.0
    assert pytest.approx(feats["Avg_payload"]) == 20.0


def test_extract_flow_features_from_packets():
    packets = []
    for i in range(25):  # 25 packets should generate 2 full 10-packet windows (5 remainder rejected)
        packets.append({
            "time": 10.0 + i * 0.05,
            "length": 100.0,
            "protocols": {"tcp"},
            "tcp": 1.0,
            "udp": 0.0,
            "icmp": 0.0,
            "dns": 0.0,
            "fin": 0.0,
            "syn": 0.0,
            "rst": 0.0,
            "psh": 0.0,
            "ack": 1.0,
            "urg": 0.0,
            "tcp_len": 40.0,
        })

    df = extract_flow_features_from_packets(packets, window_size=10)
    assert len(df) == 2  # exactly 2 full windows
    for feat in EXPECTED_FEATURES:
        assert feat in df.columns


def test_validate_feature_dataframe():
    from src.feature_extractor import validate_feature_dataframe

    # Valid feature row
    valid_row = {col: 0.0 for col in EXPECTED_FEATURES}
    valid_row["Avg_TCP_pkt"] = 1.0
    valid_row["Avg_pkts_lenght"] = 64.0
    valid_df = pd.DataFrame([valid_row])

    is_valid, issues, cleaned = validate_feature_dataframe(valid_df)
    assert is_valid is True
    assert len(issues) == 0
    assert cleaned.shape == (1, 25)

    # Missing column test
    incomplete_df = pd.DataFrame([{"Avg_syn_flag": 0.5}])
    is_valid, issues, _ = validate_feature_dataframe(incomplete_df)
    assert is_valid is False
    assert any("Missing" in msg for msg in issues)

    # Out of bounds range test (protocol sum != 1.0)
    invalid_proto_row = valid_row.copy()
    invalid_proto_row["Avg_TCP_pkt"] = 0.5  # sum = 0.5 instead of 1.0
    invalid_proto_df = pd.DataFrame([invalid_proto_row])
    is_valid, issues, _ = validate_feature_dataframe(invalid_proto_df, strict_protocol_sum=True)
    assert is_valid is False
    assert any("protocol sum" in msg for msg in issues)
