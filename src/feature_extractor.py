"""
Feature Extractor for DDoS Attack Detection.
Defines verified feature extraction logic aligned with the 10-packet window
and TCP/IP protocol specifications of the trained Random Forest model.
"""

import math
import statistics
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Exact 25 feature names required by the trained model (in order)
EXPECTED_FEATURES = [
    "Avg_syn_flag",
    "Avg_urg_flag",
    "Avg_fin_flag",
    "Avg_ack_flag",
    "Avg_psh_flag",
    "Avg_rst_flag",
    "Avg_DNS_pkt",
    "Avg_TCP_pkt",
    "Avg_UDP_pkt",
    "Avg_ICMP_pkt",
    "Duration_window_flow",
    "Avg_delta_time",
    "Min_delta_time",
    "Max_delta_time",
    "StDev_delta_time",
    "Avg_pkts_lenght",
    "Min_pkts_lenght",
    "Max_pkts_lenght",
    "StDev_pkts_lenght",
    "Avg_small_payload_pkt",
    "Avg_payload",
    "Min_payload",
    "Max_payload",
    "StDev_payload",
    "Avg_DNS_over_TCP",
]

WINDOW_SIZE = 10


def to_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    try:
        val_str = str(value).strip()
        if not val_str:
            return default
        return float(val_str)
    except (ValueError, TypeError):
        return default


def parse_flag(value: Any) -> float:
    """Parse boolean or string flag representation to 1.0 or 0.0."""
    if value is None:
        return 0.0
    val_str = str(value).strip().lower()
    if val_str in ("1", "1.0", "true", "yes", "t"):
        return 1.0
    return 0.0


def stdev_sample(values: List[float]) -> float:
    """Calculate sample standard deviation (N-1 degrees of freedom), or 0.0 if < 2 values."""
    if not values or len(values) < 2:
        return 0.0
    try:
        return float(statistics.stdev(values))
    except statistics.StatisticsError:
        return 0.0


def parse_packet(packet_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse and validate an individual packet record from TShark or CSV.
    Strictly filters for IPv4/IPv6 packets of protocol TCP, UDP, or ICMP.
    Returns None if the packet is not a valid transport-layer IP packet.
    """
    timestamp = to_float(
        packet_dict.get("frame.time_epoch") or packet_dict.get("time") or packet_dict.get("timestamp"),
        default=0.0
    )
    length = to_float(
        packet_dict.get("frame.len") or packet_dict.get("length") or packet_dict.get("len"),
        default=0.0
    )

    if timestamp <= 0 or length <= 0:
        return None

    proto_field = str(packet_dict.get("frame.protocols") or packet_dict.get("protocols") or "").lower()
    protocols = set(proto_field.split(":")) if proto_field else set()

    is_tcp = "tcp" in protocols or bool(to_float(packet_dict.get("tcp.srcport", 0)))
    is_udp = "udp" in protocols or bool(to_float(packet_dict.get("udp.srcport", 0)))
    is_icmp = bool(protocols & {"icmp", "icmpv6"})
    is_dns = "dns" in protocols or parse_flag(packet_dict.get("dns")) == 1.0

    # The training dataset strictly processes IP packets that are TCP, UDP, or ICMP
    if not (is_tcp or is_udp or is_icmp):
        return None

    tcp_len = None
    if is_tcp:
        tcp_len = to_float(packet_dict.get("tcp.len", 0.0))

    # Priority flag resolution matching the Scapy if/elif sequence:
    # FIN (0x01) -> SYN (0x02) -> RST (0x04) -> PSH (0x08) -> ACK (0x10) -> URG (0x20)
    fin, syn, rst, psh, ack, urg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if is_tcp:
        f_fin = parse_flag(packet_dict.get("tcp.flags.fin"))
        f_syn = parse_flag(packet_dict.get("tcp.flags.syn"))
        f_rst = parse_flag(packet_dict.get("tcp.flags.reset") or packet_dict.get("tcp.flags.rst"))
        f_psh = parse_flag(packet_dict.get("tcp.flags.push") or packet_dict.get("tcp.flags.psh"))
        f_ack = parse_flag(packet_dict.get("tcp.flags.ack"))
        f_urg = parse_flag(packet_dict.get("tcp.flags.urg"))

        if f_fin == 1.0:
            fin = 1.0
        elif f_syn == 1.0:
            syn = 1.0
        elif f_rst == 1.0:
            rst = 1.0
        elif f_psh == 1.0:
            psh = 1.0
        elif f_ack == 1.0:
            ack = 1.0
        elif f_urg == 1.0:
            urg = 1.0

    return {
        "time": timestamp,
        "length": length,
        "protocols": protocols,
        "tcp": 1.0 if is_tcp else 0.0,
        "udp": 1.0 if is_udp else 0.0,
        "icmp": 1.0 if is_icmp else 0.0,
        "dns": 1.0 if is_dns else 0.0,
        "fin": fin,
        "syn": syn,
        "rst": rst,
        "psh": psh,
        "ack": ack,
        "urg": urg,
        "tcp_len": tcp_len,
    }


def compute_window_features(packets: List[Dict[str, Any]], window_size: int = WINDOW_SIZE) -> Dict[str, float]:
    """
    Computes the 25 required flow features for a batch of valid packets.
    Expects exactly window_size packets (defaults to 10).
    Averages are normalized by dividing by window_size, matching the training pipeline.
    """
    n = len(packets)
    if n == 0:
        return {feat: 0.0 for feat in EXPECTED_FEATURES}

    denom = float(window_size)
    packets_sorted = sorted(packets, key=lambda p: p["time"])
    times = [p["time"] for p in packets_sorted]
    lengths = [p["length"] for p in packets_sorted]

    # Inter-arrival times
    gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
    duration = times[-1] - times[0]

    # TCP payload statistics (training dataset strictly evaluates TCP layer payloads)
    tcp_payloads = [p["tcp_len"] for p in packets_sorted if p["tcp"] == 1.0 and p["tcp_len"] is not None]

    # Small TCP payload threshold <= 32 bytes (BoNeSI botnet profile)
    small_tcp_pkts = [1.0 if (pl is not None and pl <= 32.0) else 0.0 for pl in tcp_payloads]

    # DNS over TCP normalization: ratio of total DNS packets to high-level TCP non-DNS packets
    total_dns = sum(p["dns"] for p in packets_sorted)
    high_level_tcp = sum(1.0 for p in packets_sorted if p["tcp"] == 1.0 and (p["tcp_len"] or 0) > 0 and p["dns"] == 0.0)
    dns_ratio = (total_dns / high_level_tcp) if high_level_tcp > 0 else 0.0

    features = {
        "Avg_syn_flag": sum(p["syn"] for p in packets_sorted) / denom,
        "Avg_urg_flag": sum(p["urg"] for p in packets_sorted) / denom,
        "Avg_fin_flag": sum(p["fin"] for p in packets_sorted) / denom,
        "Avg_ack_flag": sum(p["ack"] for p in packets_sorted) / denom,
        "Avg_psh_flag": sum(p["psh"] for p in packets_sorted) / denom,
        "Avg_rst_flag": sum(p["rst"] for p in packets_sorted) / denom,
        "Avg_DNS_pkt": sum(p["dns"] for p in packets_sorted) / denom,
        "Avg_TCP_pkt": sum(p["tcp"] for p in packets_sorted) / denom,
        "Avg_UDP_pkt": sum(p["udp"] for p in packets_sorted) / denom,
        "Avg_ICMP_pkt": sum(p["icmp"] for p in packets_sorted) / denom,
        "Duration_window_flow": duration,
        "Avg_delta_time": duration / denom,
        "Min_delta_time": min(gaps) if gaps else 0.0,
        "Max_delta_time": max(gaps) if gaps else 0.0,
        "StDev_delta_time": stdev_sample(gaps),
        "Avg_pkts_lenght": sum(lengths) / denom,
        "Min_pkts_lenght": min(lengths) if lengths else 0.0,
        "Max_pkts_lenght": max(lengths) if lengths else 0.0,
        "StDev_pkts_lenght": stdev_sample(lengths),
        "Avg_small_payload_pkt": sum(small_tcp_pkts) / denom,
        "Avg_payload": sum(tcp_payloads) / denom if tcp_payloads else 0.0,
        "Min_payload": min(tcp_payloads) if tcp_payloads else 0.0,
        "Max_payload": max(tcp_payloads) if tcp_payloads else 0.0,
        "StDev_payload": stdev_sample(tcp_payloads),
        "Avg_DNS_over_TCP": dns_ratio / denom,
    }

    return features


def extract_flow_features_from_packets(
    parsed_packets: List[Dict[str, Any]],
    window_size: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Groups valid packets into contiguous batches of `window_size` (10 packets)
    and computes the 25 model features. Packets that cannot form a full window
    are rejected rather than creating partial or distorted feature vectors.
    """
    if not parsed_packets or len(parsed_packets) < window_size:
        return pd.DataFrame(columns=EXPECTED_FEATURES)

    feature_rows = []
    for i in range(0, len(parsed_packets) - window_size + 1, window_size):
        window = parsed_packets[i:i + window_size]
        feats = compute_window_features(window, window_size=window_size)
        feats["window_start_time"] = window[0]["time"]
        feats["window_end_time"] = window[-1]["time"]
        feats["packet_count"] = window_size
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)


def validate_feature_dataframe(
    df: pd.DataFrame,
    strict_protocol_sum: bool = True,
) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validates that a DataFrame contains the exact 25 model features,
    in the correct order, with valid numeric data types, no NaNs/Infs,
    and value ranges conforming to the trained model's expectations.

    Validation Rules:
    1. Exact Columns & Ordering: Must contain all 25 EXPECTED_FEATURES.
    2. Data Types: All features must be numeric (float or int).
    3. No Missing Values: Null, NaN, or Inf values are flagged.
    4. Logical Value Ranges:
       - Flag averages in [0.0, 1.0].
       - Protocol averages in [0.0, 1.0].
       - Protocol sum: Avg_TCP_pkt + Avg_UDP_pkt + Avg_ICMP_pkt == 1.0 (within float tolerance).
       - Timing: Duration_window_flow >= 0, Avg_delta_time >= 0, Min_delta_time >= 0, Max_delta_time >= 0, StDev_delta_time >= 0.
       - Packet lengths: Avg_pkts_lenght >= 54.0, Min_pkts_lenght >= 54.0, Max_pkts_lenght >= Min_pkts_lenght.
       - Payloads: Avg_payload >= 0, Min_payload >= 0, Max_payload >= Min_payload, StDev_payload >= 0.
       - Small payload ratio & DNS over TCP in [0.0, 1.0].

    Returns:
        (is_valid: bool, issues: List[str], validated_df: pd.DataFrame)
    """
    issues: List[str] = []

    if df is None or df.empty:
        return False, ["DataFrame is empty or None."], pd.DataFrame(columns=EXPECTED_FEATURES)

    # 1. Check feature columns presence
    missing_cols = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing {len(missing_cols)} required feature columns: {missing_cols}")
        return False, issues, pd.DataFrame(columns=EXPECTED_FEATURES)

    # Select the exact 25 features in order
    validated = df[EXPECTED_FEATURES].copy()

    # 2. Check and convert data types
    for col in EXPECTED_FEATURES:
        if not np.issubdtype(validated[col].dtype, np.number):
            try:
                validated[col] = pd.to_numeric(validated[col], errors="coerce")
            except Exception as e:
                issues.append(f"Column '{col}' could not be converted to numeric: {e}")

    # 3. Check for NaNs and Infs
    nan_counts = validated.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if not cols_with_nan.empty:
        issues.append(f"NaN values found in columns: {cols_with_nan.to_dict()}")
        validated = validated.fillna(0.0)

    inf_mask = np.isinf(validated.values)
    if inf_mask.any():
        issues.append(f"Infinite values found and clamped in {int(inf_mask.sum())} cells.")
        validated = validated.replace([np.inf, -np.inf], 0.0)

    # 4. Range validation
    flag_cols = ["Avg_syn_flag", "Avg_urg_flag", "Avg_fin_flag", "Avg_ack_flag", "Avg_psh_flag", "Avg_rst_flag"]
    for fc in flag_cols:
        out_of_bounds = ((validated[fc] < -1e-5) | (validated[fc] > 1.00001)).sum()
        if out_of_bounds > 0:
            issues.append(f"Column '{fc}' has {out_of_bounds} rows outside [0.0, 1.0].")

    proto_cols = ["Avg_DNS_pkt", "Avg_TCP_pkt", "Avg_UDP_pkt", "Avg_ICMP_pkt", "Avg_small_payload_pkt", "Avg_DNS_over_TCP"]
    for pc in proto_cols:
        out_of_bounds = ((validated[pc] < -1e-5) | (validated[pc] > 1.00001)).sum()
        if out_of_bounds > 0:
            issues.append(f"Column '{pc}' has {out_of_bounds} rows outside [0.0, 1.0].")

    if strict_protocol_sum:
        proto_sum = validated["Avg_TCP_pkt"] + validated["Avg_UDP_pkt"] + validated["Avg_ICMP_pkt"]
        sum_mismatches = (np.abs(proto_sum - 1.0) > 1e-3).sum()
        if sum_mismatches > 0:
            issues.append(
                f"{sum_mismatches} rows have transport protocol sum != 1.0 (Avg_TCP + Avg_UDP + Avg_ICMP != 1.0)."
            )

    time_cols = ["Duration_window_flow", "Avg_delta_time", "Min_delta_time", "Max_delta_time", "StDev_delta_time"]
    for tc in time_cols:
        negative_times = (validated[tc] < -1e-6).sum()
        if negative_times > 0:
            issues.append(f"Column '{tc}' has {negative_times} negative values.")

    payload_cols = ["Avg_payload", "Min_payload", "Max_payload", "StDev_payload"]
    for plc in payload_cols:
        negative_payloads = (validated[plc] < -1e-6).sum()
        if negative_payloads > 0:
            issues.append(f"Column '{plc}' has {negative_payloads} negative values.")

    is_valid = len(issues) == 0
    return is_valid, issues, validated
