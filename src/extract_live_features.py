"""
Extract Live Flow Features Module.
Extracts network packet headers and statistics from Wi-Fi PCAP captures (via TShark),
live network interfaces, or pre-exported packet CSVs, computing traffic summaries
and 10-packet flow features strictly validated against the model's training pipeline.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.feature_extractor import (
    EXPECTED_FEATURES,
    WINDOW_SIZE,
    parse_packet,
    compute_window_features,
    extract_flow_features_from_packets,
    validate_feature_dataframe,
)
from src.utils import get_data_dir, get_outputs_dir

TSHARK = shutil.which("tshark") or r"C:\Program Files\Wireshark\tshark.exe"

FIELDS = [
    "frame.time_epoch", "frame.len", "ip.src", "ip.dst",
    "ipv6.src", "ipv6.dst", "frame.protocols",
    "tcp.flags.syn", "tcp.flags.urg", "tcp.flags.fin",
    "tcp.flags.ack", "tcp.flags.push", "tcp.flags.reset",
    "tcp.len", "udp.length", "dns",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport"
]


def list_tshark_interfaces() -> List[Tuple[str, str]]:
    """Query TShark for available network capture interfaces."""
    if not Path(TSHARK).exists():
        raise FileNotFoundError(f"TShark executable not found at: {TSHARK}")

    cmd = [TSHARK, "-D"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "Failed to list TShark interfaces.")

    interfaces = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: 1. \Device\NPF_{...} (Wi-Fi)
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            idx = m.group(1)
            desc = m.group(2)
            interfaces.append((idx, desc))
    return interfaces


def get_default_interface() -> Optional[str]:
    """Auto-detect the primary Wi-Fi or active Ethernet interface index."""
    try:
        interfaces = list_tshark_interfaces()
        # Look for Wi-Fi first, then Ethernet, then fallback to first available
        for idx, desc in interfaces:
            desc_lower = desc.lower()
            if "wi-fi" in desc_lower and "wi-fi 2" not in desc_lower and "wi-fi 5" not in desc_lower:
                return idx
        for idx, desc in interfaces:
            if "wi-fi" in desc.lower():
                return idx
        for idx, desc in interfaces:
            if "ethernet" in desc.lower():
                return idx
        if interfaces:
            return interfaces[0][0]
    except Exception:
        pass
    return "4"  # Default Wi-Fi index on Windows


def capture_live_traffic_via_tshark(
    interface: str,
    duration_sec: int = 10,
    output_pcap: Path = None,
    packet_count: Optional[int] = None,
) -> Path:
    """
    Capture live packets on a specific interface using TShark and save to PCAP.
    """
    if not Path(TSHARK).exists():
        raise FileNotFoundError(f"TShark executable not found at: {TSHARK}")

    if output_pcap is None:
        output_pcap = get_data_dir() / "live" / "wifi_capture.pcapng"
    output_pcap = Path(output_pcap)
    output_pcap.parent.mkdir(parents=True, exist_ok=True)

    cmd = [TSHARK, "-i", str(interface), "-a", f"duration:{duration_sec}", "-w", str(output_pcap)]
    if packet_count:
        cmd.extend(["-c", str(packet_count)])

    print(f"Starting TShark live capture on interface '{interface}' for {duration_sec}s...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_sec + 30)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "TShark live capture failed.")

    print(f"Live capture complete -> {output_pcap} ({output_pcap.stat().st_size / 1024:.1f} KB)")
    return output_pcap


def extract_packets_via_tshark(pcap_path: Path) -> List[Dict[str, Any]]:
    """Run TShark on pcapng file to extract packet fields."""
    if not Path(TSHARK).exists():
        raise FileNotFoundError(f"TShark executable not found at: {TSHARK}")

    cmd = [
        TSHARK, "-r", str(pcap_path), "-T", "fields",
        "-E", "separator=\t", "-E", "occurrence=f",
    ]
    for field in FIELDS:
        cmd.extend(["-e", field])

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "TShark extraction failed")

    raw_packets = []
    for line in res.stdout.splitlines():
        vals = line.split("\t")
        vals += [""] * (len(FIELDS) - len(vals))
        raw_packets.append(dict(zip(FIELDS, vals)))
    return raw_packets


def load_packets_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load packet records directly from a packet CSV file."""
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


def extract_features(
    pcap_path: Path = None,
    csv_fallback_path: Path = None,
    summary_out: Path = None,
    features_out: Path = None,
    window_packets: int = WINDOW_SIZE,
    live_capture: bool = False,
    interface: Optional[str] = None,
    capture_duration: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract windowed flow features and traffic summary.
    Groups traffic into verified 10-packet flow windows for multi-class detection,
    while maintaining 1-second interval statistics for traffic monitoring.
    """
    data_dir = get_data_dir()
    outputs_dir = get_outputs_dir()

    if pcap_path is None:
        pcap_path = data_dir / "live" / "wifi_capture.pcapng"
    if csv_fallback_path is None:
        csv_fallback_path = data_dir / "live" / "packets_for_features.csv"
    if summary_out is None:
        summary_out = outputs_dir / "traffic_summary.csv"
    if features_out is None:
        features_out = outputs_dir / "live_model_features_provisional.csv"

    pcap_path = Path(pcap_path)
    csv_fallback_path = Path(csv_fallback_path)
    summary_out = Path(summary_out)
    features_out = Path(features_out)

    # Optional: trigger live packet capture first
    if live_capture:
        if not interface:
            interface = get_default_interface()
            print(f"Auto-selected capture interface: {interface}")
        capture_live_traffic_via_tshark(
            interface=interface,
            duration_sec=capture_duration,
            output_pcap=pcap_path,
        )

    raw_packets = []

    # Try TShark if pcap exists and tshark is present
    if pcap_path.exists() and (shutil.which("tshark") or Path(TSHARK).exists()):
        print(f"Extracting packet fields from PCAP via TShark: {pcap_path.name}")
        try:
            raw_packets = extract_packets_via_tshark(pcap_path)
        except Exception as e:
            print(f"Warning: TShark capture extraction failed ({e}). Checking CSV fallback...")

    # Fallback to packet CSV if needed
    if not raw_packets:
        if csv_fallback_path.exists():
            print(f"Reading packet fields from CSV: {csv_fallback_path.name}")
            raw_packets = load_packets_from_csv(csv_fallback_path)
        else:
            raise FileNotFoundError(
                f"Neither valid TShark output nor fallback packet CSV found at {csv_fallback_path}"
            )

    parsed_packets = []
    for r in raw_packets:
        p = parse_packet(r)
        if p is not None:
            p["source"] = r.get("ip.src") or r.get("ipv6.src") or ""
            p["destination"] = r.get("ip.dst") or r.get("ipv6.dst") or ""
            parsed_packets.append(p)

    if not parsed_packets:
        sys.exit("No valid IP (TCP/UDP/ICMP) packets found for feature extraction.")

    # Sort chronologically by arrival time
    parsed_packets.sort(key=lambda p: p["time"])
    start_time = parsed_packets[0]["time"]

    # 1. 1-second interval traffic summary for bandwidth and protocol monitoring
    buckets = defaultdict(list)
    for p in parsed_packets:
        second = int(p["time"] - start_time)
        buckets[second].append(p)

    summary_rows = []
    for second, group in sorted(buckets.items()):
        lengths = [p["length"] for p in group]
        sources = {p["source"] for p in group if p["source"]}
        destinations = {p["destination"] for p in group if p["destination"]}

        summary_rows.append({
            "second": second,
            "packets": len(group),
            "bytes": sum(lengths),
            "ipv4_packets": sum(1 for p in group if "ip" in p["protocols"]),
            "ipv6_packets": sum(1 for p in group if "ipv6" in p["protocols"]),
            "tcp_packets": sum(1 for p in group if p["tcp"]),
            "udp_packets": sum(1 for p in group if p["udp"]),
            "icmp_packets": sum(1 for p in group if p["icmp"]),
            "unique_sources": len(sources),
            "unique_destinations": len(destinations),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_out, index=False)

    # 2. Extract validated 10-packet window features for model classification
    features_df = extract_flow_features_from_packets(parsed_packets, window_size=window_packets)
    
    # 3. Validate feature names, ordering, data types, and values
    is_valid, issues, validated_features = validate_feature_dataframe(
        features_df,
        strict_protocol_sum=True,
    )
    if not is_valid:
        print(f"Warning: Feature validation found {len(issues)} warnings:")
        for issue in issues[:5]:
            print(f" - {issue}")
    else:
        print("Feature Validation PASSED: All 25 features are valid, normalized, and in correct order.")

    features_out.parent.mkdir(parents=True, exist_ok=True)
    validated_features.to_csv(features_out, index=False)

    print(f"Valid IP Packets Processed  : {len(parsed_packets)}")
    print(f"Traffic 1-sec Intervals     : {len(buckets)}")
    print(f"10-Packet Windows Extracted : {len(validated_features)}")
    print(f"Saved traffic summary to    : {summary_out}")
    print(f"Saved live features to      : {features_out}")
    return summary_df, validated_features


def main():
    parser = argparse.ArgumentParser(description="Live network flow feature extraction for DDoS detection")
    parser.add_argument("--capture", action="store_true", help="Perform live packet capture using TShark")
    parser.add_argument("--interface", type=str, default=None, help="Capture interface index or name")
    parser.add_argument("--duration", type=int, default=10, help="Capture duration in seconds (default: 10)")
    parser.add_argument("--list-interfaces", action="store_true", help="List available TShark capture interfaces and exit")
    parser.add_argument("--pcap", type=str, default=None, help="Path to PCAP capture file")
    parser.add_argument("--output", type=str, default=None, help="Path to save output features CSV")
    parser.add_argument("--summary", type=str, default=None, help="Path to save traffic summary CSV")

    args = parser.parse_args()

    if args.list_interfaces:
        print("Available TShark Network Interfaces:")
        try:
            for idx, desc in list_tshark_interfaces():
                print(f"  [{idx}] {desc}")
        except Exception as e:
            print(f"Error listing interfaces: {e}")
        return

    extract_features(
        pcap_path=args.pcap,
        summary_out=args.summary,
        features_out=args.output,
        live_capture=args.capture,
        interface=args.interface,
        capture_duration=args.duration,
    )


if __name__ == "__main__":
    main()
