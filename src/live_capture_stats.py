"""
Live Capture Traffic Statistics Module.
Summarizes captured network traffic (packet count, byte rates, protocol hierarchy,
active endpoints) using TShark when available, with automatic fallback
to packet analysis from saved capture files or CSV representations.
"""

import subprocess
import shutil
import sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CAPTURE_FILE = PROJECT_DIR / "data" / "live" / "wifi_capture.pcapng"
PACKET_CSV = PROJECT_DIR / "data" / "live" / "packets_for_features.csv"
TRAFFIC_SUMMARY = PROJECT_DIR / "outputs" / "traffic_summary.csv"

TSHARK = shutil.which("tshark")
if not TSHARK:
    possible_path = Path(r"C:\Program Files\Wireshark\tshark.exe")
    if possible_path.exists():
        TSHARK = str(possible_path)


def run_tshark(arguments):
    """Execute tshark CLI command and return stdout."""
    if not TSHARK:
        raise FileNotFoundError("TShark is not installed or not in PATH.")
    command = [TSHARK, *arguments]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "TShark execution failed.")
    return result.stdout


def summarize_from_csv():
    """Fallback traffic summary using packet CSV."""
    if not PACKET_CSV.exists():
        print(f"Neither TShark nor packet CSV ({PACKET_CSV}) is available.")
        return

    df = pd.read_csv(PACKET_CSV)
    print("\n--- Fallback Offline Analysis from Packet Records ---")
    print(f"Total Packets Recorded: {len(df)}")

    # Time span
    times = pd.to_numeric(df.get("frame.time_epoch"), errors="coerce").dropna()
    if not times.empty:
        duration = times.max() - times.min()
        print(f"Capture Duration       : {duration:.2f} seconds")
        print(f"Average Packet Rate    : {len(df) / max(duration, 0.001):.2f} packets/sec")

    # Packet lengths
    lengths = pd.to_numeric(df.get("frame.len"), errors="coerce").dropna()
    if not lengths.empty:
        print(f"Total Traffic Volume   : {lengths.sum() / 1024:.2f} KB ({lengths.sum()} bytes)")
        print(f"Average Packet Size    : {lengths.mean():.2f} bytes (Min: {lengths.min()}, Max: {lengths.max()})")

    # Protocol breakdown
    proto_series = df.get("frame.protocols", pd.Series(dtype=str)).dropna().astype(str)
    is_tcp = proto_series.str.contains("tcp", case=False).sum()
    is_udp = proto_series.str.contains("udp", case=False).sum()
    is_dns = proto_series.str.contains("dns", case=False).sum()
    is_icmp = proto_series.str.contains("icmp", case=False).sum()

    print("\n--- Protocol Breakdown ---")
    print(f"TCP Packets : {is_tcp} ({is_tcp / len(df) * 100:.1f}%)")
    print(f"UDP Packets : {is_udp} ({is_udp / len(df) * 100:.1f}%)")
    print(f"DNS Packets : {is_dns} ({is_dns / len(df) * 100:.1f}%)")
    print(f"ICMP Packets: {is_icmp} ({is_icmp / len(df) * 100:.1f}%)")


def show_capture_summary():
    """Display traffic capture statistics."""
    print("\n=== DDoS Detection: Traffic Capture Monitor ===")
    print(f"Capture File : {CAPTURE_FILE.name}")

    if TSHARK and CAPTURE_FILE.exists():
        print("Mode         : TShark Deep Packet Inspection\n")
        try:
            print("--- Capture File I/O Statistics ---")
            print(run_tshark(["-r", str(CAPTURE_FILE), "-q", "-z", "io,stat,1"]))

            print("--- Protocol Hierarchy Statistics ---")
            print(run_tshark(["-r", str(CAPTURE_FILE), "-q", "-z", "io,phs"]))
            print("TShark analysis complete.")
            return
        except Exception as e:
            print(f"Note: TShark failed ({e}), switching to CSV summary...")

    summarize_from_csv()
    print("\nAnalysis complete.")
    print("Note: Traffic volume and protocol counts alone cannot confirm a DDoS attack.")


if __name__ == "__main__":
    show_capture_summary()
