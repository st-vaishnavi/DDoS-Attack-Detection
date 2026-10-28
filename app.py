"""
DDoS Attack Detection & Traffic Monitoring Dashboard
A comprehensive Streamlit application for network traffic analysis,
real-time flow classification, alert logging, and model evaluation.
"""

from pathlib import Path
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Automatically launch Streamlit if run directly with `python app.py`
if __name__ == "__main__" and not st.runtime.exists():
    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    sys.exit(stcli.main())

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "ddos_best_model.joblib"
META_PATH = BASE_DIR / "models" / "model_metadata.json"
DATA_CLEANED_PATH = BASE_DIR / "data" / "processed" / "ddos_cleaned.csv"
if not DATA_CLEANED_PATH.exists():
    DATA_CLEANED_PATH = BASE_DIR / "data" / "ddos_cleaned.csv"

TEST_DATA_PATH = BASE_DIR / "data" / "external" / "test.csv"
if not TEST_DATA_PATH.exists():
    TEST_DATA_PATH = DATA_CLEANED_PATH

TRAFFIC_SUMMARY_PATH = BASE_DIR / "outputs" / "traffic_summary.csv"
LIVE_FEATURES_PATH = BASE_DIR / "outputs" / "live_model_features_provisional.csv"
LIVE_PREDICTIONS_PATH = BASE_DIR / "outputs" / "live_predictions.csv"
ALERTS_PATH = BASE_DIR / "outputs" / "alerts.csv"
MODEL_COMP_PATH = BASE_DIR / "outputs" / "model_comparison.csv"
CONFUSION_IMG_PATH = BASE_DIR / "outputs" / "confusion_matrix.png"
REPORT_PATH = BASE_DIR / "outputs" / "classification_report.txt"
PACKETS_PATH = BASE_DIR / "data" / "live" / "packets_for_features.csv"

# Live validation artifacts
LIVE_VAL_REPORT_PATH = BASE_DIR / "outputs" / "live_validation_report.txt"
LIVE_VAL_METRICS_PATH = BASE_DIR / "outputs" / "live_validation_metrics.csv"
LIVE_VAL_CM_IMG_PATH = BASE_DIR / "outputs" / "live_validation_confusion_matrix.png"

# Configure Page
st.set_page_config(
    page_title="DDoS Shield: Network Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Cyber-Defense UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .alert-critical {
        background-color: #FEE2E2;
        border-left: 6px solid #DC2626;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .alert-normal {
        background-color: #DCFCE7;
        border-left: 6px solid #16A34A;
        padding: 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.05rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detection_assets():
    """Load model, metadata, and test dataset."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return None, None, None
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    test_df = pd.read_csv(TEST_DATA_PATH) if TEST_DATA_PATH.exists() else None
    return model, meta, test_df


model, metadata, test_df = load_detection_assets()

# Sidebar: System & Engine Status
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=64)
    st.title("DDoS Shield Engine")
    st.caption("AI-Powered Multi-Class Flooding Detection")
    st.divider()

    if model and metadata:
        st.success(f"**Model Active:** {metadata.get('model_name', 'Random Forest')}")
        st.info(f"**Classes Monitored:** {len(metadata.get('classes', []))}")
        st.write(f"**Features Analyzed:** {len(metadata.get('feature_columns', []))}")
        st.caption("Flow Window: 10 Filtered Packets")
    else:
        st.error("Model assets not loaded.")

    st.divider()
    st.markdown("### System Architecture")
    st.markdown("""
    1. 📡 **Wi-Fi Packet Capture** (TShark)
    2. 🔍 **Transport Layer Filtering** (TCP/UDP/ICMP)
    3. ⚙️ **10-Packet Flow Windowing**
    4. 🤖 **Random Forest Inference** (25 Features)
    5. 🚨 **Persistence-Verified Alerting**
    """)
    st.divider()
    st.caption("🛡️ Defense Mode: Active Monitoring (No Automatic Blocking)")


# Main Header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🛡️ DDoS Attack Detection & Traffic Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Class Flooding Detection & Network Packet Flow Monitoring</div>', unsafe_allow_html=True)

with col_status:
    # Check alert count
    alert_count = 0
    confirmed_count = 0
    if ALERTS_PATH.exists():
        try:
            alerts_df_raw = pd.read_csv(ALERTS_PATH)
            alert_count = len(alerts_df_raw)
            if "status" in alerts_df_raw.columns:
                confirmed_count = len(alerts_df_raw[alerts_df_raw["status"] == "CONFIRMED_ATTACK"])
            else:
                confirmed_count = alert_count
        except Exception:
            pass
    if confirmed_count > 0:
        st.metric("🚨 Confirmed Attacks", f"{confirmed_count} Active", delta="CRITICAL", delta_color="inverse")
    elif alert_count > 0:
        st.metric("⚠️ Threat Status", f"{alert_count} Alerts", delta="Suspect Spikes", delta_color="off")
    else:
        st.metric("✅ Threat Level", "Nominal", delta="0 Attacks")

# Tab Layout
tab_traffic, tab_detect, tab_simulator, tab_eval = st.tabs([
    "📊 Traffic Monitoring",
    "🚨 Live Flow Detection & Alerts",
    "🧪 Classifier & Flow Simulator",
    "📈 Model Analytics & Evaluation",
])

# =========================================================================
# TAB 1: Network Traffic Monitoring
# =========================================================================
with tab_traffic:
    st.subheader("Network Packet Capture & Flow Statistics")
    st.caption("Analyzed from live Wi-Fi network packet capture (`wifi_capture.pcapng` / `packets_for_features.csv`).")

    # Live capture control panel
    with st.expander("📡 TShark Live Capture Controls", expanded=False):
        c_int, c_dur, c_btn = st.columns([2, 1, 1])
        with c_int:
            st.write("**Capture Interface:** Wi-Fi (Auto-detected)")
        with c_dur:
            cap_duration = st.slider("Duration (seconds)", min_value=3, max_value=30, value=5, step=1)
        with c_btn:
            st.write("")
            st.write("")
            if st.button("🚀 Run Live Capture", type="primary"):
                with st.spinner(f"Capturing traffic on Wi-Fi for {cap_duration}s via TShark..."):
                    try:
                        from src.extract_live_features import extract_features
                        extract_features(live_capture=True, capture_duration=cap_duration)
                        st.success("Capture and feature extraction complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Live capture error: {e}")

    if TRAFFIC_SUMMARY_PATH.exists():
        traffic_df = pd.read_csv(TRAFFIC_SUMMARY_PATH)

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        total_packets = traffic_df["packets"].sum()
        total_kb = traffic_df["bytes"].sum() / 1024.0
        total_seconds = len(traffic_df)
        avg_rate = total_packets / max(total_seconds, 1)

        m1.metric("Total Capture Packets", f"{total_packets:,}")
        m2.metric("Total Volume", f"{total_kb:.1f} KB")
        m3.metric("Monitored Duration", f"{total_seconds} seconds")
        m4.metric("Avg Packet Rate", f"{avg_rate:.1f} pkts/sec")

        st.markdown("#### Packet & Volume Rates over Time")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.line_chart(traffic_df.set_index("second")[["packets"]], color="#2563EB")
            st.caption("Packets per second interval")
        with col_chart2:
            st.line_chart(traffic_df.set_index("second")[["bytes"]], color="#059669")
            st.caption("Bytes transferred per second interval")

        st.markdown("#### Protocol Distribution Across Flow Windows")
        proto_cols = ["tcp_packets", "udp_packets", "icmp_packets"]
        proto_present = [c for c in proto_cols if c in traffic_df.columns]
        if proto_present:
            st.area_chart(traffic_df.set_index("second")[proto_present])

        with st.expander("🔍 View Raw Interval Summary Data"):
            st.dataframe(traffic_df, use_container_width=True)

    else:
        st.info("No traffic summary found at outputs/traffic_summary.csv. Run `python src/extract_live_features.py`.")

    if PACKETS_PATH.exists():
        with st.expander("📦 Raw Wi-Fi Packet Header Inspector"):
            try:
                raw_pkts = pd.read_csv(PACKETS_PATH, nrows=100)
                st.dataframe(raw_pkts, use_container_width=True)
                st.caption(f"Showing first 100 captured packet headers from {PACKETS_PATH.name}")
            except Exception as e:
                st.warning(f"Unable to read packet records: {e}")


# =========================================================================
# TAB 2: Live Flow Detection & Alert Log
# =========================================================================
with tab_detect:
    st.subheader("DDoS Detection on Extracted Flow Features")
    st.caption("Inference using the trained Random Forest model on validated 10-packet flow windows.")

    pred_source = None
    if LIVE_PREDICTIONS_PATH.exists():
        pred_source = pd.read_csv(LIVE_PREDICTIONS_PATH)
    elif LIVE_FEATURES_PATH.exists() and model is not None:
        try:
            from src.predict import DDoSClassifier
            clf = DDoSClassifier()
            raw_features = pd.read_csv(LIVE_FEATURES_PATH)
            pred_source = clf.evaluate_traffic_and_alert(raw_features, log_to_alerts_file=False)
        except Exception as e:
            st.error(f"Error evaluating live features: {e}")

    if pred_source is not None and not pred_source.empty:
        total_eval = len(pred_source)
        preds = pred_source["predicted_label"]
        counts = preds.value_counts()
        normal_count = counts.get("Normal", 0)
        attack_count = total_eval - normal_count

        confirmed_count = 0
        if "threat_status" in pred_source.columns:
            confirmed_count = len(pred_source[pred_source["threat_status"] == "CONFIRMED_ATTACK"])

        # Display Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Flow Windows Evaluated", total_eval)
        c2.metric("Normal Windows", normal_count)
        c3.metric(
            "Confirmed DDoS Floods",
            confirmed_count,
            delta=f"{confirmed_count} Critical" if confirmed_count > 0 else "None",
            delta_color="inverse",
        )
        c4.metric(
            "Transient Suspect Spikes",
            attack_count - confirmed_count,
            delta="Filtered by Persistence" if (attack_count - confirmed_count) > 0 else "0 Spikes",
        )

        st.markdown("#### Flow Classification Timeline")
        timeline_df = pd.DataFrame({
            "Window": range(total_eval),
            "Status": [0 if p == "Normal" else 1 for p in preds],
            "Classification": preds,
        })
        st.scatter_chart(timeline_df, x="Window", y="Classification", color="Classification")

        st.markdown("#### Detection Details by 10-Packet Flow Window")
        core_display_cols = ["window_id", "timestamp", "predicted_label", "threat_status", "confidence", "consecutive_windows"]
        feature_cols_display = ["Avg_TCP_pkt", "Avg_UDP_pkt", "Avg_syn_flag", "Avg_ack_flag", "Avg_pkts_lenght", "Avg_payload"]
        all_cols_to_show = [c for c in core_display_cols + feature_cols_display if c in pred_source.columns]

        st.dataframe(
            pred_source[all_cols_to_show].style.format({"confidence": "{:.1%}"}, na_rep=""),
            use_container_width=True
        )

    else:
        st.warning("No flow predictions found. Run `python src/extract_live_features.py` and `python src/predict.py`.")

    st.divider()
    st.markdown("### 🚨 Verified Security Threat Log")
    if ALERTS_PATH.exists():
        try:
            alerts_df = pd.read_csv(ALERTS_PATH)
            if not alerts_df.empty:
                st.dataframe(alerts_df, use_container_width=True)

                st.markdown("#### 🛡️ Threat Mitigation Intelligence")
                st.info("ℹ️ **Defensive Safety Policy:** Automatic IP blacklisting is disabled. Mitigation recommendations are provided for manual firewall administrator authorization.")

                attack_types = alerts_df["attack_type"].unique().tolist()
                for atk in attack_types:
                    if atk == "SYN Flooding":
                        st.error(f"**SYN Flooding Mitigation:** Enable TCP SYN Cookies (`sysctl -w net.ipv4.tcp_syncookies=1`), reduce SYN-RECEIVED timeout, apply connection rate limiting at edge router.")
                    elif atk == "UDP Flooding":
                        st.warning(f"**UDP Flooding Mitigation:** Implement stateless UDP threshold throttling, block unfragmented high-volume UDP bursts, coordinate upstream IP scrubbers.")
                    elif atk == "ACK Flooding":
                        st.warning(f"**ACK Flooding Mitigation:** Enforce stateful packet inspection (SPI) firewall rules; drop out-of-order ACK frames not mapped to active sessions.")
                    elif atk == "HTTP Flooding":
                        st.warning(f"**HTTP Flooding Mitigation:** Enable WAF application rate limiting, CAPTCHA challenges, and reverse proxy caching.")
            else:
                st.success("✅ Threat log is empty. No anomalous or attack traffic detected.")
        except Exception as e:
            st.error(f"Error reading alerts: {e}")
    else:
        st.success("✅ Threat log is empty (`outputs/alerts.csv`). No attack detected.")


# =========================================================================
# TAB 3: Interactive Classifier & Attack Simulator
# =========================================================================
with tab_simulator:
    st.subheader("Interactive Flow Classifier & Attack Simulator")
    st.caption("🧪 Simulation Sandbox — Flow simulations are isolated experiments and do NOT alter the live security alert log.")

    mode = st.radio("Select Testing Mode", ["Held-out Test Split Examples", "Custom Flow Simulator"], horizontal=True)

    if mode == "Held-out Test Split Examples":
        if test_df is not None and model is not None:
            target_col = metadata.get("target_column", "Label")
            feature_cols = metadata.get("feature_columns", [])

            st.markdown("Select a real test flow from the unseen evaluation dataset:")
            available_classes = sorted(test_df[target_col].unique())
            selected_class = st.selectbox("Filter by Class:", ["All"] + available_classes)

            filtered_df = test_df if selected_class == "All" else test_df[test_df[target_col] == selected_class]
            sample_candidates = filtered_df.head(50).index.tolist()

            selected_idx = st.selectbox("Select Test Sample Index:", sample_candidates)

            if st.button("Classify Selected Sample", type="primary"):
                sample_row = test_df.loc[[selected_idx]]
                actual_label = sample_row.iloc[0][target_col]
                input_feat = sample_row[feature_cols]

                predicted_label = model.predict(input_feat)[0]

                res_col1, res_col2 = st.columns(2)
                res_col1.metric("Ground Truth Label", str(actual_label))
                res_col2.metric("Model Prediction", str(predicted_label))

                if actual_label == predicted_label:
                    st.success("✅ Perfect Match: Model correctly identified this network flow.")
                else:
                    st.error("❌ Discrepancy: Model prediction differs from ground truth.")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_feat)[0]
                    prob_df = pd.DataFrame({
                        "Class": model.classes_,
                        "Confidence Score (%)": np.round(probs * 100, 2)
                    }).sort_values("Confidence Score (%)", ascending=False)
                    st.bar_chart(prob_df.set_index("Class"))
        else:
            st.error("Test dataset not available.")

    else:
        st.markdown("#### Synthetic Flow Parameter Adjuster")
        st.caption("Adjust flow parameters to observe model classification response in real-time.")

        sim_col1, sim_col2, sim_col3 = st.columns(3)
        with sim_col1:
            sim_tcp = st.slider("Avg TCP Packet Ratio", 0.0, 1.0, 1.0, 0.1)
            sim_udp = st.slider("Avg UDP Packet Ratio", 0.0, 1.0, 0.0, 0.1)
            sim_dns = st.slider("Avg DNS Packet Ratio", 0.0, 1.0, 0.0, 0.1)
            sim_icmp = st.slider("Avg ICMP Packet Ratio", 0.0, 1.0, 0.0, 0.1)
        with sim_col2:
            sim_syn = st.slider("Avg SYN Flag", 0.0, 1.0, 0.0, 0.1)
            sim_ack = st.slider("Avg ACK Flag", 0.0, 1.0, 1.0, 0.1)
            sim_fin = st.slider("Avg FIN Flag", 0.0, 1.0, 0.0, 0.1)
            sim_rst = st.slider("Avg RST Flag", 0.0, 1.0, 0.0, 0.1)
        with sim_col3:
            sim_pkt_len = st.slider("Avg Packet Length (bytes)", 54.0, 1500.0, 750.0, 10.0)
            sim_delta = st.slider("Avg Delta Time (sec)", 0.0001, 0.5, 0.02, 0.01)
            sim_payload = st.slider("Avg Payload (bytes)", 0.0, 1460.0, 680.0, 10.0)

        # Build feature vector
        feature_cols = metadata.get("feature_columns", [])
        sim_input = {col: 0.0 for col in feature_cols}
        sim_input["Avg_TCP_pkt"] = sim_tcp
        sim_input["Avg_UDP_pkt"] = sim_udp
        sim_input["Avg_DNS_pkt"] = sim_dns
        sim_input["Avg_ICMP_pkt"] = sim_icmp
        sim_input["Avg_syn_flag"] = sim_syn
        sim_input["Avg_ack_flag"] = sim_ack
        sim_input["Avg_fin_flag"] = sim_fin
        sim_input["Avg_rst_flag"] = sim_rst
        sim_input["Avg_pkts_lenght"] = sim_pkt_len
        sim_input["Min_pkts_lenght"] = min(sim_pkt_len, 54.0)
        sim_input["Max_pkts_lenght"] = max(sim_pkt_len, 1500.0)
        sim_input["Avg_delta_time"] = sim_delta
        sim_input["Duration_window_flow"] = sim_delta * 10.0
        sim_input["Avg_payload"] = sim_payload
        sim_input["Avg_small_payload_pkt"] = 1.0 if sim_payload <= 32.0 else 0.0

        sim_df = pd.DataFrame([sim_input])[feature_cols]
        sim_pred = model.predict(sim_df)[0]

        st.divider()
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.metric("Simulated Flow Classification", sim_pred)
            if sim_pred == "Normal":
                st.success("Flow classified as BENIGN (Normal traffic)")
            else:
                st.error(f"ATTACK DETECTED: {sim_pred}")
        with sc2:
            if hasattr(model, "predict_proba"):
                sim_probs = model.predict_proba(sim_df)[0]
                sim_prob_df = pd.DataFrame({
                    "Class": model.classes_,
                    "Confidence (%)": np.round(sim_probs * 100, 2)
                }).sort_values("Confidence (%)", ascending=False)
                st.dataframe(sim_prob_df, use_container_width=True)


# =========================================================================
# TAB 4: Model Analytics & Evaluation
# =========================================================================
with tab_eval:
    st.subheader("Model Analytics, Benchmark Evaluation & Operational Validation")
    st.caption("Strict separation between offline model training benchmarks and live operational validation.")

    eval_view = st.radio(
        "Select Evaluation Perspective:",
        ["Offline Training Benchmarks (14,200 Test Samples)", "Live & Lab Operational Validation"],
        horizontal=True,
    )

    if eval_view == "Offline Training Benchmarks (14,200 Test Samples)":
        if MODEL_COMP_PATH.exists():
            st.markdown("#### Benchmark Comparison Across Evaluated Classifiers")
            comp_df = pd.read_csv(MODEL_COMP_PATH)
            st.dataframe(
                comp_df.style.format({"accuracy": "{:.4%}", "macro_f1": "{:.4%}"}),
                use_container_width=True
            )

        col_cm, col_rep = st.columns([1, 1])
        with col_cm:
            st.markdown("#### Multi-Class Confusion Matrix (Held-out Test)")
            if CONFUSION_IMG_PATH.exists():
                st.image(str(CONFUSION_IMG_PATH), caption="Confusion Matrix on Held-out Test Split (14,200 samples)", use_container_width=True)
            else:
                st.info("Run `python src/evaluate_model.py` to generate the confusion matrix plot.")

        with col_rep:
            st.markdown("#### Detailed Classification Report")
            if REPORT_PATH.exists():
                rep_text = REPORT_PATH.read_text(encoding="utf-8")
                st.code(rep_text, language="text")
            else:
                st.info("Classification report not found at outputs/classification_report.txt.")

    else:
        st.markdown("#### Operational Live & Lab Validation Results")
        st.caption("Generated by `src/validate_live_detection.py` evaluating live Wi-Fi capture + multi-class attack flows.")

        if LIVE_VAL_METRICS_PATH.exists():
            val_metrics_df = pd.read_csv(LIVE_VAL_METRICS_PATH)
            st.markdown("##### Per-Class Live Validation Metrics")
            st.dataframe(
                val_metrics_df.style.format({
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.2%}",
                }),
                use_container_width=True
            )

        col_val_cm, col_val_rep = st.columns([1, 1])
        with col_val_cm:
            st.markdown("##### Live & Lab Validation Confusion Matrix")
            if LIVE_VAL_CM_IMG_PATH.exists():
                st.image(str(LIVE_VAL_CM_IMG_PATH), caption="Operational Live/Lab Validation Matrix", use_container_width=True)
            else:
                st.info("Run `python src/validate_live_detection.py` to generate the live validation plot.")

        with col_val_rep:
            st.markdown("##### Operational Audit Report & SYN Analysis")
            if LIVE_VAL_REPORT_PATH.exists():
                val_rep_text = LIVE_VAL_REPORT_PATH.read_text(encoding="utf-8")
                st.code(val_rep_text, language="text")
            else:
                st.info("Validation report not found at outputs/live_validation_report.txt.")
