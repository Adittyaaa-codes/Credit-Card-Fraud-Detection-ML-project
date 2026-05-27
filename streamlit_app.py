import io
import os
import requests
import pandas as pd
import streamlit as st

API_BASE = "http://51.21.150.144:8080"

st.set_page_config(
    page_title="FraudShield — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #0a0f1e;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 3rem 2rem 3rem !important;
    max-width: 1200px !important;
}

/* ── Top Nav Bar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.2rem 0 1rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 2.5rem;
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.nav-brand-icon { font-size: 1.6rem; }
.nav-brand-text {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.3px;
}
.nav-brand-sub {
    font-size: 0.7rem;
    font-weight: 400;
    color: #64748b;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.nav-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.8rem;
    font-weight: 500;
}
.dot-green { color: #22c55e; font-size: 0.55rem; }
.dot-red   { color: #ef4444; font-size: 0.55rem; }
.status-text-green { color: #86efac; }
.status-text-red   { color: #fca5a5; }

/* ── Hero Section ── */
.hero {
    text-align: center;
    padding: 1rem 0 2.5rem 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.35);
    color: #c4b5fd;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #f8fafc !important;
    letter-spacing: -1px;
    line-height: 1.1;
    margin: 0 0 1rem 0;
}
.hero h1 span {
    background: linear-gradient(135deg, #818cf8, #a78bfa, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    font-size: 1.05rem;
    color: #94a3b8 !important;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ── Cards ── */
.card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
}
.card-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.75rem;
}

/* ── Stat Cards ── */
.stat-card {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.25rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(139,92,246,0.4); }
.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.stat-value {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
    color: #f1f5f9;
}
.stat-value.purple { color: #a78bfa; }
.stat-value.green  { color: #4ade80; }
.stat-value.red    { color: #f87171; }
.stat-value.amber  { color: #fbbf24; }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9 !important;
    margin-bottom: 0.35rem;
}
.section-sub {
    font-size: 0.85rem;
    color: #64748b !important;
    margin-bottom: 1.2rem;
}

/* ── Streamlit overrides ── */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 2px dashed rgba(139,92,246,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"] * { color: #cbd5e1 !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:disabled {
    background: #1e293b !important;
    color: #475569 !important;
}

[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Dataframe dark */
.stDataFrame iframe { background: #111827; }

/* Metrics */
[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem !important;
}
[data-testid="metric-container"] label { color: #94a3b8 !important; }
[data-testid="metric-container"] [data-testid="metric-value"] { color: #f1f5f9 !important; font-weight: 700 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary:hover { color: #a78bfa !important; }

/* Alert/warning/error boxes */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Download button */
[data-testid="stDownloadButton"] button {
    background: #1e293b !important;
    color: #a78bfa !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: rgba(139,92,246,0.12) !important;
}

/* Caption/small text */
.stCaption p, small { color: #64748b !important; }

/* Info box */
.info-pill {
    background: rgba(139,92,246,0.1);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #c4b5fd;
    font-size: 0.85rem;
    line-height: 1.8;
}
.info-pill code {
    background: rgba(139,92,246,0.2);
    color: #e9d5ff;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.8rem;
}

/* Separator */
.sep {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

/* Spinner override */
.stSpinner > div { border-top-color: #8b5cf6 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def check_api():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


api_online = check_api()

# ══════════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════════
if api_online:
    status_html = "<span class='dot-green'>●</span><span class='status-text-green'>API Connected</span>"
else:
    status_html = "<span class='dot-red'>●</span><span class='status-text-red'>API Offline</span>"

st.markdown(f"""
<div class="navbar">
    <div class="nav-brand">
        <span class="nav-brand-icon">🛡️</span>
        <div>
            <div class="nav-brand-text">FraudShield</div>
            <div class="nav-brand-sub">ML Detection Platform</div>
        </div>
    </div>
    <div class="nav-status">{status_html}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ Powered by Machine Learning</div>
    <h1>Detect <span>Fraudulent</span><br>Transactions Instantly</h1>
    <p>Upload your transaction data and our ML pipeline will flag suspicious activity in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT — Two columns
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<p class="section-header">📂 Upload Transaction Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Drop a CSV file with credit card transactions below</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.caption(f"{len(df_preview):,} rows · {len(df_preview.columns)} columns detected")
        st.dataframe(df_preview.head(6), use_container_width=True, hide_index=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col_btn, _ = st.columns([2, 1])
        with col_btn:
            run_predict = st.button("🔍  Analyse Transactions", disabled=not api_online)

        if run_predict:
            uploaded_file.seek(0)
            with st.spinner("Analysing transactions…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        files={"file": ("data.csv", io.BytesIO(uploaded_file.read()), "text/csv")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as ex:
                    st.error(f"❌ Prediction failed: {ex}")
                    st.stop()

            st.markdown("<hr class='sep'>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">📊 Results</p>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"""<div class="stat-card"><div class="stat-label">Total</div>
                <div class="stat-value purple">{data['total']:,}</div></div>""", unsafe_allow_html=True)
            c2.markdown(f"""<div class="stat-card"><div class="stat-label">Legitimate</div>
                <div class="stat-value green">{data['legitimate']:,}</div></div>""", unsafe_allow_html=True)
            c3.markdown(f"""<div class="stat-card"><div class="stat-label">Fraudulent</div>
                <div class="stat-value red">{data['fraudulent']:,}</div></div>""", unsafe_allow_html=True)
            c4.markdown(f"""<div class="stat-card"><div class="stat-label">Fraud Rate</div>
                <div class="stat-value amber">{data['fraud_rate_pct']}%</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            result_df = pd.DataFrame(data["predictions"])
            result_df.index = range(1, len(result_df) + 1)
            st.dataframe(result_df, use_container_width=True, height=320)

            st.download_button(
                label="⬇️  Download Full Results (CSV)",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif not uploaded_file:
        st.markdown("""
        <div style="height: 180px; display: flex; align-items: center; justify-content: center;
                    border: 2px dashed rgba(139,92,246,0.2); border-radius: 12px; margin-top: 0.5rem;">
            <div style="text-align:center; color:#475569;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">📄</div>
                <div style="font-size:0.9rem; font-weight:500; color:#64748b;">Your preview will appear here</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    # Info card
    st.markdown("""
    <div class="card" style="margin-bottom: 1.25rem;">
        <div class="card-title">📋 Expected Format</div>
        <div class="info-pill">
            Required columns:<br>
            <code>Time</code> &nbsp;·&nbsp; <code>V1</code> – <code>V28</code> &nbsp;·&nbsp; <code>Amount</code><br><br>
            <code>Class</code> is optional and will be ignored.<br><br>
            Tip: Works with Kaggle's <b>Credit Card Fraud</b> dataset directly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline card
    st.markdown("""
    <div class="card">
        <div class="card-title">🔬 ML Pipeline</div>
    """, unsafe_allow_html=True)

    steps = [
        ("📥", "Data Ingestion",      "Pulls from MongoDB Atlas"),
        ("✅", "Validation",          "Schema & drift checks"),
        ("⚙️", "Transformation",      "KNN imputation + scaling"),
        ("🤖", "Model Training",      "Best of LR · DT · RF"),
        ("📈", "MLflow Tracking",     "Logged to DagsHub"),
    ]
    for icon, title, desc in steps:
        st.markdown(f"""
        <div style="display:flex; gap:0.75rem; align-items:flex-start; padding: 0.6rem 0;
                    border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="font-size:1.1rem; margin-top:1px;">{icon}</span>
            <div>
                <div style="color:#e2e8f0; font-size:0.85rem; font-weight:600;">{title}</div>
                <div style="color:#64748b; font-size:0.78rem;">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Train expander
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("⚙️  Retrain the Model"):
        st.markdown("""<p style='color:#94a3b8; font-size:0.85rem; margin-bottom:1rem;'>
            Triggers the full pipeline end-to-end. Takes 5–15 minutes.</p>""",
            unsafe_allow_html=True)
        if st.button("▶  Start Training Pipeline", disabled=not api_online):
            with st.spinner("Pipeline running…"):
                try:
                    resp = requests.get(f"{API_BASE}/train", timeout=900)
                    resp.raise_for_status()
                    data = resp.json()
                    st.success("✅ Training complete!")
                    st.metric("Test F1",        data.get("test_f1", "—"))
                    st.metric("Test Precision",  data.get("test_precision", "—"))
                    st.metric("Test Recall",     data.get("test_recall", "—"))
                except requests.exceptions.Timeout:
                    st.error("Timed out. The pipeline may still be running.")
                except Exception as ex:
                    st.error(f"Error: {ex}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class='sep'>
<div style='text-align:center; color:#334155; font-size:0.78rem; padding-bottom:1rem;'>
    FraudShield &nbsp;·&nbsp; Credit Card Fraud Detection &nbsp;·&nbsp; 
    Built with FastAPI &amp; Streamlit
</div>
""", unsafe_allow_html=True)
