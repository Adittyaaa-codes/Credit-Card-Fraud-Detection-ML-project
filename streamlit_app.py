import os
import io
import time
import requests
import pandas as pd
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.3), rgba(168,85,247,0.3));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.hero h1 { font-size: 2.4rem; font-weight: 700; color: #fff; margin: 0; letter-spacing: -0.5px; }
.hero p  { color: rgba(255,255,255,0.65); margin: 0.5rem 0 0; font-size: 1.05rem; }

.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
}
.glass-card h3 { color: #a78bfa; font-size: 1rem; font-weight: 600;
                  text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.75rem; }

.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.metric-chip {
    flex: 1; min-width: 120px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-chip .label { font-size: 0.72rem; color: rgba(255,255,255,0.5);
                       text-transform: uppercase; letter-spacing: 1px; }
.metric-chip .value { font-size: 1.6rem; font-weight: 700; color: #a78bfa; line-height: 1.2; }

.api-online  { color: #10b981; font-weight: 700; }
.api-offline { color: #ef4444; font-weight: 700; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff; border: none; border-radius: 10px;
    padding: 0.65rem 2rem; font-weight: 600; font-size: 0.95rem;
    transition: all 0.2s ease; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99,102,241,0.45);
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04);
    border: 2px dashed rgba(99,102,241,0.4);
    border-radius: 14px; padding: 1rem;
}

hr { border-color: rgba(255,255,255,0.08) !important; }
h1,h2,h3,h4 { color: #f3f4f6 !important; }
p, li, label, span { color: #d1d5db; }
.stMarkdown p { color: #d1d5db; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def check_api() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

def api_status_badge() -> str:
    if check_api():
        return "<span class='api-online'>● API Online</span>"
    return "<span class='api-offline'>● API Offline — start app.py first</span>"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 1rem 0 1rem;'>
        <div style='font-size:3rem;'>🛡️</div>
        <div style='font-size:1.1rem; font-weight:700; color:#a78bfa;'>Fraud Shield</div>
        <div style='font-size:0.75rem; color:rgba(255,255,255,0.4); margin-top:4px;'>ML Detection Platform</div>
    </div>
    <div style='text-align:center; margin-bottom:1rem; font-size:0.82rem;'>{api_status_badge()}</div>
    """, unsafe_allow_html=True)

    st.divider()
    page = st.radio(
        "Navigation",
        ["🏠  Home", "🚀  Train Pipeline", "🔍  Predict"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(f"""
    <div style='font-size:0.72rem; color:rgba(255,255,255,0.3); text-align:center; padding-top:0.5rem;'>
        API → <code>{API_BASE}</code>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div class='hero'>
        <h1>🛡️ Credit Card Fraud Detection</h1>
        <p>Enterprise-grade ML pipeline — powered by FastAPI + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("📥 Data Ingestion",    "Pulls transactions from MongoDB Atlas, saves raw CSV to feature store, performs stratified train/test split."),
        ("✅ Validation & Drift", "Schema checks, required column validation, KS-test based dataset drift detection with YAML report."),
        ("⚙️ Transformation",     "KNN imputation pipeline fitted on training data, saves numpy arrays and serialised preprocessor."),
        ("🤖 Model Training",    "GridSearchCV over LR, DT, Random Forest. Best model selected by F1 score on 20% sub-sample."),
        ("📊 Metrics",           "Precision, Recall, F1 reported on full train & test sets after best model is selected."),
        ("🔍 Inference",         "Upload CSV → REST API call → fraud predictions with per-row labels and downloadable results."),
    ]
    for col, (title, desc) in zip([col1, col2, col3, col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class='glass-card'>
                <h3>{title}</h3>
                <p style='color:rgba(255,255,255,0.6); font-size:0.9rem;'>{desc}</p>
            </div>""", unsafe_allow_html=True)

    model_path = "final_model/model.pkl"
    st.markdown("<br>", unsafe_allow_html=True)
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        trained = time.strftime("%d %b %Y  %H:%M", time.localtime(os.path.getmtime(model_path)))
        st.markdown(f"""
        <div class='glass-card'>
            <h3>🟢 Model Status</h3>
            <div class='metric-row'>
                <div class='metric-chip'><div class='label'>Status</div>
                    <div style='color:#10b981;font-size:1rem;font-weight:700;margin-top:4px;'>Trained ✓</div></div>
                <div class='metric-chip'><div class='label'>Size</div>
                    <div class='value'>{size_kb:.1f}<span style='font-size:0.8rem;color:#6b7280;'> KB</span></div></div>
                <div class='metric-chip'><div class='label'>Last Trained</div>
                    <div style='color:#a78bfa;font-size:0.85rem;font-weight:600;margin-top:6px;'>{trained}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='glass-card'>
            <h3>🔴 Model Status</h3>
            <p style='color:#f87171;'>No trained model found. Go to <b>Train Pipeline</b> to train one.</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚀  Train Pipeline":
    st.markdown("""
    <div class='hero'>
        <h1>🚀 Training Pipeline</h1>
        <p>Calls <code>GET /train</code> on the FastAPI backend to run the full ML pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_api():
        st.error("❌ FastAPI server is not running. Start it with: `python app.py`")
        st.stop()

    st.markdown("""
    <div class='glass-card'>
        <h3>⚠️ Before You Start</h3>
        <ul style='color:rgba(255,255,255,0.65); font-size:0.9rem; line-height:1.8;'>
            <li>Ensure <code>.env</code> has a valid <code>DB_URL</code> MongoDB connection string</li>
            <li>Each run creates a new timestamped folder under <code>artifacts/</code></li>
            <li>Training uses a 20% stratified sample — expect ~5–10 minutes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        run_training = st.button("▶  Run Full Pipeline", use_container_width=True)

    if run_training:
        with st.spinner("Pipeline running… this may take several minutes"):
            try:
                resp = requests.get(f"{API_BASE}/train", timeout=600)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. The pipeline may still be running in the background.")
                st.stop()
            except Exception as ex:
                st.error(f"❌ API error: {ex}")
                st.stop()

        st.success("🎉 " + data.get("message", "Pipeline complete!"))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='glass-card'>
                <h3>🏋️ Train Metrics</h3>
                <div class='metric-row'>
                    <div class='metric-chip'><div class='label'>F1</div>
                        <div class='value'>{data['train_f1']}</div></div>
                    <div class='metric-chip'><div class='label'>Precision</div>
                        <div class='value'>{data['train_precision']}</div></div>
                    <div class='metric-chip'><div class='label'>Recall</div>
                        <div class='value'>{data['train_recall']}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='glass-card'>
                <h3>🧪 Test Metrics</h3>
                <div class='metric-row'>
                    <div class='metric-chip'><div class='label'>F1</div>
                        <div class='value'>{data['test_f1']}</div></div>
                    <div class='metric-chip'><div class='label'>Precision</div>
                        <div class='value'>{data['test_precision']}</div></div>
                    <div class='metric-chip'><div class='label'>Recall</div>
                        <div class='value'>{data['test_recall']}</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='glass-card'>
            <h3>💾 Saved Model</h3>
            <p style='font-size:0.9rem; color:rgba(255,255,255,0.6);'>
                {data.get('trained_model_path', 'artifacts/.../model.pkl')}<br>
                Final model → <code>final_model/model.pkl</code>
            </p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Predict":
    st.markdown("""
    <div class='hero'>
        <h1>🔍 Fraud Prediction</h1>
        <p>Calls <code>POST /predict</code> — upload a CSV and get instant results</p>
    </div>
    """, unsafe_allow_html=True)

    if not check_api():
        st.error("❌ FastAPI server is not running. Start it with: `python app.py`")
        st.stop()

    if not os.path.exists("final_model/model.pkl"):
        st.warning("⚠️ No trained model found. Please run the Training Pipeline first.")
        st.stop()

    col_up, col_info = st.columns([1, 1])
    with col_up:
        st.markdown("<div class='glass-card'><h3>📂 Upload Transactions CSV</h3></div>",
                    unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop CSV here",
            type=["csv"],
            label_visibility="collapsed",
        )
    with col_info:
        st.markdown("""
        <div class='glass-card'>
            <h3>ℹ️ Expected Format</h3>
            <p style='font-size:0.85rem; color:rgba(255,255,255,0.6); line-height:1.8;'>
                • Columns: <code>Time</code>, <code>V1</code>–<code>V28</code>, <code>Amount</code><br>
                • <code>Class</code> column is optional (ignored during inference)<br>
                • Any number of rows
            </p>
        </div>""", unsafe_allow_html=True)

    if uploaded_file:
        df_preview = pd.read_csv(uploaded_file)
        st.markdown(f"""
        <div class='glass-card'>
            <h3>📋 Preview</h3>
            <p style='font-size:0.8rem; color:rgba(255,255,255,0.45); margin-bottom:0.5rem;'>
                {len(df_preview):,} rows × {len(df_preview.columns)} columns
            </p>
        </div>""", unsafe_allow_html=True)
        st.dataframe(df_preview.head(10), use_container_width=True)

        col_run, _ = st.columns([1, 2])
        with col_run:
            run_predict = st.button("🔍  Run Prediction", use_container_width=True)

        if run_predict:
            uploaded_file.seek(0)
            csv_bytes = uploaded_file.read()

            with st.spinner("Sending to API…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/predict",
                        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as ex:
                    st.error(f"❌ API error: {ex}")
                    st.stop()

            n_total  = data["total"]
            n_fraud  = data["fraudulent"]
            n_legit  = data["legitimate"]
            fraud_pct = data["fraud_rate_pct"]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='glass-card'>
                <h3>📊 Prediction Summary</h3>
                <div class='metric-row'>
                    <div class='metric-chip'><div class='label'>Total</div>
                        <div class='value'>{n_total:,}</div></div>
                    <div class='metric-chip'><div class='label'>Legitimate</div>
                        <div class='value' style='color:#10b981;'>{n_legit:,}</div></div>
                    <div class='metric-chip'><div class='label'>Fraudulent</div>
                        <div class='value' style='color:#ef4444;'>{n_fraud:,}</div></div>
                    <div class='metric-chip'><div class='label'>Fraud Rate</div>
                        <div class='value' style='color:#f59e0b;'>{fraud_pct}<span style='font-size:0.8rem;'>%</span></div></div>
                </div>
            </div>""", unsafe_allow_html=True)

            result_df = pd.DataFrame(data["predictions"])
            result_df.index = range(1, len(result_df) + 1)

            st.markdown("<div class='glass-card'><h3>📄 Detailed Results</h3></div>",
                        unsafe_allow_html=True)
            st.dataframe(result_df, use_container_width=True, height=400)

            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️  Download Results CSV",
                data=csv_out,
                file_name="fraud_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
