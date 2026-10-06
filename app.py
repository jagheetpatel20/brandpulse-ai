"""
app.py
──────────────────────────────────────────────────────────────────────────────
BrandPulse AI — Real-time Brand Sentiment Analytics Dashboard
Streamlit application entry point.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Ensure project root is on sys.path ───────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
LABEL_ORDER = ["negative", "neutral", "positive"]
PALETTE: dict[str, str] = {
    "negative": "#FF4E6A",
    "neutral":  "#FFC947",
    "positive": "#4ECDC4",
}
EMOJI: dict[str, str] = {"negative": "😡", "neutral": "😐", "positive": "😊"}

MODEL_DIR  = os.path.join(PROJECT_ROOT, "models")
DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "Tweets.csv")
MAX_LEN    = 100   # must match Notebook 02


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  —  must be the very first Streamlit call
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="BrandPulse AI | Real-time Sentiment Analytics",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════
# CUSTOM CSS — dark theme + BrandPulse components
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Global base ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #1a1040 0%, #0F1117 45%, #0a1628 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141F 0%, #0F1117 100%);
    border-right: 1px solid #1E2130;
}
[data-testid="stSidebar"] * { color: #E0E0E0 !important; }

/* ── Metric cards ─────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(145deg, #1A1D2E 0%, #1E2240 100%);
    border: 1px solid #252840;
    border-radius: 18px;
    padding: 22px 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 6px;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    border-radius: 18px 18px 0 0;
}
.metric-card.blue::before  { background: linear-gradient(90deg, #7B61FF, #4ECDC4); }
.metric-card.green::before { background: linear-gradient(90deg, #4ECDC4, #44CF89); }
.metric-card.red::before   { background: linear-gradient(90deg, #FF4E6A, #FF8C42); }
.metric-card.amber::before { background: linear-gradient(90deg, #FFC947, #FF8C42); }
.metric-card:hover         { transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,0.5); }
.metric-label  { font-size: 11px; color: #6E7A9A; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
.metric-value  { font-size: 30px; font-weight: 800; color: #FFFFFF; line-height: 1.1; }
.metric-sub    { font-size: 11px; color: #4E5E7A; margin-top: 6px; }

/* ── Prediction badge ─────────────────────────────────────────── */
.pred-badge {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    padding: 16px 32px;
    border-radius: 60px;
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 1px;
    margin: 18px 0;
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
}
.badge-positive { background: linear-gradient(135deg,#0B3830,#164F44); border:2px solid #4ECDC4; color:#4ECDC4; }
.badge-neutral  { background: linear-gradient(135deg,#3A2E08,#554318); border:2px solid #FFC947; color:#FFC947; }
.badge-negative { background: linear-gradient(135deg,#380B18,#551624); border:2px solid #FF4E6A; color:#FF4E6A; }

/* ── Confidence bars ──────────────────────────────────────────── */
.conf-row        { margin: 10px 0; }
.conf-label      { display:flex; justify-content:space-between; font-size:13px; color:#C0C4D6; margin-bottom:5px; }
.conf-bar-wrap   { background:#1A1D2E; border-radius:10px; height:10px; overflow:hidden; }
.conf-bar        { height:10px; border-radius:10px; transition: width 0.5s cubic-bezier(.4,0,.2,1); }

/* ── Tweet cards ──────────────────────────────────────────────── */
.tweet-card {
    background: #1A1D2E;
    border: 1px solid #252840;
    border-left: 4px solid;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 13.5px;
    line-height: 1.65;
    color: #D0D4E8;
    transition: transform 0.15s ease;
}
.tweet-card:hover   { transform: translateX(3px); }
.tweet-card.positive { border-left-color: #4ECDC4; }
.tweet-card.neutral  { border-left-color: #FFC947; }
.tweet-card.negative { border-left-color: #FF4E6A; }

/* ── Alert / status banners ───────────────────────────────────── */
.alert-banner {
    background: linear-gradient(90deg,#380B18,#551624);
    border: 1px solid #FF4E6A44;
    border-radius: 12px;
    padding: 14px 22px;
    color: #FF4E6A;
    font-weight: 600;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    animation: pulse-glow 2.5s infinite;
    margin-bottom: 8px;
}
.ok-banner {
    background: linear-gradient(90deg,#0B3830,#164F44);
    border: 1px solid #4ECDC444;
    border-radius: 12px;
    padding: 14px 22px;
    color: #4ECDC4;
    font-weight: 600;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
@keyframes pulse-glow {
    0%,100% { box-shadow: 0 0 0   0 #FF4E6A22; }
    50%      { box-shadow: 0 0 18px 4px #FF4E6A22; }
}

/* ── Section headers ──────────────────────────────────────────── */
.section-header {
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 8px 0 4px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-sub { font-size: 12px; color: #5A6380; margin-bottom: 18px; }

/* ── Sidebar branding ─────────────────────────────────────────── */
.sidebar-logo {
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(90deg, #7B61FF, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.sidebar-sub { font-size: 11px; color: #4E5E7A !important; margin-top:2px; }

/* ── Tabs ─────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1A1D2E;
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #7A82A0 !important;
    font-weight: 500 !important;
    padding: 10px 22px !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#2A2060,#1E3D3A) !important;
    color: #FFFFFF !important;
    border-bottom: 2px solid #7B61FF !important;
}

/* ── Streamlit native overrides ───────────────────────────────── */
div[data-testid="stTextArea"] textarea {
    background: #1A1D2E !important;
    border: 1px solid #252840 !important;
    border-radius: 10px !important;
    color: #E0E0E0 !important;
    font-size: 14px !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7B61FF, #4ECDC4) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    color: #fff !important;
    padding: 10px 0 !important;
    transition: opacity 0.2s ease !important;
}
div.stButton > button[kind="primary"]:hover { opacity: 0.88 !important; }
div.stButton > button[kind="secondary"] {
    background: #1A1D2E !important;
    border: 1px solid #252840 !important;
    border-radius: 10px !important;
    color: #9EA8C0 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CACHED RESOURCE LOADERS
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🧠 Loading NLP preprocessor…")
def load_preprocessor() -> TextPreprocessor:
    """Load and cache the BrandPulse text preprocessor (spaCy / NLTK)."""
    return TextPreprocessor(spacy_model="en_core_web_sm", min_token_length=2)


@st.cache_resource(show_spinner="📦 Loading Classical TF-IDF pipeline…")
def load_classical_model():
    """Load the TF-IDF + LinearSVC joblib pipeline."""
    import joblib  # noqa: PLC0415
    path = os.path.join(MODEL_DIR, "classical_tfidf_pipeline.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_resource(show_spinner="🔬 Loading BiLSTM model…")
def load_lstm_resources():
    """Load the Keras BiLSTM model and fitted Tokenizer from disk."""
    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras.preprocessing.text import tokenizer_from_json  # noqa: PLC0415

    model_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    tok_path   = os.path.join(MODEL_DIR, "tokenizer.json")

    if not os.path.exists(model_path) or not os.path.exists(tok_path):
        return None, None

    model = tf.keras.models.load_model(model_path)
    with open(tok_path, "r", encoding="utf-8") as fh:
        tok = tokenizer_from_json(fh.read())
    return model, tok


@st.cache_data(show_spinner="📂 Loading dataset…")
def load_dataset() -> pd.DataFrame:
    """Load and lightly clean the raw Tweets CSV."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    keep = [c for c in ["airline_sentiment", "text", "airline", "tweet_created"] if c in df.columns]
    df = df[keep].dropna(subset=["text", "airline_sentiment"]).copy()
    df.rename(columns={"airline_sentiment": "label"}, inplace=True)
    if "tweet_created" in df.columns:
        df["tweet_dt"]   = pd.to_datetime(df["tweet_created"], utc=True, errors="coerce")
        df["tweet_hour"] = df["tweet_dt"].dt.floor("h")
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════

def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_classical(text: str, preprocessor: TextPreprocessor, pipeline) -> dict | None:
    """
    Predict sentiment using the TF-IDF scikit-learn pipeline.

    Parameters
    ----------
    text : str
        Raw tweet string.
    preprocessor : TextPreprocessor
        BrandPulse NLP cleaner.
    pipeline : sklearn.Pipeline
        Fitted TF-IDF + classifier pipeline.

    Returns
    -------
    dict | None
        {'label': str, 'probs': {'negative': float, 'neutral': float, 'positive': float}}
    """
    cleaned = preprocessor.clean(text)
    if not cleaned.strip():
        return {"label": "neutral", "probs": {l: (1.0 if l == "neutral" else 0.0) for l in LABEL_ORDER}}

    label  = pipeline.predict([cleaned])[0]
    clf    = pipeline.named_steps["clf"]
    tfidf  = pipeline.named_steps["tfidf"]

    if hasattr(clf, "predict_proba"):
        raw   = pipeline.predict_proba([cleaned])[0]
        classes = list(clf.classes_)
    elif hasattr(clf, "decision_function"):
        # LinearSVC → pseudo-probs via softmax on decision scores
        vec    = tfidf.transform([cleaned])
        scores = clf.decision_function(vec)[0]
        raw    = _softmax(scores)
        classes = list(clf.classes_)
    else:
        return {"label": label, "probs": {l: (1.0 if l == label else 0.0) for l in LABEL_ORDER}}

    probs = {cls: float(p) for cls, p in zip(classes, raw)}
    for lbl in LABEL_ORDER:
        probs.setdefault(lbl, 0.0)
    return {"label": label, "probs": probs}


def predict_lstm(text: str, preprocessor: TextPreprocessor, model, tokenizer) -> dict | None:
    """
    Predict sentiment using the Keras BiLSTM model.

    Parameters
    ----------
    text : str
        Raw tweet string.
    preprocessor : TextPreprocessor
        BrandPulse NLP cleaner.
    model : keras.Model
        Trained BiLSTM model.
    tokenizer : keras Tokenizer
        Fitted Tokenizer (vocab built from training data).

    Returns
    -------
    dict | None
        {'label': str, 'probs': dict}
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # noqa: PLC0415

    cleaned = preprocessor.clean(text)
    if not cleaned.strip():
        return {"label": "neutral", "probs": {l: (1.0 if l == "neutral" else 0.0) for l in LABEL_ORDER}}

    seq    = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob   = model.predict(padded, verbose=0)[0]          # shape (3,)

    probs = {lbl: float(p) for lbl, p in zip(LABEL_ORDER, prob)}
    label = LABEL_ORDER[int(np.argmax(prob))]
    return {"label": label, "probs": probs}


def run_inference(text: str, model_choice: str, resources: dict) -> dict | None:
    """Route a prediction to whichever model is selected in the sidebar."""
    preprocessor = resources["preprocessor"]

    if model_choice.startswith("Classical"):
        if resources["classical"] is None:
            return None
        return predict_classical(text, preprocessor, resources["classical"])
    else:
        if resources["lstm_model"] is None:
            return None
        return predict_lstm(text, preprocessor, resources["lstm_model"], resources["tokenizer"])


# ══════════════════════════════════════════════════════════════════
# PLOTLY CHART BUILDERS
# ══════════════════════════════════════════════════════════════════

_PLOTLY_BASE = dict(
    paper_bgcolor="#0F1117",
    plot_bgcolor="#0F1117",
    font=dict(color="#D0D4E8", family="Inter, sans-serif"),
    margin=dict(l=12, r=12, t=48, b=12),
)


def build_donut_chart(counts: dict[str, int]) -> go.Figure:
    """Animated Plotly donut chart from a {label: count} dict."""
    labels = LABEL_ORDER
    values = [counts.get(l, 0) for l in labels]
    colors = [PALETTE[l] for l in labels]
    total  = sum(values) or 1

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.68,
        marker=dict(colors=colors, line=dict(color="#0F1117", width=3)),
        textinfo="label+percent",
        textfont=dict(size=13, color="#FFFFFF"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        direction="clockwise",
        sort=False,
        pull=[0.03 if v == max(values) else 0 for v in values],
    ))
    fig.update_layout(
        **_PLOTLY_BASE,
        title=dict(text="Sentiment Distribution", font=dict(size=14, color="#FFFFFF"), x=0.5, xanchor="center"),
        showlegend=True,
        legend=dict(
            orientation="v", x=1.02, y=0.5, xanchor="left",
            font=dict(color="#C0C4D6", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        annotations=[dict(
            text=f"<b>{total:,}</b><br><span style='font-size:11px;color:#6E7A9A'>tweets</span>",
            x=0.5, y=0.5,
            font=dict(size=20, color="#FFFFFF"),
            showarrow=False,
            align="center",
        )],
        height=300,
    )
    return fig


def build_trend_chart(hourly_df: pd.DataFrame) -> go.Figure:
    """Rolling sentiment percentage line chart by hour."""
    fig = go.Figure()
    for lbl in LABEL_ORDER:
        col = f"pct_{lbl}"
        if col not in hourly_df.columns:
            continue
        color = PALETTE[lbl]
        fig.add_trace(go.Scatter(
            x=hourly_df["tweet_hour"],
            y=hourly_df[col],
            name=lbl.capitalize(),
            line=dict(color=color, width=2.5, shape="spline"),
            mode="lines+markers",
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor=f"{color}14",
            hovertemplate=f"<b>{lbl.capitalize()}</b>: %{{y:.1f}}%<br>Hour: %{{x|%Y-%m-%d %H:00}}<extra></extra>",
        ))
    fig.update_layout(
        **_PLOTLY_BASE,
        title=dict(
            text="24-Hour Hourly Sentiment Velocity",
            font=dict(size=14, color="#FFFFFF"), x=0.5, xanchor="center",
        ),
        xaxis=dict(
            showgrid=True, gridcolor="#1E2130",
            title=dict(text="Hour (UTC)", font=dict(color="#6E7A9A")),
            tickfont=dict(color="#6E7A9A"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1E2130",
            title=dict(text="Share (%)", font=dict(color="#6E7A9A")),
            tickfont=dict(color="#6E7A9A"),
            range=[0, 105],
        ),
        legend=dict(
            orientation="h", x=0.5, y=-0.18, xanchor="center",
            font=dict(color="#C0C4D6", size=12),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        height=320,
    )
    return fig


def compute_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-hour sentiment percentages from a scored DataFrame."""
    if df.empty or "tweet_hour" not in df.columns or "label" not in df.columns:
        return pd.DataFrame()
    grp = (
        df.groupby(["tweet_hour", "label"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for lbl in LABEL_ORDER:
        if lbl not in grp.columns:
            grp[lbl] = 0
    grp["total"] = grp[LABEL_ORDER].sum(axis=1)
    for lbl in LABEL_ORDER:
        grp[f"pct_{lbl}"] = (grp[lbl] / grp["total"].replace(0, 1) * 100).round(2)
    return grp.sort_values("tweet_hour").reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════
# HTML COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════

def _metric_card(label: str, value: str, sub: str = "", accent: str = "blue") -> str:
    return (
        f'<div class="metric-card {accent}">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value}</div>'
        f'  <div class="metric-sub">{sub}</div>'
        f'</div>'
    )


def _confidence_bars(probs: dict[str, float]) -> str:
    html = ""
    for lbl in LABEL_ORDER:
        pct   = probs.get(lbl, 0.0) * 100
        color = PALETTE[lbl]
        html += (
            f'<div class="conf-row">'
            f'  <div class="conf-label">'
            f'    <span>{EMOJI[lbl]} {lbl.capitalize()}</span>'
            f'    <span style="color:{color};font-weight:700">{pct:.1f}%</span>'
            f'  </div>'
            f'  <div class="conf-bar-wrap">'
            f'    <div class="conf-bar" style="width:{min(pct,100):.1f}%;background:{color};"></div>'
            f'  </div>'
            f'</div>'
        )
    return html


def _prediction_badge(label: str) -> str:
    return f'<div class="pred-badge badge-{label}">{EMOJI[label]}&nbsp;&nbsp;{label.upper()}</div>'


def _tweet_card(text: str, label: str, conf: float, show_meta: bool = True) -> str:
    color = PALETTE[label]
    meta = (
        f'<div style="font-size:11px;color:#4E5E7A;margin-bottom:7px;">'
        f'{EMOJI[label]} <span style="color:{color};font-weight:600">{label.upper()}</span>'
        f' &nbsp;·&nbsp; {conf*100:.0f}% confidence'
        f'</div>'
    ) if show_meta else ""
    display_text = text[:180] + ("…" if len(text) > 180 else "")
    return (
        f'<div class="tweet-card {label}">'
        f'{meta}'
        f'{display_text}'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════
# LOAD ALL RESOURCES
# ══════════════════════════════════════════════════════════════════

preprocessor   = load_preprocessor()
classical_pipe = load_classical_model()
lstm_model, tokenizer = load_lstm_resources()
df_full = load_dataset()

resources = {
    "preprocessor": preprocessor,
    "classical":    classical_pipe,
    "lstm_model":   lstm_model,
    "tokenizer":    tokenizer,
}


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🔍 BrandPulse AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Real-time Brand Sentiment Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("#### ⚙️ Inference Engine")
    model_choice = st.selectbox(
        "Active Model",
        ["Classical TF-IDF (LinearSVC)", "Deep Learning (Bi-LSTM)"],
        help=(
            "Classical: TF-IDF vectoriser + LinearSVC (fast, interpretable).\n"
            "Deep Learning: Bidirectional LSTM (context-aware)."
        ),
    )

    st.divider()
    st.markdown("#### 📡 Live Feed Settings")
    stream_batch_size = st.slider("Batch size (tweets)", 10, 100, 30, 10)
    stream_delay      = st.slider("Delay per tweet (s)", 0.1, 2.0, 0.5, 0.1)
    all_airlines      = ["United", "US Airways", "American", "Southwest", "Delta", "Virgin America"]
    airline_filter    = st.multiselect(
        "Filter airlines (leave empty for all)",
        all_airlines,
        default=[],
    )

    st.divider()
    classical_ok = classical_pipe is not None
    lstm_ok      = lstm_model is not None
    st.markdown("#### 🟢 Model Status")
    st.markdown(
        f"{'🟢' if classical_ok else '🔴'} **Classical pipeline** {'ready' if classical_ok else 'not found'}"
    )
    st.markdown(
        f"{'🟢' if lstm_ok else '🔴'} **BiLSTM model** {'ready' if lstm_ok else 'not found'}"
    )

    st.divider()
    st.markdown(
        '<div style="font-size:11px;color:#4E5E7A;line-height:2;">'
        "📁 <b>Dataset</b>: Twitter US Airline Sentiment<br>"
        "🧪 <b>Models</b>: TF-IDF + LinearSVC · BiLSTM<br>"
        "🔖 <b>Version</b>: 1.0.0"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
# DASHBOARD HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 6px 0 20px 0;">
    <div style="font-size:30px;font-weight:800;
                background:linear-gradient(90deg,#7B61FF 0%,#4ECDC4 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:6px;line-height:1.2;">
        🔍 BrandPulse AI
    </div>
    <div style="font-size:14px;color:#4E5E7A;">
        Real-time Brand Reputation Monitoring &nbsp;·&nbsp;
        Twitter US Airline Sentiment Dashboard
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# GLOBAL KPI CARDS
# ══════════════════════════════════════════════════════════════════

if not df_full.empty:
    vc = df_full["label"].value_counts()
    total_tw  = len(df_full)
    pct_pos   = vc.get("positive", 0) / total_tw * 100
    pct_neg   = vc.get("negative", 0) / total_tw * 100
    pct_neu   = vc.get("neutral",  0) / total_tw * 100
    net_score = pct_pos - pct_neg
    alert_on  = pct_neg > 50.0

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        st.markdown(
            _metric_card("📦 Total Tweets", f"{total_tw:,}", sub="in dataset", accent="blue"),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            _metric_card(
                "📈 Net Sentiment Score",
                f"{'+' if net_score >= 0 else ''}{net_score:.1f}%",
                sub="Positive % − Negative %",
                accent="green" if net_score >= 0 else "red",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            _metric_card(
                "😡 Negative Share",
                f"{pct_neg:.1f}%",
                sub=f"{vc.get('negative', 0):,} tweets",
                accent="red",
            ),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _metric_card(
                "🚨 Negative Alert",
                "ACTIVE" if alert_on else "CLEAR",
                sub=f"Threshold: >50% negative",
                accent="red" if alert_on else "green",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if alert_on:
        st.markdown(
            '<div class="alert-banner">'
            "🚨 &nbsp;<b>NEGATIVE ALERT ACTIVE</b> — Negative sentiment exceeds the 50% critical threshold. "
            "Immediate brand reputation review is recommended."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="ok-banner">'
            "✅ &nbsp;<b>Sentiment Health: Normal</b> — All indicators within acceptable parameters."
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🔬  Single Tweet Analyzer", "📡  Live Feed Simulator"])


# ─────────────────────────────────────────────────────────────────
# TAB 1 · SINGLE TWEET ANALYZER
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        '<div class="section-header">🔬 Single Tweet Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Enter any tweet below for real-time sentiment analysis with class-level confidence scores.</div>',
        unsafe_allow_html=True,
    )

    input_col, result_col = st.columns([1.15, 0.85], gap="large")

    # ── Input panel ───────────────────────────────────────────────
    with input_col:
        QUICK_EXAMPLES = [
            "— type your own tweet —",
            "The flight crew was exceptional and friendly!",
            "Waited 4 hours on the tarmac with no water. Terrible service.",
            "@AmericanAir my luggage has been lost for the third time this year!",
            "Flight arrived on time. Nothing special, nothing bad.",
            "@SouthwestAir thank you for the quick rebooking, you saved my trip!",
        ]
        example_choice = st.selectbox("💡 Quick example", QUICK_EXAMPLES)
        prefill = "" if example_choice.startswith("—") else example_choice

        user_tweet = st.text_area(
            "Tweet text",
            value=prefill,
            placeholder="Paste or type a tweet here…",
            height=130,
            key="user_tweet_input",
            label_visibility="collapsed",
        )

        active_model_ok = (
            classical_pipe is not None
            if model_choice.startswith("Classical")
            else lstm_model is not None
        )

        if not active_model_ok:
            st.warning(
                f"The selected model **{model_choice}** has not been trained yet. "
                "Run the corresponding notebook first.",
                icon="⚠️",
            )

        analyze_btn = st.button(
            "🔍 Analyze Sentiment",
            type="primary",
            use_container_width=True,
            disabled=not active_model_ok,
        )
        st.markdown(
            f'<div style="font-size:11px;color:#4E5E7A;margin-top:8px;">'
            f'Active engine: <span style="color:#7B61FF;font-weight:600">{model_choice}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Result panel ──────────────────────────────────────────────
    with result_col:
        st.markdown("**Prediction**")
        result_slot = st.empty()

        if analyze_btn:
            if not user_tweet.strip():
                result_slot.warning("Please enter some tweet text first.", icon="⚠️")
            else:
                with st.spinner("Running inference…"):
                    pred = run_inference(user_tweet.strip(), model_choice, resources)
                if pred:
                    label = pred["label"]
                    probs = pred["probs"]
                    result_slot.markdown(
                        _prediction_badge(label)
                        + '<div style="margin:14px 0 6px;font-size:12px;color:#6E7A9A;'
                          'text-transform:uppercase;letter-spacing:0.8px;">Confidence Breakdown</div>'
                        + _confidence_bars(probs),
                        unsafe_allow_html=True,
                    )
        else:
            result_slot.markdown("""
            <div style="text-align:center;padding:44px 20px;color:#2A2D4E;
                        border:1.5px dashed #252840;border-radius:14px;">
                <div style="font-size:42px;margin-bottom:14px;">🔍</div>
                <div style="font-size:13px;line-height:1.7;">
                    Enter a tweet above and click<br>
                    <b style="color:#7B61FF">Analyze Sentiment</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Quick batch test ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">⚡ Quick Batch Test</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Six labelled test tweets — green ✅ = correct prediction, red ❌ = mismatch.</div>',
        unsafe_allow_html=True,
    )

    BATCH_SAMPLES = [
        ("The flight crew was exceptional and friendly!", "positive"),
        ("Waited 4 hours on the tarmac with no water. Terrible service.", "negative"),
        ("@AmericanAir you lost my bag AGAIN. Completely unacceptable!", "negative"),
        ("Flight was on time, nothing special but no complaints.", "neutral"),
        ("@SouthwestAir thank you for the quick rebooking — saved my trip!", "positive"),
        ("Not bad for a budget airline. Arrived ten minutes late.", "neutral"),
    ]

    bc1, bc2 = st.columns(2, gap="medium")
    for idx, (tw, true_lbl) in enumerate(BATCH_SAMPLES):
        col = bc1 if idx % 2 == 0 else bc2
        with col:
            if active_model_ok:
                res = run_inference(tw, model_choice, resources)
                if res:
                    pred_lbl = res["label"]
                    conf_val = res["probs"][pred_lbl]
                    correct  = pred_lbl == true_lbl
                    icon     = "✅" if correct else "❌"
                    st.markdown(
                        _tweet_card(tw, pred_lbl, conf_val)
                        + f'<div style="font-size:11px;color:#4E5E7A;margin:-4px 0 12px 0;">'
                        + f'{icon} Pred: <span style="color:{PALETTE[pred_lbl]};font-weight:600">{pred_lbl}</span>'
                        + f' &nbsp;·&nbsp; True: <span style="color:{PALETTE[true_lbl]};font-weight:600">{true_lbl}</span>'
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    _tweet_card(tw, "neutral", 0.0)
                    + '<div style="font-size:11px;color:#4E5E7A;margin:-4px 0 12px 0;">'
                    + "⚠️ Model not loaded"
                    + "</div>",
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────────
# TAB 2 · LIVE FEED SIMULATOR
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="section-header">📡 Live Feed Simulator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        "Stream tweets from the dataset in real time. "
        "The donut chart and trend line update live as each tweet is classified."
        "</div>",
        unsafe_allow_html=True,
    )

    if df_full.empty:
        st.error(
            "Dataset not found. Place `Tweets.csv` in the `data/` directory and reload.",
            icon="🚫",
        )
        st.stop()

    if not active_model_ok:
        st.warning(
            f"Selected model **{model_choice}** is not available. Train it first via the corresponding notebook.",
            icon="⚠️",
        )

    # ── Build sample batch ────────────────────────────────────────
    sample_pool = df_full.copy()
    if airline_filter:
        lower_filter = [a.lower() for a in airline_filter]
        sample_pool  = sample_pool[sample_pool["airline"].str.lower().isin(lower_filter)]

    if sample_pool.empty:
        st.warning("No tweets found for the selected airline filter.", icon="⚠️")
    else:
        batch_df = sample_pool.sample(
            min(stream_batch_size, len(sample_pool)), random_state=42
        ).reset_index(drop=True)

        # ── Control row ───────────────────────────────────────────
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 4])
        with btn_col1:
            start_btn = st.button(
                "▶ Start Stream",
                type="primary",
                use_container_width=True,
                disabled=not active_model_ok,
            )
        with btn_col2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)

        if reset_btn:
            st.session_state.pop("stream_results", None)
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart area ────────────────────────────────────────────
        donut_col, trend_col = st.columns([1, 1.5], gap="large")
        with donut_col:
            st.markdown("**Sentiment Distribution**")
            donut_slot = st.empty()
        with trend_col:
            st.markdown("**24-Hour Hourly Sentiment Trend**")
            trend_slot = st.empty()

        # ── Feed area ─────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**📋 Tweet Stream**")
        feed_slot     = st.empty()
        progress_slot = st.progress(0)
        status_slot   = st.empty()

        # Pre-compute & render the dataset-wide hourly trend
        hourly_full = compute_hourly(df_full)
        if not hourly_full.empty:
            trend_slot.plotly_chart(build_trend_chart(hourly_full), use_container_width=True)

        # Render any previously streamed results from session state
        prev = st.session_state.get("stream_results", [])
        init_counts = {l: sum(1 for r in prev if r["label"] == l) for l in LABEL_ORDER}
        donut_slot.plotly_chart(build_donut_chart(init_counts), use_container_width=True)

        if prev:
            feed_slot.markdown(
                "".join(_tweet_card(r["text"], r["label"], r["conf"]) for r in reversed(prev[-8:])),
                unsafe_allow_html=True,
            )
            progress_slot.progress(1.0)
            status_slot.markdown(
                f'<span style="color:#4ECDC4;font-size:13px;">'
                f"✅ Previous stream: {len(prev)} tweets processed</span>",
                unsafe_allow_html=True,
            )

        # ── Streaming loop ────────────────────────────────────────
        if start_btn and active_model_ok:
            results_live: list[dict] = []
            live_counts  = {l: 0 for l in LABEL_ORDER}
            card_buffer:  list[str] = []
            n_total = len(batch_df)

            for i, (_, row) in enumerate(batch_df.iterrows()):
                raw_text = str(row.get("text", ""))
                pred = run_inference(raw_text, model_choice, resources)

                if pred is None:
                    continue

                lbl  = pred["label"]
                conf = pred["probs"][lbl]
                live_counts[lbl] += 1

                results_live.append({
                    "text":       raw_text,
                    "label":      lbl,
                    "conf":       conf,
                    "tweet_hour": row.get("tweet_hour"),
                })
                st.session_state["stream_results"] = results_live

                # Update donut chart
                donut_slot.plotly_chart(
                    build_donut_chart(live_counts),
                    use_container_width=True,
                )

                # Update tweet feed (last 6 cards, newest on top)
                card_buffer.append(_tweet_card(raw_text, lbl, conf))
                feed_slot.markdown(
                    "".join(reversed(card_buffer[-6:])),
                    unsafe_allow_html=True,
                )

                # Update progress + status
                progress_slot.progress((i + 1) / n_total)
                status_slot.markdown(
                    f'<span style="color:#FFC947;font-size:13px;">'
                    f"⏳ Processing {i+1}/{n_total} &nbsp;·&nbsp; "
                    f"<span style=\"color:{PALETTE[lbl]}\">{lbl.upper()}</span> "
                    f"detected ({conf*100:.0f}% confidence)</span>",
                    unsafe_allow_html=True,
                )

                time.sleep(stream_delay)

            # ── Stream complete: update trend with live scored data ─
            live_df = pd.DataFrame(results_live)
            if "tweet_hour" in live_df.columns and live_df["tweet_hour"].notna().any():
                live_hourly = compute_hourly(live_df)
                if not live_hourly.empty:
                    trend_slot.plotly_chart(
                        build_trend_chart(live_hourly),
                        use_container_width=True,
                    )

            progress_slot.progress(1.0)
            status_slot.markdown(
                f'<span style="color:#4ECDC4;font-size:13px;">'
                f"✅ Stream complete — <b>{n_total}</b> tweets processed</span>",
                unsafe_allow_html=True,
            )

            # ── Batch summary KPIs ────────────────────────────────
            st.markdown("---")
            st.markdown("**📊 Batch Summary**")
            total_s = sum(live_counts.values()) or 1
            pp = live_counts["positive"] / total_s * 100
            pn = live_counts["negative"] / total_s * 100

            sm1, sm2, sm3, sm4 = st.columns(4, gap="medium")
            with sm1:
                st.markdown(
                    _metric_card("Streamed", str(total_s), sub="tweets", accent="blue"),
                    unsafe_allow_html=True,
                )
            with sm2:
                st.markdown(
                    _metric_card(
                        "Net Score", f"{pp - pn:+.1f}%",
                        sub="Positive − Negative",
                        accent="green" if pp >= pn else "red",
                    ),
                    unsafe_allow_html=True,
                )
            with sm3:
                st.markdown(
                    _metric_card(
                        "Negative", str(live_counts["negative"]),
                        sub=f"{pn:.1f}%", accent="red",
                    ),
                    unsafe_allow_html=True,
                )
            with sm4:
                st.markdown(
                    _metric_card(
                        "Positive", str(live_counts["positive"]),
                        sub=f"{pp:.1f}%", accent="green",
                    ),
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:10px 0 6px;font-size:12px;color:#2A3050;">
    🔍 <strong style="color:#7B61FF">BrandPulse AI</strong>
    &nbsp;·&nbsp; Real-time Brand Sentiment Intelligence
    &nbsp;·&nbsp; v1.0.0
    &nbsp;·&nbsp;
    <span>Streamlit · Plotly · TensorFlow · scikit-learn · spaCy</span>
</div>
""", unsafe_allow_html=True)
