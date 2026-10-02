"""AICS | SOC Console (v2)

Run:   pip install streamlit pandas pyarrow requests plotly
       streamlit run dashboard.py
Keeps the same Flask contract as before: POST /predict with one flow's features.
"""
from __future__ import annotations

import html
import json
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ----------------------------- Configuration -----------------------------
st.set_page_config(
    page_title="AICS | SOC Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
API_URL = "http://127.0.0.1:5000/predict"
API_ROOT = API_URL.rsplit("/", 1)[0] + "/"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = "llama3.2:3b"

DATASETS = {
    "Normal Traffic": "Benign-Monday-no-metadata.parquet",
    "DDoS Traffic": "DDoS-Friday-no-metadata.parquet",
}
MIXED = "Mixed traffic"
SOURCES = list(DATASETS.keys()) + [MIXED]
MAX_HISTORY = 300

SEV_WEIGHT = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
SEV_COLOR = {
    "Critical": "#ff3b5c",
    "High": "#ff7a45",
    "Medium": "#ffc247",
    "Low": "#5ad6a4",
    "Unknown": "#8fa1bb",
}
OK_COLOR = "#3ddc97"
ATTACK_COLOR = "#ff5d6c"
ACCENT = "#7aa2ff"

TACTICS = [
    "Reconnaissance", "Resource Development", "Initial Access", "Execution",
    "Persistence", "Privilege Escalation", "Defense Evasion", "Credential Access",
    "Discovery", "Lateral Movement", "Collection", "Command and Control",
    "Exfiltration", "Impact",
]
STATUSES = ["Open", "Investigating", "Contained", "Resolved", "False positive"]
OWNERS = ["Unassigned", "Tier 1", "Tier 2", "Incident lead"]

# ----------------------------- Styling -----------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root { color-scheme: dark; --ink:#0a0f1a; --panel:#111a2b; --line:#1f2c44; --text:#e6edf7; --muted:#8fa1bb; --accent:#7aa2ff; --ok:#3ddc97; --bad:#ff5d6c; }
    .stApp {
        background:
          radial-gradient(900px 420px at 8% -5%, rgba(122,162,255,.14), transparent 60%),
          radial-gradient(700px 380px at 100% 0%, rgba(255,93,108,.06), transparent 60%),
          #0a0f1a;
        color: var(--text);
        font-family: 'Manrope', sans-serif;
    }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1480px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
    h1, h2, h3, h4 { color: var(--text) !important; letter-spacing: -.02em; font-weight: 700 !important; }
    p, label, .stMarkdown, .stCaption { color: #c3d0e2; }
    [data-testid="stSidebar"] { background: #080d17; border-right: 1px solid var(--line); }

    .mono { font-family: 'IBM Plex Mono', monospace; font-size: .8rem; }
    .dim { color: var(--muted); }

    /* top bar */
    .topbar { display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
        padding: 18px 22px; border:1px solid var(--line); border-radius:14px;
        background: linear-gradient(110deg, #111c30, #0d1524); margin-bottom: 14px; }
    .brand { font-size: 1.55rem; font-weight: 800; letter-spacing:-.03em; }
    .brand-sub { color: var(--muted); font-size: .9rem; margin-top: 2px; }
    .chip { display:inline-flex; align-items:center; gap:7px; padding:4px 11px; border-radius:999px; margin:2px 0 2px 6px;
        border:1px solid color-mix(in srgb, var(--c, #8fa1bb) 45%, transparent);
        background: color-mix(in srgb, var(--c, #8fa1bb) 12%, transparent);
        color: var(--c, #c3d0e2); font-size:.78rem; font-weight:600; }
    .dot { width:7px; height:7px; border-radius:50%; background: var(--c, #8fa1bb); display:inline-block; }
    .dot.live { animation: pulse 1.8s ease-in-out infinite; }
    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.3; } }
    @media (prefers-reduced-motion: reduce) { .dot.live { animation:none; } }

    /* tabs */
    button[data-baseweb="tab"] { font-weight:600; color: var(--muted); }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
    div[data-baseweb="tab-highlight"] { background: var(--accent); }

    /* metrics + buttons */
    div[data-testid="stMetric"] { background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
    div[data-testid="stMetricLabel"] { color: var(--muted) !important; }
    div[data-testid="stMetricValue"] { color: var(--text) !important; font-weight:700; }
    div[data-testid="stButton"] > button, div[data-testid="stDownloadButton"] > button {
        border-radius:10px; border:1px solid #2f4a7a; background:#16315a; color:#fff; font-weight:600; min-height:2.5rem; }
    div[data-testid="stButton"] > button:hover, div[data-testid="stDownloadButton"] > button:hover { border-color: var(--accent); background:#1d3f73; }
    div[data-testid="stButton"] > button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

    /* panels */
    .panel { background: var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin-bottom:10px; }
    .panel-title { font-weight:700; color: var(--text); margin-bottom:4px; }
    .muted { color: var(--muted); font-size:.88rem; }
    .section-note { color: var(--muted); font-size:.82rem; margin:-4px 0 8px; }

    /* event banner */
    .banner { border-radius:12px; padding:16px 20px; margin: 6px 0 14px; border:1px solid var(--line); border-left-width:5px; }
    .banner.attack { border-color: rgba(255,93,108,.45); border-left-color: var(--bad); background: rgba(255,93,108,.08); }
    .banner.normal { border-color: rgba(61,220,151,.35); border-left-color: var(--ok); background: rgba(61,220,151,.06); }
    .banner-title { font-size:1.45rem; font-weight:800; letter-spacing:-.02em; margin: 2px 0 4px; }

    /* ticker */
    .ticker { border:1px solid var(--line); border-radius:12px; overflow:hidden; background: var(--panel); }
    .tick { display:grid; grid-template-columns: 74px 84px 84px 1fr 72px; gap:10px; align-items:center;
        padding:9px 14px; border-bottom:1px solid #16233a; font-size:.9rem; }
    .tick:last-child { border-bottom:none; }
    .tick.attack { background: rgba(255,93,108,.05); }
    .tick-type { font-weight:600; }
    .empty { padding:18px; color: var(--muted); text-align:center; }

    /* pipeline */
    .pipe { display:flex; gap:8px; flex-wrap:wrap; }
    .stage { flex:1 1 150px; border-top:3px solid var(--line); background: var(--panel); border-radius:0 0 10px 10px; padding:10px 12px; }
    .stage.done { border-top-color: var(--accent); }
    .stage.skip { border-top-style:dashed; opacity:.5; }
    .stage .t { font-weight:700; font-size:.9rem; }
    .stage .d { color: var(--muted); font-size:.78rem; margin-top:2px; }

    /* MITRE grid */
    .tactics { display:grid; grid-template-columns: repeat(7, minmax(0,1fr)); gap:8px; }
    @media (max-width: 1100px) { .tactics { grid-template-columns: repeat(3, minmax(0,1fr)); } .tick { grid-template-columns: 70px 80px 1fr; } .tick > :nth-child(3), .tick > :nth-child(5) { display:none; } }
    .tactic { border:1px solid var(--line); background: var(--panel); border-radius:10px; padding:10px 10px 12px; min-height:84px; }
    .tactic .n { font-size:.82rem; font-weight:700; color: var(--muted); }
    .tactic.hit { border-color: rgba(255,93,108,.6); background: rgba(255,93,108,.1); }
    .tactic.hit .n { color: var(--text); }
    .tactic .ids { margin-top:6px; font-size:.75rem; color:#ffb3bb; font-family:'IBM Plex Mono', monospace; }
    .tactic .c { font-size:1.2rem; font-weight:800; color: var(--text); }

    div[data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background: rgba(13,21,36,.7); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- State -------------------------------------
def fresh_state() -> dict:
    return {
        "history": [],        # newest first
        "flow_cache": {},     # event_id -> raw feature dict
        "total_flows": 0,
        "attack_count": 0,
        "normal_count": 0,
        "latest": None,
        "last_error": None,
        "incidents": {},      # event_id -> {status, owner, notes}
        "chat": [],
        "pending_q": None,
        "event_seq": 0,
        "cursor": {},         # source -> next row for sequential mode
    }


for _k, _v in fresh_state().items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ----------------------------- Helpers -----------------------------------
def is_attack(r: dict) -> bool:
    return str(r.get("prediction", "")).strip().lower() == "attack"


def sev_of(r: dict) -> str:
    return str(r.get("severity") or "Unknown").strip().title()


def esc(x) -> str:
    return html.escape(str(x))


@st.cache_data(ttl=5, show_spinner=False)
def probe(url: str) -> bool:
    try:
        return requests.get(url, timeout=1.5).status_code < 500
    except requests.RequestException:
        return False


@st.cache_resource(show_spinner="Loading dataset…")
def load_dataset(label: str) -> pd.DataFrame:
    path = DATASET_DIR / DATASETS[label]
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"The dataset '{label}' is empty.")
    return df


@st.cache_resource(show_spinner=False)
def benign_baseline():
    """Mean/std of every numeric feature in benign traffic, used for the deviation chart."""
    try:
        df = load_dataset("Normal Traffic")
    except (FileNotFoundError, ValueError):
        return None
    num = df.drop(columns=["Label"], errors="ignore").select_dtypes("number")
    return num.mean(), num.std().replace(0, float("nan"))


def fetch_flow(source: str, row_idx: int, randomize: bool, attack_ratio: float):
    if source == MIXED:
        source = "DDoS Traffic" if random.random() < attack_ratio else "Normal Traffic"
        randomize = True
    df = load_dataset(source)
    n = len(df)
    if randomize:
        idx = random.randrange(n)
    else:
        idx = max(0, row_idx) % n  # wraps to the start after the last row
    row = df.iloc[idx]
    truth = str(row["Label"]) if "Label" in df.columns else None
    feats = row.drop(labels=["Label"], errors="ignore").to_dict()
    return source, idx, feats, truth


def call_api(features: dict):
    t0 = time.perf_counter()
    resp = requests.post(API_URL, json=features, timeout=180)
    latency = (time.perf_counter() - t0) * 1000
    if not resp.ok:
        raise RuntimeError(f"Prediction API returned HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json(), latency


def register(result: dict, source: str, idx: int, feats: dict, truth, latency: float) -> dict:
    ss = st.session_state
    ss.event_seq += 1
    result.update(
        event_id=f"EVT-{ss.event_seq:04d}",
        dataset=source,
        row_number=idx,
        analyzed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        latency_ms=round(latency),
        ground_truth=truth,
    )
    ss.latest = result
    ss.total_flows += 1
    if is_attack(result):
        ss.attack_count += 1
    else:
        ss.normal_count += 1
    ss.history.insert(0, result)
    ss.flow_cache[result["event_id"]] = feats
    if len(ss.history) > MAX_HISTORY:
        for old in ss.history[MAX_HISTORY:]:
            ss.flow_cache.pop(old["event_id"], None)
        ss.history = ss.history[:MAX_HISTORY]
    ss.last_error = None
    return result


def process_flow(source, row_idx, randomize, attack_ratio) -> dict:
    src, idx, feats, truth = fetch_flow(source, row_idx, randomize, attack_ratio)
    result, latency = call_api(feats)
    return register(result, src, idx, feats, truth, latency)


def score_of(events: list) -> float:
    """Attack share weighted by severity, 0-100."""
    if not events:
        return 0.0
    attacks = [e for e in events if is_attack(e)]
    if not attacks:
        return 0.0
    ratio = len(attacks) / len(events)
    sev = sum(SEV_WEIGHT.get(sev_of(e), 2) for e in attacks) / len(attacks) / 4
    return round(100 * ratio * (0.5 + 0.5 * sev), 1)


def posture(window: int = 20):
    ev = st.session_state.history[:window]
    if not ev:
        return 0.0, "Idle", "#8fa1bb"
    s = score_of(ev)
    if s < 10:
        return s, "Stable", OK_COLOR
    if s < 35:
        return s, "Elevated", SEV_COLOR["Medium"]
    if s < 65:
        return s, "High", SEV_COLOR["High"]
    return s, "Critical", SEV_COLOR["Critical"]


def events_df() -> pd.DataFrame:
    rows = []
    for e in st.session_state.history:
        m = e.get("mitre") or {}
        att = is_attack(e)
        rows.append({
            "Event": e["event_id"],
            "Time": e["analyzed_at"],
            "Source": e["dataset"],
            "Row": e["row_number"],
            "Prediction": e.get("prediction"),
            "Attack type": e.get("attack_type") or "—",
            "Severity": sev_of(e) if att else "—",
            "Technique": m.get("technique_id", "—") if att else "—",
            "Latency (ms)": e.get("latency_ms"),
            "Dataset label": e.get("ground_truth") or "—",
        })
    return pd.DataFrame(rows)


def style_fig(fig: go.Figure, h: int = 300) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#c3d0e2"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#1a2740", zerolinecolor="#1a2740")
    fig.update_yaxes(gridcolor="#1a2740", zerolinecolor="#1a2740")
    return fig


def chip(label: str, color: str, live: bool = False) -> str:
    return f'<span class="chip" style="--c:{color}"><span class="dot{" live" if live else ""}"></span>{esc(label)}</span>'


def ticker_html(events: list, n: int = 8) -> str:
    if not events:
        return '<div class="ticker"><div class="empty">No flows analyzed yet. Pick a traffic source and run an analysis.</div></div>'
    rows = ""
    for e in events[:n]:
        att = is_attack(e)
        sev = sev_of(e)
        tag = chip(sev, SEV_COLOR.get(sev, "#8fa1bb")) if att else chip("Normal", OK_COLOR)
        rows += (
            f'<div class="tick {"attack" if att else "normal"}">'
            f'<span class="mono dim">{esc(e["analyzed_at"][-8:])}</span>'
            f'<span class="mono">{esc(e["event_id"])}</span>'
            f'<span>{tag}</span>'
            f'<span class="tick-type">{esc(e.get("attack_type") or "Benign flow")}</span>'
            f'<span class="mono dim">{esc(e.get("latency_ms", "—"))} ms</span>'
            f'</div>'
        )
    return f'<div class="ticker">{rows}</div>'


def pipeline_html() -> str:
    latest = st.session_state.latest
    stages = [
        ("Network flow", "CIC-IDS2017 record"),
        ("Preprocessing", "Feature order and scaling"),
        ("Anomaly detection", "Isolation Forest"),
        ("Attack classification", "Random Forest"),
        ("Threat context", "MITRE ATT&CK mapping"),
        ("AI analyst", "Local Llama 3.2"),
    ]
    out = ""
    for i, (t, d) in enumerate(stages):
        if latest is None:
            cls, note = "", d
        elif i < 3 or is_attack(latest):
            cls, note = "done", d
        else:
            cls, note = "skip", "Skipped: flow looked normal"
        out += f'<div class="stage {cls}"><div class="t">{t}</div><div class="d">{note}</div></div>'
    return f'<div class="pipe">{out}</div>'


# ----------------------------- Analysis runner ---------------------------
def run_analysis(mode, source, row, randomize, count, attack_ratio, delay, advance=False):
    ss = st.session_state
    live = mode == "Live stream"
    total = 1 if mode == "Single flow" else count
    randomize = randomize or live
    ticker = st.empty()
    bar = st.progress(0.0, text="Starting analysis…") if total > 1 else None
    done = attacks = 0
    problems = []

    for i in range(total):
        try:
            if total == 1:
                with st.spinner("Running the detection pipeline…"):
                    res = process_flow(source, row, randomize, attack_ratio)
            else:
                res = process_flow(source, row + i, randomize, attack_ratio)
        except requests.ConnectionError:
            ss.last_error = (
                f"Can't reach the prediction API at {API_URL}. "
                "Start the Flask server, then run the analysis again."
            )
            break
        except (requests.RequestException, RuntimeError) as exc:
            problems.append(f"Flow {i + 1}: {exc}")
            continue
        done += 1
        attacks += int(is_attack(res))
        if total > 1:
            ticker.markdown(ticker_html(ss.history, 6), unsafe_allow_html=True)
            bar.progress((i + 1) / total, text=f"Analyzed {i + 1} of {total} flows. Attack flows also run the local LLM, so they take longer.")
            if live and i < total - 1:
                time.sleep(delay)

    if advance and done:
        ss.cursor[source] = row + done
    ticker.empty()
    if bar:
        bar.empty()
    for p in problems[:3]:
        st.warning(p)
    if done:
        st.toast(f"Analyzed {done} flow{'s' if done != 1 else ''} · {attacks} attack{'s' if attacks != 1 else ''}", icon="🛡️")


# ----------------------------- Sidebar -----------------------------------
api_online = probe(API_ROOT)
llm_online = probe(OLLAMA_URL + "/")

with st.sidebar:
    st.markdown("## 🛡️ AI-powered SOC analyst console")
    st.divider()
    st.markdown("#### Traffic")
    source = st.selectbox("Traffic source", SOURCES, help="Mixed traffic blends benign and DDoS flows, like a real feed.")
    attack_ratio = 0.3
    if source == MIXED:
        attack_ratio = st.slider("Attack share", 0.0, 1.0, 0.3, 0.05, format="%.2f")
    mode = st.radio("Mode", ["Row range", "Single flow", "Batch", "Live stream"], horizontal=False,
                    help="Row range analyzes every row between two numbers you enter. Batch analyzes a set number of flows. Live stream samples random flows with a delay between each.")
    count, delay = 1, 1.0
    cursor = st.session_state.cursor
    randomize, advance, row_number = True, False, 0
    range_blocked = False
    if mode == "Row range":
        if source == MIXED:
            st.warning("Row ranges need a single dataset. Choose Normal Traffic or DDoS Traffic.")
            range_blocked = True
        else:
            try:
                n_rows = len(load_dataset(source))
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
                n_rows, range_blocked = 0, True
            if n_rows:
                st.caption(f"{source} has {n_rows:,} rows (0 to {n_rows - 1:,}).")
                r1, r2 = st.columns(2)
                range_start = int(r1.number_input("From row", min_value=0, max_value=n_rows - 1, value=0, step=1))
                range_end = int(r2.number_input("To row", min_value=0, max_value=n_rows - 1, value=min(99, n_rows - 1), step=1))
                if range_end < range_start:
                    st.warning("'To row' must be the same as or after 'From row'.")
                    range_blocked = True
                else:
                    count = range_end - range_start + 1
                    randomize, advance, row_number = False, False, range_start
                    st.caption(f"Will analyze {count:,} flows (both rows included). Attack flows also run the local LLM, so large ranges take a while.")
    elif mode in ("Single flow", "Batch") and source != MIXED:
        pick = st.radio(
            "Which flows?", ["Random", "Next in sequence"], horizontal=True,
            help="Random samples any row automatically. Next in sequence continues from where the last run stopped.",
        )
        if pick == "Next in sequence":
            randomize, advance = False, True
            row_number = cursor.get(source, 0)
            st.caption(f"Starts at row {row_number:,}")
        with st.expander("Advanced: pick a row manually"):
            manual = st.checkbox("Use a specific row", value=False)
            manual_row = int(st.number_input("Dataset row", min_value=0, value=0, step=1, disabled=not manual))
            if manual:
                randomize, advance, row_number = False, False, manual_row
    if mode == "Batch":
        count = st.slider("Flows to analyze", 2, 50, 8)
    elif mode == "Live stream":
        count = st.slider("Flows in this stream", 5, 100, 20)
        delay = st.slider("Seconds between flows", 0.0, 3.0, 1.0, 0.1)
        st.caption("To stop a stream early, click Reset or change any control.")
    st.markdown("")
    run_clicked = st.button("Analyze traffic", use_container_width=True, type="primary", disabled=range_blocked)
    reset_clicked = st.button("Reset session", use_container_width=True)
    st.divider()
    st.markdown("#### System")
    st.markdown(chip("Flask API", OK_COLOR if api_online else ATTACK_COLOR, api_online) , unsafe_allow_html=True)
    st.markdown(chip(f"Ollama · {OLLAMA_MODEL}", OK_COLOR if llm_online else SEV_COLOR["Medium"], llm_online), unsafe_allow_html=True)
    st.caption("The API runs Isolation Forest, Random Forest and MITRE mapping. Ollama powers the copilot tab.")

if reset_clicked:
    for k, v in fresh_state().items():
        st.session_state[k] = v
    st.rerun()

# ----------------------------- Header ------------------------------------
st.markdown(
    f"""
    <div class="topbar">
      <div>
        <div class="brand">Intelligent Security Analytics Platform</div>
        <div class="brand-sub">Detect anomalies. Classify threats. Investigate incidents.</div>
      </div>
      <div>
        {chip("Prediction API", OK_COLOR if api_online else ATTACK_COLOR, api_online)}
        {chip("Local LLM", OK_COLOR if llm_online else SEV_COLOR["Medium"], llm_online)}
        {chip("CIC-IDS2017 benchmark data", ACCENT)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

live_zone = st.container()
if run_clicked:
    with live_zone:
        try:
            run_analysis(mode, source, row_number, randomize, count, attack_ratio, delay, advance)
        except (requests.RequestException, RuntimeError, ValueError, FileNotFoundError) as exc:
            st.session_state.last_error = str(exc)

if st.session_state.last_error:
    st.error(st.session_state.last_error)


# ----------------------------- Tabs --------------------------------------
def tab_overview():
    ss = st.session_state
    score, label, color = posture()
    left, right = st.columns([1, 1.7])

    with left:
        st.markdown("#### Threat posture")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number={"font": {"size": 46, "color": "#e6edf7"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#41526d", "tickfont": {"color": "#8fa1bb", "size": 10}},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 10], "color": "rgba(61,220,151,.16)"},
                    {"range": [10, 35], "color": "rgba(255,194,71,.14)"},
                    {"range": [35, 65], "color": "rgba(255,122,69,.14)"},
                    {"range": [65, 100], "color": "rgba(255,59,92,.16)"},
                ],
            },
        ))
        fig.update_layout(height=210, margin=dict(l=24, r=24, t=8, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div style='text-align:center;margin-top:-14px'>{chip(label, color, label in ('High', 'Critical'))}</div>", unsafe_allow_html=True)
        st.markdown("<div class='section-note' style='text-align:center;margin-top:8px'>Share of attacks in the last 20 flows, weighted by severity.</div>", unsafe_allow_html=True)

    with right:
        st.markdown("#### Session at a glance")
        attacks = [e for e in ss.history if is_attack(e)]
        open_inc = sum(1 for e in attacks if ss.incidents.get(e["event_id"], {}).get("status", "Open") == "Open")
        lat = [e["latency_ms"] for e in ss.history[:50] if e.get("latency_ms") is not None]
        k1, k2, k3 = st.columns(3)
        k1.metric("Flows analyzed", ss.total_flows)
        k2.metric("Attacks detected", ss.attack_count)
        k3.metric("Normal flows", ss.normal_count)
        k4, k5, k6 = st.columns(3)
        k4.metric("Detection rate", f"{ss.attack_count / ss.total_flows * 100:.1f}%" if ss.total_flows else "—",
                  help="Share of analyzed flows classified as attacks. This is not the model's benchmark recall or accuracy.")
        k5.metric("Avg response time", f"{sum(lat) / len(lat) / 1000:.1f} s" if lat else "—",
                  help="Average API round trip over the last 50 flows. Attack flows include LLM time.")
        k6.metric("Open incidents", open_inc)

    st.markdown("#### Live event feed")
    st.markdown(ticker_html(ss.history, 8), unsafe_allow_html=True)

    st.markdown("#### Detection pipeline")
    st.markdown("<div class='section-note'>Stages the latest flow passed through.</div>", unsafe_allow_html=True)
    st.markdown(pipeline_html(), unsafe_allow_html=True)

    st.markdown("#### Analytics")
    if not ss.history:
        st.info("Charts appear after you analyze some flows.")
        return

    c1, c2 = st.columns([1.6, 1])
    with c1:
        events = list(reversed(ss.history[:60]))
        xs, ys, cols, txt = [], [], [], []
        for i, e in enumerate(events):
            xs.append(i + 1)
            ys.append(score_of(events[max(0, i - 4):i + 1]))
            cols.append(SEV_COLOR.get(sev_of(e), ATTACK_COLOR) if is_attack(e) else OK_COLOR)
            txt.append(f"{e['event_id']} · {e.get('attack_type') or 'Benign'}")
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="lines+markers", text=txt,
            line=dict(color=ACCENT, width=2), fill="tozeroy", fillcolor="rgba(122,162,255,.10)",
            marker=dict(size=9, color=cols, line=dict(width=1, color="#0a0f1a")),
            hovertemplate="%{text}<br>Rolling score %{y:.0f}<extra></extra>",
        ))
        fig.update_layout(title=dict(text="Threat timeline (5-flow rolling score, dots show each flow)", font=dict(size=14)),
                          yaxis=dict(range=[0, 105], title="Score"), xaxis=dict(title="Flow sequence"), showlegend=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with c2:
        sev_counts = pd.Series([sev_of(e) for e in ss.history if is_attack(e)]).value_counts()
        if sev_counts.empty:
            st.info("No attacks yet, so there is no severity mix to show.")
        else:
            fig = go.Figure(go.Pie(
                labels=sev_counts.index, values=sev_counts.values, hole=0.62, sort=False,
                marker=dict(colors=[SEV_COLOR.get(s, "#8fa1bb") for s in sev_counts.index], line=dict(color="#0a0f1a", width=2)),
                textinfo="value",
            ))
            fig.update_layout(title=dict(text="Attacks by severity", font=dict(size=14)))
            st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        types = pd.Series([e.get("attack_type") or "Unknown" for e in ss.history if is_attack(e)]).value_counts()
        if types.empty:
            st.info("No attack types observed yet.")
        else:
            fig = go.Figure(go.Bar(x=types.values[::-1], y=types.index[::-1], orientation="h", marker_color=ATTACK_COLOR))
            fig.update_layout(title=dict(text="Attack types observed", font=dict(size=14)))
            st.plotly_chart(style_fig(fig, 260), use_container_width=True)
    with c4:
        pc = pd.Series([("Attack" if is_attack(e) else "Normal") for e in ss.history]).value_counts()
        fig = go.Figure(go.Bar(x=pc.index, y=pc.values, marker_color=[ATTACK_COLOR if k == "Attack" else OK_COLOR for k in pc.index]))
        fig.update_layout(title=dict(text="Prediction distribution", font=dict(size=14)))
        st.plotly_chart(style_fig(fig, 260), use_container_width=True)


def deviation_fig(feats: dict):
    base = benign_baseline()
    if base is None:
        return None
    mean, std = base
    s = pd.to_numeric(pd.Series(feats), errors="coerce")
    z = ((s - mean) / std).replace([float("inf"), float("-inf")], float("nan")).dropna()
    if z.empty:
        return None
    top = z.reindex(z.abs().sort_values(ascending=False).index).head(10)[::-1]
    shown = top.clip(-100, 100)
    fig = go.Figure(go.Bar(
        x=shown.values, y=shown.index, orientation="h",
        marker_color=[ATTACK_COLOR if v > 0 else ACCENT for v in shown.values],
        customdata=[[s[k], z[k]] for k in shown.index],
        hovertemplate="%{y}<br>Value %{customdata[0]:,.2f}<br>%{customdata[1]:,.1f}σ from benign mean<extra></extra>",
    ))
    fig.update_layout(title=dict(text="Features furthest from the benign baseline (standard deviations, capped at ±100)", font=dict(size=14)))
    return style_fig(fig, 360)


def tab_latest():
    e = st.session_state.latest
    if e is None:
        st.info("No flow analyzed yet. Choose a traffic source in the sidebar and click **Analyze traffic**.")
        return
    att = is_attack(e)
    sev = sev_of(e)
    title = e.get("attack_type") or ("No attack classified" if not att else "Attack type unavailable")
    st.markdown(
        f"""
        <div class="banner {'attack' if att else 'normal'}">
          {chip('Attack detected' if att else 'Normal traffic', ATTACK_COLOR if att else OK_COLOR, att)}
          <div class="banner-title">{esc(title)}</div>
          <div class="muted"><span class="mono">{esc(e['event_id'])}</span> · Severity {esc(sev if att else '—')} · {esc(e['dataset'])}, row {esc(e['row_number'])} · {esc(e['analyzed_at'])} · {esc(e.get('latency_ms', '—'))} ms</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    truth = e.get("ground_truth")
    if truth:
        truth_attack = truth.strip().lower() not in ("benign", "normal")
        if truth_attack == att:
            st.success(f"The dataset label is **{truth}**, so the model got this one right.")
        else:
            st.warning(f"The dataset label is **{truth}**, so the model {'raised a false alarm' if att else 'missed an attack'} here.")

    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("**What happened**")
            st.write(e.get("description", "No description returned."))
    with b:
        with st.container(border=True):
            st.markdown("**Recommended action**")
            st.write(e.get("recommended_action", "No recommended action returned."))

    if att:
        m = e.get("mitre") or {}
        x, y, z = st.columns(3)
        x.metric("Technique ID", m.get("technique_id", "—"))
        y.metric("Technique", m.get("technique_name", "—"))
        z.metric("Tactic", m.get("tactic", "—"))
        st.markdown("#### AI analyst report")
        with st.container(border=True):
            st.markdown(e.get("llm_analysis") or "No LLM analysis was returned.")
    else:
        st.success("No AI incident report was requested for this normal flow.")

    st.markdown("#### Flow fingerprint")
    st.markdown("<div class='section-note'>Compares this flow with the average benign flow. This shows what looks unusual; it is not the model's own explanation.</div>", unsafe_allow_html=True)
    feats = st.session_state.flow_cache.get(e["event_id"])
    if feats:
        fig = deviation_fig(feats)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("The benign dataset isn't available, so no baseline comparison can be drawn.")
        with st.expander("All raw features"):
            q = st.text_input("Filter features", key="feat_filter", placeholder="e.g. packet, flag, duration")
            fdf = pd.DataFrame({"Feature": list(feats.keys()), "Value": list(feats.values())})
            if q.strip():
                fdf = fdf[fdf["Feature"].str.contains(q.strip(), case=False, na=False)]
            st.dataframe(fdf, use_container_width=True, hide_index=True)


def incident_report(e: dict) -> str:
    m = e.get("mitre") or {}
    inc = st.session_state.incidents.get(e["event_id"], {})
    return f"""# Incident {e['event_id']}: {e.get('attack_type') or 'Unknown attack'}

- **Detected:** {e['analyzed_at']}
- **Severity:** {sev_of(e)}
- **Status:** {inc.get('status', 'Open')}
- **Owner:** {inc.get('owner', 'Unassigned')}
- **Source:** {e['dataset']}, row {e['row_number']}
- **MITRE ATT&CK:** {m.get('technique_id', '—')} {m.get('technique_name', '')} ({m.get('tactic', '—')})

## What happened
{e.get('description', '—')}

## Recommended action
{e.get('recommended_action', '—')}

## AI analyst report
{e.get('llm_analysis') or '—'}

## Analyst notes
{inc.get('notes') or '—'}

---
Generated by AICS from CIC-IDS2017 benchmark data. Validate before acting.
"""


def tab_incidents():
    ss = st.session_state
    attacks = [e for e in ss.history if is_attack(e)]
    if not attacks:
        st.info("Incidents open automatically when the model flags an attack.")
        return
    rows = []
    for e in attacks:
        rec = ss.incidents.setdefault(e["event_id"], {"status": "Open", "owner": "Unassigned", "notes": ""})
        rows.append({
            "Incident": e["event_id"], "Time": e["analyzed_at"],
            "Attack type": e.get("attack_type") or "Unknown", "Severity": sev_of(e),
            "Status": rec["status"], "Owner": rec["owner"], "Notes": rec["notes"],
        })
    df = pd.DataFrame(rows)

    counts = df["Status"].value_counts()
    cols = st.columns(len(STATUSES))
    for col, s in zip(cols, STATUSES):
        col.metric(s, int(counts.get(s, 0)))

    f1, f2 = st.columns([2, 1])
    status_filter = f1.multiselect("Show statuses", STATUSES, default=[s for s in STATUSES if s not in ("Resolved", "False positive")])
    sev_filter = f2.multiselect("Severity", list(SEV_WEIGHT.keys()), default=[])
    view = df[df["Status"].isin(status_filter)] if status_filter else df
    if sev_filter:
        view = view[view["Severity"].isin(sev_filter)]
    if view.empty:
        st.info("No incidents match these filters.")
        return

    st.markdown("<div class='section-note'>Edit status, owner and notes directly in the table. Changes save automatically.</div>", unsafe_allow_html=True)
    key = f"inc_editor_{len(attacks)}_{'-'.join(status_filter)}_{'-'.join(sev_filter)}"
    edited = st.data_editor(
        view, key=key, hide_index=True, use_container_width=True,
        disabled=["Incident", "Time", "Attack type", "Severity"],
        column_config={
            "Status": st.column_config.SelectboxColumn(options=STATUSES, required=True),
            "Owner": st.column_config.SelectboxColumn(options=OWNERS, required=True),
            "Notes": st.column_config.TextColumn(width="large"),
        },
    )
    for _, r in edited.iterrows():
        ss.incidents[r["Incident"]] = {"status": r["Status"], "owner": r["Owner"], "notes": r["Notes"] or ""}

    st.markdown("#### Incident report")
    by_id = {e["event_id"]: e for e in attacks}
    pick = st.selectbox("Choose an incident", list(by_id.keys()),
                        format_func=lambda i: f"{i} · {by_id[i].get('attack_type') or 'Unknown'} · {sev_of(by_id[i])}")
    report = incident_report(by_id[pick])
    with st.expander("Preview report", expanded=False):
        st.markdown(report)
    d1, d2 = st.columns(2)
    d1.download_button("Download report (.md)", report, file_name=f"{pick}.md", mime="text/markdown", use_container_width=True)
    d2.download_button("Download all incidents (.csv)", df.to_csv(index=False), file_name="incidents.csv", mime="text/csv", use_container_width=True)


def tab_mitre():
    attacks = [e for e in st.session_state.history if is_attack(e)]
    if not attacks:
        st.info("Run some traffic with attacks to see which ATT&CK tactics and techniques show up.")
        return
    tactic_hits: dict[str, dict] = {t: {"n": 0, "ids": set()} for t in TACTICS}
    tech: dict[tuple, dict] = {}
    for e in attacks:
        m = e.get("mitre") or {}
        tid, name, tac = m.get("technique_id"), m.get("technique_name"), str(m.get("tactic") or "")
        for t in TACTICS:
            if t.lower() in tac.lower():
                tactic_hits[t]["n"] += 1
                if tid:
                    tactic_hits[t]["ids"].add(tid)
        if tid:
            rec = tech.setdefault((tid, name, tac), {"count": 0, "max": 0})
            rec["count"] += 1
            rec["max"] = max(rec["max"], SEV_WEIGHT.get(sev_of(e), 2))

    st.markdown("#### Tactics observed")
    st.markdown("<div class='section-note'>Highlighted tactics were mapped from attacks in this session.</div>", unsafe_allow_html=True)
    cells = ""
    for t in TACTICS:
        h = tactic_hits[t]
        if h["n"]:
            cells += f'<div class="tactic hit"><div class="n">{esc(t)}</div><div class="c">{h["n"]}</div><div class="ids">{esc(", ".join(sorted(h["ids"])))}</div></div>'
        else:
            cells += f'<div class="tactic"><div class="n">{esc(t)}</div></div>'
    st.markdown(f'<div class="tactics">{cells}</div>', unsafe_allow_html=True)

    st.markdown("#### Techniques")
    inv = {v: k for k, v in SEV_WEIGHT.items()}
    rows = [{
        "Technique": tid, "Name": name or "—", "Tactic": tac or "—", "Alerts": r["count"],
        "Highest severity": inv.get(r["max"], "Unknown"),
        "Reference": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}/",
    } for (tid, name, tac), r in tech.items()]
    st.dataframe(
        pd.DataFrame(rows).sort_values("Alerts", ascending=False),
        use_container_width=True, hide_index=True,
        column_config={"Reference": st.column_config.LinkColumn(display_text="Open")},
    )


def tab_health():
    ss = st.session_state
    st.markdown("#### Model check against dataset labels")
    st.markdown("<div class='section-note'>Each CIC-IDS2017 row carries a ground-truth label, so you can see how the pipeline did on this session's flows. Small samples are noisy; this isn't a benchmark.</div>", unsafe_allow_html=True)
    labeled = [e for e in ss.history if e.get("ground_truth")]
    if not labeled:
        st.info("No labeled flows yet.")
    else:
        tp = fp = tn = fn = 0
        for e in labeled:
            truth = e["ground_truth"].strip().lower() not in ("benign", "normal")
            pred = is_attack(e)
            tp += truth and pred
            fp += (not truth) and pred
            tn += (not truth) and (not pred)
            fn += truth and (not pred)
        n = tp + fp + tn + fn
        c1, c2 = st.columns([1, 1.4])
        with c1:
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{(tp + tn) / n * 100:.1f}%")
            m2.metric("Precision", f"{tp / (tp + fp) * 100:.1f}%" if tp + fp else "—")
            m3, m4 = st.columns(2)
            m3.metric("Recall", f"{tp / (tp + fn) * 100:.1f}%" if tp + fn else "—")
            m4.metric("False alarm rate", f"{fp / (fp + tn) * 100:.1f}%" if fp + tn else "—")
            st.caption(f"Based on {n} labeled flows.")
        with c2:
            z = [[tn, fp], [fn, tp]]
            fig = go.Figure(go.Heatmap(
                z=z, x=["Predicted normal", "Predicted attack"], y=["Actually normal", "Actually attack"],
                text=z, texttemplate="%{text}", colorscale=[[0, "#111a2b"], [1, "#2f5fb8"]], showscale=False,
                textfont=dict(size=20),
            ))
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(title=dict(text="Confusion matrix", font=dict(size=14)))
            st.plotly_chart(style_fig(fig, 280), use_container_width=True)

    st.markdown("#### Response time")
    lat = [(e["event_id"], e["latency_ms"], is_attack(e)) for e in reversed(ss.history) if e.get("latency_ms") is not None]
    if not lat:
        st.info("No timing data yet.")
        return
    s = pd.Series([x[1] for x in lat])
    a, b, c = st.columns(3)
    a.metric("Median", f"{s.median() / 1000:.2f} s")
    b.metric("95th percentile", f"{s.quantile(.95) / 1000:.2f} s")
    atk = [x[1] for x in lat if x[2]]
    nrm = [x[1] for x in lat if not x[2]]
    c.metric("Attack vs normal", f"{(sum(atk) / len(atk) / 1000 if atk else 0):.1f}s / {(sum(nrm) / len(nrm) / 1000 if nrm else 0):.1f}s",
             help="Attack flows also wait for the local LLM report.")
    fig = go.Figure(go.Bar(
        x=[x[0] for x in lat], y=[x[1] for x in lat],
        marker_color=[ATTACK_COLOR if x[2] else OK_COLOR for x in lat],
        hovertemplate="%{x}<br>%{y} ms<extra></extra>",
    ))
    fig.update_layout(title=dict(text="API round trip per flow (ms)", font=dict(size=14)), xaxis=dict(showticklabels=False))
    st.plotly_chart(style_fig(fig, 260), use_container_width=True)


def session_context() -> str:
    ss = st.session_state
    keep = ("event_id", "analyzed_at", "prediction", "attack_type", "severity")
    latest = None
    if ss.latest:
        latest = {k: ss.latest.get(k) for k in keep + ("description", "recommended_action", "mitre")}
    recent = [{**{k: e.get(k) for k in keep}, "mitre": (e.get("mitre") or {}).get("technique_id")} for e in ss.history[:15]]
    return json.dumps({
        "totals": {"flows": ss.total_flows, "attacks": ss.attack_count, "normal": ss.normal_count},
        "latest_event": latest, "recent_events": recent,
        "incident_status": {k: v["status"] for k, v in list(ss.incidents.items())[:30]},
    }, default=str)


def ask_copilot(question: str) -> str:
    system = (
        "You are AICS Copilot, a concise SOC analyst assistant. Answer using only the session data below. "
        "If the data doesn't contain the answer, say so. Never invent IP addresses, hosts or CVEs. "
        "Data comes from the CIC-IDS2017 benchmark, not live traffic.\n\nSESSION DATA:\n" + session_context()
    )
    msgs = [{"role": "system", "content": system}] + st.session_state.chat[-8:] + [{"role": "user", "content": question}]
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={"model": OLLAMA_MODEL, "messages": msgs, "stream": False}, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.ConnectionError:
        return f"I can't reach Ollama at {OLLAMA_URL}. Start it with `ollama serve`, then ask again."
    except (requests.RequestException, KeyError, ValueError) as exc:
        return f"The local model returned an error: {exc}. Check that `{OLLAMA_MODEL}` is pulled (`ollama pull {OLLAMA_MODEL}`)."


def tab_copilot():
    ss = st.session_state
    st.markdown("#### Ask the copilot about this session")
    st.markdown("<div class='section-note'>Answers come from your local model and only see this session's alerts.</div>", unsafe_allow_html=True)
    quick = {
        "Summarize the session": "Summarize this session for a shift handover in under 120 words.",
        "Explain latest alert": "Explain the latest alert to a non-technical manager in 3 sentences.",
        "What should I do first?": "Which open alert should I triage first, and why? Give a short containment checklist.",
    }
    cols = st.columns(len(quick))
    for col, (label, prompt) in zip(cols, quick.items()):
        if col.button(label, use_container_width=True):
            ss.pending_q = prompt
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    typed = st.chat_input("Ask about the alerts, techniques or next steps…")
    q = typed or ss.pending_q
    if q:
        ss.pending_q = None
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = ask_copilot(q)
            st.markdown(answer)
        ss.chat.append({"role": "user", "content": q})
        ss.chat.append({"role": "assistant", "content": answer})
    if ss.chat and st.button("Clear conversation"):
        ss.chat = []
        st.rerun()


def tab_log():
    ss = st.session_state
    if not ss.history:
        st.info("Events will be listed here as you analyze flows.")
        return
    df = events_df()
    f1, f2, f3 = st.columns([2, 1.2, 1])
    q = f1.text_input("Search", placeholder="Event ID, attack type, technique…")
    sevs = f2.multiselect("Severity", ["Critical", "High", "Medium", "Low", "Unknown"])
    only_attacks = f3.toggle("Attacks only", value=False)
    if q.strip():
        mask = df.astype(str).apply(lambda c: c.str.contains(q.strip(), case=False, na=False)).any(axis=1)
        df = df[mask]
    if sevs:
        df = df[df["Severity"].isin(sevs)]
    if only_attacks:
        df = df[df["Prediction"].astype(str).str.lower() == "attack"]
    st.dataframe(df, use_container_width=True, hide_index=True)

    d1, d2 = st.columns(2)
    d1.download_button("Download log (.csv)", df.to_csv(index=False), file_name="aics_events.csv", mime="text/csv", use_container_width=True)
    d2.download_button("Download raw events (.json)", json.dumps(ss.history, default=str, indent=2), file_name="aics_events.json", mime="application/json", use_container_width=True)

    by_id = {e["event_id"]: e for e in ss.history}
    ids = [i for i in df["Event"] if i in by_id]
    if ids:
        pick = st.selectbox("Inspect event", ids, format_func=lambda i: f"{i} · {by_id[i].get('prediction')} · {by_id[i].get('attack_type') or 'Normal'}")
        st.json(by_id[pick])


t_over, t_latest, t_inc, t_mitre, t_health, t_ai, t_log = st.tabs(
    ["Overview", "Latest alert", "Incidents", "MITRE ATT&CK", "Model health", "AI copilot", "Event log"]
)
with t_over:
    tab_overview()
with t_latest:
    tab_latest()
with t_inc:
    tab_incidents()
with t_mitre:
    tab_mitre()
with t_health:
    tab_health()
with t_ai:
    tab_copilot()
with t_log:
    tab_log()

with st.expander("About this prototype"):
    st.write(
        "This dashboard is a benchmark-data prototype using CIC-IDS2017 samples. "
        "It isn't connected to live production network traffic. Model false positives "
        "and missed detections are possible; validate alerts before taking action."
    )
    st.code(
        "CIC-IDS2017 → Flask API → StandardScaler → Isolation Forest → "
        "Random Forest (if flagged) → MITRE ATT&CK mapping → Local Llama 3.2 → Dashboard",
        language="text",
    )
