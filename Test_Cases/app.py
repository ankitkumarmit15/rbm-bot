"""
QA Test Cases Agent — Streamlit UI
Light theme, production-grade, multi-agent LangGraph pipeline.
"""

import datetime
import hashlib
import re
from collections import Counter
import streamlit as st

from modules.extraction import extract_text_from_file
from modules.pipeline import execute_pipeline_with_agent, validate_test_case_rows
from modules.excel_builder import build_excel_workbook, build_full_strategy_workbook
from modules.agent import generate_strategy_content
from modules.utils import file_hash, render_test_case_preview
from modules.chunking import smart_split, split_excel_rows
from modules.config import (
    CHROMA_AVAILABLE, SUPPORTED_MODELS,
    OLLAMA_AVAILABLE, list_ollama_models,
    GROQ_AVAILABLE, GROQ_MODELS, GROQ_API_KEY_SET,
)
from modules.rag import (
    add_to_kb, get_kb_docs, delete_from_kb, get_kb_doc_names, query_kb,
)
from modules.gherkin_builder import build_gherkin
import modules.timing as timing

try:
    from modules.warmup import warmup_gemini
except Exception:
    def warmup_gemini(model_name=None):  # noqa: E306
        del model_name
        return False


# ─────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QA Test Cases Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Light theme CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════
   COLOUR PALETTE
   bg:      #f8fafc   (slate-50  — page background)
   surface: #ffffff   (white     — cards, sidebar)
   border:  #e2e8f0   (slate-200 — all dividers)
   text:    #1e293b   (slate-800 — primary text)
   muted:   #64748b   (slate-500 — secondary text)
   blue:    #2563eb   (blue-600  — primary accent)
   blue-lt: #eff6ff   (blue-50   — tinted surfaces)
   green:   #059669   (emerald   — download / success)
═══════════════════════════════════════════════════════════ */

/* ── Page & sidebar ──────────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: #f8fafc !important;
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 12px rgba(37,99,235,.06) !important;
}
/* Sidebar inner content — keep text dark and readable */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] .stMarkdown p {
    color: #1e293b !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #64748b !important;
}

/* ── Typography ──────────────────────────────────────────── */
h1, h2, h3, h4 { color: #1e293b !important; }
p, li, span     { color: #1e293b; }

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.30) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(37,99,235,.40) !important;
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #2563eb !important;
    border: 1.5px solid #2563eb !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #eff6ff !important;
    transform: translateY(-1px) !important;
}
/* Plain / small buttons (KB delete etc.) */
.stButton > button[kind="tertiary"] {
    background: #f8fafc !important;
    color: #64748b !important;
    border: 1px solid #e2e8f0 !important;
    font-size: 12px !important;
}

/* ── Download button ─────────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(5,150,105,.25) !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(5,150,105,.35) !important;
}

/* ── File uploader — force light everywhere ──────────────── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] > div {
    background: #eff6ff !important;
    border: 2px dashed #93c5fd !important;
    border-radius: 12px !important;
    color: #1e293b !important;
}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploader"]:hover section {
    background: #dbeafe !important;
    border-color: #2563eb !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #2563eb !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] svg {
    fill: #2563eb !important;
}
/* Uploaded file name chip */
[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* ── Inputs & selects ────────────────────────────────────── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextInput > div > div {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #1e293b !important;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 8px rgba(37,99,235,.06) !important;
    margin: 6px 0 !important;
    overflow: hidden !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 16px rgba(37,99,235,.10) !important;
}
/* Expander header text */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* ── Alerts — consistent blue-tinted style ───────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
}
/* Info */
[data-testid="stAlert"][data-type="info"],
div.stAlert > div[data-testid="stMarkdownContainer"] {
    color: #1e293b !important;
}
/* Warning */
[data-testid="stAlert"][data-type="warning"] {
    background: #fffbeb !important;
    border-color: #fde68a !important;
}
/* Success */
[data-testid="stAlert"][data-type="success"] {
    background: #f0fdf4 !important;
    border-color: #bbf7d0 !important;
}
/* Error */
[data-testid="stAlert"][data-type="error"] {
    background: #fef2f2 !important;
    border-color: #fecaca !important;
}

/* ── Code block (execution log) ──────────────────────────── */
[data-testid="stCode"] > div,
pre, code {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #1e293b !important;
    font-size: 12.5px !important;
    line-height: 1.65 !important;
}

/* ── Chat messages ───────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.06) !important;
}

/* ── Data editor ─────────────────────────────────────────── */
[data-testid="stDataEditor"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 10px rgba(37,99,235,.06) !important;
}

/* ── Checkbox ────────────────────────────────────────────── */
.stCheckbox label span { color: #1e293b !important; font-weight: 500 !important; }

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: #e2e8f0 !important; margin: 12px 0 !important; }

/* ── Spinner ─────────────────────────────────────────────── */
[data-testid="stSpinner"] p { color: #2563eb !important; }

/* ── Hide Streamlit default top header bar ──────────────── */
[data-testid="stHeader"],
[data-testid="stDecoration"],
#stDecoration { display: none !important; }
.stDeployButton { display: none !important; }
.main .block-container { padding-top: 1.5rem !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }

/* ── Sidebar branding ───────────────────────────────────── */
.qa-brand {
    display: flex; align-items: center; gap: 11px;
    padding: 8px 0 14px;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 4px;
}
.qa-brand-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 19px; flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(37,99,235,.35);
}
.qa-brand-name { font-size: 14px; font-weight: 800; color: #1e293b; line-height: 1.2; }
.qa-brand-sub  { font-size: 10px; color: #94a3b8; text-transform: uppercase;
                 letter-spacing: .7px; margin-top: 2px; }

/* ── Sidebar section header (enhanced) ─────────────────── */
.qa-sb-hdr {
    display: flex; align-items: center; gap: 7px;
    padding: 7px 11px;
    background: #f1f5f9;
    border-radius: 8px;
    border-left: 3px solid #2563eb;
    margin: 14px 0 8px;
}
.qa-sb-hdr-icon { font-size: 13px; }
.qa-sb-hdr-text {
    font-size: 11px; font-weight: 700; color: #1e293b;
    text-transform: uppercase; letter-spacing: .5px;
}

/* ── Provider selector pills (radio row) ───────────────── */
/* Override Streamlit radio to look like pill tabs */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 9px !important;
    padding: 7px 12px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    color: #475569 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
    color: #1d4ed8 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div label,
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #eff6ff !important;
    border-color: #2563eb !important;
    color: #1d4ed8 !important;
}

/* ── Status chips ───────────────────────────────────────── */
.qa-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 600; border: 1px solid;
}
.qa-chip-ok   { background: #f0fdf4; color: #166534; border-color: #bbf7d0; }
.qa-chip-warn { background: #fef3c7; color: #92400e; border-color: #fde68a; }
.qa-chip-err  { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.qa-chip-info { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; }

/* ── Model info card ────────────────────────────────────── */
.qa-model-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 13px;
    margin: 8px 0;
    box-shadow: 0 1px 4px rgba(37,99,235,.05);
}
.qa-model-name { font-size: 13px; font-weight: 700; color: #1e293b; }
.qa-model-meta { font-size: 11px; color: #64748b; margin-top: 3px; }

/* ═══════════════════════════════════════════════════════════
   ANIMATIONS
═══════════════════════════════════════════════════════════ */
@keyframes qa-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
@keyframes qa-slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes qa-fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ═══════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════════════════════════════ */

/* Agent pipeline badges */
.qa-pipeline {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 14px 0 4px;
    animation: qa-slideUp 0.5s ease;
}
.qa-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 6px 14px;
    border-radius: 22px;
    font-size: 13px;
    font-weight: 600;
    white-space: nowrap;
    border: 1.5px solid;
    transition: all 0.25s ease;
}
.qa-badge-idle   { background: rgba(255,255,255,.18); color: rgba(255,255,255,.85);
                   border-color: rgba(255,255,255,.30); }
.qa-badge-active { background: #fef3c7; color: #92400e; border-color: #fcd34d;
                   animation: qa-pulse 1.1s infinite; }
.qa-badge-done   { background: #dcfce7; color: #166534; border-color: #86efac; }
.qa-arrow        { color: rgba(255,255,255,.40); font-size: 18px; font-weight: 300; }

/* Metric cards */
.qa-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 18px 0;
    animation: qa-slideUp 0.5s ease;
}
.qa-metric {
    background: #ffffff;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(37,99,235,.07);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.qa-metric:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 18px rgba(37,99,235,.12);
}
.qa-metric-val   { font-size: 2rem; font-weight: 800; color: #2563eb; line-height: 1.15; }
.qa-metric-label { font-size: 11px; color: #94a3b8; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 0.6px; margin-top: 4px; }

/* KB doc rows */
.qa-kb-doc {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 7px 11px;
    font-size: 12.5px;
    color: #374151;
    margin: 3px 0;
    animation: qa-fadeIn 0.3s ease;
}
.qa-kb-small { font-size: 11px; color: #94a3b8; margin-top: 2px; }

/* Section header banner */
.qa-section {
    background: #eff6ff;
    border-radius: 12px;
    padding: 13px 18px;
    margin: 14px 0 10px;
    border-left: 4px solid #2563eb;
    animation: qa-fadeIn 0.4s ease;
}
.qa-section h3 { margin: 0; color: #1e3a8a; font-size: 15px; font-weight: 700; }
.qa-section p  { margin: 4px 0 0; color: #475569; font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
_STATE_DEFAULTS = {
    "selected_model": None,
    "warmed_model": None,
    "warmup_success": False,
    "workbook_bytes": None,
    "strategy_bytes": None,        # Full 7-tab strategy workbook
    "output_format": "📋 Test Cases Only",
    "extracted_data": None,
    "preview_rows": [],
    "last_file_hash": None,
    "spec_content_hash": None,
    "pipeline_meta": {},
    "gherkin_bytes": None,
    "run_history": [],
    "processed_kb_file_ids": set(),
}


# ─────────────────────────────────────────────────────────────
# UI helper — horizontal bar chart (no extra deps)
# ─────────────────────────────────────────────────────────────

def _bars_html(counts: dict, colors: dict, default_color: str = "#2563eb") -> str:
    """Render a horizontal bar chart as HTML with count + % labels."""
    total = max(sum(counts.values()), 1)
    rows = ""
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = round(cnt / total * 100)
        color = colors.get(label, default_color)
        rows += (
            "<div style='margin:4px 0;display:flex;align-items:center;gap:6px;'>"
            "<div style='width:100px;font-size:11px;color:#475569;text-align:right;"
            "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>" + label + "</div>"
            "<div style='flex:1;background:#f1f5f9;border-radius:4px;height:16px;overflow:hidden;'>"
            "<div style='background:" + color + ";width:" + str(pct) + "%;height:100%;border-radius:4px;'></div>"
            "</div>"
            "<div style='width:50px;font-size:11px;font-weight:600;color:#1e293b;white-space:nowrap;'>"
            + str(cnt) + " <span style='color:#94a3b8;font-weight:400;'>(" + str(pct) + "%)</span></div>"
            "</div>"
        )
    return "<div style='padding:4px 0;'>" + rows + "</div>"
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #7c3aed 100%);
    border-radius: 20px;
    padding: 30px 36px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(37,99,235,.22);
">
  <h1 style="color:#fff !important; font-size:2rem; font-weight:800; margin:0;
             -webkit-text-fill-color:#fff !important;">
      ⚡ QA Test Cases Agent
  </h1>
  <p style="color:rgba(255,255,255,.82); margin:8px 0 18px; font-size:1rem;">
      Intelligent multi-agent generation powered by Gemini · Groq · Ollama &amp; LangGraph
  </p>
  <div class="qa-pipeline">
    <span class="qa-badge qa-badge-idle">📄 Document Parser</span>
    <span class="qa-arrow">→</span>
    <span class="qa-badge qa-badge-idle">🤖 Test Generator</span>
    <span class="qa-arrow">→</span>
    <span class="qa-badge qa-badge-idle">🔍 Reviewer</span>
    <span class="qa-arrow" style="color:rgba(255,255,255,.35)">↻</span>
    <span class="qa-badge qa-badge-idle">⚙️ Refiner</span>
    <span class="qa-arrow">→</span>
    <span class="qa-badge qa-badge-idle">✅ Finalizer</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Sidebar branding ─────────────────────────────────────
    st.markdown(
        "<div class='qa-brand'>"
        "<div class='qa-brand-icon'>⚡</div>"
        "<div>"
        "<div class='qa-brand-name'>QA Test Agent</div>"
        "<div class='qa-brand-sub'>Multi-Agent · LangGraph</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    def _sidebar_section(icon: str, title: str):
        st.markdown(
            f"<div class='qa-sb-hdr'>"
            f"<span class='qa-sb-hdr-icon'>{icon}</span>"
            f"<span class='qa-sb-hdr-text'>{title}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Model selection ──────────────────────────────────────
    _sidebar_section("🤖", "AI Model")
    # ── Level 1: Provider ────────────────────────────────────
    _ollama_models_list = []
    if OLLAMA_AVAILABLE:
        try:
            _ollama_models_list = list_ollama_models()
        except Exception:
            _ollama_models_list = []

    _PROVIDER_DEFS = [
        ("🔵  Google Gemini", "gemini"),
        ("⚡  Groq LPU",      "groq"),
        ("🦙  Ollama Local",  "ollama"),
    ]
    _prov_labels  = [p[0] for p in _PROVIDER_DEFS]
    _prov_key_map = {p[0]: p[1] for p in _PROVIDER_DEFS}

    _prov_sel = st.radio(
        "Provider",
        _prov_labels,
        key="_ui_provider",
        label_visibility="collapsed",
    )
    _prov_key = _prov_key_map[_prov_sel]

    st.markdown(
        "<div style='height:1px;background:#e2e8f0;margin:8px 0 10px;'></div>",
        unsafe_allow_html=True,
    )

    # ── Level 2: Model ───────────────────────────────────────
    _GROQ_MODEL_INFO = {
        "llama-3.3-70b-versatile": "Best quality · 128k context",
        "llama-3.1-8b-instant":    "Fastest inference · 128k context",
        "mixtral-8x7b-32768":      "Long context · 32k tokens",
        "gemma2-9b-it":            "Google Gemma · 8k context",
    }

    if _prov_key == "gemini":
        _gem_model = st.selectbox(
            "Gemini model", SUPPORTED_MODELS,
            key="_ui_gem_model", label_visibility="collapsed",
        )
        _custom = st.text_input(
            "Custom model ID",
            value="",
            placeholder="e.g. gemini-2.0-flash-exp",
            key="_ui_custom_model",
            help="Overrides the dropdown above",
        )
        st.session_state.selected_model = _custom.strip() if _custom.strip() else _gem_model

    elif _prov_key == "groq":
        if not GROQ_AVAILABLE:
            st.markdown(
                "<div class='qa-chip qa-chip-err' style='margin:4px 0;'>"
                "❌ langchain-groq not installed</div>",
                unsafe_allow_html=True,
            )
            st.caption("Fix: `pip install langchain-groq` then restart")
        elif not GROQ_API_KEY_SET:
            st.markdown(
                "<div class='qa-chip qa-chip-warn' style='margin:4px 0;'>"
                "⚠ GROQ_API_KEY not set</div>",
                unsafe_allow_html=True,
            )
            st.caption("Add `GROQ_API_KEY=gsk_...` to your `.env` file")
            st.caption("Free key at console.groq.com (2 min signup)")
        else:
            _groq_model = st.selectbox(
                "Groq model", GROQ_MODELS,
                key="_ui_groq_model", label_visibility="collapsed",
            )
            st.session_state.selected_model = f"groq:{_groq_model}"
            _ginfo = _GROQ_MODEL_INFO.get(_groq_model, "")
            if _ginfo:
                st.markdown(
                    f"<div class='qa-chip qa-chip-info' style='margin:4px 0;'>"
                    f"ℹ {_ginfo}</div>",
                    unsafe_allow_html=True,
                )

    elif _prov_key == "ollama":
        if not OLLAMA_AVAILABLE:
            st.markdown(
                "<div class='qa-chip qa-chip-warn' style='margin:4px 0;'>"
                "⚠ Ollama not detected</div>",
                unsafe_allow_html=True,
            )
            st.caption("Install from ollama.com")
        elif not _ollama_models_list:
            st.markdown(
                "<div class='qa-chip qa-chip-warn' style='margin:4px 0;'>"
                "⚠ No local models found</div>",
                unsafe_allow_html=True,
            )
            st.caption("Run: `ollama pull llama3`")
        else:
            _oll_model = st.selectbox(
                "Ollama model", _ollama_models_list,
                key="_ui_ollama_model", label_visibility="collapsed",
            )
            st.session_state.selected_model = f"ollama:{_oll_model}"
            st.markdown(
                "<div class='qa-chip qa-chip-info' style='margin:4px 0;'>"
                "🦙 Local · No API key needed</div>",
                unsafe_allow_html=True,
            )

    # ── Warm-up / connection test ─────────────────────────────
    if (
        st.session_state.selected_model is not None
        and st.session_state.warmed_model != st.session_state.selected_model
    ):
        with st.spinner(f"🔥 Connecting…"):
            try:
                st.session_state.warmup_success = warmup_gemini(st.session_state.selected_model)
            except Exception:
                st.session_state.warmup_success = False
        st.session_state.warmed_model = st.session_state.selected_model

    _sel_m = st.session_state.selected_model or ""
    if not _sel_m:
        st.info("Select a provider and model above.")
    elif not st.session_state.warmup_success:
        if _sel_m.startswith("groq:"):
            st.error("Connection failed — check GROQ_API_KEY")
        elif _sel_m.startswith("ollama:"):
            st.error("Connection failed — is Ollama running?")
        else:
            st.error("Connection failed — check GOOGLE_API_KEY")
    else:
        _disp_m = _sel_m
        if _disp_m.startswith("groq:"):
            _disp_m = "⚡ " + _disp_m.replace("groq:", "")
        elif _disp_m.startswith("ollama:"):
            _disp_m = "🦙 " + _disp_m.replace("ollama:", "")
        else:
            _disp_m = "🔵 " + _disp_m
        st.markdown(
            f"<div class='qa-chip qa-chip-ok' style='margin-top:8px;font-size:12px;"
            f"padding:6px 12px;'>✅ {_disp_m} ready</div>",
            unsafe_allow_html=True,
        )

    # ── Spec file upload (multi-file) ────────────────────────
    _sidebar_section("📂", "Specification")
    uploaded_files = st.file_uploader(
        "Drop PDF, DOCX, DOC, or TXT — select multiple files to combine",
        type=["pdf", "docx", "doc", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        _file_texts: list[str] = []
        _file_names: list[str] = []
        with st.spinner(f"Extracting text from {len(uploaded_files)} file(s)..."):
            for _uf in uploaded_files:
                _ft = extract_text_from_file(_uf)
                if _ft.strip():
                    _file_texts.append(f"=== Document: {_uf.name} ===\n\n{_ft}")
                    _file_names.append(_uf.name)
                else:
                    st.warning(f"No text extracted from **{_uf.name}** — skipped.")

        if not _file_texts:
            st.error("No text could be extracted from any file.")
            for _k in ["raw_text", "extracted_data", "preview_rows", "workbook_bytes", "last_file_hash"]:
                st.session_state[_k] = None if _k != "preview_rows" else []
        else:
            raw_text = "\n\n".join(_file_texts)
            _label = (
                _file_names[0] if len(_file_names) == 1
                else f"{_file_names[0]} + {len(_file_names) - 1} more"
            )
            _new_spec_hash = file_hash(raw_text)
            # Only invalidate pipeline results when the file content genuinely changed
            if _new_spec_hash != st.session_state.get("spec_content_hash"):
                st.session_state.extracted_data = None
                st.session_state.preview_rows = []
                st.session_state.workbook_bytes = None
                st.session_state.last_file_hash = None
                st.session_state.pipeline_meta = {}
                st.session_state.spec_content_hash = _new_spec_hash
            st.session_state.raw_text = raw_text
            st.session_state.file_name = _label
            for _fn in _file_names:
                st.success(f"**{_fn}**")
            _char_count = len(raw_text)
            _word_count = len(raw_text.split())
            _page_est   = max(1, round(_char_count / 2000))
            _info = (
                f"{_char_count:,} chars · ~{_word_count:,} words · ~{_page_est} page(s)"
            )
            if len(_file_names) > 1:
                _info += f" from {len(_file_names)} documents"
            st.info(_info)
            with st.expander("Preview extracted text"):
                st.text_area("", value=raw_text[:3000], height=200, label_visibility="collapsed")

    st.markdown("---")

    # ── Knowledge Base ───────────────────────────────────────
    with st.expander("📚 Knowledge Base (RAG)", expanded=False):

        if not CHROMA_AVAILABLE:
            st.warning("ChromaDB not installed — knowledge base unavailable.")
        else:
            # ── Section 1: Requirements ──────────────────────
            st.markdown(
                "<div style='background:#eff6ff;border-left:3px solid #2563eb;"
                "border-radius:6px;padding:7px 10px;margin-bottom:8px;'>"
                "<span style='color:#1e3a8a;font-weight:700;font-size:13px;'>📋 Requirements</span>"
                "<div style='color:#475569;font-size:11px;margin-top:2px;'>"
                "Spec docs, PRDs, user stories — used for domain accuracy</div></div>",
                unsafe_allow_html=True,
            )
            req_file = st.file_uploader(
                "Upload requirements doc",
                type=["pdf", "docx", "doc", "txt"],
                key="kb_req_upload",
                label_visibility="collapsed",
            )
            if req_file:
                _req_fid = getattr(req_file, "file_id", req_file.name)
                if _req_fid not in st.session_state.processed_kb_file_ids:
                    with st.spinner(f"Storing {req_file.name}..."):
                        try:
                            req_text = extract_text_from_file(req_file)
                            if req_text.strip():
                                req_chunks = smart_split(req_text, chunk_size=2000, chunk_overlap=100)
                                req_doc_id = hashlib.md5(req_text.encode()).hexdigest()
                                n = add_to_kb(req_chunks, req_file.name, req_doc_id, "requirements")
                                st.session_state.processed_kb_file_ids = (
                                    st.session_state.processed_kb_file_ids | {_req_fid}
                                )
                                st.success(f"✅ {n} chunks stored from **{req_file.name}**")
                            else:
                                st.warning("Could not extract text from this file.")
                        except Exception as _e:
                            st.error(f"KB error: {_e}")

            try:
                req_docs = get_kb_docs("requirements")
                if req_docs:
                    for doc in req_docs:
                        _c1, _c2 = st.columns([5, 1])
                        _c1.markdown(
                            f"<div class='qa-kb-doc'>📄 {doc['name'][:28]}"
                            f"<div class='qa-kb-small'>{doc['chunks']} chunks</div></div>",
                            unsafe_allow_html=True,
                        )
                        if _c2.button("✕", key=f"del_req_{doc['id']}", help="Remove"):
                            delete_from_kb(doc["id"], "requirements")
                            st.rerun()
                else:
                    st.caption("No requirements stored yet.")
            except Exception:
                st.caption("Requirements KB unavailable.")

            st.markdown(
                "<hr style='margin:10px 0;border-color:#e2e8f0;'>",
                unsafe_allow_html=True,
            )

            # ── Section 2: Previous Test Cases ───────────────
            st.markdown(
                "<div style='background:#f0fdf4;border-left:3px solid #059669;"
                "border-radius:6px;padding:7px 10px;margin-bottom:8px;'>"
                "<span style='color:#064e3b;font-weight:700;font-size:13px;'>🧪 Previous Test Cases</span>"
                "<div style='color:#475569;font-size:11px;margin-top:2px;'>"
                "Excel/CSV with past test cases — used as output format &amp; style reference</div></div>",
                unsafe_allow_html=True,
            )
            tc_file = st.file_uploader(
                "Upload test cases file",
                type=["xlsx", "xls", "pdf", "docx", "doc", "txt"],
                key="kb_tc_upload",
                label_visibility="collapsed",
            )

            # Link dropdown — show requirement docs already in KB
            _req_names = get_kb_doc_names("requirements")
            _no_link = "— Auto-detect (semantic match) —"
            _link_options = [_no_link] + _req_names
            _selected_link = st.selectbox(
                "Link to requirement doc",
                _link_options,
                key="kb_tc_req_link",
                help=(
                    "Link these test cases to the requirement doc they came from. "
                    "At generation time the agent will prefer these examples when "
                    "processing a similar requirement. Auto-detect uses semantic "
                    "similarity if no link is set."
                ),
            )
            _source_req_doc = _selected_link if _selected_link != _no_link else None

            if tc_file:
                _tc_fid = getattr(tc_file, "file_id", tc_file.name)
                if _tc_fid not in st.session_state.processed_kb_file_ids:
                    with st.spinner(f"Storing {tc_file.name}..."):
                        try:
                            tc_text = extract_text_from_file(tc_file)
                            if tc_text.strip():
                                _ext = tc_file.name.split(".")[-1].lower()
                                if _ext in ("xlsx", "xls"):
                                    tc_chunks = split_excel_rows(tc_text)
                                else:
                                    tc_chunks = smart_split(tc_text, chunk_size=2000, chunk_overlap=100)
                                tc_doc_id = hashlib.md5(tc_text.encode()).hexdigest()
                                n = add_to_kb(tc_chunks, tc_file.name, tc_doc_id, "testcases",
                                              source_req_doc=_source_req_doc)
                                st.session_state.processed_kb_file_ids = (
                                    st.session_state.processed_kb_file_ids | {_tc_fid}
                                )
                                st.success(f"✅ {n} chunks stored from **{tc_file.name}**")
                            else:
                                st.warning("Could not extract text from this file.")
                        except Exception as _e:
                            st.error(f"KB error: {_e}")

            try:
                tc_docs = get_kb_docs("testcases")
                if tc_docs:
                    for doc in tc_docs:
                        _c1, _c2 = st.columns([5, 1])
                        _link_label = doc.get("source_req_doc")
                        _link_html = (
                            f"<div class='qa-kb-small' style='color:#059669;'>"
                            f"🔗 linked → {_link_label[:30]}</div>"
                            if _link_label else
                            "<div class='qa-kb-small' style='color:#94a3b8;'>"
                            "🔍 auto-detect (no link set)</div>"
                        )
                        _c1.markdown(
                            f"<div class='qa-kb-doc' style='border-color:#bbf7d0;'>"
                            f"🧪 {doc['name'][:28]}"
                            f"<div class='qa-kb-small'>{doc['chunks']} chunks</div>"
                            f"{_link_html}</div>",
                            unsafe_allow_html=True,
                        )
                        if _c2.button("✕", key=f"del_tc_{doc['id']}", help="Remove"):
                            delete_from_kb(doc["id"], "testcases")
                            st.rerun()
                else:
                    st.caption("No test cases stored yet.")
            except Exception:
                st.caption("Test Cases KB unavailable.")

            # ── KB Search Preview ────────────────────────────
            if CHROMA_AVAILABLE:
                st.markdown(
                    "<hr style='margin:10px 0;border-color:#e2e8f0;'>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<span style='color:#1e293b;font-weight:700;font-size:13px;'>🔎 Test KB Retrieval</span>"
                    "<div style='color:#475569;font-size:11px;margin-bottom:6px;'>"
                    "Type a query to preview what the agent will retrieve</div>",
                    unsafe_allow_html=True,
                )
                _kbq = st.text_input(
                    "KB query", placeholder="e.g. MSISDN change fee validation",
                    key="kb_search_query", label_visibility="collapsed",
                )
                if st.button("Search KB", key="kb_search_btn"):
                    if _kbq.strip():
                        with st.spinner("Searching..."):
                            _kr = query_kb(_kbq, "requirements", n_results=3)
                            _kt = query_kb(_kbq, "testcases",    n_results=3)
                        if _kr:
                            st.markdown("**Requirements KB:**")
                            for _chunk in _kr:
                                st.markdown(
                                    f"<div class='qa-kb-doc'>{_chunk[:200]}…</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No requirements hits.")
                        if _kt:
                            st.markdown("**Test Cases KB:**")
                            for _chunk in _kt:
                                st.markdown(
                                    f"<div class='qa-kb-doc' style='border-color:#bbf7d0;'>"
                                    f"{_chunk[:200]}…</div>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.caption("No test-case hits.")
                    else:
                        st.warning("Enter a query above.")

    # ── Output Format ────────────────────────────────────────
    _sidebar_section("📁", "Output Format")
    _fmt_choice = st.radio(
        "Output format",
        ["📋 Test Cases Only", "📁 Full Test Strategy (7 tabs)"],
        key="output_format",
        label_visibility="collapsed",
    )
    if st.session_state.get("output_format", "") == "📁 Full Test Strategy (7 tabs)":
        st.markdown(
            "<div class='qa-chip qa-chip-info' style='margin-top:6px;font-size:10px;"
            "padding:4px 9px;'>7 tabs: Cover · Overview · Timelines · Scenarios · "
            "Data · Test Cases · Samples · (~20s extra)</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='qa-chip qa-chip-ok' style='margin-top:6px;font-size:10px;"
            "padding:4px 9px;'>Single sheet · Fast · Template-formatted</div>",
            unsafe_allow_html=True,
        )

    # ── Pipeline settings ────────────────────────────────────
    _sidebar_section("⚙️", "Pipeline")
    fast_mode = False
    with st.expander("Advanced Settings", expanded=False):
        fast_mode = st.checkbox("Fast mode (≤ 3 chunks, lower latency)", value=False)
        st.markdown("**Generation controls**")
        st.slider("Max review iterations", 1, 3, 2, key="pipeline_max_iter",
                  help="How many Refiner passes before finalizing (more = higher quality, slower)")
        st.slider("Chunk size (chars)", 1000, 6000, 4000, step=500, key="pipeline_chunk_size",
                  help="Larger = fewer LLM calls; smaller = finer-grained test cases")
        st.slider("Temperature", 0.0, 0.5, 0.1, step=0.05, key="pipeline_temperature",
                  help="0.0 = deterministic output; 0.5 = more varied/creative")
        st.markdown(f"**RAG:** {'✅ ChromaDB enabled' if CHROMA_AVAILABLE else '❌ Not installed'}")
        st.markdown(
            "**Agent graph:**  \n"
            "1. 📄 Document Parser → chunks + RAG  \n"
            "2. 🤖 Test Generator → per-chunk + KB context  \n"
            "3. 🔍 Reviewer → quality score (0–100)  \n"
            "4. ⚙️ Refiner → gap-fill (configurable passes)  \n"
            "5. ✅ Finalizer → deduplicate & output"
        )
        st.markdown("---")
        if st.button("🗂 Show timing cache", key="timing_debug"):
            _cf = getattr(timing, "_CACHE_FILE", None)
            try:
                if _cf and _cf.exists():
                    st.code(_cf.read_text(encoding="utf-8"))
                else:
                    st.info("No timing cache yet.")
            except Exception as _e:
                st.error(f"Could not read cache: {_e}")

    # ── ETA panel ────────────────────────────────────────────
    _sidebar_section("⏱", "Timings")
    _timing_keys = [
        ("warmup", "Warmup"),
        ("generate_chunk", "Per-chunk generation"),
        ("validate_quality", "Quality review"),
        ("refinement", "Refinement"),
    ]
    for _key, _label in _timing_keys:
        _avg = timing.get_avg_seconds(_key)
        if _avg is None:
            st.write(f"- **{_label}:** — (no data yet)")
        else:
            st.write(f"- **{_label}:** {_avg:.1f}s avg")

    # ── Run History ──────────────────────────────────────────
    _history = st.session_state.get("run_history", [])
    if _history:
        _sidebar_section("🕐", "Run History")
        with st.expander(
            f"🕐 Run History ({len(_history)} run{'s' if len(_history) != 1 else ''})",
            expanded=False,
        ):
            for _hi, _he in enumerate(_history):
                _qs   = _he["meta"].get("quality_score", "—")
                _cp   = (_he["meta"].get("coverage_report") or {}).get("coverage_pct", "—")
                _conf_list = _he["meta"].get("confidence_scores", [])
                _ac   = (
                    round(sum(s["score"] for s in _conf_list) / max(len(_conf_list), 1))
                    if _conf_list else "—"
                )
                _tag  = "Current" if _hi == 0 else f"Run -{_hi}"
                _hc1, _hc2 = st.columns([5, 1])
                _hc1.markdown(
                    f"<div class='qa-kb-doc'>"
                    f"<span style='font-weight:700;color:#1e293b;'>{_tag}</span>"
                    f" <span style='color:#94a3b8;font-size:10px;'>{_he['timestamp']}</span>"
                    f"<div class='qa-kb-small'>{_he['file_name']}</div>"
                    f"<div class='qa-kb-small'>"
                    f"{_he['case_count']} cases · Q:{_qs} · Cov:{_cp}% · Conf:{_ac}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if _hi > 0 and _hc2.button("↩", key=f"restore_run_{_hi}", help="Restore this run"):
                    st.session_state.preview_rows    = _he["test_cases"]
                    st.session_state.pipeline_meta   = _he["meta"]
                    st.session_state.extracted_data  = {"Test Cases": _he["test_cases"]}
                    st.session_state.workbook_bytes  = None
                    st.session_state.gherkin_bytes   = None
                    st.session_state.last_file_hash  = None
                    st.rerun()


# ─────────────────────────────────────────────────────────────
# Main — Run Pipeline
# ─────────────────────────────────────────────────────────────
_col_run, _col_reset, _col_hint = st.columns([2, 1, 4])
with _col_run:
    run_clicked = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
with _col_reset:
    if st.button("🔄 Reset", type="secondary", use_container_width=True, help="Clear all results and start fresh"):
        for _k, _v in _STATE_DEFAULTS.items():
            st.session_state[_k] = _v
        for _extra in ["raw_text", "file_name"]:
            if _extra in st.session_state:
                del st.session_state[_extra]
        st.session_state.strategy_bytes = None
        st.rerun()
with _col_hint:
    if not st.session_state.get("raw_text"):
        st.info("Upload a spec document in the sidebar, then click **Run Pipeline**.")
    elif not st.session_state.selected_model:
        st.warning("Select a model in the sidebar before running.")
    elif st.session_state.preview_rows:
        st.success(f"✅ {len(st.session_state.preview_rows)} test cases ready — re-run to regenerate.")

# ── Onboarding panel (empty state) ───────────────────────────
if not st.session_state.preview_rows and not run_clicked:
    _s1 = "✅" if st.session_state.selected_model else "⬜"
    _s2 = "✅" if st.session_state.get("raw_text") else "⬜"
    _s3 = "✅" if st.session_state.preview_rows else "⬜"
    _step3_bg     = "#f0fdf4" if _s3 == "✅" else "#eff6ff"
    _step3_border = "#bbf7d0" if _s3 == "✅" else "#bfdbfe"
    _step3_color  = "#166534" if _s3 == "✅" else "#1e3a8a"
    _step3_sub    = (
        f"{len(st.session_state.preview_rows)} test cases generated"
        if _s3 == "✅" else "Click Run Pipeline above"
    )
    st.markdown(
        f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:18px;
                    padding:32px 36px;margin:24px 0;box-shadow:0 2px 12px rgba(37,99,235,.07);">
          <h2 style="color:#1e293b;margin:0 0 6px;font-size:1.35rem;">Get started in 3 steps</h2>
          <p style="color:#64748b;margin:0 0 24px;font-size:14px;">
            Generate professional QA test cases from any spec document in minutes.
          </p>
          <div style="display:flex;gap:18px;flex-wrap:wrap;margin-bottom:28px;">
            <div style="flex:1;min-width:160px;background:#f8fafc;border-radius:12px;
                        padding:16px;border:1px solid #e2e8f0;">
              <div style="font-size:1.5rem;margin-bottom:6px;">{_s1}</div>
              <div style="font-weight:700;color:#1e293b;font-size:13px;">Select a model</div>
              <div style="color:#64748b;font-size:12px;margin-top:3px;">
                Choose Gemini from the sidebar
              </div>
            </div>
            <div style="flex:1;min-width:160px;background:#f8fafc;border-radius:12px;
                        padding:16px;border:1px solid #e2e8f0;">
              <div style="font-size:1.5rem;margin-bottom:6px;">{_s2}</div>
              <div style="font-weight:700;color:#1e293b;font-size:13px;">Upload your spec</div>
              <div style="color:#64748b;font-size:12px;margin-top:3px;">
                PDF, DOCX, TXT — or multiple files
              </div>
            </div>
            <div style="flex:1;min-width:160px;background:{_step3_bg};border-radius:12px;
                        padding:16px;border:1px solid {_step3_border};">
              <div style="font-size:1.5rem;margin-bottom:6px;">{_s3}</div>
              <div style="font-weight:700;color:{_step3_color};font-size:13px;">Run the pipeline</div>
              <div style="color:#475569;font-size:12px;margin-top:3px;">
                {_step3_sub}
              </div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
            <div style="text-align:center;padding:12px 8px;background:#f0fdf4;border-radius:10px;
                        border:1px solid #bbf7d0;">
              <div style="font-size:1.3rem;">🤖</div>
              <div style="font-size:11px;font-weight:600;color:#166534;margin-top:4px;">
                Multi-agent AI
              </div>
            </div>
            <div style="text-align:center;padding:12px 8px;background:#eff6ff;border-radius:10px;
                        border:1px solid #bfdbfe;">
              <div style="font-size:1.3rem;">📊</div>
              <div style="font-size:11px;font-weight:600;color:#1e3a8a;margin-top:4px;">
                Req coverage
              </div>
            </div>
            <div style="text-align:center;padding:12px 8px;background:#fdf4ff;border-radius:10px;
                        border:1px solid #e9d5ff;">
              <div style="font-size:1.3rem;">🥒</div>
              <div style="font-size:11px;font-weight:600;color:#6b21a8;margin-top:4px;">
                BDD / Gherkin
              </div>
            </div>
            <div style="text-align:center;padding:12px 8px;background:#fff7ed;border-radius:10px;
                        border:1px solid #fed7aa;">
              <div style="font-size:1.3rem;">📚</div>
              <div style="font-size:11px;font-weight:600;color:#9a3412;margin-top:4px;">
                RAG knowledge base
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_clicked:
    raw = (st.session_state.get("raw_text") or "").strip()
    if not raw:
        st.warning("Upload a spec document in the sidebar first.")
    elif not st.session_state.selected_model:
        st.warning("Select a model in the sidebar first.")
    else:
        fhash = file_hash(raw)

        if fhash == st.session_state.last_file_hash and st.session_state.extracted_data:
            st.toast("⚡ Using cached result — results shown below.", icon="⚡")
        else:
            with st.chat_message("user"):
                st.write(
                    f"Process **{st.session_state.get('file_name', 'document')}** "
                    f"and generate QA test cases using the multi-agent pipeline."
                )

            with st.chat_message("assistant"):
                progress_placeholder = st.empty()
                with st.expander("🔬 Execution Log", expanded=False):
                    log_placeholder = st.empty()

                try:
                    extracted = execute_pipeline_with_agent(
                        raw,
                        log_placeholder,
                        fast_mode=fast_mode,
                        selected_tabs=["Test Cases"],
                        model_name=st.session_state.selected_model,
                        chunk_size=st.session_state.get("pipeline_chunk_size", 4000),
                        max_iterations=st.session_state.get("pipeline_max_iter", 2),
                        temperature=st.session_state.get("pipeline_temperature", 0.1),
                        progress_placeholder=progress_placeholder,
                    )
                    st.session_state.last_file_hash = fhash

                    # Stash metadata (pop private keys before storing extracted_data)
                    st.session_state.pipeline_meta = {
                        "quality_score":       extracted.pop("_quality_score", 0),
                        "chunks_count":        extracted.pop("_chunks_count", 0),
                        "review_iterations":   extracted.pop("_review_iterations", 0),
                        "coverage_report":     extracted.pop("_coverage_report", {}),
                        "confidence_scores":   extracted.pop("_confidence_scores", []),
                        "parsed_requirements": extracted.pop("_parsed_requirements", []),
                        "fast_mode":           fast_mode,
                        "run_timestamp":       datetime.datetime.now().strftime("%b %d, %H:%M"),
                    }
                    st.session_state.extracted_data = extracted
                    st.session_state.gherkin_bytes = None

                    # Validate and store rows now so results section can read from session state
                    _new_rows, _ = validate_test_case_rows(extracted.get("Test Cases", []))
                    st.session_state.preview_rows = _new_rows
                    st.session_state.workbook_bytes = None
                    st.session_state.strategy_bytes = None

                    # ── Full Test Strategy: one extra LLM call ──
                    _is_full_strategy = st.session_state.get("output_format", "") == "📁 Full Test Strategy (7 tabs)"
                    if _is_full_strategy and _new_rows:
                        _strat_status = st.empty()
                        _strat_status.info("📁 Generating Full Test Strategy content (Cover Page, Scenarios…)")
                        try:
                            _strategy_content = generate_strategy_content(
                                raw,
                                model_name=st.session_state.selected_model,
                            )
                            st.session_state.pipeline_meta["strategy_content"] = _strategy_content
                            _strat_status.success(
                                f"✅ Strategy content ready — "
                                f"{len(_strategy_content.get('test_scenarios', []))} scenarios generated"
                            )
                        except Exception as _se:
                            _strat_status.warning(f"Strategy content generation failed: {_se}. "
                                                  "Workbook will use placeholder text.")
                            st.session_state.pipeline_meta["strategy_content"] = {}

                    # ── Save to run history (keep last 3) ───────
                    _run_entry = {
                        "timestamp": datetime.datetime.now().strftime("%m/%d %H:%M"),
                        "file_name": st.session_state.get("file_name", "document"),
                        "case_count": len(_new_rows),
                        "test_cases": list(_new_rows),
                        "meta": dict(st.session_state.pipeline_meta),
                    }
                    _prev_history = list(st.session_state.get("run_history", []))
                    _prev_history.insert(0, _run_entry)
                    st.session_state.run_history = _prev_history[:3]

                except Exception as _exc:
                    st.error(f"Pipeline error: {_exc}")
                    st.stop()

# ── Results — always rendered from session state, survive re-runs ──
if st.session_state.preview_rows:
        _test_rows = st.session_state.preview_rows
        has_content = bool(_test_rows)

        if not has_content:
            st.warning("No valid test cases were extracted. Check the log above for details.")
        else:
            _meta = st.session_state.pipeline_meta
            _cov  = _meta.get("coverage_report", {})
            _conf = _meta.get("confidence_scores", [])

            # ── Metrics row ───────────────────────────────────
            _cov_pct  = _cov.get("coverage_pct", 0) if _cov else 0
            _avg_conf = (
                round(sum(s["score"] for s in _conf) / max(len(_conf), 1))
                if _conf else "—"
            )
            _qs = _meta.get("quality_score", 0) or 0
            _qs_color = "#059669" if _qs >= 80 else "#d97706" if _qs >= 60 else "#dc2626"
            _qs_label = "Excellent" if _qs >= 80 else "Good" if _qs >= 60 else "Needs work"
            _conf_color = (
                "#059669" if isinstance(_avg_conf, int) and _avg_conf >= 80
                else "#d97706" if isinstance(_avg_conf, int) and _avg_conf >= 60
                else "#dc2626" if isinstance(_avg_conf, int) else "#64748b"
            )
            _cov_color = "#059669" if _cov_pct >= 70 else "#d97706" if _cov_pct >= 50 else "#dc2626"
            _metric_tips = {
                "tc":   "Total unique test cases generated after deduplication",
                "qs":   "0–100 score from the Reviewer agent: 80+ Excellent · 60–79 Good · &lt;60 Needs work",
                "conf": "Average confidence score across all test cases (0–100). Low = ambiguous steps or missing expected results",
                "cov":  "% of parsed requirements that have at least one test case. 70%+ is good coverage",
                "chk":  "Number of document chunks processed by the Test Generator agent",
            }
            st.markdown(
                f"""
                <div class="qa-metrics" style="grid-template-columns:repeat(5,1fr);">
                  <div class="qa-metric" title="{_metric_tips['tc']}">
                    <div class="qa-metric-val">{len(_test_rows)}</div>
                    <div class="qa-metric-label">Test Cases</div>
                  </div>
                  <div class="qa-metric" title="{_metric_tips['qs']}">
                    <div class="qa-metric-val" style="color:{_qs_color};">{_qs}/100</div>
                    <div class="qa-metric-label">Quality — <em>{_qs_label}</em></div>
                  </div>
                  <div class="qa-metric" title="{_metric_tips['conf']}">
                    <div class="qa-metric-val" style="color:{_conf_color};">{_avg_conf}</div>
                    <div class="qa-metric-label">Avg Confidence</div>
                  </div>
                  <div class="qa-metric" title="{_metric_tips['cov']}">
                    <div class="qa-metric-val" style="color:{_cov_color};">{_cov_pct}%</div>
                    <div class="qa-metric-label">Req Coverage</div>
                  </div>
                  <div class="qa-metric" title="{_metric_tips['chk']}">
                    <div class="qa-metric-val">{_meta.get('chunks_count','—')}</div>
                    <div class="qa-metric-label">Chunks Processed</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Processed banner ──────────────────────────────
            _fname_raw = st.session_state.get("file_name", "document")
            _fname = _fname_raw if len(_fname_raw) <= 45 else _fname_raw[:42] + "…"
            _ts    = _meta.get("run_timestamp", "")
            _ts_str = f" · {_ts}" if _ts else ""
            st.markdown(
                f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;"
                f"padding:8px 14px;margin:10px 0 4px;font-size:12px;color:#166534;'>"
                f"📄 <strong>{_fname}</strong> · {len(_test_rows)} test cases generated{_ts_str}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Coverage data for traceability matrix ─────────
            _cov_data = None
            if _cov and _cov.get("mapping") and _meta.get("parsed_requirements"):
                _cov_data = {
                    "mapping": _cov.get("mapping", {}),
                    "parsed_requirements": _meta.get("parsed_requirements", []),
                }

            # ── Auto-build artefacts ───────────────────────────
            _is_full_strategy = st.session_state.get("output_format", "") == "📁 Full Test Strategy (7 tabs)"

            if st.session_state.workbook_bytes is None:
                with st.spinner("📊 Assembling Excel workbook..."):
                    st.session_state.workbook_bytes = build_excel_workbook(
                        {"Test Cases": st.session_state.preview_rows},
                        ["Test Cases"],
                        coverage_data=_cov_data,
                    )

            if _is_full_strategy and st.session_state.strategy_bytes is None:
                with st.spinner("📁 Building Full Test Strategy workbook (7 tabs)..."):
                    _sc = _meta.get("strategy_content") or {}
                    st.session_state.strategy_bytes = build_full_strategy_workbook(
                        st.session_state.preview_rows,
                        strategy_content=_sc,
                        coverage_data=_cov_data,
                    )
            if st.session_state.gherkin_bytes is None:
                st.session_state.gherkin_bytes = build_gherkin(
                    st.session_state.preview_rows
                ).encode("utf-8")

            # ── Export helpers ─────────────────────────────────
            import json as _json
            _rows_for_export = st.session_state.preview_rows

            def _to_json_bytes(rows):
                return _json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")

            def _to_markdown_bytes(rows):
                if not rows:
                    return b""
                fields = list(rows[0].keys())
                header = "| " + " | ".join(fields) + " |"
                sep    = "| " + " | ".join("---" for _ in fields) + " |"
                lines  = [header, sep]
                for r in rows:
                    cells = [str(r.get(f,"")).replace("|","\\|").replace("\n"," ") for f in fields]
                    lines.append("| " + " | ".join(cells) + " |")
                return "\n".join(lines).encode("utf-8")

            def _to_plaintext_bytes(rows):
                lines = []
                for tc in rows:
                    lines += [
                        f"{tc.get('Test Case ID','')} — {tc.get('Test Case Name','')}",
                        f"Priority: {tc.get('Priority','')}  |  Category: {tc.get('Test Category','')}  |  Level: {tc.get('Test Level','')}",
                        f"Requirement: {tc.get('Requirement ID','')}",
                        f"Description: {tc.get('Description','')}",
                        f"Precondition: {tc.get('Precondition','')}",
                        f"Steps:\n{tc.get('Test Step Description','')}",
                        f"Expected Result:\n{tc.get('Test Step Expected Result','')}",
                        "-" * 60,
                    ]
                return "\n".join(lines).encode("utf-8")

            # ── Confidence weak set (used in Tab 1 filters) ───
            _weak_set = {s["tc_name"] for s in _conf if s.get("needs_review")}

            # ══════════════════════════════════════════════════
            # TABS  (live counts in labels)
            # ══════════════════════════════════════════════════
            _tab_tc, _tab_analysis, _tab_cov = st.tabs([
                f"🧾 Test Cases ({len(_test_rows)})",
                "📈 Analysis",
                f"📊 Coverage ({_cov_pct}%)" if _cov and _cov.get("total_requirements", 0) > 0 else "📊 Coverage",
            ])

            # ─── TAB 1: Test Cases ────────────────────────────
            with _tab_tc:
                _all_rows = st.session_state.preview_rows
                _all_cats = sorted({r.get("Test Category","") for r in _all_rows if r.get("Test Category")})
                _all_lvls = sorted({r.get("Test Level","")    for r in _all_rows if r.get("Test Level")})
                _all_pris = sorted({r.get("Priority","")      for r in _all_rows if r.get("Priority")})
                _all_reqs = sorted({r.get("Requirement ID","") for r in _all_rows if r.get("Requirement ID")})

                # ── Search + filters + sort ───────────────────
                _fsearch = st.text_input(
                    "🔍 Search test cases",
                    key="flt_search",
                    placeholder="Search by name, description, steps, TC ID…",
                    label_visibility="collapsed",
                )
                _fc1, _fc2, _fc3, _fc4, _fc5, _fc6 = st.columns([2, 2, 2, 2, 2, 2])
                with _fc1:
                    _sel_cats = st.multiselect("Category", _all_cats, key="flt_cat",
                                               placeholder="All categories")
                with _fc2:
                    _sel_lvls = st.multiselect("Level", _all_lvls, key="flt_lvl",
                                               placeholder="All levels")
                with _fc3:
                    _sel_pris = st.multiselect("Priority", _all_pris, key="flt_pri",
                                               placeholder="All priorities")
                with _fc4:
                    _sel_reqs = st.multiselect("Requirement", _all_reqs, key="flt_req",
                                               placeholder="All requirements")
                with _fc5:
                    _flt_weak = st.checkbox("Low confidence only", key="flt_weak", value=False)
                with _fc6:
                    _sort_by = st.selectbox("Sort by", ["Risk (default)", "Priority", "Category", "Level", "TC ID"],
                                            key="tc_sort", label_visibility="collapsed")

                _is_filtered = bool(_fsearch or _sel_cats or _sel_lvls or _sel_pris or _sel_reqs or _flt_weak)
                _display_rows = list(_all_rows)
                if _fsearch:
                    _q = _fsearch.lower()
                    _search_fields = ["Test Case Name", "Description", "Test Step Description",
                                      "Test Step Expected Result", "Precondition", "Test Case ID"]
                    _display_rows = [
                        r for r in _display_rows
                        if any(_q in str(r.get(f,"")).lower() for f in _search_fields)
                    ]
                if _sel_cats:
                    _display_rows = [r for r in _display_rows if r.get("Test Category","") in _sel_cats]
                if _sel_lvls:
                    _display_rows = [r for r in _display_rows if r.get("Test Level","") in _sel_lvls]
                if _sel_pris:
                    _display_rows = [r for r in _display_rows if r.get("Priority","") in _sel_pris]
                if _sel_reqs:
                    _display_rows = [r for r in _display_rows if r.get("Requirement ID","") in _sel_reqs]
                if _flt_weak:
                    _display_rows = [r for r in _display_rows if r.get("Test Case Name","") in _weak_set]

                # Apply sort
                _PRI_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
                if _sort_by == "Priority":
                    _display_rows = sorted(_display_rows, key=lambda r: _PRI_ORDER.get(r.get("Priority",""), 9))
                elif _sort_by == "Category":
                    _display_rows = sorted(_display_rows, key=lambda r: r.get("Test Category",""))
                elif _sort_by == "Level":
                    _display_rows = sorted(_display_rows, key=lambda r: r.get("Test Level",""))
                elif _sort_by == "TC ID":
                    _display_rows = sorted(_display_rows, key=lambda r: r.get("Test Case ID",""))

                if _is_filtered:
                    st.caption(
                        f"Showing **{len(_display_rows)}** of **{len(_all_rows)}** "
                        "— edits disabled while filters/search are active"
                    )

                # ── Empty state ───────────────────────────────
                if not _display_rows:
                    st.markdown(
                        "<div style='text-align:center;padding:40px 20px;color:#64748b;'>"
                        "<div style='font-size:40px;margin-bottom:12px;'>🔍</div>"
                        "<div style='font-size:15px;font-weight:600;margin-bottom:6px;'>No test cases match your filters</div>"
                        "<div style='font-size:13px;'>Try clearing the search box or removing a filter above.</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    preview_rows = render_test_case_preview(_display_rows)
                    if not _is_filtered:
                        st.session_state.preview_rows = preview_rows

                    if preview_rows:
                        _rb_label = (
                            "📦 Rebuild Test Cases workbook from edits"
                            if not _is_full_strategy
                            else "📦 Rebuild both workbooks from edits"
                        )
                        if st.button(_rb_label, type="secondary"):
                            _rows_to_use = st.session_state.preview_rows if _is_filtered else preview_rows
                            with st.spinner("Assembling..."):
                                st.session_state.workbook_bytes = build_excel_workbook(
                                    {"Test Cases": _rows_to_use},
                                    ["Test Cases"],
                                    coverage_data=_cov_data,
                                )
                                st.session_state.gherkin_bytes = build_gherkin(
                                    _rows_to_use
                                ).encode("utf-8")
                                if _is_full_strategy:
                                    _sc2 = _meta.get("strategy_content") or {}
                                    st.session_state.strategy_bytes = build_full_strategy_workbook(
                                        _rows_to_use,
                                        strategy_content=_sc2,
                                        coverage_data=_cov_data,
                                    )
                            st.toast("✅ Workbook(s) & Gherkin rebuilt.", icon="✅")

                    # ── Test case detail panel ────────────────
                    _tc_names = [r.get("Test Case ID","") + " — " + r.get("Test Case Name","")
                                 for r in _display_rows]
                    if _tc_names:
                        with st.expander("🔎 View test case detail", expanded=False):
                            _sel_tc = st.selectbox("Select a test case", _tc_names,
                                                   key="detail_tc_sel", label_visibility="collapsed")
                            _sel_idx = _tc_names.index(_sel_tc) if _sel_tc in _tc_names else 0
                            _tc = _display_rows[_sel_idx]
                            _pri_c = {"Critical":"#dc2626","High":"#d97706","Medium":"#2563eb","Low":"#64748b"}.get(
                                _tc.get("Priority",""), "#64748b")
                            st.markdown(
                                f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;'>"
                                f"<div style='font-size:15px;font-weight:700;color:#1e293b;margin-bottom:8px;'>"
                                f"{_tc.get('Test Case ID','')} &nbsp; {_tc.get('Test Case Name','')}</div>"
                                f"<div style='display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;'>"
                                f"<span style='background:{_pri_c};color:white;font-size:11px;font-weight:600;"
                                f"padding:2px 8px;border-radius:12px;'>{_tc.get('Priority','')}</span>"
                                f"<span style='background:#eff6ff;color:#1d4ed8;font-size:11px;padding:2px 8px;border-radius:12px;'>"
                                f"{_tc.get('Test Category','')}</span>"
                                f"<span style='background:#f0fdf4;color:#166534;font-size:11px;padding:2px 8px;border-radius:12px;'>"
                                f"{_tc.get('Test Level','')}</span>"
                                f"<span style='background:#fefce8;color:#854d0e;font-size:11px;padding:2px 8px;border-radius:12px;'>"
                                f"REQ: {_tc.get('Requirement ID','—')}</span>"
                                f"</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                            _d1, _d2 = st.columns(2)
                            with _d1:
                                st.markdown("**Description**")
                                st.markdown(_tc.get("Description","—"))
                                st.markdown("**Precondition**")
                                st.markdown(_tc.get("Precondition","—"))
                            with _d2:
                                st.markdown("**Test Steps**")
                                st.markdown(_tc.get("Test Step Description","—"))
                                st.markdown("**Expected Result**")
                                st.markdown(_tc.get("Test Step Expected Result","—"))

                # ── Export ────────────────────────────────────
                st.markdown("---")

                # ── Full Test Strategy primary download ────────
                if _is_full_strategy and st.session_state.strategy_bytes:
                    st.markdown(
                        "<div style='background:linear-gradient(135deg,#1e3a8a,#2563eb);"
                        "border-radius:12px;padding:14px 18px;margin-bottom:12px;'>"
                        "<div style='color:white;font-weight:700;font-size:14px;margin-bottom:4px;'>"
                        "📁 Full Test Strategy — 7-Tab Workbook Ready</div>"
                        "<div style='color:rgba(255,255,255,.75);font-size:12px;'>"
                        "Cover Page · Project Overview · BOT Timelines · Test Scenarios · "
                        "Test Data · Test Cases · Samples"
                        "</div></div>",
                        unsafe_allow_html=True,
                    )
                    _fname_base = st.session_state.get("file_name", "document")
                    _fname_base = re.sub(r"\.[^.]+$", "", _fname_base)[:40]
                    st.download_button(
                        "📥 Download Full Test Strategy (.xlsx)",
                        data=st.session_state.strategy_bytes,
                        file_name=f"BOT_Test_Strategy_{_fname_base}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        type="primary",
                    )
                    st.markdown("---")

                _ex_l, _ex_r = st.columns([3, 4])
                with _ex_l:
                    st.markdown("**⬇️ Primary exports**")
                    _p1, _p2 = st.columns(2)
                    with _p1:
                        if st.session_state.workbook_bytes:
                            _tc_label = "📊 Test Cases (.xlsx)" if _is_full_strategy else "📊 Excel (.xlsx)"
                            st.download_button(_tc_label,
                                data=st.session_state.workbook_bytes,
                                file_name="QA_Test_Cases.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)
                    with _p2:
                        if st.session_state.gherkin_bytes:
                            st.download_button("🥒 Gherkin (.feature)",
                                data=st.session_state.gherkin_bytes,
                                file_name="QA_Test_Cases.feature",
                                mime="text/plain",
                                use_container_width=True)
                with _ex_r:
                    st.markdown("**More formats**")
                    _m1, _m2, _m3 = st.columns(3)
                    with _m1:
                        st.download_button("📋 JSON",
                            data=_to_json_bytes(_rows_for_export),
                            file_name="QA_Test_Cases.json",
                            mime="application/json",
                            use_container_width=True)
                    with _m2:
                        st.download_button("📝 Markdown",
                            data=_to_markdown_bytes(_rows_for_export),
                            file_name="QA_Test_Cases.md",
                            mime="text/markdown",
                            use_container_width=True)
                    with _m3:
                        st.download_button("📄 Plain Text",
                            data=_to_plaintext_bytes(_rows_for_export),
                            file_name="QA_Test_Cases.txt",
                            mime="text/plain",
                            use_container_width=True)

                # ── Copy-to-clipboard (Gherkin + JSON) ────────
                if st.session_state.gherkin_bytes or _rows_for_export:
                    with st.expander("📋 Copy to clipboard", expanded=False):
                        _cp1, _cp2 = st.tabs(["🥒 Gherkin", "📋 JSON"])
                        with _cp1:
                            if st.session_state.gherkin_bytes:
                                st.code(
                                    st.session_state.gherkin_bytes.decode("utf-8"),
                                    language="gherkin",
                                )
                        with _cp2:
                            if _rows_for_export:
                                import json as _json2
                                st.code(
                                    _json2.dumps(_rows_for_export, indent=2, ensure_ascii=False),
                                    language="json",
                                )

            # ─── TAB 2: Analysis ──────────────────────────────
            with _tab_analysis:
                # Suite balance charts
                _cat_counts = Counter(r.get("Test Category","Unknown") for r in _test_rows)
                _lvl_counts = Counter(r.get("Test Level","Unknown")    for r in _test_rows)
                _pri_counts = Counter(r.get("Priority","Unknown")      for r in _test_rows)
                _CAT_C = {"Functional":"#2563eb","Negative":"#dc2626","Security":"#7c3aed",
                           "Boundary":"#d97706","Performance":"#059669"}
                _PRI_C = {"Critical":"#dc2626","High":"#d97706","Medium":"#2563eb","Low":"#64748b"}
                _LVL_C = {"Unit":"#94a3b8","Integration":"#2563eb","System":"#7c3aed","Acceptance":"#059669"}
                _sa1, _sa2, _sa3 = st.columns(3)
                with _sa1:
                    st.markdown("**By Category**")
                    st.markdown(_bars_html(_cat_counts, _CAT_C), unsafe_allow_html=True)
                with _sa2:
                    st.markdown("**By Priority**")
                    st.markdown(_bars_html(_pri_counts, _PRI_C), unsafe_allow_html=True)
                with _sa3:
                    st.markdown("**By Level**")
                    st.markdown(_bars_html(_lvl_counts, _LVL_C), unsafe_allow_html=True)

                _func_pct = round(_cat_counts.get("Functional",0) / max(len(_test_rows),1) * 100)
                _neg_pct  = round(_cat_counts.get("Negative",0)   / max(len(_test_rows),1) * 100)
                if _func_pct > 60:
                    st.warning(f"⚠ {_func_pct}% Functional — consider more Negative/Boundary/Security cases.")
                elif _neg_pct >= 20 and _cat_counts.get("Security",0) >= 1:
                    st.success("✅ Well-balanced test suite.")

                # Confidence issues
                if _conf:
                    _weak = [s for s in _conf if s["needs_review"]]
                    if _weak:
                        st.markdown("---")
                        st.markdown(f"**🔎 Confidence Issues — {len(_weak)} case(s) below threshold**")
                        for _s in _weak:
                            st.markdown(
                                f"<div class='qa-kb-doc' style='border-color:#fde68a;background:#fffbeb;'>"
                                f"<span style='font-weight:600;color:#92400e;'>{_s['tc_name'][:50]}</span>"
                                f" <span style='float:right;color:#d97706;font-weight:700;'>{_s['score']}/100</span>"
                                f"<div class='qa-kb-small'>" + " · ".join(_s["issues"]) + "</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # Extraction summary
                st.markdown("---")
                st.markdown("**🔍 Extraction Summary**")
                _extracted_data = st.session_state.get("extracted_data") or {}
                for _tab_name, _tab_rws in _extracted_data.items():
                    if not _tab_name.startswith("_"):
                        st.write(f"**{_tab_name}** → `{len(_tab_rws)}` row(s) extracted")
                _iter = _meta.get("review_iterations", 0)
                st.caption(
                    f"Pipeline: {_iter} review pass{'es' if _iter != 1 else ''} · "
                    f"Quality score {_meta.get('quality_score','—')}/100 · "
                    f"{_meta.get('chunks_count','—')} chunks"
                )

            # ─── TAB 3: Coverage ──────────────────────────────
            with _tab_cov:
                if not _cov or _cov.get("total_requirements", 0) == 0:
                    st.markdown(
                        """
                        <div style='background:#fefce8;border:1px solid #fde68a;border-radius:10px;
                                    padding:16px 20px;margin:8px 0;'>
                          <div style='font-weight:700;color:#92400e;font-size:14px;margin-bottom:8px;'>
                            📊 Coverage analysis unavailable for this document
                          </div>
                          <div style='color:#78350f;font-size:13px;line-height:1.6;'>
                            Coverage tracking works when your document contains
                            <strong>structured requirements</strong> — sentences that describe
                            a specific actor, action, or expected system behaviour.<br><br>
                            <strong>Tips to get coverage data:</strong>
                          </div>
                          <ul style='color:#78350f;font-size:13px;margin:8px 0 0 16px;line-height:1.8;'>
                            <li>Use numbered requirements: <em>"REQ-01: The system shall…"</em></li>
                            <li>Use user story format: <em>"As a user, I want to… so that…"</em></li>
                            <li>Use acceptance criteria: <em>"Given… When… Then…"</em></li>
                            <li>Upload the requirements file to the <strong>Knowledge Base</strong>
                                (sidebar → Requirements KB) before running</li>
                          </ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    if _meta.get("fast_mode"):
                        st.warning(
                            "⚡ Fast mode active — coverage % is a partial estimate."
                        )
                    _bar_color = (
                        "#059669" if _cov_pct >= 70
                        else "#d97706" if _cov_pct >= 50
                        else "#dc2626"
                    )
                    st.markdown(
                        f"**{_cov_pct}% coverage** "
                        f"({len(_cov.get('covered_ids',[]))}/{_cov.get('total_requirements',0)} requirements)"
                    )
                    st.markdown(
                        f"<div style='background:#f1f5f9;border-radius:8px;height:14px;"
                        f"overflow:hidden;margin:4px 0 18px;'>"
                        f"<div style='background:{_bar_color};width:{_cov_pct}%;height:100%;"
                        f"border-radius:8px;transition:width 0.6s ease;'></div></div>",
                        unsafe_allow_html=True,
                    )

                    _risk_gaps = _cov.get("risk_gaps", [])
                    if _risk_gaps:
                        st.markdown(
                            f"<div style='background:#fef2f2;border:1px solid #fecaca;"
                            f"border-radius:8px;padding:10px 14px;margin-bottom:10px;'>"
                            f"<span style='color:#dc2626;font-weight:700;'>⚠ {len(_risk_gaps)} "
                            f"Critical/High requirement(s) with no test cases</span></div>",
                            unsafe_allow_html=True,
                        )
                        for _gap in _risk_gaps:
                            st.markdown(
                                f"<div class='qa-kb-doc' style='border-color:#fecaca;margin:3px 0;'>"
                                f"<span style='color:#dc2626;font-weight:600;'>{_gap['req_id']}</span>"
                                f" · <span style='color:#7f1d1d;font-size:11px;'>{_gap['priority']}</span>"
                                f"<div class='qa-kb-small'>{_gap.get('action','')}</div></div>",
                                unsafe_allow_html=True,
                            )

                    _partial = _cov.get("partial_detail", [])
                    if _partial:
                        st.markdown(
                            f"<div style='background:#fffbeb;border:1px solid #fde68a;"
                            f"border-radius:8px;padding:10px 14px;margin-bottom:10px;'>"
                            f"<span style='color:#92400e;font-weight:700;'>⚠ {len(_partial)} "
                            f"requirement(s) covered by Functional tests only</span></div>",
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"🟡 Partial coverage ({len(_partial)})"):
                            for _p in _partial:
                                _tc_list = ", ".join(_p.get("tc_names",[])[:2])
                                st.markdown(
                                    f"<div class='qa-kb-doc' style='border-color:#fde68a;'>"
                                    f"<span style='font-weight:600;color:#92400e;'>{_p['req_id']}</span>"
                                    f" <span style='color:#64748b;font-size:11px;'>[{_p.get('priority','')}]</span>"
                                    f"<div class='qa-kb-small'>{_p.get('action','')}</div>"
                                    f"<div class='qa-kb-small' style='color:#d97706;'>Covered by: {_tc_list}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    _uncovered = _cov.get("uncovered_detail", [])
                    if _uncovered:
                        with st.expander(f"📋 All uncovered ({len(_uncovered)})"):
                            for _u in _uncovered:
                                _pc = {"Critical":"#dc2626","High":"#d97706",
                                       "Medium":"#2563eb","Low":"#64748b"}.get(
                                    _u.get("priority","Medium"), "#64748b")
                                st.markdown(
                                    f"<div class='qa-kb-doc' style='margin:3px 0;'>"
                                    f"<span style='font-weight:600;color:{_pc};'>{_u['req_id']}</span>"
                                    f" <span style='color:#64748b;font-size:11px;'>[{_u.get('priority','')}]</span>"
                                    f"<div class='qa-kb-small'>{_u.get('source_text','')}</div></div>",
                                    unsafe_allow_html=True,
                                )


# ─────────────────────────────────────────────────────────────
# Pipeline status footer
# ─────────────────────────────────────────────────────────────
if st.session_state.workbook_bytes and st.session_state.preview_rows:
    _meta_f = st.session_state.get("pipeline_meta", {})
    if _meta_f:
        _iter_f  = _meta_f.get("review_iterations", 0)
        _score_f = _meta_f.get("quality_score", "—")
        st.markdown(
            f"""
            <div style="background:#f0fdf4;border-radius:10px;padding:10px 16px;
                        border:1px solid #bbf7d0;margin-top:12px;">
              <span style="color:#166534;font-weight:600;font-size:13px;">✅ Pipeline complete</span>
              &nbsp;·&nbsp;
              <span style="color:#475569;font-size:12px;">
                Quality score <strong>{_score_f}/100</strong>
                &nbsp;·&nbsp; {_iter_f} review pass{'es' if _iter_f != 1 else ''}
                &nbsp;·&nbsp; {len(st.session_state.preview_rows)} unique test cases
                &nbsp;·&nbsp; <em>Export options in the 🧾 Test Cases tab above</em>
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
