"""
Warmup: LLM health check and initial timing measurement.
"""

import streamlit as st
from .config import get_llm, DEFAULT_MODEL
from .timing import Timer


@st.cache_resource(show_spinner="🔥 Warming up model...")
def warmup_gemini(model_name: str | None = None):
    """Make a single LLM call to warm up the connection and record latency for ETA estimates."""
    try:
        llm = get_llm(model_name or DEFAULT_MODEL)
        with Timer("warmup"):
            llm.invoke("hi")
        return True
    except Exception as exc:
        try:
            st.warning(f"Model warm-up failed: {exc}")
        except Exception:
            pass
        return False
