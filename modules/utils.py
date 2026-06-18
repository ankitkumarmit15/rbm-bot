"""
Utilities: UI helpers and data handling functions.
"""

import hashlib
import streamlit as st
import pandas as pd
from .json_parser import normalize_test_case_rows


def file_hash(text: str) -> str:
    """MD5 hash of text — used as a cache key for uploaded files."""
    return hashlib.md5(text.encode()).hexdigest()


def render_test_case_preview(test_cases: list) -> list:
    """
    Render an editable DataFrame preview of test cases in Streamlit.
    Returns the (possibly edited) rows as a list of dicts.
    """
    normalized = normalize_test_case_rows(test_cases)
    if not normalized:
        st.warning("No test cases to display. Verify the input document and re-run.")
        return []

    df = pd.DataFrame(normalized)
    st.caption("Review and edit test cases below before building the workbook.")

    if hasattr(st, "data_editor"):
        edited = st.data_editor(df, use_container_width=True, num_rows="dynamic",
                                key="test_case_preview")
    elif hasattr(st, "experimental_data_editor"):
        edited = st.experimental_data_editor(df, use_container_width=True, num_rows="dynamic",
                                              key="test_case_preview")
    else:
        st.dataframe(df)
        edited = df

    edited = edited.fillna("") if hasattr(edited, "fillna") else pd.DataFrame(edited).fillna("")
    return edited.to_dict(orient="records")
