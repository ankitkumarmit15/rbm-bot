"""
JSON Parser: extract, normalize, and validate JSON from LLM output.
Handles both raw strings and LangChain AIMessage objects.
"""

import re
import json
from .config import TAB_SCHEMAS


def _to_str(raw) -> str:
    """Coerce LangChain AIMessage or any object to a plain string."""
    if hasattr(raw, "content"):
        return str(raw.content)
    return str(raw)


def normalize_test_case_rows(rows: list) -> list:
    """Ensure every row is a dict with all schema fields as strings."""
    fields = TAB_SCHEMAS["Test Cases"]
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized.append({
            field: str(row.get(field, "") or "").strip()
            for field in fields
        })
    return normalized


def extract_json_from_response(raw, tab_name: str, fields: list) -> dict:
    """
    Robustly extract JSON from LLM output.
    Accepts AIMessage objects or plain strings.
    Falls back through 4 strategies before giving up.
    """
    raw_str = _to_str(raw)
    cleaned = raw_str.replace("```json", "").replace("```", "").strip()

    # Strategy 1: locate the outermost JSON object
    if not cleaned.startswith("{"):
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if m:
            cleaned = m.group(0)

    try:
        data = json.loads(cleaned)
        if tab_name in data and isinstance(data[tab_name], list):
            return {tab_name: normalize_test_case_rows(data[tab_name])}
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract a bare JSON array
    try:
        m = re.search(r'\[.*\]', cleaned, re.DOTALL)
        if m:
            data = {tab_name: json.loads(m.group(0))}
            if isinstance(data[tab_name], list):
                return {tab_name: normalize_test_case_rows(data[tab_name])}
    except json.JSONDecodeError:
        pass

    # Strategy 3: try common truncation suffixes
    for suffix in [']}', '"}]}', '"]}', '  ]}']:
        try:
            data = json.loads(cleaned + suffix)
            if tab_name in data and isinstance(data[tab_name], list):
                return {tab_name: normalize_test_case_rows(data[tab_name])}
        except json.JSONDecodeError:
            continue

    # Strategy 4: regex field extraction (last resort)
    extracted = {f: "" for f in fields}
    found_any = False
    for field in fields:
        m = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', cleaned, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if val:
                extracted[field] = val
                found_any = True

    if found_any:
        return {tab_name: [extracted]}

    return {tab_name: [{f: "" for f in fields}]}
