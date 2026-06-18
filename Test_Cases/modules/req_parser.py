"""
Requirement Parser: extract structured testable assertions from requirements text.

Each assertion: req_id, actor, action, constraint, expected_behavior, priority, source_text

For long specs (> 5 000 chars) the text is chunked and parsed in parallel — wall time equals
one LLM call regardless of document length.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import get_llm
from .json_parser import _to_str

_PRIORITY_WORDS = {
    "Critical": ["shall", "must", "will not", "required", "mandatory"],
    "High":     ["should", "needs to", "is expected to", "is required"],
    "Medium":   ["may", "can", "optionally", "is able to"],
    "Low":      ["nice to have", "desired", "preferred", "consider"],
}


def _infer_priority(text: str) -> str:
    lower = text.lower()
    for level, words in _PRIORITY_WORDS.items():
        if any(w in lower for w in words):
            return level
    return "Medium"


def _extract_json(raw_str: str) -> list:
    m = re.search(r'\{.*\}', raw_str, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            reqs = data.get("requirements", [])
            if isinstance(reqs, list) and reqs:
                return reqs
        except Exception:
            pass
    return []


def _parse_single_chunk(text: str, model_name: str | None) -> list[dict]:
    """Parse one chunk. Returns raw list — req_ids are reassigned by the caller."""
    prompt = (
        "You are a senior SDET. Extract EVERY distinct testable assertion from the text below.\n\n"
        "For EACH requirement or testable statement return:\n"
        "  req_id           : placeholder like REQ-001 (will be renumbered)\n"
        "  actor            : who performs the action — User, System, Admin, API, etc.\n"
        "  action           : the operation being tested\n"
        "  constraint       : business rule, limit, or condition (empty string if none)\n"
        "  expected_behavior: the verifiable outcome\n"
        "  priority         : Critical | High | Medium | Low\n"
        "                     (shall/must=Critical, should=High, may=Medium, nice-to-have=Low)\n"
        "  source_text      : exact original sentence\n\n"
        'Output ONLY valid JSON: {"requirements": [...]}\n\n'
        f"Text:\n{text}"
    )
    try:
        raw = get_llm(model_name).invoke(prompt)
        reqs = _extract_json(_to_str(raw))
        if reqs:
            return reqs
    except Exception:
        pass

    # Fallback: sentence splitting
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    return [
        {
            "req_id": f"REQ-{i+1:03d}",
            "actor": "System",
            "action": s[:80],
            "constraint": "",
            "expected_behavior": s,
            "priority": _infer_priority(s),
            "source_text": s,
        }
        for i, s in enumerate(sentences[:40])
    ]


def parse_requirements(text: str, model_name: str | None = None) -> list[dict]:
    """
    Extract every testable assertion from requirements text.

    For short documents (≤ 5 000 chars) a single LLM call is made.
    For longer documents the text is chunked and all chunks are parsed
    in parallel (max 3 concurrent LLM calls) so wall time stays low.

    Returns a merged, deduplicated, renumbered list.
    """
    from .chunking import smart_split

    chunks = smart_split(text, chunk_size=4500, chunk_overlap=100)

    if len(chunks) <= 1:
        raw_results = [_parse_single_chunk(text[:5000], model_name)]
    else:
        ordered: list[list[dict] | None] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=min(len(chunks), 3)) as ex:
            futures = {
                ex.submit(_parse_single_chunk, chunk, model_name): i
                for i, chunk in enumerate(chunks)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    ordered[idx] = fut.result() or []
                except Exception:
                    ordered[idx] = []
        raw_results = [r for r in ordered if r is not None]

    # Merge, deduplicate on source_text fingerprint, renumber sequentially
    all_reqs: list[dict] = []
    seen: set[str] = set()
    counter = 0

    for chunk_reqs in raw_results:
        for r in chunk_reqs:
            key = (r.get("source_text") or r.get("action") or "")[:60].lower().strip()
            if key and key not in seen:
                seen.add(key)
                counter += 1
                r["req_id"] = f"REQ-{counter:03d}"
                all_reqs.append(r)

    return all_reqs
