"""
Coverage tracking: map generated test cases back to parsed requirements
using semantic similarity (ChromaDB) with keyword fallback.

Returns a coverage report: covered %, uncovered requirements, risk gaps.
"""
from __future__ import annotations

from .rag import CHROMA_AVAILABLE

_LINK_THRESHOLD = 0.72  # cosine distance — below this = test case covers requirement


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def compute_coverage(requirements: list, test_cases: list) -> dict:
    """
    Link test cases to requirements and return a full coverage report.

    Returns:
      covered_ids      : list of req_ids that have at least one test case
      uncovered_ids    : list of req_ids with no test case
      coverage_pct     : int 0-100
      mapping          : {req_id: [test_case_names]}
      risk_gaps        : Critical/High requirements with no test cases
      total_requirements: int
    """
    if not requirements:
        return _empty_report()

    if CHROMA_AVAILABLE:
        return _semantic_coverage(requirements, test_cases)
    return _keyword_coverage(requirements, test_cases)


# ─────────────────────────────────────────────────────────────
# Semantic coverage (preferred)
# ─────────────────────────────────────────────────────────────

def _semantic_coverage(requirements: list, test_cases: list) -> dict:
    import uuid
    import chromadb
    from chromadb.utils import embedding_functions

    try:
        client = chromadb.Client()
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        coll_name = f"req_cov_{uuid.uuid4().hex[:10]}"
        coll = client.create_collection(coll_name, embedding_function=embed_fn)

        req_ids = [r.get("req_id", f"REQ-{i}") for i, r in enumerate(requirements)]
        req_docs = [
            f"{r.get('req_id', '')} {r.get('action', '')} {r.get('expected_behavior', '')}"[:300]
            for r in requirements
        ]
        coll.add(documents=req_docs, ids=[f"r_{i}" for i in range(len(requirements))])

        mapping: dict[str, list] = {rid: [] for rid in req_ids}

        for tc in test_cases:
            tc_text = (
                f"{tc.get('Test Case Name', '')} "
                f"{tc.get('Description', '')} "
                f"{tc.get('Test Step Description', '')}"
            )[:300]
            try:
                results = coll.query(
                    query_texts=[tc_text],
                    n_results=min(3, len(requirements)),
                    include=["distances"],
                )
                for rid_key, dist in zip(
                    results.get("ids", [[]])[0],
                    results.get("distances", [[]])[0],
                ):
                    idx = int(rid_key.replace("r_", ""))
                    if dist < _LINK_THRESHOLD and idx < len(req_ids):
                        mapping[req_ids[idx]].append(tc.get("Test Case Name", ""))
            except Exception:
                pass

        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

        return _build_report(requirements, mapping, test_cases)

    except Exception:
        return _keyword_coverage(requirements, test_cases)


# ─────────────────────────────────────────────────────────────
# Keyword fallback
# ─────────────────────────────────────────────────────────────

def _keyword_coverage(requirements: list, test_cases: list) -> dict:
    mapping: dict[str, list] = {}
    for r in requirements:
        req_id = r.get("req_id", "")
        mapping[req_id] = []
        keywords = [
            w for w in r.get("action", "").lower().split()
            if len(w) > 4
        ][:6]
        for tc in test_cases:
            tc_lower = (
                f"{tc.get('Test Case Name', '')} {tc.get('Description', '')}"
            ).lower()
            if any(kw in tc_lower for kw in keywords):
                mapping[req_id].append(tc.get("Test Case Name", ""))
    return _build_report(requirements, mapping, test_cases)


# ─────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────

_POSITIVE_ONLY_CATEGORIES = {"Functional", ""}


def _detect_partial_coverage(covered_ids: list, mapping: dict, test_cases: list) -> list[str]:
    """
    Return req_ids that are 'covered' but ONLY by positive/functional test cases.
    A requirement truly covered should have at least one non-functional test
    (Negative, Boundary, Security, or Performance).
    """
    tc_by_name = {tc.get("Test Case Name", ""): tc for tc in test_cases}
    partial = []
    for rid in covered_ids:
        tc_names = mapping.get(rid, [])
        categories = {
            tc_by_name.get(name, {}).get("Test Category", "Functional")
            for name in tc_names
            if name in tc_by_name
        }
        if categories and categories.issubset(_POSITIVE_ONLY_CATEGORIES):
            partial.append(rid)
    return partial


def _build_report(requirements: list, mapping: dict, test_cases: list = None) -> dict:
    covered_ids = [rid for rid, tcs in mapping.items() if tcs]
    uncovered_ids = [rid for rid, tcs in mapping.items() if not tcs]
    total = max(len(requirements), 1)
    coverage_pct = round(len(covered_ids) / total * 100)

    req_lookup = {r.get("req_id", ""): r for r in requirements}

    # Partial coverage: covered but only by happy-path tests
    partial_ids = _detect_partial_coverage(covered_ids, mapping, test_cases or [])

    risk_gaps = [
        {
            "req_id": rid,
            "priority": req_lookup.get(rid, {}).get("priority", ""),
            "action": req_lookup.get(rid, {}).get("action", ""),
            "source_text": req_lookup.get(rid, {}).get("source_text", "")[:120],
        }
        for rid in uncovered_ids
        if req_lookup.get(rid, {}).get("priority") in ("Critical", "High")
    ]

    uncovered_detail = [
        {
            "req_id": rid,
            "priority": req_lookup.get(rid, {}).get("priority", "Medium"),
            "action": req_lookup.get(rid, {}).get("action", ""),
            "source_text": req_lookup.get(rid, {}).get("source_text", "")[:120],
        }
        for rid in uncovered_ids
    ]

    partial_detail = [
        {
            "req_id": rid,
            "priority": req_lookup.get(rid, {}).get("priority", "Medium"),
            "action": req_lookup.get(rid, {}).get("action", ""),
            "source_text": req_lookup.get(rid, {}).get("source_text", "")[:120],
            "tc_names": mapping.get(rid, []),
        }
        for rid in partial_ids
    ]

    return {
        "covered_ids": covered_ids,
        "uncovered_ids": uncovered_ids,
        "partial_ids": partial_ids,
        "uncovered_detail": uncovered_detail,
        "partial_detail": partial_detail,
        "coverage_pct": coverage_pct,
        "mapping": mapping,
        "risk_gaps": risk_gaps,
        "total_requirements": len(requirements),
    }


def _empty_report() -> dict:
    return {
        "covered_ids": [],
        "uncovered_ids": [],
        "partial_ids": [],
        "uncovered_detail": [],
        "partial_detail": [],
        "coverage_pct": 0,
        "mapping": {},
        "risk_gaps": [],
        "total_requirements": 0,
    }
