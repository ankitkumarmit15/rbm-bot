"""
Multi-Agent System: LangGraph StateGraph
Flow: Document Parser → Test Generator → Reviewer ↻ Refiner → Finalizer

New in v3:
  - Requirement parsing (structured assertions with Actor/Action/Constraint/Expected/Priority)
  - Per-case confidence scoring (0-100 per test case, flags weak ones)
  - Semantic deduplication in Finalizer (embedding similarity, keeps higher-confidence case)
  - Risk-based output ordering (Security/Negative first, sorted by criticality)
  - Requirement coverage report (which requirements are covered, uncovered, risk gaps)
"""
from __future__ import annotations

import hashlib
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

from .config import TAB_SCHEMAS, get_llm
from .json_parser import extract_json_from_response
from .prompts import make_system_prompt, make_gap_prompt, make_improve_prompt, make_strategy_prompt
from .chunking import smart_split
from .rag import (
    build_vector_store, retrieve_similar_chunks,
    query_kb, _infer_category, find_nearest_req_doc, CHROMA_AVAILABLE,
)
from .req_parser import parse_requirements
from .coverage import compute_coverage
from . import timing as _timing


# ─────────────────────────────────────────────────────────────
# Shared graph state
# ─────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    # Inputs
    requirements_text: str
    model_name: Optional[str]
    fast_mode: bool
    chunk_size: int        # chars per chunk (default 4000)
    max_iterations: int    # max refiner passes (default 2)
    temperature: float     # LLM temperature (default 0.1)
    # Parser
    chunks: list
    collection: Any
    doc_id: str
    parsed_requirements: list   # structured assertions from req_parser
    # Generator
    all_test_cases: list
    generated_names: list
    # Reviewer
    quality_score: int
    quality_valid: bool
    quality_issues: list
    review_iterations: int
    confidence_scores: list     # per-case: {tc_name, score, issues, needs_review}
    # Output
    final_test_cases: list
    coverage_report: dict       # requirement → test case mapping + gaps
    # Log
    log: list


# ─────────────────────────────────────────────────────────────
# LLM invocation with retry + rate-limit backoff
# ─────────────────────────────────────────────────────────────

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "ratequota")
_TRANSIENT_MARKERS  = _RATE_LIMIT_MARKERS + ("503", "500 internal", "timeout", "unavailable")


def _llm_invoke(model_name: str | None, prompt: str, temperature: float = 0.1, max_retries: int = 2):
    """
    Call the LLM and retry on rate-limit or transient errors.
    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return get_llm(model_name, temperature=temperature).invoke(prompt)
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            is_rate_limit  = any(m in err for m in _RATE_LIMIT_MARKERS)
            is_transient   = any(m in err for m in _TRANSIENT_MARKERS)
            if attempt < max_retries and is_transient:
                wait = (10 if is_rate_limit else 3) * (2 ** attempt)
                time.sleep(wait)
                continue
            break
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────
# RAG helpers
# ─────────────────────────────────────────────────────────────

def _find_relevant_reqs(chunk: str, parsed_requirements: list, max_reqs: int = 4) -> str:
    """
    Find parsed requirements relevant to this chunk via keyword overlap.
    Returns a formatted string ready for injection into the system prompt,
    or empty string if nothing relevant is found.
    """
    if not parsed_requirements:
        return ""
    chunk_lower = chunk.lower()
    scored: list[tuple[int, dict]] = []
    for req in parsed_requirements:
        keywords = (
            req.get("action", "") + " "
            + req.get("constraint", "") + " "
            + req.get("expected_behavior", "")
        ).lower().split()
        meaningful = [k for k in keywords if len(k) > 4]
        score = sum(1 for k in meaningful if k in chunk_lower)
        if score > 0:
            scored.append((score, req))
    scored.sort(key=lambda x: -x[0])
    top = [r for _, r in scored[:max_reqs]]
    if not top:
        return ""
    lines = [
        f"[{r.get('req_id','')} | {r.get('priority','')}] "
        f"Actor: {r.get('actor','')} | "
        f"Action: {r.get('action','')} | "
        f"Expected: {r.get('expected_behavior','')}"
        for r in top
    ]
    return "\n".join(lines)


def _focused_query(chunk: str) -> str:
    return chunk[:300].strip()


def _dedup_chunks(chunks: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for c in chunks:
        key = c[:80].strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(c)
    return result


def _parallel_kb_query(query_text: str, category_hint: str | None = None):
    """
    Query both KB collections concurrently. Returns (req_hits, tc_hits).

    The TC thread also calls find_nearest_req_doc so it can filter test-case
    examples to those linked to the most similar stored requirement doc.
    Both lookups run in parallel — zero extra wall-clock cost.
    """
    if not CHROMA_AVAILABLE:
        return [], []

    def _req():
        try:
            return query_kb(query_text, "requirements", n_results=2)
        except Exception:
            return []

    def _tc():
        try:
            source_req_doc = find_nearest_req_doc(query_text)
            return query_kb(
                query_text, "testcases", n_results=2,
                category_hint=category_hint,
                source_req_doc=source_req_doc,
            )
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_req, f_tc = ex.submit(_req), ex.submit(_tc)
        return f_req.result() or [], f_tc.result() or []


# ─────────────────────────────────────────────────────────────
# Per-case confidence scoring
# ─────────────────────────────────────────────────────────────

_VAGUE_EXPECTED = {"success", "pass", "ok", "works", "correct", "passed", "done"}
_GENERIC_NAMES  = {"test", "test case", "n/a", "tbd", "todo", "untitled"}


def _score_test_case(tc: dict) -> dict:
    """
    Score a single test case 0–100.
    Returns {tc_name, score, issues, needs_review}
    """
    score = 100
    issues: list[str] = []

    name = str(tc.get("Test Case Name", "")).strip()
    if not name:
        score -= 20; issues.append("Missing test case name")
    elif len(name) < 8:
        score -= 10; issues.append(f"Name too short: '{name}'")
    elif name.lower() in _GENERIC_NAMES:
        score -= 15; issues.append("Generic placeholder name")

    desc = str(tc.get("Description", "")).strip()
    if not desc or len(desc) < 10:
        score -= 15; issues.append("Missing or too brief description")

    steps = str(tc.get("Test Step Description", "")).strip()
    if not steps:
        score -= 25; issues.append("Missing test steps")
    elif not re.search(r'\d+[\.\)]', steps):
        score -= 10; issues.append("Steps are not numbered")
    elif len(steps) < 20:
        score -= 12; issues.append("Steps too vague")

    expected = str(tc.get("Test Step Expected Result", "")).strip()
    if not expected:
        score -= 25; issues.append("Missing expected result")
    elif expected.lower() in _VAGUE_EXPECTED:
        score -= 15; issues.append("Expected result not verifiable")
    elif len(expected) < 15:
        score -= 10; issues.append("Expected result too vague")

    precond = str(tc.get("Precondition", "")).strip()
    if not precond or len(precond) < 5:
        score -= 5; issues.append("Missing precondition")

    final = max(0, score)
    return {
        "tc_name": name,
        "score": final,
        "issues": issues,
        "needs_review": final < 60,
    }


# ─────────────────────────────────────────────────────────────
# Risk scoring & ordering
# ─────────────────────────────────────────────────────────────

_RISK_CATEGORY = {"Security": 10, "Negative": 9, "Boundary": 7, "Performance": 6, "Functional": 5}
_RISK_LEVEL    = {"System": 10, "Acceptance": 9, "Integration": 7, "Unit": 5}


def _risk_score(tc: dict) -> int:
    return (
        _RISK_CATEGORY.get(tc.get("Test Category", "Functional"), 5)
        + _RISK_LEVEL.get(tc.get("Test Level", "Integration"), 5)
    )


# ─────────────────────────────────────────────────────────────
# Semantic deduplication
# ─────────────────────────────────────────────────────────────

def _semantic_dedup(cases: list, conf_scores: list) -> list:
    """
    Remove near-duplicate test cases using embedding cosine distance.
    When two cases are duplicates, keep the one with the higher confidence score.
    Falls back to name-based dedup if ChromaDB is unavailable.
    """
    if not cases:
        return cases

    if not CHROMA_AVAILABLE or len(cases) <= 1:
        return _name_dedup(cases)

    try:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.Client()
        embed_fn = embedding_functions.DefaultEmbeddingFunction()
        coll_name = f"dedup_{uuid.uuid4().hex[:8]}"
        coll = client.create_collection(coll_name, embedding_function=embed_fn)

        docs = [
            f"{tc.get('Test Case Name', '')} {tc.get('Description', '')}"[:200]
            for tc in cases
        ]
        coll.add(documents=docs, ids=[f"tc_{i}" for i in range(len(cases))])

        conf_lookup = {s["tc_name"]: s["score"] for s in (conf_scores or [])}
        to_remove: set[int] = set()

        for i in range(len(cases)):
            if i in to_remove:
                continue
            results = coll.query(
                query_texts=[docs[i]],
                n_results=min(6, len(cases)),
                include=["distances"],
            )
            for sid, dist in zip(
                results.get("ids", [[]])[0],
                results.get("distances", [[]])[0],
            ):
                j = int(sid.replace("tc_", ""))
                if j != i and j not in to_remove and dist < 0.28:
                    name_i = cases[i].get("Test Case Name", "")
                    name_j = cases[j].get("Test Case Name", "")
                    if conf_lookup.get(name_i, 50) >= conf_lookup.get(name_j, 50):
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        try:
            client.delete_collection(coll_name)
        except Exception:
            pass

        return [tc for i, tc in enumerate(cases) if i not in to_remove]

    except Exception:
        return _name_dedup(cases)


def _name_dedup(cases: list) -> list:
    seen: set[str] = set()
    result: list[dict] = []
    for tc in cases:
        key = str(tc.get("Test Case Name", "")).strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(tc)
        elif not key:
            result.append(tc)
    return result


# ─────────────────────────────────────────────────────────────
# Node 1 — Document Parser Agent
# ─────────────────────────────────────────────────────────────

def document_parser_node(state: GraphState) -> dict:
    """Chunks spec, builds RAG vector store, and extracts structured requirements."""
    log = list(state.get("log", []))
    text = state["requirements_text"]
    fast_mode = state.get("fast_mode", False)
    model_name = state.get("model_name")

    log.append("📄 [Document Parser Agent] Analyzing specification...")

    chunk_size = state.get("chunk_size", 4000)
    chunks = smart_split(text, chunk_size=chunk_size, chunk_overlap=200)
    if fast_mode:
        chunks = chunks[:3]

    log.append(f"   ✅ {len(chunks)} semantic chunks created")

    doc_id = hashlib.md5(text.encode()).hexdigest()
    collection = None

    if CHROMA_AVAILABLE:
        log.append("   📦 Building in-memory vector store...")
        collection = build_vector_store(chunks, doc_id)
        log.append(f"   ✅ Vector store ready — {len(chunks)} chunks embedded")
    else:
        log.append("   ⚠  ChromaDB unavailable — RAG disabled")

    # ── Requirement extraction ──
    log.append("   🔍 Extracting structured requirements...")
    parsed_requirements: list = []
    try:
        parsed_requirements = parse_requirements(text, model_name)
        log.append(f"   ✅ {len(parsed_requirements)} testable assertions identified")
    except Exception as e:
        log.append(f"   ⚠  Requirement parsing failed: {str(e)[:60]}")

    log.append("   🏁 Document Parser complete\n")
    return {
        "chunks": chunks,
        "collection": collection,
        "doc_id": doc_id,
        "parsed_requirements": parsed_requirements,
        "log": log,
    }


# ─────────────────────────────────────────────────────────────
# Node 2 — Test Generator Agent
# ─────────────────────────────────────────────────────────────

def test_generator_node(state: GraphState) -> dict:
    """Generates test cases per chunk, enriched with RAG + dual KB context."""
    log = list(state.get("log", []))
    chunks = state.get("chunks", [])
    collection = state.get("collection")
    model_name = state.get("model_name")
    generated_names = list(state.get("generated_names", []))
    existing = list(state.get("all_test_cases", []))

    parsed_requirements = state.get("parsed_requirements", [])
    log.append(f"🤖 [Test Generator Agent] Processing {len(chunks)} chunk(s)...")

    fields = TAB_SCHEMAS["Test Cases"]
    new_cases = list(existing)
    temperature = state.get("temperature", 0.1)

    for i, chunk in enumerate(chunks):
        try:
            # ── Focused query + parallel KB (category-aware) ──
            query_text = _focused_query(chunk)
            category_hint = _infer_category(query_text)

            doc_hits: list[str] = []
            if collection is not None:
                doc_hits = retrieve_similar_chunks(collection, query_text, n_results=2)

            req_kb_hits, tc_kb_hits = _parallel_kb_query(query_text, category_hint)

            req_context    = "\n---\n".join(_dedup_chunks(doc_hits + req_kb_hits))
            tc_context     = "\n---\n".join(_dedup_chunks(tc_kb_hits))
            req_assertions = _find_relevant_reqs(chunk, parsed_requirements)

            # ── ETA hint ──
            avg = _timing.get_avg_seconds("generate_chunk")
            eta_str = f" | ETA ~{(len(chunks) - i - 1) * avg:.0f}s" if (avg and i < len(chunks) - 1) else ""
            log.append(f"   📝 Chunk {i + 1}/{len(chunks)}{eta_str}")

            # ── Build prompt ──
            system_prompt = make_system_prompt(
                fields,
                req_context=req_context,
                tc_context=tc_context,
                req_assertions=req_assertions,
            )
            if generated_names:
                system_prompt += (
                    "\n\nAlready generated — avoid ALL duplicates:\n"
                    + ", ".join(generated_names[-12:])
                )
            prompt = system_prompt + "\n\n" + chunk

            # ── LLM call (with retry + rate-limit backoff) ──
            t0 = time.time()
            raw = _llm_invoke(model_name, prompt, temperature=temperature)
            _timing.record_timing("generate_chunk", time.time() - t0)

            # ── Parse ──
            data = extract_json_from_response(raw, "Test Cases", fields)
            rows = data.get("Test Cases", [])
            non_empty = [r for r in rows if any(str(v).strip() for v in r.values())]

            new_cases.extend(non_empty)
            for row in non_empty:
                name = str(row.get("Test Case Name", "")).strip()
                if name:
                    generated_names.append(name)

            log.append(f"      ✅ {len(non_empty)} test case(s) extracted")

        except Exception as exc:
            log.append(f"      ❌ Chunk {i + 1} error: {str(exc)[:80]}")

    log.append(f"   🎯 Generator done — {len(new_cases)} total cases\n")
    return {"all_test_cases": new_cases, "generated_names": generated_names, "log": log}


# ─────────────────────────────────────────────────────────────
# Node 3 — Reviewer Agent
# ─────────────────────────────────────────────────────────────

def reviewer_node(state: GraphState) -> dict:
    """Scores quality at batch level + per-case confidence. Routes to finalizer or refiner."""
    log = list(state.get("log", []))
    rows = state.get("all_test_cases", [])
    iteration = state.get("review_iterations", 0) + 1

    log.append(f"🔍 [Reviewer Agent] Quality review — pass {iteration}...")

    if not rows:
        log.append("   ❌ No test cases to review\n")
        return {
            "quality_score": 0, "quality_valid": False,
            "quality_issues": ["No test cases generated"],
            "confidence_scores": [],
            "review_iterations": iteration, "log": log,
        }

    # ── Per-case confidence scores ──
    confidence_scores = [_score_test_case(tc) for tc in rows]
    needs_review_count = sum(1 for s in confidence_scores if s["needs_review"])
    avg_confidence = round(sum(s["score"] for s in confidence_scores) / max(len(confidence_scores), 1))

    # ── Batch quality score (field penalty) ──
    weights = {
        "Test Case Name": 15,
        "Description": 10,
        "Test Step Description": 20,
        "Test Step Expected Result": 20,
    }
    issues: list[str] = []
    penalty = 0.0
    n = max(len(rows), 1)

    for i, row in enumerate(rows):
        for field, w in weights.items():
            val = str(row.get(field, "")).strip()
            if not val or val.lower() in ("none", "n/a", "tbd"):
                issues.append(f"Row {i + 1}: empty {field}")
                penalty += w / n
        name = str(row.get("Test Case Name", "")).strip()
        if 0 < len(name) < 5:
            issues.append(f"Row {i + 1}: name too short ('{name}')")
            penalty += 3 / n

    score = max(0, min(100, round(100 - penalty)))
    empty_ratio = len([x for x in issues if "empty" in x]) / max(n * len(weights), 1)
    valid = score >= 72 and empty_ratio < 0.20

    log.append(
        f"   📊 Batch score: {score}/100  |  Cases: {len(rows)}"
        f"  |  Avg confidence: {avg_confidence}/100"
        f"  |  Needs review: {needs_review_count}"
    )
    for issue in issues[:3]:
        log.append(f"      ⚠  {issue}")

    if valid:
        log.append("   ✅ Quality approved — proceeding to Finalizer\n")
    else:
        log.append(f"   🔄 Below threshold — routing to Refiner (pass {iteration})\n")

    try:
        _timing.record_timing("validate_quality", 0.05)
    except Exception:
        pass

    return {
        "quality_score": score,
        "quality_valid": valid,
        "quality_issues": issues[:6],
        "confidence_scores": confidence_scores,
        "review_iterations": iteration,
        "log": log,
    }


# ─────────────────────────────────────────────────────────────
# Node 4 — Refiner Agent
# ─────────────────────────────────────────────────────────────

def refiner_node(state: GraphState) -> dict:
    """Detects coverage gaps, generates additional test cases, then re-reviews."""
    log = list(state.get("log", []))
    current = list(state.get("all_test_cases", []))
    collection = state.get("collection")
    model_name = state.get("model_name")
    chunks = state.get("chunks", [])
    generated_names = list(state.get("generated_names", []))
    quality_issues = state.get("quality_issues", [])

    conf_scores   = state.get("confidence_scores", [])
    temperature   = state.get("temperature", 0.1)
    log.append("⚙️  [Refiner Agent] Analyzing coverage gaps...")

    current_names = [str(r.get("Test Case Name", "")).strip() for r in current if r.get("Test Case Name")]
    fields = TAB_SCHEMAS["Test Cases"]

    # ── Step 1: Rewrite low-confidence cases (score < 50) ──────
    conf_lookup = {s["tc_name"]: s for s in conf_scores}
    weak_indices = [
        i for i, tc in enumerate(current)
        if conf_lookup.get(tc.get("Test Case Name", ""), {}).get("score", 100) < 50
    ][:3]  # cap at 3 rewrites per pass to keep latency reasonable

    if weak_indices:
        log.append(f"   🔧 Rewriting {len(weak_indices)} low-confidence test case(s)...")
        for idx in weak_indices:
            tc = current[idx]
            name = tc.get("Test Case Name", "")
            issues = conf_lookup.get(name, {}).get("issues", [])
            try:
                improve_raw = _llm_invoke(model_name, make_improve_prompt(tc, issues, fields), temperature=temperature)
                data = extract_json_from_response(improve_raw, "Test Cases", fields)
                new_rows = data.get("Test Cases", [])
                if new_rows and any(str(v).strip() for v in new_rows[0].values()):
                    current[idx] = new_rows[0]
                    log.append(f"      ✅ Rewrote: '{name[:40]}'")
            except Exception as exc:
                log.append(f"      ⚠  Could not rewrite '{name[:30]}': {str(exc)[:50]}")

    # ── Step 2: Gap analysis for missing scenarios ──────────────
    try:
        t0 = time.time()
        gap_text = make_gap_prompt(current_names, quality_issues)
        missing_raw = _llm_invoke(model_name, gap_text, temperature=temperature)
        _timing.record_timing("refinement", time.time() - t0)

        missing_text = (
            missing_raw.content if hasattr(missing_raw, "content") else str(missing_raw)
        )
        log.append("   🔎 Gaps identified — generating additional scenarios...")

        # ── Context for gap filling ──
        gap_query = _focused_query(missing_text)
        category_hint = _infer_category(gap_query)

        doc_hits: list[str] = []
        if collection:
            doc_hits = retrieve_similar_chunks(collection, gap_query, n_results=3)
        elif chunks:
            doc_hits = [chunks[0][:1500]]

        req_kb_hits, tc_hits = _parallel_kb_query(gap_query, category_hint)

        req_context = "\n---\n".join(_dedup_chunks(doc_hits + req_kb_hits))
        tc_context  = "\n---\n".join(_dedup_chunks(tc_hits))

        system_prompt = make_system_prompt(fields, req_context=req_context, tc_context=tc_context)
        if generated_names:
            system_prompt += (
                "\n\nAlready generated — NO duplicates allowed:\n"
                + ", ".join(generated_names[-15:])
            )
        refine_prompt = (
            system_prompt
            + f"\n\nGenerate test cases ONLY for these missing scenarios:\n{missing_text[:800]}"
        )

        raw2 = _llm_invoke(model_name, refine_prompt, temperature=temperature)
        data = extract_json_from_response(raw2, "Test Cases", fields)
        new_rows = data.get("Test Cases", [])
        non_empty = [r for r in new_rows if any(str(v).strip() for v in r.values())]

        current.extend(non_empty)
        for row in non_empty:
            name = str(row.get("Test Case Name", "")).strip()
            if name:
                generated_names.append(name)

        log.append(f"   ✅ {len(non_empty)} additional case(s) added\n")

    except Exception as exc:
        log.append(f"   ❌ Refiner error: {str(exc)[:80]}\n")

    return {"all_test_cases": current, "generated_names": generated_names, "log": log}


# ─────────────────────────────────────────────────────────────
# Node 5 — Finalizer
# ─────────────────────────────────────────────────────────────

def finalizer_node(state: GraphState) -> dict:
    """
    1. Semantic deduplication (keeps higher-confidence case)
    2. Attach confidence score to each case (_confidence field)
    3. Risk-based ordering (Security/Negative first)
    4. Requirement coverage report
    """
    log = list(state.get("log", []))
    all_cases = state.get("all_test_cases", [])
    conf_scores = state.get("confidence_scores", [])
    parsed_reqs = state.get("parsed_requirements", [])

    log.append("🧹 [Finalizer] Semantic deduplication + risk ordering...")

    # ── Semantic dedup ──
    unique = _semantic_dedup(all_cases, conf_scores)
    removed = len(all_cases) - len(unique)
    if removed:
        log.append(f"   🗑  Removed {removed} near-duplicate case(s)")

    # ── Attach confidence score to each case ──
    conf_lookup = {s["tc_name"]: s["score"] for s in conf_scores}
    for tc in unique:
        name = str(tc.get("Test Case Name", "")).strip()
        tc["_confidence"] = conf_lookup.get(name, 80)

    # ── Risk-based ordering ──
    unique.sort(key=_risk_score, reverse=True)
    log.append(f"   ✅ {len(unique)} unique test cases — ordered by risk priority")

    # ── Stamp sequential Test Case IDs (after sort so order is stable) ──
    for i, tc in enumerate(unique):
        tc["Test Case ID"] = f"TC-{i + 1:03d}"

    # ── Priority fallback — fill blank priority from Test Category ──
    _CAT_PRIORITY = {
        "Security": "High", "Negative": "Medium", "Boundary": "Medium",
        "Performance": "Medium", "Functional": "Low",
    }
    for tc in unique:
        if not str(tc.get("Priority", "")).strip():
            tc["Priority"] = _CAT_PRIORITY.get(tc.get("Test Category", "Functional"), "Medium")

    # ── Requirement coverage ──
    coverage_report: dict = {}
    if parsed_reqs:
        log.append("   📊 Computing requirement coverage...")
        try:
            coverage_report = compute_coverage(parsed_reqs, unique)
            pct = coverage_report.get("coverage_pct", 0)
            total = coverage_report.get("total_requirements", 0)
            gaps = len(coverage_report.get("risk_gaps", []))
            log.append(
                f"   📈 Coverage: {pct}% ({len(coverage_report.get('covered_ids',[]))}/{total} requirements)"
            )
            if gaps:
                log.append(f"   ⚠  {gaps} Critical/High requirement(s) have no test cases")
        except Exception as e:
            log.append(f"   ⚠  Coverage computation failed: {str(e)[:60]}")

    # ── Stamp Requirement IDs from coverage mapping ──
    if coverage_report.get("mapping"):
        tc_to_reqs: dict[str, list] = {}
        for req_id, tc_names in coverage_report["mapping"].items():
            for n in tc_names:
                tc_to_reqs.setdefault(n, []).append(req_id)
        for tc in unique:
            n = str(tc.get("Test Case Name", "")).strip()
            linked = tc_to_reqs.get(n, [])
            if linked:
                tc["Requirement ID"] = ", ".join(linked[:3])

    log.append("\n🎯 All agents complete!")
    return {
        "final_test_cases": unique,
        "coverage_report": coverage_report,
        "log": log,
    }


# ─────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────

def _route_after_review(state: GraphState) -> str:
    max_iter = state.get("max_iterations", 2)
    if state.get("quality_valid", False) or state.get("review_iterations", 0) >= max_iter:
        return "finalize"
    return "refine"


# ─────────────────────────────────────────────────────────────
# Build & compile
# ─────────────────────────────────────────────────────────────

def _build_graph():
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph is required. Run: pip install 'langgraph>=0.2.0'")
    g = StateGraph(GraphState)
    g.add_node("document_parser", document_parser_node)
    g.add_node("test_generator", test_generator_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("refiner", refiner_node)
    g.add_node("finalizer", finalizer_node)

    g.set_entry_point("document_parser")
    g.add_edge("document_parser", "test_generator")
    g.add_edge("test_generator", "reviewer")
    g.add_conditional_edges(
        "reviewer", _route_after_review,
        {"finalize": "finalizer", "refine": "refiner"},
    )
    g.add_edge("refiner", "reviewer")
    g.add_edge("finalizer", END)
    return g.compile()


# ─────────────────────────────────────────────────────────────
# Progress bar helpers
# ─────────────────────────────────────────────────────────────

_PROG_STEPS = [
    ("📄", "Parse"),
    ("🤖", "Generate"),
    ("🔍", "Review"),
    ("⚙️", "Refine"),
    ("🧹", "Finalize"),
]


def _progress_from_log(log_lines: list) -> tuple:
    """Scan log in reverse to determine (pct, step_label, detail)."""
    if not log_lines:
        return 2, "Starting", "Initializing pipeline..."
    for line in reversed(log_lines):
        ls = line.strip()
        if "🎯 All agents complete" in ls:
            return 100, "Complete", "All agents complete!"
        if "🧹 [Finalizer]" in ls:
            return 88, "Finalize", "Finalizing & deduplicating..."
        if "⚙️" in ls and "Refiner" in ls:
            return 72, "Refine", "Refining test cases..."
        if "✅ Quality approved" in ls or "🔄 Below threshold" in ls:
            return 63, "Review", "Review done — routing next step..."
        if "🔍 [Reviewer" in ls:
            return 55, "Review", "Reviewing quality..."
        if "🎯 Generator done" in ls:
            return 50, "Generate", "Test cases generated"
        if "📝 Chunk" in ls:
            m = re.search(r"Chunk (\d+)/(\d+)", ls)
            if m:
                cur, tot = int(m.group(1)), int(m.group(2))
                pct = 22 + int(cur / max(tot, 1) * 26)
                return pct, "Generate", f"Generating — chunk {cur}/{tot}"
        if "🤖 [Test Generator" in ls:
            return 22, "Generate", "Generating test cases..."
        if "🏁 Document Parser complete" in ls:
            return 20, "Parse", "Document parsed"
        if "📄 [Document Parser" in ls:
            return 8, "Parse", "Parsing document..."
    return 2, "Starting", "Starting pipeline..."


def _render_progress_html(pct: int, step_label: str, detail: str) -> str:
    dots_html = []
    for i, (icon, label) in enumerate(_PROG_STEPS):
        step_done  = label == step_label
        step_past  = False
        # determine order
        labels = [s[1] for s in _PROG_STEPS]
        try:
            cur_i = labels.index(step_label)
        except ValueError:
            cur_i = -1
        is_done    = i < cur_i
        is_active  = i == cur_i
        is_pending = i > cur_i

        if is_done:
            dot_bg  = "#22c55e"; dot_fc = "white"; icon_show = "✓"; lc = "#22c55e"
        elif is_active:
            dot_bg  = "#3b82f6"; dot_fc = "white"; icon_show = icon; lc = "#93c5fd"
        else:
            dot_bg  = "#1e293b"; dot_fc = "#475569"; icon_show = icon; lc = "#475569"

        dots_html.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;gap:3px;">'
            f'<div style="width:30px;height:30px;border-radius:50%;background:{dot_bg};'
            f'display:flex;align-items:center;justify-content:center;font-size:13px;color:{dot_fc};">'
            f'{icon_show}</div>'
            f'<span style="font-size:10px;color:{lc};white-space:nowrap;">{label}</span>'
            f'</div>'
        )
        if i < len(_PROG_STEPS) - 1:
            lc2 = "#22c55e" if i < cur_i else "#1e293b"
            dots_html.append(
                f'<div style="flex:1;height:2px;background:{lc2};margin-top:14px;"></div>'
            )

    if pct >= 100:
        bar_style = "background:#22c55e;width:100%;"
    else:
        bar_style = (
            "background:linear-gradient(90deg,#1d4ed8 0%,#6366f1 60%,#1d4ed8 100%);"
            f"width:{pct}%;background-size:200% 100%;"
            "animation:qa_shimmer 1.6s linear infinite;"
        )

    return (
        "<style>@keyframes qa_shimmer{"
        "0%{background-position:200% 0}100%{background-position:-200% 0}}"
        "</style>"
        '<div style="background:#0f172a;border-radius:12px;padding:14px 18px;margin:6px 0;">'
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:12px;">'
        + "".join(dots_html) +
        '</div>'
        '<div style="background:#1e293b;border-radius:6px;height:8px;overflow:hidden;">'
        f'<div style="{bar_style}height:100%;border-radius:6px;transition:width 0.5s ease;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        f'<span style="color:#94a3b8;font-size:11px;font-family:monospace;">⚡ {detail}</span>'
        f'<span style="color:#64748b;font-size:11px;font-family:monospace;">{pct}%</span>'
        '</div>'
        '</div>'
    )


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def run_agent_graph(
    requirements_text: str,
    log_placeholder,
    fast_mode: bool = False,
    model_name: str | None = None,
    chunk_size: int = 4000,
    max_iterations: int = 2,
    temperature: float = 0.1,
    progress_placeholder=None,
) -> dict:
    """
    Execute the full multi-agent pipeline.
    Streams log + progress updates after each node.

    Returns:
      {
        "Test Cases"          : list of test case dicts (with _confidence field),
        "_quality_score"      : int,
        "_chunks_count"       : int,
        "_review_iterations"  : int,
        "_coverage_report"    : dict,
        "_confidence_scores"  : list,
        "_parsed_requirements": list,
      }
    """
    compiled = _build_graph()

    initial: GraphState = {
        "requirements_text": requirements_text,
        "model_name": model_name,
        "fast_mode": fast_mode,
        "chunk_size": chunk_size,
        "max_iterations": max_iterations,
        "temperature": temperature,
        "chunks": [],
        "collection": None,
        "doc_id": "",
        "parsed_requirements": [],
        "all_test_cases": [],
        "generated_names": [],
        "quality_score": 0,
        "quality_valid": False,
        "quality_issues": [],
        "review_iterations": 0,
        "confidence_scores": [],
        "final_test_cases": [],
        "coverage_report": {},
        "log": ["🚀 Multi-Agent Pipeline starting...\n"],
    }

    # Show initial progress
    if progress_placeholder is not None:
        progress_placeholder.markdown(
            _render_progress_html(2, "Starting", "Initializing pipeline..."),
            unsafe_allow_html=True,
        )

    final_state: dict = initial  # type: ignore[assignment]
    for snapshot in compiled.stream(initial, stream_mode="values"):
        final_state = snapshot
        log_lines = snapshot.get("log", [])
        if log_placeholder is not None and log_lines:
            log_placeholder.code("\n".join(log_lines))
        if progress_placeholder is not None and log_lines:
            pct, step_lbl, detail = _progress_from_log(log_lines)
            progress_placeholder.markdown(
                _render_progress_html(pct, step_lbl, detail),
                unsafe_allow_html=True,
            )

    # Final 100% state
    if progress_placeholder is not None:
        progress_placeholder.markdown(
            _render_progress_html(100, "Complete", "All agents complete!"),
            unsafe_allow_html=True,
        )

    return {
        "Test Cases":              final_state.get("final_test_cases", []),
        "_quality_score":          final_state.get("quality_score", 0),
        "_chunks_count":           len(final_state.get("chunks", [])),
        "_review_iterations":      final_state.get("review_iterations", 0),
        "_coverage_report":        final_state.get("coverage_report", {}),
        "_confidence_scores":      final_state.get("confidence_scores", []),
        "_parsed_requirements":    final_state.get("parsed_requirements", []),
    }


# ─────────────────────────────────────────────────────────────
# Full Test Strategy content generator (one extra LLM call)
# ─────────────────────────────────────────────────────────────

def generate_strategy_content(
    requirements_text: str,
    model_name: str | None = None,
) -> dict:
    """
    One post-pipeline LLM call that extracts project-level metadata and
    test scenarios from the requirements doc for the 7-tab strategy workbook.

    Returns a dict with keys:
      project_name, jira_refs, feature_summary, bot_activities, test_scenarios
    """
    import json as _json

    fallback = {
        "project_name": "BOT Test Documentation",
        "jira_refs": "",
        "feature_summary": "See requirements document for full feature details.",
        "bot_activities": "1. Verify configuration\n2. Run bill flow\n3. Validate reports",
        "test_scenarios": [],
    }

    try:
        prompt = make_strategy_prompt(requirements_text)
        raw = _llm_invoke(model_name, prompt, temperature=0.1, max_retries=2)
        text = raw.content if hasattr(raw, "content") else str(raw)

        # Strip any markdown fences
        text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()

        # Find JSON object boundaries
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return fallback

        data = _json.loads(text[start:end])

        # Validate required keys
        for key in ("project_name", "jira_refs", "feature_summary", "bot_activities", "test_scenarios"):
            if key not in data:
                data[key] = fallback[key]

        # Ensure test_scenarios is a list of dicts with required keys
        cleaned_scenarios = []
        for scen in data.get("test_scenarios", []):
            if isinstance(scen, dict):
                cleaned_scenarios.append({
                    "sl_no":          str(scen.get("sl_no", "")),
                    "scenario_name":  str(scen.get("scenario_name", "")),
                    "expected_result": str(scen.get("expected_result", "")),
                })
        data["test_scenarios"] = cleaned_scenarios
        return data

    except Exception:
        return fallback


# ─────────────────────────────────────────────────────────────
# Backward-compatibility shims
# ─────────────────────────────────────────────────────────────

class AgentState:
    def __init__(self, collection, generated_names=None, log_list=None):
        self.collection = collection
        self.generated_names = generated_names or []
        self.log_list = log_list or []


def validate_test_case_quality(rows: list) -> dict:
    if not rows:
        return {"valid": False, "issues": ["No test cases"], "score": 0, "total_rows": 0}
    scores = [_score_test_case(r) for r in rows]
    avg = round(sum(s["score"] for s in scores) / max(len(scores), 1))
    issues = [i for s in scores for i in s["issues"]]
    return {"valid": avg >= 70, "issues": issues[:5], "score": avg, "total_rows": len(rows)}
