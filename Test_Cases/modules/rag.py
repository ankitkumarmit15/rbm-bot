"""
RAG module: ChromaDB vector store operations.

Two persistent KB collections:
  - kb_requirements_v1  : uploaded requirement / spec documents
  - kb_testcases_v1     : uploaded previous test-case Excel files

One ephemeral in-memory collection per pipeline run (per-document RAG).

Improvements:
  - Distance threshold filtering (score < 0.75 only)
  - Category-aware retrieval for test cases KB
  - Row-level metadata stored per chunk
"""

import re
from pathlib import Path

import streamlit as st

try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

_KB_DIR = Path(__file__).resolve().parent.parent / "chroma_db"

_KB_COLLECTIONS = {
    "requirements": "kb_requirements_v1",
    "testcases":    "kb_testcases_v1",
}

_DISTANCE_THRESHOLD = 0.75  # cosine distance — lower = more similar; ignore above this

# Keywords used to infer test category from a chunk of text
_CATEGORY_KEYWORDS = {
    "Security":    ["security", "auth", "permission", "role", "token", "credential",
                    "privilege", "access control", "xss", "injection", "encrypt"],
    "Negative":    ["invalid", "negative", "error", "fail", "reject", "exception",
                    "wrong", "missing", "corrupt", "bad request", "unauthoriz"],
    "Boundary":    ["boundary", "limit", "max", "min", "overflow", "threshold",
                    "edge case", "upper", "lower", "range"],
    "Performance": ["performance", "load", "stress", "latency", "throughput",
                    "concurren", "timeout", "response time", "spike"],
}

# Field name aliases for external test case formats (Jira, TestRail, etc.)
_TC_FIELD_ALIASES: dict[str, list[str]] = {
    "test_category": [
        "Test Category", "Category", "Type", "Issue Type", "Test Type",
        "Test Classification", "Classification", "Test Kind",
    ],
    "test_level": [
        "Test Level", "Level", "Test Phase", "Phase", "Test Layer",
        "Scope", "Testing Level", "Test Scope",
    ],
}


def _infer_category(text: str) -> str | None:
    """Return the most likely test category for a text snippet, or None."""
    lower = text.lower()
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return cat
    return None


def _extract_tc_metadata(row_text: str) -> dict:
    """
    Extract test_category and test_level from a row of text.

    Supports:
    - Our own pipe-separated format: "Test Category: Functional | ..."
    - External formats (Jira, TestRail): tries all known field name aliases
    - Keyword fallback: infers category from row content if no field matched
    """
    meta: dict = {}

    for meta_key, aliases in _TC_FIELD_ALIASES.items():
        for alias in aliases:
            m = re.search(rf"{re.escape(alias)}:\s*([^|,\n]+)", row_text, re.IGNORECASE)
            if m:
                meta[meta_key] = m.group(1).strip()
                break

    # Keyword fallback for test_category if no field matched
    if "test_category" not in meta:
        lower = row_text.lower()
        for cat, keywords in _CATEGORY_KEYWORDS.items():
            if any(k in lower for k in keywords):
                meta["test_category"] = cat
                break

    return meta


# ─────────────────────────────────────────────────────────────
# In-memory client (per-document, ephemeral)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_inmem_client():
    if not CHROMA_AVAILABLE:
        return None, None
    client = chromadb.Client()
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client, embed_fn


def build_vector_store(chunks: list, doc_id: str):
    """Embed document chunks into an in-memory ChromaDB collection."""
    if not CHROMA_AVAILABLE:
        return None
    client, embed_fn = _get_inmem_client()
    if client is None:
        return None
    coll_name = f"doc_{doc_id[:16]}"
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(name=coll_name, embedding_function=embed_fn)
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])
    return collection


def retrieve_similar_chunks(collection, query: str, n_results: int = 3) -> list:
    """
    Return semantically similar chunks from an in-memory collection.
    Filters out results with cosine distance >= _DISTANCE_THRESHOLD.
    """
    if collection is None:
        return []
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "distances"],
        )
        docs = results["documents"][0] if results.get("documents") else []
        dists = results["distances"][0] if results.get("distances") else []
        return [d for d, s in zip(docs, dists) if s < _DISTANCE_THRESHOLD]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Persistent Knowledge Base — internal helpers
# ─────────────────────────────────────────────────────────────

def _get_persistent_client():
    if not CHROMA_AVAILABLE:
        return None, None
    _KB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(_KB_DIR))
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    return client, embed_fn


def _coll_name(coll_type: str) -> str:
    return _KB_COLLECTIONS.get(coll_type, _KB_COLLECTIONS["requirements"])


# ─────────────────────────────────────────────────────────────
# Persistent Knowledge Base — public API
# ─────────────────────────────────────────────────────────────

def add_to_kb(
    chunks: list,
    doc_name: str,
    doc_id: str,
    coll_type: str = "requirements",
    source_req_doc: str | None = None,
) -> int:
    """
    Store chunks in a persistent KB collection.

    source_req_doc : name of the requirement doc these test cases were generated from.
                     Only meaningful when coll_type="testcases". Stored as metadata so
                     the generator can retrieve only test-case examples that are linked
                     to the most similar requirement doc.
    """
    client, embed_fn = _get_persistent_client()
    if client is None:
        return 0
    try:
        coll = client.get_or_create_collection(
            name=_coll_name(coll_type), embedding_function=embed_fn
        )
        try:
            existing = coll.get(where={"doc_id": doc_id})
            if existing.get("ids"):
                coll.delete(ids=existing["ids"])
        except Exception:
            pass

        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta: dict = {
                "doc_name": doc_name,
                "doc_id": doc_id,
                "chunk_idx": i,
                "coll_type": coll_type,
            }
            if coll_type == "testcases":
                meta.update(_extract_tc_metadata(chunk))
                if source_req_doc:
                    meta["source_req_doc"] = source_req_doc
            metadatas.append(meta)

        coll.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)
    except Exception:
        return 0


def get_kb_docs(coll_type: str = "requirements") -> list:
    """
    Return list of documents stored in the given KB collection.
    For testcases, also returns source_req_doc if set.
    """
    client, _ = _get_persistent_client()
    if client is None:
        return []
    try:
        coll = client.get_collection(_coll_name(coll_type))
        all_items = coll.get(include=["metadatas"])
        seen: dict = {}
        for meta in (all_items.get("metadatas") or []):
            if not meta:
                continue
            name = meta.get("doc_name", "unknown")
            did = meta.get("doc_id", "")
            if name not in seen:
                seen[name] = {
                    "name": name,
                    "id": did,
                    "chunks": 0,
                    "source_req_doc": meta.get("source_req_doc"),
                }
            seen[name]["chunks"] += 1
        return list(seen.values())
    except Exception:
        return []


def get_kb_doc_names(coll_type: str = "requirements") -> list[str]:
    """Return just the document names stored in a KB collection."""
    return [d["name"] for d in get_kb_docs(coll_type)]


def delete_from_kb(doc_id: str, coll_type: str = "requirements") -> bool:
    """Remove all chunks for a document from the given KB collection."""
    client, _ = _get_persistent_client()
    if client is None:
        return False
    try:
        coll = client.get_collection(_coll_name(coll_type))
        existing = coll.get(where={"doc_id": doc_id})
        if existing.get("ids"):
            coll.delete(ids=existing["ids"])
        return True
    except Exception:
        return False


def query_kb(
    query: str,
    coll_type: str = "requirements",
    n_results: int = 3,
    category_hint: str | None = None,
    source_req_doc: str | None = None,
) -> list:
    """
    Semantic search against a persistent KB collection.
    Filters by distance threshold.

    For testcases collection:
      - category_hint : filters by test_category metadata (e.g. "Security")
      - source_req_doc: filters by the requirement doc these test cases are linked to.
                        Falls back to unfiltered if the filtered result is empty.
    """
    client, embed_fn = _get_persistent_client()
    if client is None:
        return []
    try:
        coll = client.get_collection(_coll_name(coll_type))

        # Build where filter
        where: dict | None = None
        if coll_type == "testcases":
            filters = []
            if source_req_doc:
                filters.append({"source_req_doc": {"$eq": source_req_doc}})
            if category_hint:
                filters.append({"test_category": {"$eq": category_hint}})
            if len(filters) == 1:
                where = filters[0]
            elif len(filters) > 1:
                where = {"$and": filters}

        def _run_query(w):
            kw: dict = {
                "query_texts": [query],
                "n_results": n_results,
                "include": ["documents", "distances"],
            }
            if w:
                kw["where"] = w
            r = coll.query(**kw)
            docs  = r.get("documents", [[]])[0]
            dists = r.get("distances",  [[]])[0]
            return [d for d, s in zip(docs, dists) if s < _DISTANCE_THRESHOLD]

        results = _run_query(where)

        # Cascade fallback: linked+category → linked-only → unfiltered
        if not results and where and source_req_doc and category_hint:
            results = _run_query({"source_req_doc": {"$eq": source_req_doc}})
        if not results and where:
            results = _run_query(None)

        return results
    except Exception:
        return []


def find_nearest_req_doc(query: str) -> str | None:
    """
    Find the name of the most similar requirement doc in the KB for a given query.
    Used at generation time to auto-select linked test case examples.
    Returns None if nothing is close enough.
    """
    client, embed_fn = _get_persistent_client()
    if client is None:
        return None
    try:
        coll = client.get_collection(_coll_name("requirements"))
        results = coll.query(
            query_texts=[query],
            n_results=1,
            include=["metadatas", "distances"],
        )
        dists = results.get("distances", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if dists and metas and dists[0] < 0.62:  # tight threshold — only return a confident match
            return metas[0].get("doc_name")
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Backward-compatibility aliases
# ─────────────────────────────────────────────────────────────

def add_to_knowledge_base(chunks, doc_name, doc_id) -> int:
    return add_to_kb(chunks, doc_name, doc_id, "requirements")

def get_knowledge_base_docs() -> list:
    return get_kb_docs("requirements")

def delete_from_knowledge_base(doc_id) -> bool:
    return delete_from_kb(doc_id, "requirements")

def query_knowledge_base(query, n_results=3) -> list:
    return query_kb(query, "requirements", n_results)

def get_chroma_client():
    return _get_inmem_client()
