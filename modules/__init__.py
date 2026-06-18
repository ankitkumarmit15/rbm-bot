"""
QA Test Cases Agent — modules package
──────────────────────────────────────
Multi-agent test case generation powered by LangGraph + Google Gemini.

Architecture (LangGraph StateGraph):
  DocumentParserAgent → TestGeneratorAgent → ReviewerAgent ↻ RefinerAgent → Finalizer

Modules:
  config          — LLM factory, model constants, schema definitions
  extraction      — File parsing (PDF, DOCX, TXT)
  chunking        — Structure-aware text splitting
  rag             — ChromaDB vector store (in-memory + persistent knowledge base)
  prompts         — System prompt builders for each agent role
  json_parser     — Robust JSON extraction from LLM responses
  agent           — LangGraph StateGraph (all 5 agent nodes + graph compiler)
  pipeline        — Thin orchestration shim (calls agent.run_agent_graph)
  excel_builder   — Professional Excel workbook generation
  utils           — UI helpers (preview, file hash)
  warmup          — LLM health check + latency measurement
  timing          — Persistent timing cache for ETA estimates
"""

__version__ = "2.0.0"

try:
    from .config import TAB_SCHEMAS, CHROMA_AVAILABLE
    from .extraction import extract_text_from_file
    from .chunking import smart_split
    from .pipeline import execute_pipeline_with_agent
except ImportError:
    pass

__all__ = [
    "TAB_SCHEMAS",
    "CHROMA_AVAILABLE",
    "extract_text_from_file",
    "smart_split",
    "execute_pipeline_with_agent",
]
