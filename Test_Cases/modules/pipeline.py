"""
Pipeline: thin orchestration layer.
Delegates all work to the LangGraph multi-agent system in agent.py.
"""

from .agent import run_agent_graph
from .json_parser import normalize_test_case_rows


def execute_pipeline_with_agent(
    requirements_text: str,
    log_placeholder,
    fast_mode: bool = False,
    selected_tabs: list | None = None,
    model_name: str | None = None,
    chunk_size: int = 4000,
    max_iterations: int = 2,
    temperature: float = 0.1,
    progress_placeholder=None,
) -> dict:
    """
    Run the full multi-agent pipeline.
    Returns {"Test Cases": [...], "_quality_score": N, "_chunks_count": N, "_review_iterations": N}.
    `selected_tabs` is accepted for API compatibility but is always ["Test Cases"].
    """
    return run_agent_graph(
        requirements_text=requirements_text,
        log_placeholder=log_placeholder,
        fast_mode=fast_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        max_iterations=max_iterations,
        temperature=temperature,
        progress_placeholder=progress_placeholder,
    )


def validate_test_case_rows(rows: list) -> tuple:
    """Normalize and validate test case rows. Returns (normalized_rows, has_content)."""
    normalized = normalize_test_case_rows(rows)
    has_content = any(any(v for v in row.values()) for row in normalized)
    return normalized, has_content
