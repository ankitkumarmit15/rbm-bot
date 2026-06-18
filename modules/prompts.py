"""
Prompts: build system prompts for each agent role.
"""

import json
from .config import TAB_SCHEMAS


def make_system_prompt(
    fields: list,
    req_context: str = "",
    tc_context: str = "",
    req_assertions: str = "",
) -> str:
    """
    System prompt for the Test Generator Agent.

    req_context    : RAG chunks from requirements KB + per-doc vector store
    tc_context     : RAG chunks from test-cases KB (style/format reference)
    req_assertions : structured assertions from req_parser (Actor/Action/Expected/Priority)
                     — direct targets the generator must cover
    """
    example = {field: f"<{field}>" for field in fields}
    example_json = json.dumps({"Test Cases": [example]}, indent=2)

    extra = ""
    if req_assertions.strip():
        extra += (
            "\n\n## Structured requirements — generate test cases that DIRECTLY cover each assertion below:\n"
            + req_assertions.strip()
        )
    if req_context.strip():
        extra += (
            "\n\n## Related context — use for domain accuracy, do NOT duplicate:\n"
            + req_context.strip()
        )
    if tc_context.strip():
        extra += (
            "\n\n## Example test cases — use as format/style reference only:\n"
            + tc_context.strip()
        )

    return (
        "You are a senior QA analyst specialising in telecom and billing systems.\n"
        "Read the provided specification and generate high-quality, executable test cases.\n\n"
        "Rules:\n"
        "- Output ONLY a JSON object with key 'Test Cases' and an array of objects.\n"
        "- Use EXACTLY these fields in this order: "
        + ", ".join(fields)
        + ".\n"
        "- One row per distinct testable requirement or scenario.\n"
        "- Populate ALL fields — never leave core fields blank.\n"
        "- Test Case ID: leave BLANK — the system auto-assigns it.\n"
        "- Test Case Name: short, action-oriented (e.g. 'Verify MSISDN Change Fee Applied').\n"
        "- Priority: Critical | High | Medium | Low — based on test risk.\n"
        "  (Security/data-integrity tests → High or Critical; happy-path Functional → Low or Medium)\n"
        "- Test Method: Manual | Automated | Semi-Automated.\n"
        "- Test Level: Unit | Integration | System | Acceptance.\n"
        "- Test Category: Functional | Negative | Boundary | Performance | Security.\n"
        "- Precondition: realistic system state before the test.\n"
        "- Test Step Description: numbered steps (1. ... 2. ...).\n"
        "- Test Step Expected Result: concrete, verifiable outcome.\n"
        "- Requirement ID: leave BLANK — the system auto-links it.\n"
        "- Do NOT output markdown fences, explanations, or extra keys.\n"
        "- Do NOT reproduce the example placeholder values below.\n\n"
        f"Example output format:\n{example_json}"
        + extra
    )


def make_gap_prompt(current_names: list, quality_issues: list) -> str:
    """Prompt for the Refiner Agent's gap-analysis step."""
    names_block = "\n".join(f"- {n}" for n in current_names[:15])
    issues_block = "\n".join(f"- {i}" for i in quality_issues[:4])
    return (
        "You are a senior QA analyst. Existing test case names:\n"
        + names_block
        + ("\n\nKnown quality issues:\n" + issues_block if issues_block else "")
        + "\n\nList 3-5 SPECIFIC test scenarios that are MISSING. Focus on:\n"
        "- Edge / boundary values\n"
        "- Negative / error paths\n"
        "- Security and permission checks\n"
        "- Performance under sustained load\n"
        "- Cross-module integration points\n"
        "Be specific about what to test."
    )


def make_strategy_prompt(requirements_text: str) -> str:
    """
    Prompt that extracts project-level metadata and test scenarios
    from a requirements document for the Full Test Strategy workbook.
    """
    return (
        "You are a senior QA analyst reading a requirements/specification document.\n"
        "Extract the following information and return ONLY a JSON object — no markdown, no explanation.\n\n"
        "Required JSON structure:\n"
        "{\n"
        '  "project_name": "short project / feature name (e.g. CAP-16627: Wi-Fi Auth Fee)",\n'
        '  "jira_refs": "JIRA ticket IDs found in the doc (e.g. CAP-16627 | WDTB-2173, WDTB-2259)",\n'
        '  "feature_summary": "2-4 sentence summary of what the feature does and its purpose",\n'
        '  "bot_activities": "numbered list of BOT testing activities (1. Verify config...\\n2. Run bill flow...)",\n'
        '  "test_scenarios": [\n'
        '    {"sl_no": "1", "scenario_name": "Top-level scenario name", "expected_result": "Expected outcome"},\n'
        '    {"sl_no": "1.1", "scenario_name": "Sub-scenario name", "expected_result": "Expected outcome"},\n'
        '    {"sl_no": "2", "scenario_name": "Another top-level scenario", "expected_result": "Expected outcome"}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- project_name: extract from document title, header, or first JIRA reference\n"
        "- jira_refs: list all JIRA/CAP ticket IDs found (CAP-XXXXX, WDTB-XXXX format)\n"
        "- test_scenarios: generate 15-30 scenarios covering all major functional areas\n"
        "  Use hierarchical numbering: top-level (1, 2, 3...) and sub-scenarios (1.1, 1.2...)\n"
        "  Cover: Configuration, Integration, Billing/Invoicing, Reporting, Error/Negative cases\n"
        "- Do NOT output markdown fences or any text outside the JSON\n\n"
        "REQUIREMENTS DOCUMENT:\n"
        + requirements_text[:12000]
    )


def make_improve_prompt(weak_tc: dict, issues: list, fields: list) -> str:
    """
    Prompt for the Refiner Agent to rewrite a specific low-confidence test case.
    Preserves the original scenario intent but fixes every listed quality issue.
    """
    # Strip internal keys before showing to LLM
    display_tc = {k: v for k, v in weak_tc.items() if not k.startswith("_")}
    tc_json = json.dumps(display_tc, indent=2)
    issues_text = "\n".join(f"- {i}" for i in issues)
    example = {field: f"<{field}>" for field in fields}
    example_json = json.dumps({"Test Cases": [example]}, indent=2)

    return (
        "You are a senior QA analyst. The following test case has quality issues that must be fixed.\n\n"
        f"ORIGINAL TEST CASE:\n{tc_json}\n\n"
        f"QUALITY ISSUES TO FIX:\n{issues_text}\n\n"
        "Rewrite this test case to resolve ALL listed issues. Rules:\n"
        "- Keep the same test scenario intent — do NOT change what is being tested\n"
        "- Number all test steps (1. ... 2. ...)\n"
        "- Make the expected result concrete and verifiable (not 'success' or 'pass')\n"
        "- Write a meaningful description (at least one sentence)\n"
        "- Include a realistic precondition\n"
        "- Populate ALL fields completely\n\n"
        f"Output ONLY a JSON object: {example_json}"
    )
