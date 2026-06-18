# QA Test Cases Agent ⚡

Streamlit app that reads a product/engineering specification and generates structured QA test cases, exported as a styled Excel workbook.

Powered by **Google Gemini**, **LangGraph multi-agent**, and **ChromaDB RAG**.

---

## How it works

Five specialized agents run in a LangGraph `StateGraph`:

```
Document Parser → Test Generator → Reviewer ──→ Finalizer
                                       ↑↓ (up to 2 passes)
                                    Refiner
```

| Agent | What it does |
|---|---|
| **Document Parser** | Chunks spec, builds RAG vector store, extracts structured requirements (Actor/Action/Constraint/Expected/Priority) |
| **Test Generator** | Generates test cases per chunk — parallel KB queries, category-aware retrieval, focused 300-char embedding query |
| **Reviewer** | Batch quality score (0–100) + per-case confidence score per test case, flags weak cases |
| **Refiner** | Detects coverage gaps via LLM gap analysis, generates missing scenarios |
| **Finalizer** | Semantic deduplication (embedding similarity), risk-based ordering, requirement coverage report |

### What the output includes
- **Excel workbook** — all test cases, risk-ordered (Security/Negative first)
- **Gherkin .feature file** — BDD format (Given/When/Then) grouped by Test Category
- **Coverage dashboard** — which requirements are covered, uncovered, and Critical/High gaps
- **Per-case confidence** — every test case scored 0–100 with specific issues flagged
- **Risk gaps** — Critical/High priority requirements with zero test coverage highlighted

### Knowledge Base (RAG)
Two separate persistent ChromaDB collections — both injected into the generator's context on every run:

| Collection | What to upload | How it's used |
|---|---|---|
| **Requirements** | Spec docs, PRDs, user stories (PDF/DOCX/TXT) | Domain accuracy — ensures generated cases match your domain |
| **Test Cases** | Previous test case Excel sheets (XLSX/XLS) | Output format & style reference — LLM mirrors your naming conventions, field structure, and detail level |

---

## Project structure

```
Test_Cases/
├── app.py                  — Streamlit UI entry point
├── requirements.txt        — Python dependencies
├── .env.example            — API key template
├── .gitignore
├── cache/
│   └── timing_cache.json   — Auto-generated ETA cache (safe to delete)
│
├── modules/
│   ├── agent.py            — LangGraph StateGraph (all 5 agent nodes)
│   ├── pipeline.py         — Thin shim: calls agent.run_agent_graph
│   ├── config.py           — LLM factory, model list, schema definitions
│   ├── extraction.py       — PDF / DOCX / TXT text extraction
│   ├── chunking.py         — Structure-aware text splitting
│   ├── rag.py              — ChromaDB (in-memory per-doc + persistent KB)
│   ├── prompts.py          — System prompt builders per agent role
│   ├── json_parser.py      — Robust JSON extraction from LLM responses
│   ├── excel_builder.py    — Styled Excel workbook generation
│   ├── utils.py            — UI helpers (editable preview, file hash)
│   ├── warmup.py           — LLM health check + latency measurement
│   └── timing.py           — Persistent timing cache for ETA estimates
│
└── scripts/
    └── pin_requirements.py — Print installed package versions
```

---

## Setup

### Requirements
- Python 3.10+
- Google API key → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Install

```bash
cd Test_Cases
pip install -r requirements.txt
```

### Configure

Create a `.env` file (copy from `.env.example`):

```
GOOGLE_API_KEY=your_key_here
```

### Run

```bash
python -m streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Usage

1. **Select model** in the sidebar (default: `gemini-2.5-flash`)
2. **Upload spec** — PDF, DOCX, DOC, or TXT
3. Optionally add to the **Knowledge Base** via the sidebar:
   - **Requirements** section → upload past spec/PRD docs (PDF, DOCX, TXT)
   - **Previous Test Cases** section → upload existing test case Excel sheets (XLSX, XLS)
4. Click **Run Pipeline** — watch the 5 agents execute with live log
5. **Review / edit** test cases in the editable table
6. **Download** the Excel workbook

### Fast mode
Tick **Fast mode** in pipeline settings to process only the first 3 chunks — useful for quick previews on large documents.

---

## Output fields

Each generated test case has 8 fields:

| Field | Description |
|---|---|
| Test Case Name | Short, action-oriented title |
| Description | What is being tested |
| Precondition | Required system state before execution |
| Test Step Description | Numbered steps |
| Test Step Expected Result | Concrete, verifiable outcome |
| Test Method | Manual / Automated / Semi-Automated |
| Test Level | Unit / Integration / System / Acceptance |
| Test Category | Functional / Negative / Boundary / Performance / Security |

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `langchain-google-genai` | Google Gemini API |
| `langchain-core` | LangChain base types |
| `langchain-text-splitters` | Smart text chunking |
| `langgraph` | Multi-agent StateGraph orchestration |
| `chromadb` | Vector embeddings (RAG) |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX/DOC text extraction |
| `openpyxl` | Excel workbook read/write (test case output + KB upload) |
| `pandas` | DataFrame preview |
| `openpyxl` | Excel workbook generation + KB Excel upload |
| `python-dotenv` | Load `.env` API key |

---

## Local / Ollama models

If `ollama` is installed and running, the sidebar will show available local models automatically. Select one to run fully offline.

---

## Troubleshooting

**Warm-up failed** — Check that `GOOGLE_API_KEY` is set correctly and the key has Gemini API access enabled.

**No test cases extracted** — The spec may be too short or the LLM response didn't parse. Try a larger or more structured document.

**ChromaDB errors** — Delete the `chroma_db/` folder at the project root and restart.

**Slow generation** — Enable Fast mode (3 chunks max) for a quick preview.
