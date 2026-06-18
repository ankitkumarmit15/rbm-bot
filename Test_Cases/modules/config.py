"""
Configuration module: LLM settings, constants, and schemas
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import subprocess
import shlex

load_dotenv()  # loads GOOGLE_API_KEY / GROQ_API_KEY from a .env file if present

# ─────────────────────────────────────────────
# Groq availability
# ─────────────────────────────────────────────
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_API_KEY_SET = bool(os.environ.get("GROQ_API_KEY"))

GROQ_MODELS = [
    "llama-3.3-70b-versatile",   # best quality, great for complex test cases
    "llama-3.1-8b-instant",      # fastest, good for quick runs
    "mixtral-8x7b-32768",        # long context (32k tokens)
    "gemma2-9b-it",              # Google Gemma via Groq
]

# ─────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────
# Requires GOOGLE_API_KEY env var (or .env file) to be set
# Get a key at: https://aistudio.google.com/apikey
DEFAULT_MODEL = "gemini-2.5-flash"
# NOTE: Many model identifiers are experimental or depend on your Google account/API-version.
# Keep a minimal, known-safe list by default to avoid confusing 'NOT_FOUND' errors.
SUPPORTED_MODELS = [
    DEFAULT_MODEL,
]


def get_llm(model_name: str | None = None, temperature: float = 0.1):
    """Factory that returns the right LLM for the requested model.
    Supports Google Gemini (default), Groq (groq: prefix), and Ollama (ollama: prefix).
    """
    model = model_name or DEFAULT_MODEL
    # Ollama adapter if model_name explicitly references an Ollama model
    if isinstance(model, str) and model.startswith("ollama:"):
        # lazy: OllamaAdapter is defined later in this module
        name = model.split("ollama:", 1)[1]
        return OllamaAdapter(name)

    # Groq LPU adapter
    if isinstance(model, str) and model.startswith("groq:"):
        if not GROQ_AVAILABLE:
            raise ImportError(
                "langchain-groq is not installed. Run: pip install langchain-groq"
            )
        groq_model = model.split("groq:", 1)[1]
        return ChatGroq(
            model=groq_model,
            api_key=os.environ.get("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=8192,
        )
    # If the user explicitly requested a local model, use a local Llama adapter.
    # Otherwise, remote Gemini requires GOOGLE_API_KEY on Streamlit Cloud.
    use_local = False
    if model and isinstance(model, str) and model.startswith("local:"):
        use_local = True

    if not os.environ.get("GOOGLE_API_KEY") and not use_local:
        raise RuntimeError(
            "GOOGLE_API_KEY is required for remote Gemini on Streamlit Cloud. "
            "Set GOOGLE_API_KEY in Streamlit secrets or .env, or use a local model with 'local:<model>'."
        )

    if use_local:
        # Prefer llama-cpp-python if available
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise ImportError("Local Llama requested but 'llama_cpp' is not installed: " + str(e))

        class LocalLlamaAdapter:
            def __init__(self, model_path: str | None = None, temperature: float = 0.1, max_tokens: int = 2048):
                self.model_path = model_path or os.environ.get("LLAMA_MODEL_PATH") or "models/ggml-model.bin"
                self.temperature = temperature
                self.max_tokens = max_tokens
                self._client = Llama(model_path=self.model_path)

            def invoke(self, prompt_or_payload):
                # Accept either a raw string prompt or a dict with 'source_document'
                if isinstance(prompt_or_payload, dict):
                    prompt = prompt_or_payload.get("source_document") or prompt_or_payload.get("prompt") or ""
                else:
                    prompt = str(prompt_or_payload)

                resp = self._client.create(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                # llama-cpp-python returns a dict with choices/text
                try:
                    return resp["choices"][0]["text"]
                except Exception:
                    # some versions return 'choices' as list of dicts with 'text'
                    try:
                        return resp.choices[0].text
                    except Exception:
                        return str(resp)

        return LocalLlamaAdapter()

    # Default: return Google Gemini LLM wrapper
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=temperature,
        max_output_tokens=8192,
    )


def _detect_ollama() -> bool:
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


OLLAMA_AVAILABLE = _detect_ollama()


def list_ollama_models() -> list[str]:
    """Return a list of Ollama model names (e.g. 'neural-chat:7b')."""
    if not OLLAMA_AVAILABLE:
        return []
    try:
        cp = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = cp.stdout.splitlines()
        models = []
        # Skip header lines until a line starting with NAME or similar
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Expect lines like: NAME ID SIZE MODIFIED
            parts = line.split()
            # If the first token contains a colon (model:tag) assume it's a model line
            first = parts[0]
            if ":" in first:
                models.append(first)
        return models
    except Exception:
        return []


class OllamaAdapter:
    """Adapter that calls the `ollama` CLI to generate text from a local model."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, prompt_or_payload):
        if isinstance(prompt_or_payload, dict):
            prompt = prompt_or_payload.get("source_document") or prompt_or_payload.get("prompt") or ""
        else:
            prompt = str(prompt_or_payload)

        try:
            # Use ollama run <model> "prompt"
            # Use shlex to ensure prompt is passed as single arg if needed
            cp = subprocess.run(["ollama", "run", self.model_name, prompt], capture_output=True, text=True, check=True)
            return cp.stdout.strip()
        except Exception as e:
            return f"Ollama invoke error: {e}"

# Backwards-compat: expose a module-level `llm` using the default model
llm = get_llm(DEFAULT_MODEL)

# ─────────────────────────────────────────────
# ChromaDB / RAG Availability
# ─────────────────────────────────────────────
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# ─────────────────────────────────────────────
# Tab Schemas (Field definitions for output)
# ─────────────────────────────────────────────
TAB_SCHEMAS = {
    "Test Cases": [
        "Test Case ID", "Test Case Name", "Priority", "Description", "Precondition",
        "Test Step Description", "Test Step Expected Result",
        "Test Method", "Test Level", "Test Category", "Requirement ID",
    ],
}
