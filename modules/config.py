"""
Configuration module: LLM settings, constants, and schemas
"""

import os
import time
import subprocess
import urllib.request
import json as _json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

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
        name = model.split("ollama:", 1)[1]
        return OllamaAdapter(name, temperature=temperature)

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
    """
    Adapter for local Ollama models.

    Uses the Ollama HTTP REST API (http://localhost:11434/api/generate) which
    keeps the model hot in RAM between calls — eliminates the cold-start model
    reload that killed performance when using subprocess.

    Falls back to subprocess only if the HTTP call fails.
    """
    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

    def _invoke_http(self, prompt: str) -> str:
        payload = _json.dumps({
            "model":      self.model_name,
            "prompt":     prompt,
            "stream":     False,
            "keep_alive": "15m",           # keep model warm in RAM for 15 minutes
            "options": {
                "temperature": self.temperature,
                "num_ctx":     8192,        # generous context window for chunked prompts
                "num_predict": 4096,        # max tokens to generate
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")

    def _invoke_subprocess(self, prompt: str) -> str:
        cp = subprocess.run(
            ["ollama", "run", self.model_name, prompt],
            capture_output=True, text=True, check=True, timeout=600,
        )
        return cp.stdout.strip()

    def invoke(self, prompt_or_payload):
        if isinstance(prompt_or_payload, dict):
            prompt = (
                prompt_or_payload.get("source_document")
                or prompt_or_payload.get("prompt")
                or ""
            )
        else:
            prompt = str(prompt_or_payload)

        # Try HTTP first (keeps model loaded, no process spawn overhead)
        try:
            return self._invoke_http(prompt)
        except Exception:
            pass

        # Fallback: subprocess
        try:
            return self._invoke_subprocess(prompt)
        except Exception as e:
            return f"Ollama invoke error: {e}"

# ─────────────────────────────────────────────
# Shared LLM invocation with retry + rate-limit backoff
# ─────────────────────────────────────────────

_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "quota", "rate limit", "ratequota")
_TRANSIENT_MARKERS  = _RATE_LIMIT_MARKERS + ("503", "500 internal", "timeout", "unavailable")


def llm_invoke(
    model_name: str | None,
    prompt: str,
    temperature: float = 0.1,
    max_retries: int = 2,
):
    """
    Invoke the LLM with automatic retry on rate-limit or transient errors.

    Centralised here so every agent (generator, refiner, req_parser, strategy)
    gets identical retry / backoff behaviour without duplicating code.
    Raises the last exception when all retries are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return get_llm(model_name, temperature=temperature).invoke(prompt)
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            is_rate_limit = any(m in err for m in _RATE_LIMIT_MARKERS)
            is_transient  = any(m in err for m in _TRANSIENT_MARKERS)
            if attempt < max_retries and is_transient:
                wait = (10 if is_rate_limit else 3) * (2 ** attempt)
                time.sleep(wait)
                continue
            break
    raise last_exc  # type: ignore[misc]


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
