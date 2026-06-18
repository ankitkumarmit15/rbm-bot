"""
Print installed versions of all project dependencies.
Usage: python scripts/pin_requirements.py
"""
import importlib.metadata as m

PACKAGES = [
    "streamlit",
    "pandas",
    "pypdf",
    "python-docx",
    "openpyxl",
    "langchain-google-genai",
    "langchain-core",
    "langchain-text-splitters",
    "langgraph",
    "chromadb",
    "python-dotenv",
]

for pkg in PACKAGES:
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except m.PackageNotFoundError:
        print(f"# NOT INSTALLED: {pkg}")
