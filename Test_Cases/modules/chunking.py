"""
Chunking module: Smart text splitting respecting document structure
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def smart_split(text: str, chunk_size: int = 4000, chunk_overlap: int = 0) -> list[str]:
    """
    Smart chunking: respects document structure.
    First tries Markdown header splitting to keep sections intact.
    Falls back to recursive character splitting if that fails.
    """
    # Attempt header-aware split
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "H1"), ("##", "H2"), ("###", "H3"),
        ],
        strip_headers=False,
    )
    
    try:
        md_chunks = header_splitter.split_text(text)
        if len(md_chunks) > 1:
            # Further split large sections
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            final = []
            for doc in md_chunks:
                content = doc.page_content
                if len(content) > chunk_size:
                    final.extend(char_splitter.split_text(content))
                else:
                    final.append(content)
            return [c for c in final if c.strip()]
    except Exception:
        pass

    # Fallback: requirement-boundary aware splitting
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\nREQ-", "\n#", "\n##", "\n-", "\n*", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def split_excel_rows(text: str) -> list[str]:
    """
    Split Excel-extracted text into per-row chunks.
    Each row from extraction is already a complete semantic unit (pipe-separated fields).
    Sheet header lines (=== Sheet: ... ===) are dropped.
    """
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("===")
    ]
