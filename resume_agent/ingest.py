"""Resume ingestion: .tex, .pdf, or plain text -> raw text + stripped plain text."""

import os
import re
from dataclasses import dataclass


@dataclass
class IngestedResume:
    raw: str          # original content (LaTeX source, or extracted text)
    text: str         # plain, human-readable text for analysis/scoring
    fmt: str          # "tex" | "pdf" | "text"


def extract_pdf_text(path: str) -> str:
    """Extract text from a PDF using pdfplumber (falls back to pypdf)."""
    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts).strip()
    except ImportError:
        pass
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages).strip()
    except ImportError as e:
        raise RuntimeError(
            "No PDF library available. Install pdfplumber or pypdf."
        ) from e


_COMMAND_RE = re.compile(r"\\[a-zA-Z@]+\*?")
_BRACES_RE = re.compile(r"[{}]")


def strip_latex(tex: str) -> str:
    """Best-effort LaTeX -> readable text (for scoring, not for fidelity)."""
    s = tex
    s = re.sub(r"(?<!\\)%.*", "", s)              # comments
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\url\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(item|section|subsection|textbf|textit|emph|large|LARGE|bfseries|hfill|hspace|vspace)\*?", " ", s)
    s = _COMMAND_RE.sub(" ", s)
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    s = _BRACES_RE.sub(" ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def ingest(path_or_text: str, is_text: bool = False) -> IngestedResume:
    """Ingest a resume from a file path or a raw text string.

    Format is inferred from the extension when a path is given.
    """
    if is_text:
        return IngestedResume(raw=path_or_text, text=path_or_text, fmt="text")

    path = path_or_text
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume not found: {path}")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        text = extract_pdf_text(path)
        return IngestedResume(raw=text, text=text, fmt="pdf")

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()

    if ext in (".tex", ".latex"):
        return IngestedResume(raw=content, text=strip_latex(content), fmt="tex")

    return IngestedResume(raw=content, text=content, fmt="text")
