"""FastAPI app: the /tailor API, an optional /compile endpoint, and the web UI.

Run:
    uvicorn resume_agent.api:app --reload      # then open http://127.0.0.1:8000
Set RESUME_AGENT_MOCK=1 to run in demo mode (no API key needed).
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .pipeline import tailor

logger = logging.getLogger("resume_agent.api")


def _configure_logging() -> None:
    level_name = os.getenv("RESUME_AGENT_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


_configure_logging()

app = FastAPI(title="Resume Tailor Agent", version="0.3.0")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.exception_handler(Exception)
async def _log_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("%s %s -> unhandled %s", request.method, request.url.path, type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc) or type(exc).__name__}"},
    )


def _mock_mode() -> bool:
    return bool(os.getenv("RESUME_AGENT_MOCK"))


def _build_llm():
    if _mock_mode():
        from .llm import MockLLM
        from .demo import DEMO_RESPONSES

        return MockLLM(DEMO_RESPONSES)
    return None


def _pdflatex_available() -> bool:
    return shutil.which("pdflatex") is not None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "resume-tailor",
        "mock": _mock_mode(),
        "pdf": _pdflatex_available(),
    }


@app.post("/tailor")
async def tailor_endpoint(
    jd_text: str = Form(..., description="The job description text"),
    resume_text: Optional[str] = Form(None, description="Resume as raw text"),
    resume_file: Optional[UploadFile] = File(None, description="Resume file (.tex/.pdf/.txt)"),
    proofs: Optional[str] = Form(None, description="Comma-separated proof URLs"),
):
    proof_urls = [u.strip() for u in (proofs or "").split(",") if u.strip()]
    llm = _build_llm()
    using_file = resume_file is not None and bool(resume_file.filename)
    logger.info(
        "/tailor start: resume=%s (chars=%d) jd_chars=%d proofs=%d",
        resume_file.filename if using_file else "text",
        len(resume_text or ""), len(jd_text or ""), len(proof_urls),
    )
    started = time.monotonic()
    try:
        if using_file:
            suffix = os.path.splitext(resume_file.filename)[1] or ".tex"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await resume_file.read())
                path = tmp.name
            result = tailor(path, jd_text, proof_urls=proof_urls, llm=llm)
        elif resume_text and resume_text.strip():
            result = tailor(resume_text, jd_text, proof_urls=proof_urls, llm=llm, is_text=True)
        else:
            raise HTTPException(status_code=400, detail="Provide a resume file or resume text.")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("/tailor FAILED: %s: %s", type(e).__name__, str(e) or "(no message)")
        raise HTTPException(status_code=500, detail=str(e) or type(e).__name__)
    logger.info(
        "/tailor done in %.1fs: ats %d->%d changes=%d cost=₹%.4f",
        time.monotonic() - started, result.match.score_before, result.match.score,
        len(result.changes), result.cost_inr,
    )
    return result.to_dict()


@app.post("/compile")
async def compile_tex(payload: dict):
    """Compile LaTeX to PDF (optional; only if pdflatex is installed).

    Shell-escape is disabled, so the LaTeX cannot run system commands.
    """
    tex = (payload or {}).get("tex") or ""
    if not tex.strip():
        raise HTTPException(status_code=400, detail="No LaTeX provided.")
    if not _pdflatex_available():
        raise HTTPException(
            status_code=503,
            detail="LaTeX is not installed on the server. Use the .tex download and Overleaf.",
        )
    with tempfile.TemporaryDirectory() as d:
        tex_path = os.path.join(d, "resume.tex")
        with open(tex_path, "w", encoding="utf-8") as fh:
            fh.write(tex)
        try:
            subprocess.run(
                ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                 "-halt-on-error", "resume.tex"],
                cwd=d, capture_output=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="LaTeX compilation timed out.")
        pdf_path = os.path.join(d, "resume.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=500, detail="LaTeX failed to produce a PDF.")
        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()
    return Response(content=pdf_bytes, media_type="application/pdf")


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
