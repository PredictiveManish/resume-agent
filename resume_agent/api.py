"""FastAPI app: the /tailor API plus the web UI served at /.

Run:
    uvicorn resume_agent.api:app --reload
    # then open http://127.0.0.1:8000

Set RESUME_AGENT_MOCK=1 to run in demo mode (no API key needed).
"""

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import tailor

logger = logging.getLogger("resume_agent.api")


def _configure_logging() -> None:
    """Human-readable logs with timestamps. Override with RESUME_AGENT_LOG_LEVEL=DEBUG."""
    level_name = os.getenv("RESUME_AGENT_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


_configure_logging()

app = FastAPI(title="Resume Tailor Agent", version="0.2.0")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.exception_handler(Exception)
async def _log_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log the FULL traceback for anything the
    endpoints didn't catch themselves (response serialization, etc.)."""
    logger.exception(
        "%s %s -> unhandled %s: %s",
        request.method, request.url.path, type(exc).__name__, str(exc) or "(no message)",
    )
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc) or type(exc).__name__}"},
    )


def _mock_mode() -> bool:
    return bool(os.getenv("RESUME_AGENT_MOCK"))


def _build_llm():
    """Return a MockLLM in demo mode, else None (pipeline builds the real client)."""
    if _mock_mode():
        from .llm import MockLLM
        from .demo import DEMO_RESPONSES

        return MockLLM(DEMO_RESPONSES)
    return None


@app.get("/health")
def health():
    return {"status": "ok", "service": "resume-tailor", "mock": _mock_mode()}


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
    source = resume_file.filename if using_file else "text"
    logger.info(
        "/tailor start: resume=%s (chars=%d) jd_chars=%d proofs=%d mode=%s",
        source, len(resume_text or ""), len(jd_text or ""), len(proof_urls),
        "mock" if llm else os.getenv("SARVAM_MODEL", "sarvam-105b"),
    )
    started = time.monotonic()
    try:
        if using_file:
            suffix = os.path.splitext(resume_file.filename)[1] or ".tex"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await resume_file.read())
                path = tmp.name
            logger.info("/tailor: upload saved to temp file %s", path)
            result = tailor(path, jd_text, proof_urls=proof_urls, llm=llm)
        elif resume_text and resume_text.strip():
            result = tailor(resume_text, jd_text, proof_urls=proof_urls, llm=llm, is_text=True)
        else:
            logger.warning("/tailor rejected (400): no resume file or text provided")
            raise HTTPException(status_code=400, detail="Provide a resume file or resume text.")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        # Logs the exception type, message, AND the full traceback at the
        # exact line that failed. This is where you find the real cause.
        logger.exception("/tailor FAILED: %s: %s", type(e).__name__, str(e) or "(no message)")
        raise HTTPException(status_code=500, detail=str(e) or type(e).__name__)
    logger.info(
        "/tailor done in %.1fs: score=%s changes=%d",
        time.monotonic() - started, result.match.score, len(result.changes),
    )
    return result.to_dict()


# Serve the single-page UI at "/" (registered AFTER the API routes).
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
