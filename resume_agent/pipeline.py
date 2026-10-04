"""The agent pipeline: resume + JD (+ proofs) -> score, preview, and a .tex file."""

import re
from typing import List, Optional, Tuple

from . import latex
from .config import Settings
from .ingest import ingest
from .llm import MockLLM, SarvamClient, extract_json
from .models import (
    ChangeItem,
    MatchReport,
    PipelineResult,
    ProofResult,
    ResumeData,
)
from .proof import check_proofs

ANALYZE_SYSTEM = """You are an expert technical recruiter and resume analyst.
Given a candidate's RESUME text and a JOB DESCRIPTION, return STRICT JSON only:
{
  "jd_profile": {"role_title": str, "keywords": [str], "requirements": [str]},
  "match": {"matched_keywords": [str], "missing_keywords": [str], "gaps": [str], "suggestions": [str]}
}
Rules:
- keywords: 15-30 concrete skills, tools, or qualifications the JD emphasises.
- matched_keywords: subset present in the resume.
- missing_keywords: subset absent but relevant.
- gaps: specific weaknesses of THIS resume for THIS JD.
- suggestions: concrete, truthful improvements (never suggest inventing anything).
Return only the JSON object."""

REWRITE_SYSTEM = """You are an expert resume writer who tailors a resume to one job.
HARD RULES:
- NEVER invent experience, employers, dates, metrics, skills, or links. Only reframe
  and reorganise what the candidate actually has. Do not fabricate numbers.
- Emphasise the skills the JD wants that the candidate genuinely has.
- Keep the content to roughly one page. Use strong action verbs.
Given the original resume, the job description, the identified gaps, and any VERIFIED
proof links, return STRICT JSON only:
{
 "resume": {"name": str, "email": str, "phone": str, "location": str, "links": [str],
            "summary": str, "skills": {group: [items]},
            "experience": [{"title": str, "org": str, "dates": str, "location": str, "bullets": [str]}],
            "projects": [{"name": str, "link": str, "tech": str, "bullets": [str]}],
            "education": [{"degree": str, "institution": str, "dates": str, "details": str}]},
 "changes": [{"section": str, "before": str, "after": str, "reason": str}]
}
Return only the JSON object."""


def compute_score(resume_text: str, keywords: List[str]) -> Tuple[int, dict, List[str], List[str]]:
    """Deterministic, reproducible JD-match score (a heuristic, not an official ATS score)."""
    text = (resume_text or "").lower()
    kws = [k for k in keywords if k and k.strip()]
    matched = [k for k in kws if k.lower() in text]
    missing = [k for k in kws if k.lower() not in text]
    coverage = len(matched) / max(1, len(kws))

    fmt = 100
    if not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", resume_text or ""):
        fmt -= 25
    if not re.search(r"(\+?\d[\d\s-]{7,})", resume_text or ""):
        fmt -= 15
    low = text
    for sec in ("experience", "education", "skill", "project"):
        if sec not in low:
            fmt -= 10
    fmt = max(0, fmt)

    numbers = len(re.findall(r"\b\d+(?:[%+.]\d+)?\b", resume_text or ""))
    impact = min(100, numbers * 8)

    score = round(0.55 * coverage * 100 + 0.25 * fmt + 0.20 * impact)
    subscores = {
        "keyword_coverage": round(coverage * 100),
        "formatting": fmt,
        "quantified_impact": impact,
    }
    return score, subscores, matched, missing


def _format_proofs(proofs: List[ProofResult]) -> str:
    if not proofs:
        return "(none provided)"
    return "\n".join(
        f"- {p.url} -> {'OK' if p.ok else 'FAILED'} {p.detail}".strip() for p in proofs
    )


def tailor(
    resume_path_or_text: str,
    jd_text: str,
    proof_urls: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
    llm=None,
    is_text: bool = False,
) -> PipelineResult:
    """Run the full pipeline. Pass `llm` to inject a mock for testing."""
    settings = settings or Settings.load()
    proof_urls = proof_urls or []

    resume = ingest(resume_path_or_text, is_text=is_text)
    proofs = check_proofs(proof_urls) if proof_urls else []

    if llm is None:
        llm = SarvamClient(settings)

    # --- Call 1: analyse (JD profile + match) -----------------------------
    analyze_user = (
        f"RESUME:\n{resume.text}\n\nJOB DESCRIPTION:\n{jd_text}\n\n"
        "Return the JSON object."
    )
    a = extract_json(llm.chat(ANALYZE_SYSTEM, analyze_user, task="ANALYZE", want_json=True))
    jd_profile = a.get("jd_profile", {})
    match_block = a.get("match", {})
    keywords = jd_profile.get("keywords", []) or []

    score, subscores, matched, missing = compute_score(resume.text, keywords)
    # Prefer the model's matched/missing if it provided them; else use computed.
    matched = match_block.get("matched_keywords") or matched
    missing = match_block.get("missing_keywords") or missing

    # --- Call 2: rewrite (structured resume + change list) ----------------
    rewrite_user = (
        f"ORIGINAL RESUME:\n{resume.text}\n\n"
        f"JOB DESCRIPTION:\n{jd_text}\n\n"
        f"IDENTIFIED GAPS:\n- " + "\n- ".join(match_block.get("gaps", []) or ["(none)"]) + "\n\n"
        f"VERIFIED PROOF LINKS:\n{_format_proofs(proofs)}\n\n"
        "Return the JSON object."
    )
    r = extract_json(llm.chat(REWRITE_SYSTEM, rewrite_user, task="REWRITE", want_json=True))

    resume_data = ResumeData.from_dict(r.get("resume", {}))
    if not resume_data.links and proof_urls:
        resume_data.links = [p.url for p in proofs if p.ok]
    changes = [
        ChangeItem(
            section=c.get("section", ""),
            before=c.get("before", ""),
            after=c.get("after", ""),
            reason=c.get("reason", ""),
        )
        for c in r.get("changes", [])
    ]

    new_tex = latex.render_resume(resume_data)

    match = MatchReport(
        score=score,
        subscores=subscores,
        matched_keywords=list(matched),
        missing_keywords=list(missing),
        gaps=list(match_block.get("gaps", []) or []),
        suggestions=list(match_block.get("suggestions", []) or []),
    )

    return PipelineResult(
        match=match,
        changes=changes,
        proofs=proofs,
        resume_data=resume_data,
        new_tex=new_tex,
        old_text=resume.text,
        model=getattr(getattr(llm, "settings", None), "model", "mock"),
        input_format=resume.fmt,
    )
