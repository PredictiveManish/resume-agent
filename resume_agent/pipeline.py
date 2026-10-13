"""The agent pipeline: resume + JD (+ proofs) -> ATS score, preview, .tex, cost.

Cost design:
  * Default is ONE consolidated call (analysis + rewrite together), which sends
    the resume and JD to the model once instead of twice.
  * Thinking mode is off (config), max_tokens is capped.
  * A two-pass mode (cheap model for analysis, main model for the rewrite) is
    available for higher quality via RESUME_AGENT_TWO_PASS=1.
  * Token usage is turned into an INR figure and returned with the result.
"""

import re
from typing import List, Optional, Tuple

from . import latex
from .config import Settings, estimate_cost_inr
from .ingest import ingest
from .llm import MockLLM, SarvamClient, extract_json
from .models import ChangeItem, MatchReport, PipelineResult, ProofResult, ResumeData
from .proof import check_proofs

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TAILOR_SYSTEM = """You are an expert technical recruiter AND resume writer, working as one.
Given a candidate's RESUME (plain text) and a JOB DESCRIPTION, produce ONE JSON object
that both (a) analyses the fit and (b) rewrites the resume to maximise the candidate's
chance of passing screening for THIS job.

HARD RULES — never break these:
- NEVER invent employers, dates, job titles, skills, metrics, or links. Only reframe,
  reorder, emphasise, and tighten what the candidate actually has. Do not fabricate numbers.
- Keep the rewritten content to roughly ONE page.
- Where the candidate genuinely has a skill the JD asks for, use the JD's own wording.

Return ONLY this JSON:
{
 "jd_profile": {"role_title": str, "keywords": [15-30 concrete skills/tools/qualifications the JD emphasises]},
 "match": {"matched_keywords": [str], "missing_keywords": [str],
           "gaps": [specific weaknesses of THIS resume for THIS JD],
           "suggestions": [concrete, truthful improvements]},
 "resume": {"name": str, "email": str, "phone": str, "location": str, "links": [str],
            "summary": str, "skills": {group: [items]},
            "experience": [{"title": str, "org": str, "dates": str, "location": str, "bullets": [str]}],
            "projects": [{"name": str, "link": str, "tech": str, "bullets": [str]}],
            "education": [{"degree": str, "institution": str, "dates": str, "details": str}]},
 "changes": [{"section": str, "before": str, "after": str, "reason": str}]
}
Bullet rules: start with a strong action verb; include a number ONLY if the source
supports it; keep each bullet under ~28 words. Return only the JSON object."""

ANALYZE_SYSTEM = """You are an expert technical recruiter. Given a RESUME and a JOB
DESCRIPTION, return STRICT JSON only:
{"jd_profile": {"role_title": str, "keywords": [15-30 concrete skills/tools/quals the JD emphasises]},
 "match": {"matched_keywords": [str], "missing_keywords": [str], "gaps": [str], "suggestions": [str]}}
Return only the JSON object."""

REWRITE_SYSTEM = """You are an expert resume writer who tailors a resume to ONE job.
HARD RULES: never invent experience, employers, dates, metrics, skills, or links — only
reframe and reorganise what the candidate has. Keep to ~one page. Use strong action verbs.
Given the original resume, the JD, the identified gaps, and any verified proof links,
return STRICT JSON only:
{"resume": {"name": str, "email": str, "phone": str, "location": str, "links": [str],
            "summary": str, "skills": {group: [items]},
            "experience": [{"title": str, "org": str, "dates": str, "location": str, "bullets": [str]}],
            "projects": [{"name": str, "link": str, "tech": str, "bullets": [str]}],
            "education": [{"degree": str, "institution": str, "dates": str, "details": str}]},
 "changes": [{"section": str, "before": str, "after": str, "reason": str}]}
Return only the JSON object."""


# ---------------------------------------------------------------------------
# Deterministic ATS score (a heuristic — labelled as such everywhere)
# ---------------------------------------------------------------------------

def compute_score(
    resume_text: str, keywords: List[str]
) -> Tuple[int, dict, List[str], List[str]]:
    """Reproducible JD-match score. Keyword coverage + formatting + quantified impact."""
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
    for sec in ("experience", "education", "skill", "project"):
        if sec not in text:
            fmt -= 10
    fmt = max(0, fmt)

    numbers = len(re.findall(r"\b\d+(?:[%+.]\d+)?\b", resume_text or ""))
    impact = min(100, numbers * 8)

    score = round(0.55 * coverage * 100 + 0.25 * fmt + 0.20 * impact)
    return score, {
        "keyword_coverage": round(coverage * 100),
        "formatting": fmt,
        "quantified_impact": impact,
    }, matched, missing


def _format_proofs(proofs: List[ProofResult]) -> str:
    if not proofs:
        return "(none provided)"
    return "\n".join(
        f"- {p.url} -> {'OK' if p.ok else 'FAILED'} {p.detail}".strip() for p in proofs
    )


def _chat(llm, system, user, task, model, **kw):
    """Call the LLM and always return (content, usage_dict)."""
    try:
        content, usage = llm.chat(
            system, user, task=task, want_json=True, model=model, return_usage=True, **kw
        )
        return content, usage
    except TypeError:
        # A client without usage support (older mock) — degrade gracefully.
        return llm.chat(system, user, task=task, want_json=True), {}


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

    proofs_txt = _format_proofs(proofs)
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
    calls = 0

    def _accumulate(usage: dict):
        nonlocal calls
        calls += 1
        for k in usage_total:
            usage_total[k] += int(usage.get(k, 0) or 0)

    if settings.two_pass:
        # Higher-quality mode: cheap model analyses, main model rewrites.
        a_user = f"RESUME:\n{resume.text}\n\nJOB DESCRIPTION:\n{jd_text}\n\nReturn the JSON object."
        a_content, a_usage = _chat(llm, ANALYZE_SYSTEM, a_user, "ANALYZE", settings.fast_model)
        _accumulate(a_usage)
        a = extract_json(a_content)
        jd_profile = a.get("jd_profile", {})
        match_block = a.get("match", {})

        r_user = (
            f"ORIGINAL RESUME:\n{resume.text}\n\nJOB DESCRIPTION:\n{jd_text}\n\n"
            f"IDENTIFIED GAPS:\n- " + "\n- ".join(match_block.get("gaps", []) or ["(none)"]) + "\n\n"
            f"VERIFIED PROOF LINKS:\n{proofs_txt}\n\nReturn the JSON object."
        )
        r_content, r_usage = _chat(llm, REWRITE_SYSTEM, r_user, "REWRITE", settings.model)
        _accumulate(r_usage)
        r = extract_json(r_content)
    else:
        # Default: ONE consolidated call (resume + JD sent once).
        user = (
            f"RESUME (plain text):\n{resume.text}\n\n"
            f"JOB DESCRIPTION:\n{jd_text}\n\n"
            f"VERIFIED PROOF LINKS:\n{proofs_txt}\n\nReturn the JSON object."
        )
        content, usage = _chat(llm, TAILOR_SYSTEM, user, "TAILOR", settings.model)
        _accumulate(usage)
        data = extract_json(content)
        jd_profile = data.get("jd_profile", {})
        match_block = data.get("match", {})
        r = data

    keywords = jd_profile.get("keywords", []) or []

    # ATS score: BEFORE (original resume) and AFTER (tailored resume).
    score_before, subs_before, _, _ = compute_score(resume.text, keywords)
    resume_data = ResumeData.from_dict(r.get("resume", {}))
    if not resume_data.links and proof_urls:
        resume_data.links = [p.url for p in proofs if p.ok]
    new_text = resume_data.to_text()
    score_after, subs_after, matched, missing = compute_score(new_text, keywords)

    matched = match_block.get("matched_keywords") or matched
    missing = match_block.get("missing_keywords") or missing

    changes = [
        ChangeItem(
            section=c.get("section", ""), before=c.get("before", ""),
            after=c.get("after", ""), reason=c.get("reason", ""),
        )
        for c in r.get("changes", [])
    ]

    new_tex = latex.render_resume(resume_data)
    cost = estimate_cost_inr(
        settings.model,
        usage_total["prompt_tokens"],
        usage_total["completion_tokens"],
        usage_total["cached_tokens"],
    )

    match = MatchReport(
        score=score_after,
        subscores=subs_after,
        score_before=score_before,
        subscores_before=subs_before,
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
        model=settings.model,
        input_format=resume.fmt,
        cost_inr=cost,
        prompt_tokens=usage_total["prompt_tokens"],
        completion_tokens=usage_total["completion_tokens"],
        calls=calls,
    )
