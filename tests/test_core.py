"""Tests. The pipeline is tested end-to-end with a MockLLM — no API key needed."""

import json

from resume_agent.config import estimate_cost_inr, price_for
from resume_agent.ingest import ingest, strip_latex
from resume_agent.latex import latex_escape, render_resume
from resume_agent.llm import MockLLM, extract_json
from resume_agent.models import ResumeData
from resume_agent.pipeline import compute_score, tailor

TAILOR_RESP = json.dumps(
    {
        "jd_profile": {
            "role_title": "Backend Engineer",
            "keywords": ["Python", "FastAPI", "Docker", "SQL", "REST"],
        },
        "match": {
            "matched_keywords": ["Python", "Docker"],
            "missing_keywords": ["FastAPI", "SQL"],
            "gaps": ["No API frameworks mentioned", "No SQL shown"],
            "suggestions": ["Lead with Python", "Quantify impact"],
        },
        "resume": {
            "name": "Asha Rao",
            "email": "asha@example.com",
            "phone": "+91 90000 00000",
            "location": "Bengaluru, India",
            "links": ["https://github.com/asharao"],
            "summary": "Backend engineer focused on Python services and Docker.",
            "skills": {"Languages": ["Python", "SQL"], "Tools": ["Docker", "FastAPI"]},
            "experience": [
                {"title": "Software Engineer", "org": "Acme", "dates": "2023 - Present",
                 "location": "Remote",
                 "bullets": ["Built Python services handling 10k req/day",
                             "Containerised with Docker"]}
            ],
            "projects": [
                {"name": "SafeRun", "link": "https://github.com/asharao/saferun",
                 "tech": "Python, Docker", "bullets": ["Sandboxed untrusted code"]}
            ],
            "education": [
                {"degree": "B.Tech CSE", "institution": "IIT", "dates": "2019-2023",
                 "details": "CGPA 9.1"}
            ],
        },
        "changes": [
            {"section": "Skills", "before": "Python", "after": "Python (lead), SQL, Docker, FastAPI",
             "reason": "Match JD keywords the candidate has"},
            {"section": "Summary", "before": "(none)", "after": "Backend engineer...",
             "reason": "Add a JD-aligned summary"},
        ],
    }
)

ANALYZE_RESP = json.dumps(
    {
        "jd_profile": {"role_title": "Backend Engineer", "keywords": ["Python", "FastAPI", "Docker", "SQL", "REST"]},
        "match": {"matched_keywords": ["Python", "Docker"], "missing_keywords": ["FastAPI", "SQL"],
                  "gaps": ["No API frameworks mentioned"], "suggestions": ["Lead with Python"]},
    }
)
REWRITE_RESP = json.dumps({"resume": json.loads(TAILOR_RESP)["resume"],
                           "changes": json.loads(TAILOR_RESP)["changes"]})


def test_latex_escape_special_chars():
    assert latex_escape("A & B_100% #1") == r"A \& B\_100\% \#1"
    assert latex_escape("back\\slash") == r"back\textbackslash{}slash"


def test_strip_latex_removes_commands():
    tex = r"\section*{Skills} \item Python \& SQL \href{https://x.com}{link}"
    out = strip_latex(tex)
    assert "Skills" in out and "Python" in out and "\\item" not in out


def test_ingest_text():
    r = ingest("hello resume", is_text=True)
    assert r.fmt == "text" and r.text == "hello resume"


def test_render_resume_compilable():
    data = ResumeData.from_dict(json.loads(TAILOR_RESP)["resume"])
    tex = render_resume(data)
    assert tex.startswith("\\documentclass") and "\\end{document}" in tex
    assert "Asha Rao" in tex and "\\section*{Experience}" in tex


def test_render_escapes_dangerous_chars():
    tex = render_resume(ResumeData(name="A & B_100%", email="x@y.com"))
    assert r"A \& B\_100\%" in tex


def test_compute_score_range_and_keywords():
    score, subs, matched, missing = compute_score(
        "Python and Docker engineer, email a@b.com, 5 projects",
        ["Python", "Docker", "Kubernetes"],
    )
    assert 0 <= score <= 100
    assert "Python" in matched and "Kubernetes" in missing
    assert subs["keyword_coverage"] == 67


def test_extract_json_handles_fences_and_truncation():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('prose {"b": 2} more') == {"b": 2}
    # truncated JSON is salvaged by closing braces
    assert extract_json('{"a": {"b": 1}') == {"a": {"b": 1}}


def test_cost_math():
    # 1M input tokens on sarvam-105b = ₹29.28
    assert abs(estimate_cost_inr("sarvam-105b", 1_000_000, 0) - 29.28) < 1e-6
    assert abs(estimate_cost_inr("sarvam-105b", 0, 1_000_000) - 73.20) < 1e-6
    assert price_for("unknown-model") == price_for("sarvam-105b")


def test_pipeline_end_to_end_single_call():
    llm = MockLLM({"TAILOR": TAILOR_RESP})
    result = tailor(
        "Asha Rao\nEmail: asha@example.com\nPython, Docker",
        "We need a Python FastAPI backend engineer with SQL and Docker.",
        llm=llm, is_text=True,
    )
    assert result.new_tex.startswith("\\documentclass")
    assert result.match.score >= 0
    assert result.match.score_before >= 0
    assert len(result.changes) == 2
    assert result.calls == 1                       # single consolidated call
    assert result.cost_inr > 0                     # cost is computed
    assert [c["task"] for c in llm.calls] == ["TAILOR"]


def test_pipeline_two_pass_mode():
    from resume_agent.config import Settings
    llm = MockLLM({"ANALYZE": ANALYZE_RESP, "REWRITE": REWRITE_RESP})
    settings = Settings(api_key="x", two_pass=True)
    result = tailor("Asha Rao\nPython", "Python FastAPI Docker", llm=llm,
                    settings=settings, is_text=True)
    assert result.calls == 2
    assert [c["task"] for c in llm.calls] == ["ANALYZE", "REWRITE"]


def test_ats_after_beats_before_for_keyword_rich_rewrite():
    """The tailored resume contains JD keywords the original lacked -> higher ATS."""
    llm = MockLLM({"TAILOR": TAILOR_RESP})
    result = tailor("Asha Rao\nPython, Docker", "Python FastAPI Docker SQL REST",
                    llm=llm, is_text=True)
    assert result.match.score >= result.match.score_before
