"""Tests. The pipeline is tested end-to-end with a MockLLM — no API key needed."""

import json

from resume_agent.ingest import ingest, strip_latex
from resume_agent.latex import latex_escape, render_resume
from resume_agent.llm import MockLLM, extract_json
from resume_agent.models import ResumeData
from resume_agent.pipeline import compute_score, tailor

ANALYZE_RESP = json.dumps(
    {
        "jd_profile": {
            "role_title": "Backend Engineer",
            "keywords": ["Python", "FastAPI", "Docker", "SQL", "REST"],
            "requirements": ["3+ years Python"],
        },
        "match": {
            "matched_keywords": ["Python", "Docker"],
            "missing_keywords": ["FastAPI", "SQL"],
            "gaps": ["No mention of API frameworks", "No SQL experience shown"],
            "suggestions": ["Move Python to the top of skills", "Quantify impact"],
        },
    }
)

REWRITE_RESP = json.dumps(
    {
        "resume": {
            "name": "Asha Rao",
            "email": "asha@example.com",
            "phone": "+91 90000 00000",
            "location": "Bengaluru, India",
            "links": ["https://github.com/asharao"],
            "summary": "Backend engineer focused on Python services and Docker.",
            "skills": {"Languages": ["Python", "SQL"], "Tools": ["Docker", "FastAPI"]},
            "experience": [
                {
                    "title": "Software Engineer",
                    "org": "Acme",
                    "dates": "2023 - Present",
                    "location": "Remote",
                    "bullets": ["Built Python services handling 10k req/day", "Containerised with Docker"],
                }
            ],
            "projects": [
                {"name": "SafeRun", "link": "https://github.com/asharao/saferun",
                 "tech": "Python, Docker", "bullets": ["Sandboxed untrusted code"]}
            ],
            "education": [
                {"degree": "B.Tech CSE", "institution": "IIT", "dates": "2019-2023", "details": "CGPA 9.1"}
            ],
        },
        "changes": [
            {"section": "Skills", "before": "Python", "after": "Python (lead), SQL, Docker, FastAPI",
             "reason": "Match JD keywords the candidate genuinely has"},
            {"section": "Summary", "before": "(none)", "after": "Backend engineer...",
             "reason": "Add a JD-aligned summary"},
        ],
    }
)


def test_latex_escape_special_chars():
    assert latex_escape("A & B_100% #1") == r"A \& B\_100\% \#1"
    assert latex_escape("back\\slash") == r"back\textbackslash{}slash"


def test_strip_latex_removes_commands():
    tex = r"\section*{Skills} \item Python \& SQL \href{https://x.com}{link}"
    out = strip_latex(tex)
    assert "Skills" in out
    assert "Python" in out
    assert "\\item" not in out


def test_ingest_text():
    r = ingest("hello resume", is_text=True)
    assert r.fmt == "text" and r.text == "hello resume"


def test_render_resume_compilable():
    data = ResumeData.from_dict(json.loads(REWRITE_RESP)["resume"])
    tex = render_resume(data)
    assert tex.startswith("\\documentclass")
    assert "\\end{document}" in tex
    assert "Asha Rao" in tex
    assert "\\section*{Experience}" in tex
    # escaping applied to data
    assert "9.1" in tex


def test_render_escapes_dangerous_chars():
    data = ResumeData(name="A & B_100%", email="x@y.com")
    tex = render_resume(data)
    assert r"A \& B\_100\%" in tex


def test_compute_score_range_and_keywords():
    score, subs, matched, missing = compute_score(
        "Python and Docker engineer, email a@b.com, 5 projects",
        ["Python", "Docker", "Kubernetes"],
    )
    assert 0 <= score <= 100
    assert "Python" in matched and "Kubernetes" in missing
    assert subs["keyword_coverage"] == 67  # 2 of 3


def test_extract_json_handles_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('prose {"b": 2} more') == {"b": 2}


def test_pipeline_end_to_end_with_mock():
    llm = MockLLM({"ANALYZE": ANALYZE_RESP, "REWRITE": REWRITE_RESP})
    result = tailor(
        "Asha Rao\nEmail: asha@example.com\nPython, Docker",
        "We need a Python FastAPI backend engineer with SQL and Docker.",
        proof_urls=[],
        llm=llm,
        is_text=True,
    )
    assert result.new_tex.startswith("\\documentclass")
    assert "\\end{document}" in result.new_tex
    assert result.match.score >= 0
    assert len(result.changes) == 2
    assert result.resume_data.name == "Asha Rao"
    # the two LLM calls were made, in order
    assert [c["task"] for c in llm.calls] == ["ANALYZE", "REWRITE"]
