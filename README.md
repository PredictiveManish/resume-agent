# Resume Tailor Agent

An AI agent that takes a student's resume and a job description, checks the
student's proof links, and returns **a preview of what will change and why**,
a **JD-match score**, and a **ready-to-paste LaTeX resume** for Overleaf.

Built on the **Sarvam AI** chat API (`sarvam-105b`).

---

## What it does

```
resume (.tex | .pdf | text)  +  JD  +  proof links (e.g. GitHub)
        │
        ▼
   [1] ingest resume  ──► plain text (+ raw LaTeX kept)
   [2] check proof links (GitHub API / HTTP reachability)
   [3] ANALYSE  (LLM call 1): JD keywords, matched/missing, gaps, suggestions
   [4] SCORE    (deterministic, reproducible)
   [5] REWRITE  (LLM call 2): structured resume + a change list (before/after/why)
   [6] RENDER   structured data ──► clean, compilable LaTeX
        │
        ▼
   preview + score + improved .tex
```

## Two design decisions worth knowing

1. **The model never writes raw LaTeX.** It returns *structured JSON*; this
   code renders the LaTeX from a template. That means the output can never
   fail to compile, and formatting stays consistent. (See `resume_agent/latex.py`.)
2. **The score is deterministic and honest.** Keyword coverage, formatting
   health, and quantified-impact are computed in code, not by the model — so
   the same resume always scores the same. It is labelled a *JD-match score*,
   not an official "ATS score" (no such universal score exists).

## Truthfulness guardrail

The rewrite prompt forbids inventing experience, employers, dates, metrics, or
skills. The agent **reframes real experience**; it does not fabricate. This
protects students from getting caught, and it is a product requirement, not a
nice-to-have.

---

## Install

```bash
pip install -r requirements.txt
cp .env.example .env      # then put your Sarvam key in .env
```

## Web UI (recommended — no terminal needed)

```bash
uvicorn resume_agent.api:app --reload
# then open http://127.0.0.1:8000
```

The browser UI lets you **drag & drop your resume**, paste the job description,
add proof links, and click one button. It shows the match score, the keyword
gaps, and a **before/after preview of every change with the reason**, plus a
one-click **Download .tex** for Overleaf.

**Try it with no API key** (demo mode, canned example):

```bash
RESUME_AGENT_MOCK=1 uvicorn resume_agent.api:app --reload
```

When demo mode is on, the page shows a "Demo mode — no API key" badge.

## Use — CLI

```bash
# Tailor, print a preview, and write the .tex
python -m resume_agent.cli tailor \
  --resume examples/sample_resume.tex \
  --jd examples/sample_jd.txt \
  --proof https://github.com/asharao/saferun \
  --out tailored.tex \
  --json result.json
```

Options:
- `--resume <path>` (.tex / .pdf / .txt) **or** `--resume-text "<text>"`
- `--jd <path>` **or** `--jd-text "<text>"`
- `--proof <url>` (repeatable) — evidence for claims, e.g. a GitHub repo
- `--out tailored.tex` — the improved LaTeX to paste into Overleaf
- `--json result.json` — the full structured result

## Use — API

```bash
uvicorn resume_agent.api:app --reload
# POST /tailor  (multipart form)
#   jd_text, resume_text | resume_file, proofs (comma-separated)
```

---

## Cost (Sarvam pricing, per 1M tokens: input / cached / output)

| Model | Input | Output |
|---|---|---|
| `sarvam-105b` (default) | ₹29.28 | ₹73.20 |
| DeepSeek V4 Flash (beta) | ₹19.80 | ₹59.40 |
| GLM 5.3 Flash (beta) | ₹13.50 | ₹45.00 |

The pipeline makes **2 LLM calls** per tailoring run. With thinking mode
disabled (the default here — reasoning tokens bill as output), a run costs
roughly **₹0.5** on `sarvam-105b`. At 100 users × 10 updates/month that is
about **₹500/month** in model cost — comfortably inside a ₹100/month
subscription.

## Tests

```bash
pytest -q          # runs end-to-end with a MockLLM — no API key needed
```

---

## Roadmap (not yet built)

- PDF input is supported, but PDF → LaTeX *structure* is impossible; we parse
  to text and rebuild from the template (by design).
- Direct Overleaf write-back requires Overleaf's paid Git integration + a
  per-user token; the current flow outputs a `.tex` to paste in.
- Web UI, auth, usage metering, and application tracking are next.
