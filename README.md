# Resume Tailor Agent

An AI agent that takes a student's resume and a job description, checks the
student's proof links, and returns **an ATS score before and after**, a
**preview of every change and why**, and a **ready-to-paste LaTeX resume**
(plus a one-click PDF).

Built on the **Sarvam AI** chat API.

---

## What it does

```
resume (.tex | .pdf | text)  +  JD  +  proof links (e.g. GitHub)
        │
        ▼
   ingest → check proofs → [ONE consolidated LLM call]
        → ATS score BEFORE and AFTER → preview of changes → LaTeX (+ PDF)
        │
        ▼
   ATS score · changes · .tex / .pdf · cost
```

## Design decisions worth knowing

1. **The model never writes raw LaTeX.** It returns structured JSON; this code
   renders the LaTeX from a template. Output can never fail to compile.
2. **The ATS score is deterministic and shown twice** — on the original resume
   and on the tailored one — so the student sees the improvement, not just a
   number. It is a *heuristic JD-match score*, labelled as such.
3. **Cost is a first-class feature.** One consolidated call by default, thinking
   mode off, `max_tokens` capped, and the INR cost of every run is computed from
   the API's token usage and shown to the user.
4. **Truthfulness guardrail.** The prompt forbids inventing experience,
   employers, dates, metrics, or skills. The agent *reframes*; it never fabricates.

## Cost

Sarvam pricing (INR per 1M tokens: input / cached / output):

| Model | Input | Output | Notes |
|---|---|---|---|
| `sarvam-105b` (default) | ₹29.28 | ₹73.20 | Sarvam's only first-party chat model |
| `deepseekv4.1-flash` | ₹25.00 | ₹105.00 | open-weight, **beta**, always reasons |
| `gemma4` | ₹35.00 | ₹90.00 | open-weight, **beta**, no reasoning |
| `glm5.3` | ₹126.00 | ₹390.00 | open-weight, **beta**, reasons; costlier than 105b |

Open-weight models are served on `/v2` and require per-key beta whitelisting;
this client routes to the right endpoint automatically. There is **no cheaper
non-beta Sarvam chat model**, so the default everywhere is `sarvam-105b`.

- **Default (one consolidated call):** ~**₹0.15–0.30 per tailoring** on
  `sarvam-105b` — the resume + JD are sent to the model **once**, not twice.
- **Two-pass mode** (`RESUME_AGENT_TWO_PASS=1`): an analysis call, then a rewrite
  call; ~**₹0.25–0.45** (both on `sarvam-105b` by default).
- **The two things that used to blow the bill up** and are now off by default:
  *thinking mode* (its tokens bill as output) and an *uncapped* `max_tokens`.

Every response includes the real cost, token counts, and call count, so you can
see it rather than guess. The UI shows it under the results.

## Install & run

```bash
pip install -r requirements.txt
cp .env.example .env        # put your Sarvam key in .env (never share it)

# Web UI (recommended)
uvicorn resume_agent.api:app --reload     # open http://127.0.0.1:8000

# Demo mode — no API key needed
RESUME_AGENT_MOCK=1 uvicorn resume_agent.api:app --reload
```

If `pdflatex` is installed on the machine, the UI offers a **Download PDF**
button (LaTeX shell-escape is disabled, so the document cannot run commands).
Otherwise use the `.tex` download and paste into Overleaf.

## CLI

```bash
python -m resume_agent.cli tailor \
  --resume examples/sample_resume.tex \
  --jd examples/sample_jd.txt \
  --proof https://github.com/you/repo \
  --out tailored.tex --json result.json
```

The CLI prints the ATS score before → after, the sub-scores, the change list,
and the cost.

## Tests

```bash
pytest -q        # end-to-end with a MockLLM — no API key needed
```

## Notes

- **PDF → LaTeX structure is not recoverable.** PDF input is parsed to *text* and
  rebuilt from the template — by design, not a limitation to fix.
- Direct Overleaf write-back needs Overleaf's paid Git integration; the flow
  outputs a `.tex` to paste in.
