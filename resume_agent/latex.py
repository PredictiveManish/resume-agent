"""LaTeX rendering. Structured data -> a clean, ATS-friendly, compilable .tex file.

Rendering from structured data (rather than asking the model to emit raw LaTeX)
is the key design decision: the model can never produce a file that fails to
compile, because it only fills in fields and this code owns the LaTeX.
"""

import re
from pathlib import Path
from typing import List, Optional

from jinja2 import DictLoader, Environment, FileSystemLoader

from .models import ResumeData

# Single-pass escaping: each special char is replaced exactly once, so the
# braces inside a replacement (e.g. \textbackslash{}) are not re-escaped.
_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_ESCAPE_RE = re.compile(r"[\\&%$#_{}~^]")


def latex_escape(value) -> str:
    if value is None:
        return ""
    return _ESCAPE_RE.sub(lambda m: _ESCAPE_MAP[m.group(0)], str(value))


DEFAULT_TEMPLATE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.55in]{geometry}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\normalsize\bfseries}{}{0em}{}[\vspace{-3pt}\hrule]
\titlespacing*{\section}{0pt}{7pt}{3pt}
\setlist[itemize]{leftmargin=1.2em,nosep,topsep=1pt,parsep=0pt}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}

\begin{center}
{\LARGE\bfseries {{ name|tex }}}\\[3pt]
{% if contact %}{{ contact }}{% endif %}
{% if links %}\\[2pt] {{ links }}{% endif %}
\end{center}

{% if summary %}
\section*{Summary}
{{ summary|tex }}
{% endif %}

{% if skills %}
\section*{Skills}
\begin{itemize}
{% for group, items in skills.items() %}
  \item \textbf{ {{ group|tex }}: } {{ items|map('tex')|join(', ') }}
{% endfor %}
\end{itemize}
{% endif %}

{% if experience %}
\section*{Experience}
{% for e in experience %}
\textbf{ {{ e.title|tex }} }{% if e.org %} --- {{ e.org|tex }}{% endif %}\hfill {{ e.dates|tex }}\\
{% if e.location %}{{ e.location|tex }}\\{% endif %}
\begin{itemize}
{% for b in e.bullets %}
  \item {{ b|tex }}
{% endfor %}
\end{itemize}
{% endfor %}
{% endif %}

{% if projects %}
\section*{Projects}
{% for p in projects %}
\textbf{ {{ p.name|tex }} }{% if p.tech %} ({{ p.tech|tex }}){% endif %}{% if p.link %} \hfill {{ p.link|tex }}{% endif %}\\
\begin{itemize}
{% for b in p.bullets %}
  \item {{ b|tex }}
{% endfor %}
\end{itemize}
{% endfor %}
{% endif %}

{% if education %}
\section*{Education}
{% for e in education %}
\textbf{ {{ e.degree|tex }} }{% if e.institution %} --- {{ e.institution|tex }}{% endif %}\hfill {{ e.dates|tex }}\\
{% if e.details %}{{ e.details|tex }}\\{% endif %}
{% endfor %}
{% endif %}

\end{document}
"""


def _env(template_dir: Optional[str]) -> Environment:
    loader = (
        FileSystemLoader(template_dir)
        if template_dir
        else DictLoader({"resume.tex.j2": DEFAULT_TEMPLATE})
    )
    env = Environment(loader=loader, trim_blocks=True, lstrip_blocks=True, autoescape=False)
    env.filters["tex"] = latex_escape
    return env


def _render_links(links: List[str]) -> str:
    out = []
    for link in links:
        link = (link or "").strip()
        if not link:
            continue
        if link.lower().startswith("http"):
            out.append(r"\url{%s}" % link)
        else:
            out.append(latex_escape(link))
    return r" $|$ ".join(out)


def render_resume(data: ResumeData, template_path: Optional[str] = None) -> str:
    """Render structured resume data into LaTeX."""
    if template_path is None:
        template_path = str(
            Path(__file__).resolve().parent.parent / "templates" / "resume.tex.j2"
        )
    tpl_path = Path(template_path)
    if tpl_path.exists():
        env = _env(str(tpl_path.parent))
        tpl = env.get_template(tpl_path.name)
    else:
        env = _env(None)
        tpl = env.get_template("resume.tex.j2")

    contact_parts = [p for p in [data.email, data.phone, data.location] if p]
    contact = latex_escape(" · ".join(contact_parts))

    return tpl.render(
        name=data.name,
        contact=contact,
        links=_render_links(data.links),
        summary=data.summary,
        skills=data.skills,
        experience=data.experience,
        projects=data.projects,
        education=data.education,
    )
