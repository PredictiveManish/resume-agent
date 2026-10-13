"""Data models for the resume-tailoring pipeline."""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


@dataclass
class Experience:
    title: str = ""
    org: str = ""
    dates: str = ""
    location: str = ""
    bullets: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Experience":
        return cls(
            title=d.get("title", ""), org=d.get("org", ""), dates=d.get("dates", ""),
            location=d.get("location", ""),
            bullets=[str(b) for b in d.get("bullets", [])],
        )


@dataclass
class Project:
    name: str = ""
    link: str = ""
    tech: str = ""
    bullets: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Project":
        return cls(
            name=d.get("name", ""), link=d.get("link", ""), tech=d.get("tech", ""),
            bullets=[str(b) for b in d.get("bullets", [])],
        )


@dataclass
class Education:
    degree: str = ""
    institution: str = ""
    dates: str = ""
    details: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Education":
        return cls(
            degree=d.get("degree", ""), institution=d.get("institution", ""),
            dates=d.get("dates", ""), details=d.get("details", ""),
        )


@dataclass
class ResumeData:
    """Structured resume. Rendered into LaTeX — structure guarantees it compiles."""

    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    links: List[str] = field(default_factory=list)
    summary: str = ""
    skills: Dict[str, List[str]] = field(default_factory=dict)
    experience: List[Experience] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    education: List[Education] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResumeData":
        skills = d.get("skills", {}) or {}
        if isinstance(skills, list):
            skills = {"Skills": skills}
        return cls(
            name=d.get("name", ""), email=d.get("email", ""), phone=d.get("phone", ""),
            location=d.get("location", ""),
            links=[str(x) for x in d.get("links", [])],
            summary=d.get("summary", ""),
            skills={k: [str(i) for i in v] for k, v in skills.items()},
            experience=[Experience.from_dict(e) for e in d.get("experience", [])],
            projects=[Project.from_dict(p) for p in d.get("projects", [])],
            education=[Education.from_dict(e) for e in d.get("education", [])],
        )

    def to_text(self) -> str:
        """Plain-text projection, for scoring (not for display)."""
        parts = [self.name, self.summary]
        for group, items in self.skills.items():
            parts.append(f"{group}: " + ", ".join(items))
        for e in self.experience:
            parts.append(f"{e.title} {e.org} {e.dates} {e.location}")
            parts.extend(e.bullets)
        for p in self.projects:
            parts.append(f"{p.name} {p.tech} {p.link}")
            parts.extend(p.bullets)
        for e in self.education:
            parts.append(f"{e.degree} {e.institution} {e.dates} {e.details}")
        return "\n".join(x for x in parts if x)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchReport:
    """ATS-style match report. `score` is the score AFTER tailoring."""

    score: int
    subscores: Dict[str, int]
    score_before: int = 0
    subscores_before: Dict[str, int] = field(default_factory=dict)
    matched_keywords: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def delta(self) -> int:
        return self.score - self.score_before

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["delta"] = self.delta
        return d


@dataclass
class ChangeItem:
    section: str
    before: str
    after: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProofResult:
    url: str
    ok: bool
    title: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    match: MatchReport
    changes: List[ChangeItem]
    proofs: List[ProofResult]
    resume_data: ResumeData
    new_tex: str
    old_text: str
    model: str
    input_format: str
    cost_inr: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match": self.match.to_dict(),
            "changes": [c.to_dict() for c in self.changes],
            "proofs": [p.to_dict() for p in self.proofs],
            "resume_data": self.resume_data.to_dict(),
            "new_tex": self.new_tex,
            "model": self.model,
            "input_format": self.input_format,
            "cost": {
                "cost_inr": round(self.cost_inr, 4),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "calls": self.calls,
                "model": self.model,
            },
        }
