"""Canned LLM responses so the UI can be demoed with no API key (RESUME_AGENT_MOCK=1)."""

import json

DEMO_RESPONSES = {
    "ANALYZE": json.dumps(
        {
            "jd_profile": {
                "role_title": "Backend Engineer (Python)",
                "keywords": [
                    "Python", "FastAPI", "Docker", "SQL", "PostgreSQL",
                    "REST API", "CI/CD", "cloud", "data structures",
                    "testing", "algorithms", "containerisation",
                ],
                "requirements": [
                    "2+ years Python",
                    "FastAPI or similar",
                    "Docker experience",
                    "SQL databases",
                ],
            },
            "match": {
                "matched_keywords": ["Python", "Docker"],
                "missing_keywords": ["FastAPI", "PostgreSQL", "REST API", "CI/CD", "testing"],
                "gaps": [
                    "No API framework experience is mentioned (JD asks for FastAPI).",
                    "SQL is listed as 'basic' — the JD wants working database knowledge.",
                    "Experience bullets describe duties, not impact (no numbers).",
                ],
                "suggestions": [
                    "Lead the skills section with Python and add any real API work.",
                    "Quantify the backend work you did (requests/day, latency, users).",
                    "Mention testing practices you have actually used.",
                ],
            },
        }
    ),
    "REWRITE": json.dumps(
        {
            "resume": {
                "name": "Asha Rao",
                "email": "asha@example.com",
                "phone": "+91 90000 00000",
                "location": "Bengaluru, India",
                "links": ["https://github.com/asharao"],
                "summary": (
                    "Backend engineer focused on Python services and containerised "
                    "deployment, with hands-on Docker experience and a track record "
                    "of shipping production code."
                ),
                "skills": {
                    "Languages": ["Python", "SQL"],
                    "Tools": ["Docker", "Git", "REST APIs"],
                    "Practices": ["Testing", "CI/CD"],
                },
                "experience": [
                    {
                        "title": "Software Engineer",
                        "org": "Acme",
                        "dates": "2023 - Present",
                        "location": "Bengaluru",
                        "bullets": [
                            "Built and maintained Python backend services for the core product.",
                            "Containerised services with Docker, simplifying deployment.",
                        ],
                    }
                ],
                "projects": [
                    {
                        "name": "SafeRun",
                        "link": "https://github.com/asharao/saferun",
                        "tech": "Python, Docker",
                        "bullets": [
                            "Built a Docker-based sandbox to execute untrusted Python safely.",
                        ],
                    }
                ],
                "education": [
                    {
                        "degree": "B.Tech Computer Science",
                        "institution": "IIT",
                        "dates": "2019 - 2023",
                        "details": "CGPA 9.1",
                    }
                ],
            },
            "changes": [
                {
                    "section": "Skills",
                    "before": "Python, Docker, Git, basic SQL",
                    "after": "Python, SQL · Docker, Git, REST APIs · Testing, CI/CD",
                    "reason": "Surfaces the JD's keywords the candidate genuinely has, grouped for scanability.",
                },
                {
                    "section": "Summary",
                    "before": "(no summary)",
                    "after": "Backend engineer focused on Python services and containerised deployment…",
                    "reason": "Adds a JD-aligned summary so a recruiter sees fit in the first 5 seconds.",
                },
                {
                    "section": "Experience",
                    "before": "Worked on backend services in Python",
                    "after": "Built and maintained Python backend services for the core product",
                    "reason": "Replaces a passive duty with an active, ownership-oriented bullet.",
                },
            ],
        }
    ),
}
