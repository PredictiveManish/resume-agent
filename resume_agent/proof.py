"""Proof checking — verify a student's claims have a live, reachable source.

The student supplies links (e.g. a GitHub repo) as evidence for statements on
their resume. We check the link resolves and, for GitHub, enrich it with repo
metadata so the agent can confirm the work is real before it frames it.
"""

import re
from typing import List, Optional

import requests

from .models import ProofResult

GITHUB_RE = re.compile(r"github\.com/([^/\s]+)/([^/\s#?]+)", re.IGNORECASE)


def _is_github(url: str) -> Optional[tuple]:
    m = GITHUB_RE.search(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).replace(".git", "")
    return owner, repo


def check_proofs(urls: List[str], timeout: int = 15) -> List[ProofResult]:
    """Check each proof URL. Never raises — returns a result per URL."""
    results: List[ProofResult] = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
        gh = _is_github(url)
        if gh:
            owner, repo = gh
            try:
                api = f"https://api.github.com/repos/{owner}/{repo}"
                r = requests.get(
                    api,
                    timeout=timeout,
                    headers={"Accept": "application/vnd.github+json"},
                )
                if r.status_code == 200:
                    d = r.json()
                    desc = d.get("description") or ""
                    lang = d.get("language") or ""
                    stars = d.get("stargazers_count", 0)
                    detail = f"{lang} · {stars}★ · {desc}".strip(" ·")
                    results.append(
                        ProofResult(url=url, ok=True, title=d.get("full_name", url), detail=detail)
                    )
                elif r.status_code == 404:
                    results.append(
                        ProofResult(url=url, ok=False, detail="GitHub repo not found (404)")
                    )
                else:
                    results.append(
                        ProofResult(url=url, ok=False, detail=f"GitHub returned {r.status_code}")
                    )
                continue
            except requests.RequestException as e:
                results.append(ProofResult(url=url, ok=False, detail=f"Network error: {e}"))
                continue

        # Generic URL reachability
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
            ok = r.status_code < 400
            title = ""
            m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.IGNORECASE | re.DOTALL)
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()[:120]
            results.append(
                ProofResult(url=url, ok=ok, title=title, detail=f"HTTP {r.status_code}")
            )
        except requests.RequestException as e:
            results.append(ProofResult(url=url, ok=False, detail=f"Unreachable: {e}"))
    return results
