"""Configuration. Reads from environment / .env — never hardcode keys."""

import os
from dataclasses import dataclass
from typing import Optional


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


@dataclass
class Settings:
    api_key: Optional[str] = None
    base_url: str = "https://api.sarvam.ai/v1"
    model: str = "sarvam-105b"
    temperature: float = 0.3
    max_tokens: int = 4096
    # Sarvam thinking mode is ON by default and reasoning tokens bill as output.
    # For resume rewriting we do not need it; None disables it (cheaper).
    reasoning_effort: Optional[str] = None
    timeout: int = 180

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1"),
            model=os.getenv("SARVAM_MODEL", "sarvam-105b"),
            reasoning_effort=os.getenv("SARVAM_REASONING_EFFORT") or None,
        )
