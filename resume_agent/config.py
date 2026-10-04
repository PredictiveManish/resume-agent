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
    # Sarvam thinking mode is ON by default and reasoning tokens bill as output
    # and count against max_tokens. If reasoning burns the whole budget, the
    # final content comes back EMPTY -> keep headroom here (SARVAM_MAX_TOKENS).
    max_tokens: int = 16384
    # reasoning_effort: None disables thinking mode (cheaper).
    reasoning_effort: Optional[str] = None
    timeout: int = 180

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1"),
            model=os.getenv("SARVAM_MODEL", "sarvam-105b"),
            max_tokens=int(os.getenv("SARVAM_MAX_TOKENS", "16384")),
            reasoning_effort=os.getenv("SARVAM_REASONING_EFFORT") or None,
        )
