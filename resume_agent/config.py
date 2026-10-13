"""Configuration. Reads from environment / .env — never hardcode keys.

Model facts (verified against Sarvam docs):
  * Sarvam's only first-party chat model is `sarvam-105b` (and
    `sarvam-105b-conversations`). Sarvam-M and Sarvam-30B are deprecated.
  * Sarvam *also* serves a few open-weight models (GLM-5.3, Gemma 4 31B,
    DeepSeek V4.1 Flash) on `/v2/chat/completions` with the same key/credits —
    but they are BETA, require per-key whitelisting, and GLM-5.3 / DeepSeek
    reason on every call (billed as output).
  * There is therefore NO cheaper, non-beta, first-party Sarvam chat model.
    The safe default everywhere is `sarvam-105b`.

Cost control lives here: thinking off by default, max_tokens capped, and token
usage turned into an INR figure.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


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


# Open-weight models are served on /v2 and need beta whitelisting.
OPEN_WEIGHT_MODELS = {"glm5.3", "gemma4", "deepseekv4.1-flash"}
SARVAM_CHAT_MODELS = {"sarvam-105b", "sarvam-105b-conversations"}

V1_ENDPOINT = "https://api.sarvam.ai/v1/chat/completions"
V2_ENDPOINT = "https://api.sarvam.ai/v2/chat/completions"

# INR per 1M tokens: (input, cached_input, output). APPROXIMATE — verify against
# Sarvam's pricing page, and note open-weight cached rates are not published, so
# input is used as a conservative stand-in for them.
PRICING: Dict[str, Tuple[float, float, float]] = {
    "sarvam-105b": (29.28, 10.98, 73.20),
    "sarvam-105b-conversations": (29.28, 10.98, 73.20),
    "glm5.3": (126.0, 126.0, 390.0),
    "gemma4": (35.0, 35.0, 90.0),
    "deepseekv4.1-flash": (25.0, 25.0, 105.0),
}
DEFAULT_PRICING: Tuple[float, float, float] = PRICING["sarvam-105b"]


def price_for(model: str) -> Tuple[float, float, float]:
    return PRICING.get(model, DEFAULT_PRICING)


def endpoint_for(model: str, base_url: Optional[str] = None) -> str:
    """Open-weight models must go to /v2; Sarvam chat models to /v1."""
    if model in OPEN_WEIGHT_MODELS:
        return V2_ENDPOINT
    if base_url:
        return base_url.rstrip("/") + "/chat/completions"
    return V1_ENDPOINT


def estimate_cost_inr(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """Cost in INR for one call, from token counts."""
    p_in, p_cached, p_out = price_for(model)
    cached_tokens = max(0, min(cached_tokens, prompt_tokens))
    billable_in = max(0, prompt_tokens - cached_tokens)
    return (
        billable_in * p_in + cached_tokens * p_cached + completion_tokens * p_out
    ) / 1_000_000.0


@dataclass
class Settings:
    api_key: Optional[str] = None
    base_url: str = "https://api.sarvam.ai/v1"
    # Main model. sarvam-105b is the safe, non-beta, first-party default.
    model: str = "sarvam-105b"
    # Two-pass mode analysis model. Defaults to the SAME model — there is no
    # cheaper non-beta Sarvam chat model. If you have beta access you may try
    # "deepseekv4.1-flash", but it reasons on every call.
    fast_model: str = "sarvam-105b"
    two_pass: bool = False
    temperature: float = 0.3
    max_tokens: int = 4096
    # None => thinking mode OFF (cheaper). Set low/medium/high to enable.
    reasoning_effort: Optional[str] = None
    timeout: int = 180

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv()
        return cls(
            api_key=os.getenv("SARVAM_API_KEY"),
            base_url=os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1"),
            model=os.getenv("SARVAM_MODEL", "sarvam-105b"),
            fast_model=os.getenv("SARVAM_FAST_MODEL", "sarvam-105b"),
            two_pass=os.getenv("RESUME_AGENT_TWO_PASS", "").lower()
            in ("1", "true", "yes", "on"),
            max_tokens=int(os.getenv("SARVAM_MAX_TOKENS", "4096")),
            reasoning_effort=os.getenv("SARVAM_REASONING_EFFORT") or None,
        )
