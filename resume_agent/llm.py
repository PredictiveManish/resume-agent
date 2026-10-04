"""LLM client for Sarvam's OpenAI-compatible chat completions API.

Also provides a MockLLM so the whole pipeline can be tested with no API key.
"""

import json
import re
from typing import Dict, List, Optional

import requests

from .config import Settings


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Dict:
    """Robustly pull a JSON object out of a model response.

    Handles ```json fences and leading/trailing prose.
    """
    if not text:
        raise LLMError("Empty response from model")
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...}
    start = cleaned.find("{")
    if start == -1:
        raise LLMError(f"No JSON found in response: {text[:200]}")
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : i + 1])
    raise LLMError(f"Unbalanced JSON in response: {text[:200]}")


class SarvamClient:
    """Thin OpenAI-compatible client for Sarvam chat completions."""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise LLMError(
                "SARVAM_API_KEY is not set. Put it in your environment or .env file."
            )
        self.settings = settings
        self.endpoint = settings.base_url.rstrip("/") + "/chat/completions"

    def chat(
        self,
        system: str,
        user: str,
        task: str = "",
        want_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        s = self.settings
        headers = {
            "Content-Type": "application/json",
            # Send both header styles for compatibility across Sarvam endpoints.
            "Authorization": f"Bearer {s.api_key}",
            "api-subscription-key": s.api_key,
        }
        payload = {
            "model": s.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": s.temperature if temperature is None else temperature,
            "max_tokens": s.max_tokens if max_tokens is None else max_tokens,
        }
        # reasoning_effort=None disables thinking mode (billed as output tokens).
        if s.reasoning_effort is not None:
            payload["reasoning_effort"] = s.reasoning_effort
        else:
            payload["reasoning_effort"] = None
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                self.endpoint, headers=headers, json=payload, timeout=s.timeout
            )
        except requests.RequestException as e:
            raise LLMError(f"Request to Sarvam failed: {e}") from e

        if resp.status_code >= 400:
            # Some deployments reject response_format; retry once without it.
            if want_json and "response_format" in payload:
                payload.pop("response_format", None)
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=s.timeout
                )
            if resp.status_code >= 400:
                raise LLMError(f"Sarvam API error {resp.status_code}: {resp.text[:400]}")

        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:400]}") from e


class MockLLM:
    """Deterministic stand-in for tests. Routes on the `task` label."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls: List[Dict[str, str]] = []

    def chat(self, system, user, task="", want_json=False, **kwargs) -> str:
        self.calls.append({"task": task, "system": system, "user": user})
        if task in self.responses:
            return self.responses[task]
        return self.responses.get("default", "{}")
