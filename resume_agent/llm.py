"""LLM client for Sarvam's OpenAI-compatible chat completions API.

Also provides a MockLLM so the whole pipeline can be tested with no API key.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional

import requests

from .config import Settings

logger = logging.getLogger("resume_agent.llm")


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

        logger.info(
            "LLM request: model=%s endpoint=%s chars_in=%d max_tokens=%s json=%s",
            s.model, self.endpoint, len(system) + len(user), payload["max_tokens"], want_json,
        )
        started = time.monotonic()
        try:
            resp = requests.post(
                self.endpoint, headers=headers, json=payload, timeout=s.timeout
            )
        except requests.RequestException as e:
            logger.exception("LLM request to %s failed: %s", self.endpoint, e)
            raise LLMError(f"Request to Sarvam failed: {e}") from e

        logger.info(
            "LLM response: %s -> HTTP %d in %.1fs (task=%s)",
            s.model, resp.status_code, time.monotonic() - started, task or "-",
        )
        if resp.status_code >= 400:
            # Some deployments reject response_format; retry once without it.
            if want_json and "response_format" in payload:
                payload.pop("response_format", None)
                logger.warning("Retrying without response_format...")
                resp = requests.post(
                    self.endpoint, headers=headers, json=payload, timeout=s.timeout
                )
            if resp.status_code >= 400:
                logger.error("Sarvam API error %d: %s", resp.status_code, resp.text[:1000])
                raise LLMError(f"Sarvam API error {resp.status_code}: {resp.text[:400]}")

        try:
            data = resp.json()
        except ValueError as e:
            logger.error("Non-JSON body from Sarvam (HTTP %d): %s", resp.status_code, resp.text[:1000])
            raise LLMError(f"Sarvam returned non-JSON body: {resp.text[:200]}") from e
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Unexpected response shape: %s", str(data)[:1000])
            raise LLMError(f"Unexpected response shape: {str(data)[:400]}") from e
        if content is None or not str(content).strip():
            # Log the FULL raw response — it shows whether tokens went to
            # reasoning/thinking fields instead of the final answer.
            logger.error(
                "Empty content from %s (task=%s). Raw API response: %s",
                s.model, task or "-", str(data)[:2000],
            )
            raise LLMError(
                "Model returned an empty response (HTTP 200, no content). "
                "Usual cause: thinking/reasoning consumed the whole max_tokens budget. "
                "See server log for the raw API response; try a larger SARVAM_MAX_TOKENS."
            )
        logger.info("LLM response: %d chars of content", len(str(content)))
        return content


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
