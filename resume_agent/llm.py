"""LLM client for Sarvam's OpenAI-compatible chat completions API.

Returns token usage alongside the content so the pipeline can report cost.
Also provides a MockLLM so the whole pipeline can be tested with no API key.
"""

import json
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import requests

from .config import OPEN_WEIGHT_MODELS, Settings, endpoint_for

logger = logging.getLogger("resume_agent.llm")


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Dict:
    """Robustly pull a JSON object out of a model response.

    Handles ```json fences, leading/trailing prose, and truncation.
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
    # Truncated JSON (hit max_tokens) — try to salvage by closing braces.
    salvaged = cleaned[start:]
    salvaged += "}" * max(0, salvaged.count("{") - salvaged.count("}"))
    try:
        return json.loads(salvaged)
    except json.JSONDecodeError as e:
        raise LLMError(
            "Model response was not valid JSON (likely truncated by max_tokens). "
            f"Raise SARVAM_MAX_TOKENS or lower the prompt size. Detail: {e}"
        ) from e


def _usage_from(data: Dict, model: str) -> Dict:
    u = data.get("usage") or {}
    cached = 0
    details = u.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = int(details.get("cached_tokens") or 0)
    if not cached:
        cached = int(u.get("cached_tokens") or 0)
    return {
        "model": model,
        "prompt_tokens": int(u.get("prompt_tokens") or u.get("input_tokens") or 0),
        "completion_tokens": int(
            u.get("completion_tokens") or u.get("output_tokens") or 0
        ),
        "cached_tokens": cached,
    }


class SarvamClient:
    """Thin OpenAI-compatible client for Sarvam chat completions."""

    def __init__(self, settings: Settings):
        if not settings.api_key:
            raise LLMError(
                "SARVAM_API_KEY is not set. Put it in your environment or .env file."
            )
        self.settings = settings

    def chat(
        self,
        system: str,
        user: str,
        task: str = "",
        want_json: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        return_usage: bool = False,
    ):
        s = self.settings
        model = model or s.model
        # Open-weight models (glm5.3 / gemma4 / deepseekv4.1-flash) live on /v2;
        # Sarvam chat models (sarvam-105b) live on /v1.
        endpoint = endpoint_for(model, s.base_url)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {s.api_key}",
            "api-subscription-key": s.api_key,
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": s.temperature if temperature is None else temperature,
            "max_tokens": s.max_tokens if max_tokens is None else max_tokens,
            # reasoning_effort=None disables thinking mode on sarvam-105b.
            "reasoning_effort": s.reasoning_effort,
        }
        # Open-weight reasoning models reason on EVERY call (default = max), so
        # unless the user asked for a specific effort, keep it cheap.
        if model in OPEN_WEIGHT_MODELS and s.reasoning_effort is None:
            payload["reasoning_effort"] = "low"
        if want_json:
            payload["response_format"] = {"type": "json_object"}

        logger.info(
            "LLM request: model=%s chars_in=%d max_tokens=%s json=%s",
            model, len(system) + len(user), payload["max_tokens"], want_json,
        )
        started = time.monotonic()
        try:
            resp = requests.post(
                endpoint, headers=headers, json=payload, timeout=s.timeout
            )
        except requests.RequestException as e:
            logger.exception("LLM request failed: %s", e)
            raise LLMError(f"Request to Sarvam failed: {e}") from e

        if resp.status_code >= 400:
            if want_json and "response_format" in payload:
                payload.pop("response_format", None)
                logger.warning("Retrying without response_format...")
                resp = requests.post(
                    endpoint, headers=headers, json=payload, timeout=s.timeout
                )
            if resp.status_code >= 400:
                logger.error("Sarvam API error %d: %s", resp.status_code, resp.text[:800])
                raise LLMError(f"Sarvam API error {resp.status_code}: {resp.text[:400]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise LLMError(f"Sarvam returned non-JSON body: {resp.text[:200]}") from e

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error("Unexpected response shape: %s", str(data)[:800])
            raise LLMError(f"Unexpected response shape: {str(data)[:400]}") from e

        usage = _usage_from(data, model)
        logger.info(
            "LLM response: %s in %.1fs, tokens in/out=%d/%d (task=%s)",
            model, time.monotonic() - started,
            usage["prompt_tokens"], usage["completion_tokens"], task or "-",
        )
        if content is None or not str(content).strip():
            logger.error("Empty content (task=%s). Raw: %s", task, str(data)[:1500])
            raise LLMError(
                "Model returned an empty response. Usual cause: thinking mode "
                "consumed the whole max_tokens budget. Set SARVAM_REASONING_EFFORT "
                "empty (off) or raise SARVAM_MAX_TOKENS."
            )
        content = str(content)
        return (content, usage) if return_usage else content


class MockLLM:
    """Deterministic stand-in for tests. Routes on the `task` label."""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.calls: List[Dict[str, str]] = []

    def chat(self, system, user, task="", want_json=False, model=None,
             return_usage=False, **kwargs):
        self.calls.append({"task": task, "system": system, "user": user})
        content = self.responses.get(task, self.responses.get("default", "{}"))
        if return_usage:
            return content, {
                "model": model or "mock",
                "prompt_tokens": 1200,
                "completion_tokens": 800,
                "cached_tokens": 0,
            }
        return content
