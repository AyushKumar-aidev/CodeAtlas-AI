"""
rag/llm.py

Talks to Google's Gemini REST API with only the standard library. The API key is
sent in a header, is never logged, and never appears in error messages.
Anything with a `generate(system, contents) -> (text, model)` method can stand in
for it (the tests use a fake).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

from codeatlas.config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Message is safe to show to the user."""


class GeminiClient:
    def __init__(self, settings: Settings, sleep=time.sleep) -> None:
        self._s = settings
        self._sleep = sleep

    def _call(self, model: str, system: str, contents: list[dict]):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.2},
        }
        url = f"{self._s.gemini_base_url.rstrip('/')}/{model}:generateContent"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"x-goog-api-key": self._s.gemini_api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._s.llm_timeout) as r:
                return 200, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, None
        except Exception:
            return 0, None

    @staticmethod
    def _text(data: dict) -> str:
        fb = data.get("promptFeedback") or {}
        if fb.get("blockReason"):
            raise LLMError("The model declined to answer that question.")
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError("The model returned no answer. Try rephrasing.")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        if not text:
            raise LLMError("The model returned an empty answer. Try rephrasing.")
        return text

    def generate(self, system: str, contents: list[dict]) -> tuple[str, str]:
        models = [self._s.llm_model]
        if self._s.llm_fallback_model and self._s.llm_fallback_model != self._s.llm_model:
            models.append(self._s.llm_fallback_model)
        last = 0
        for model in models:
            for attempt in range(2):
                code, data = self._call(model, system, contents)
                last = code
                if code == 200:
                    return self._text(data), model
                if code in (429, 500, 502, 503, 504, 0):
                    log.warning("LLM %s returned %s", model, code)
                    if attempt == 0:
                        self._sleep(1.5)
                    continue
                break  # 400/401/403/404: retrying the same model will not help
            if last in (400, 401, 403):
                break
        if last in (401, 403):
            raise LLMError("The Gemini API key was rejected. Check GEMINI_API_KEY.")
        if last == 429:
            raise LLMError("The free LLM quota is used up for now. Try again in a minute.")
        raise LLMError("The language model is unavailable right now. Please try again.")
