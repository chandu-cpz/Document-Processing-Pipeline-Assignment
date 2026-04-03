"""
app/llm/client.py
-----------------
OpenRouter text-only client + LLMPipeline facade.

LLMPipeline is the single object injected into all graph nodes:
  - vision_pages_to_markdown  → GeminiClient  (Google AI direct API)
  - markdown_to_json          → OpenRouterClient (OpenAI-compat API)

Keeping the two clients separate allows independent retries, timeouts,
and model configuration while presenting a unified interface to nodes.
"""
import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from app.core.config import settings

if TYPE_CHECKING:
    from app.llm.gemini_client import GeminiClient

logger = logging.getLogger("uvicorn.error")


# ── Shared error base ─────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base error for all LLM call failures (OpenRouter + Gemini)."""


# ── OpenRouter client (text model only) ──────────────────────────────────────

class OpenRouterClient:
    """
    Wraps OpenRouter's OpenAI-compatible chat completions API.
    Used ONLY for text extraction (Markdown → JSON).

    Args:
        http_client: Shared httpx.AsyncClient — created once at app startup.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client
        self._headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/doc-processing-pipeline",
            "X-Title": settings.app_name,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal retry layer
    # ─────────────────────────────────────────────────────────────────────────

    async def _post(self, payload: dict[str, Any]) -> str:
        """
        POST to OpenRouter with exponential backoff on 429 / 5xx.

        Returns the raw content string from choices[0].message.content.
        Raises LLMError after all retries are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(settings.max_retries):
            try:
                response = await self._client.post(
                    settings.openrouter_base_url,
                    json=payload,
                    headers=self._headers,
                )

                if response.status_code == 429 or response.status_code >= 500:
                    wait = settings.retry_backoff_base ** attempt
                    logger.warning(
                        f"OpenRouter {response.status_code} — "
                        f"attempt {attempt + 1}/{settings.max_retries}, "
                        f"retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.debug(
                    f"OpenRouter OK — model={payload.get('model','?')} "
                    f"len={len(content)} chars"
                )
                return content

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                wait = settings.retry_backoff_base ** attempt
                logger.warning(
                    f"OpenRouter network error attempt {attempt + 1}: {exc!r} — "
                    f"retrying in {wait:.1f}s"
                )
                last_exc = exc
                await asyncio.sleep(wait)

            except httpx.HTTPStatusError as exc:
                raise LLMError(f"OpenRouter HTTP error: {exc}") from exc

        raise LLMError(
            f"OpenRouter call failed after {settings.max_retries} attempts. "
            f"Last error: {last_exc!r}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def markdown_to_json(
        self,
        system_prompt: str,
        markdown_text: str,
    ) -> dict[str, Any]:
        """
        Feed Markdown text to the text model → free-form JSON dict.

        JSON is enforced via the prompt itself (no response_format parameter)
        for maximum model compatibility.

        Returns a dict — empty dict on parse failure (logged as warning).
        """
        payload = {
            "model": settings.text_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Here is the Markdown transcription of the document:\n\n"
                        f"---\n{markdown_text}\n---\n\n"
                        "Extract all relevant information and return it as a single "
                        "valid JSON object. Output ONLY the JSON — no markdown fences, "
                        "no explanation, no preamble."
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": settings.text_max_tokens,
        }
        raw = await self._post(payload)

        # Robust JSON extraction: find the outermost { ... } block
        cleaned = raw.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                f"JSON parse failed for text model output: {exc} | "
                f"Raw (first 300 chars): {raw[:300]!r}"
            )
            return {}


# ── Unified pipeline facade ───────────────────────────────────────────────────

class LLMPipeline:
    """
    Unified facade injected into all graph nodes via functools.partial.

    Routes:
      vision_pages_to_markdown  → GeminiClient  (Google AI, gemini-3-flash-preview)
      markdown_to_json          → OpenRouterClient (qwen/qwen3.6-plus:free)
    """

    def __init__(
        self,
        gemini: "GeminiClient",
        openrouter: OpenRouterClient,
    ) -> None:
        self._gemini = gemini
        self._openrouter = openrouter

    async def vision_pages_to_markdown(
        self,
        pages: list[dict],
        prompt: str,
    ) -> str:
        """Delegates to GeminiClient — raises GeminiError (subclass of LLMError) on failure."""
        return await self._gemini.vision_pages_to_markdown(pages, prompt)

    async def markdown_to_json(
        self,
        system_prompt: str,
        markdown_text: str,
    ) -> dict[str, Any]:
        """Delegates to OpenRouterClient — raises LLMError on failure."""
        return await self._openrouter.markdown_to_json(system_prompt, markdown_text)
