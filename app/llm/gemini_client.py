import asyncio
import logging
from typing import Any

import httpx

from app.core.config import settings
from app.llm.client import LLMError

logger = logging.getLogger("uvicorn.error")

# Gemini generateContent endpoint template
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/{model}:generateContent"
)


class GeminiError(LLMError):
    """Raised when a Gemini API call fails — subclass of LLMError so nodes
    that catch LLMError automatically handle Gemini failures too."""


class GeminiClient:
    """
    Thin async wrapper around the Gemini generateContent REST API.

    Args:
        http_client: Shared httpx.AsyncClient created once at app startup.
    """

    def __init__(self, http_client: httpx.AsyncClient) -> None:
        self._client = http_client
        self._model = settings.vision_model      # e.g. "gemini-3-flash-preview"
        self._api_key = settings.google_api_key

    # ──────────────────────────────────────────────────────────────────────────
    # Internal retry layer
    # ──────────────────────────────────────────────────────────────────────────

    async def _post(self, payload: dict[str, Any]) -> str:
        """
        POST to Gemini with exponential backoff on 429 / 5xx.

        Returns the raw text from candidates[0].content.parts[0].text.
        Raises GeminiError (logs reason first) after all retries.
        """
        url = (
            f"{_GEMINI_ENDPOINT.format(model=self._model)}"
            f"?key={self._api_key}"
        )
        last_exc: Exception | None = None

        for attempt in range(settings.max_retries):
            try:
                response = await self._client.post(url, json=payload)

                if response.status_code == 429 or response.status_code >= 500:
                    wait = settings.retry_backoff_base ** attempt
                    snippet = response.text[:300]
                    logger.warning(
                        f"Gemini HTTP {response.status_code} on attempt "
                        f"{attempt + 1}/{settings.max_retries} — "
                        f"body: {snippet!r} — retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()

                # Extract text from Gemini response envelope
                try:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError, TypeError) as exc:
                    reason = (
                        f"Unexpected Gemini response structure: {exc!r} | "
                        f"top-level keys: {list(data.keys())} | "
                        f"raw (first 400 chars): {str(data)[:400]}"
                    )
                    logger.error(f"Gemini response parse error — {reason}")
                    raise GeminiError(reason) from exc

                logger.debug(
                    f"Gemini OK — model={self._model!r} "
                    f"response_len={len(text)} chars"
                )
                return text

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                wait = settings.retry_backoff_base ** attempt
                logger.warning(
                    f"Gemini network error on attempt {attempt + 1}"
                    f"/{settings.max_retries}: {exc!r} — "
                    f"retrying in {wait:.1f}s"
                )
                last_exc = exc
                await asyncio.sleep(wait)

            except httpx.HTTPStatusError as exc:
                body = exc.response.text[:400]
                reason = (
                    f"Gemini HTTP {exc.response.status_code}: {body!r}"
                )
                logger.error(f"Gemini fatal HTTP error — {reason}")
                raise GeminiError(reason) from exc

        reason = (
            f"Gemini call failed after {settings.max_retries} attempts. "
            f"Last error: {last_exc!r}"
        )
        logger.error(reason)
        raise GeminiError(reason)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API (matches OpenRouterClient interface for drop-in use)
    # ──────────────────────────────────────────────────────────────────────────

    async def vision_pages_to_markdown(
        self,
        pages: list[dict],  # list of {page_idx, b64_image, mime_type}
        prompt: str,
    ) -> str:
        """
        Send ALL page images in ONE Gemini call.

        Parts layout:
          [system_prompt_text, "## Page 0", image_0, "## Page 1", image_1, ...]

        Args:
            pages:  Sorted list of page dicts (page_idx, b64_image, mime_type).
            prompt: Instruction text sent as the first part.

        Returns:
            Raw string response from Gemini.
        Raises:
            GeminiError (logs reason before raising) on any failure.
        """
        parts: list[dict] = [{"text": prompt}]

        for page in sorted(pages, key=lambda p: p["page_idx"]):
            idx = page["page_idx"]
            b64 = page["b64_image"]
            mime = page.get("mime_type", "image/jpeg")
            parts.append({"text": f"## Page {idx}"})
            parts.append({
                "inline_data": {
                    "mime_type": mime,
                    "data": b64,
                }
            })

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": settings.vision_max_tokens,
            },
        }

        logger.info(
            f"Gemini vision call → model={self._model!r}, "
            f"pages={len(pages)}, max_tokens={settings.vision_max_tokens}"
        )
        return await self._post(payload)
