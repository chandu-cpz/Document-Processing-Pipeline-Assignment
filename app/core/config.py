"""
app/core/config.py
------------------
Type-safe settings loaded from .env via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── OpenRouter (text extraction) ────────────────────────────────────────
    openrouter_api_key: str

    # Text model: Markdown → structured JSON
    text_model: str = "qwen/qwen3.6-plus:free"

    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"

    # ── Google Gemini (vision / OCR) ────────────────────────────────────────
    google_api_key: str

    # Vision model: PDF page images → Markdown / classification
    vision_model: str = "gemini-3-flash-preview"

    # ── Token limits (generous — let the model decide when to stop) ─────────
    # Vision: 18-page transcription can easily need 15k-30k tokens
    vision_max_tokens: int = 65536

    # Text: JSON extraction output; very liberal upper bound
    text_max_tokens: int = 32768

    # ── HTTP / retry ────────────────────────────────────────────────────────
    max_retries: int = 3
    retry_backoff_base: float = 2.0   # wait = base ** attempt (seconds)
    http_max_connections: int = 10
    http_max_keepalive: int = 5
    http_timeout: float = 300.0        # 5 min — generous for large PDFs

    # ── PDF rendering ───────────────────────────────────────────────────────
    pdf_dpi: int = 150
    max_pdf_size_mb: int = 50

    # ── App meta ────────────────────────────────────────────────────────────
    app_name: str = "Document Processing Pipeline"
    app_version: str = "1.0.0"
    debug: bool = False


settings = Settings()
