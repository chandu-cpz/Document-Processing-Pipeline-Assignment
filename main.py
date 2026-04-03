"""
main.py
-------
FastAPI application entry point.

Run with:  fastapi dev main.py
       or: uvicorn main:app --reload

Logging uses uvicorn's built-in logger — no custom dictConfig.
All modules log via logging.getLogger("uvicorn.error").
"""
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.graph.workflow import compile_graph
from app.llm.client import OpenRouterClient, LLMPipeline
from app.llm.gemini_client import GeminiClient

logger = logging.getLogger("uvicorn.error")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup / shutdown lifecycle.

    One httpx.AsyncClient shared across all requests (connection pooling).
    Both LLM clients share it. LangGraph graph compiled once and reused.
    """
    logger.info("Starting up Document Processing Pipeline...")

    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive,
        ),
        timeout=httpx.Timeout(settings.http_timeout),
    )

    gemini_client = GeminiClient(http_client)
    openrouter_client = OpenRouterClient(http_client)
    llm_pipeline = LLMPipeline(gemini=gemini_client, openrouter=openrouter_client)

    graph = compile_graph(llm_pipeline)

    app.state.graph = graph
    app.state.settings = settings
    app.state.llm_pipeline = llm_pipeline

    logger.info(
        f"Pipeline ready — "
        f"vision={settings.vision_model!r}  text={settings.text_model!r}"
    )

    yield

    logger.info("Shutting down — closing HTTP client...")
    await http_client.aclose()
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered medical insurance document processing pipeline. "
        "Classifies all PDF pages in a single Gemini vision call, then routes "
        "relevant pages to extraction agents (one Gemini call per agent for "
        "transcription + one Qwen call per agent for JSON extraction). "
        "Returns free-form JSON — structure determined by the LLMs."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api", tags=["Processing"])


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    return {
        "status": "ok",
        "version": settings.app_version,
        "vision_model": settings.vision_model,
        "text_model": settings.text_model,
    }
