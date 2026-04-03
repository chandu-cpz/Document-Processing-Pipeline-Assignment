"""
app/api/routes.py
-----------------
POST /api/process — the single API endpoint.

Accepts:
  - claim_id (str, form field)
  - file     (PDF, multipart upload)

Flow:
  1. Validate file type
  2. Read raw bytes from UploadFile (async)
  3. Convert PDF → base64 PNG pages (async, thread-offloaded)
  4. Build initial PipelineState
  5. Run graph.ainvoke(state) — fully async LangGraph execution
  6. Return final_output as JSON

All errors are surfaced as appropriate HTTP exceptions with clear messages.
"""
import logging

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.core.pdf_utils import pdf_to_pages
from app.graph.state import PipelineState
from app.schemas.models import ErrorResponse, ProcessingResponse

logger = logging.getLogger("uvicorn.error")

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/octet-stream",   # some clients send this for PDFs
    "application/x-pdf",
}


@router.post(
    "/process",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file or input"},
        422: {"model": ErrorResponse, "description": "PDF processing failed"},
        500: {"model": ErrorResponse, "description": "Pipeline execution error"},
    },
    summary="Process a medical insurance claim PDF",
    description=(
        "Upload a PDF and a claim ID. The pipeline will:\n"
        "1. Classify each page using a vision LLM (9 document categories)\n"
        "2. Route relevant pages to the appropriate extraction agents\n"
        "3. Use a vision model to transcribe each page to Markdown\n"
        "4. Use a text model to extract structured information\n"
        "5. Return all extracted data plus raw Markdown for debugging"
    ),
)
async def process_document(
    request: Request,
    claim_id: str = Form(..., description="Unique claim identifier", min_length=1),
    file: UploadFile = File(..., description="PDF document to process"),
) -> JSONResponse:
    logger.info(f"Processing request: claim_id={claim_id!r}, file={file.filename!r}")

    # ── Validate content type ─────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                "Only PDF files are accepted."
            ),
        )

    # ── Read and validate file bytes ──────────────────────────────────────────
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    max_size_bytes = request.app.state.settings.max_pdf_size_mb * 1024 * 1024
    if len(pdf_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded PDF exceeds the maximum size of {request.app.state.settings.max_pdf_size_mb} MB."
        )

    logger.info(f"PDF received: {len(pdf_bytes):,} bytes")

    # ── Convert PDF → pages ───────────────────────────────────────────────────
    dpi = request.app.state.settings.pdf_dpi
    try:
        pages = await pdf_to_pages(pdf_bytes, dpi=dpi)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}")
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=f"PDF render error: {exc}")

    if not pages:
        raise HTTPException(status_code=422, detail="PDF contains no readable pages.")

    logger.info(f"PDF converted: {len(pages)} pages")

    # ── Build initial pipeline state ──────────────────────────────────────────
    initial_state: PipelineState = {
        "claim_id": claim_id,
        "pages": pages,
        "page_assignments": {},
        "_sends": [],
        "agent_results": [],
        "final_output": {},
        "errors": [],
    }

    # ── Execute LangGraph pipeline ────────────────────────────────────────────
    graph = request.app.state.graph
    try:
        result = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.exception(f"Pipeline execution failed for claim_id={claim_id!r}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {exc}",
        )

    final_output = result.get("final_output", {})
    logger.info(
        f"Pipeline complete: claim_id={claim_id!r}, "
        f"status={final_output.get('processing_status')}, "
        f"pages={final_output.get('total_pages')}"
    )

    return JSONResponse(content=final_output)
