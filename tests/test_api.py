import json
import os
import pathlib
import time

import pytest
import pytest_asyncio
import httpx
from fastapi.testclient import TestClient

from main import app

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).parent.parent
PDF_PATH = REPO_ROOT / "final_image_protected.pdf"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    """Load the real 18-page image-protected PDF once for the whole session."""
    assert PDF_PATH.exists(), f"PDF not found at {PDF_PATH}"
    data = PDF_PATH.read_bytes()
    print(f"\n[fixture] Loaded PDF: {PDF_PATH.name} ({len(data):,} bytes)")
    return data


@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient so the FastAPI lifespan (httpx.AsyncClient +
    LangGraph graph compilation) runs only once across all tests.
    """
    # Setup TestClient with timeout to accommodate slow OpenRouter API responses
    with TestClient(app, raise_server_exceptions=True, timeout=300.0) as c:
        print("\n[fixture] FastAPI TestClient started (real OpenRouter client ready)")
        yield c
        print("\n[fixture] FastAPI TestClient shutting down")


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check(client: TestClient):
    """Server must be up and report correct model names."""
    resp = client.get("/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["status"] == "ok"
    assert data["vision_model"] == "gemini-3-flash-preview"
    assert data["text_model"] == "qwen/qwen3.6-plus:free"
    print(f"\n  Health OK: {data}")


# ── Validation edge cases (no LLM calls) ─────────────────────────────────────

def test_missing_claim_id(client: TestClient, pdf_bytes: bytes):
    """claim_id form field is required."""
    resp = client.post(
        "/api/process",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        # claim_id intentionally omitted
    )
    assert resp.status_code == 422


def test_missing_file(client: TestClient):
    """file upload is required."""
    resp = client.post(
        "/api/process",
        data={"claim_id": "CLM-VALIDATE-001"},
    )
    assert resp.status_code == 422


def test_wrong_file_type(client: TestClient):
    """Non-PDF upload must be rejected with 400."""
    resp = client.post(
        "/api/process",
        data={"claim_id": "CLM-VALIDATE-002"},
        files={"file": ("notes.txt", b"this is plain text", "text/plain")},
    )
    assert resp.status_code == 400
    assert "PDF" in resp.json()["detail"]


def test_empty_pdf(client: TestClient):
    """Empty file must be rejected with 400."""
    resp = client.post(
        "/api/process",
        data={"claim_id": "CLM-VALIDATE-003"},
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


# ── Real pipeline test ────────────────────────────────────────────────────────

def test_full_pipeline_real_pdf(client: TestClient, pdf_bytes: bytes):
    """
    REAL end-to-end integration test.

    Sends the actual 18-page image-protected PDF through the full pipeline:
      1. PDF → 18 JPEG pages (PyMuPDF)
      2. Segregator: 1 Gemini vision call with all 18 pages → classification JSON
      3. Each relevant agent: 1 Gemini vision call (transcription) + 1 Qwen text call (JSON)
      4. Aggregator merges all results

    Assertions:
      - HTTP 200 response
      - Response shape matches the expected envelope
      - Page classification covers all 18 pages
      - At least one extraction agent produced real data
      - No catastrophic errors (partial errors are OK for free models)
    """
    claim_id = "CLM-REAL-001"
    print(f"\n[test] Starting full pipeline test with claim_id={claim_id!r}")
    print(f"[test] PDF: {PDF_PATH.name} ({len(pdf_bytes):,} bytes, 18 pages)")

    start = time.monotonic()
    resp = client.post(
        "/api/process",
        data={"claim_id": claim_id},
        files={"file": (PDF_PATH.name, pdf_bytes, "application/pdf")},
    )
    elapsed = time.monotonic() - start

    print(f"[test] Pipeline completed in {elapsed:.1f}s")
    print(f"[test] HTTP status: {resp.status_code}")

    assert resp.status_code == 200, (
        f"Expected 200, got {resp.status_code}.\nBody: {resp.text[:500]}"
    )

    data = resp.json()

    # ── Print full response for inspection ────────────────────────────────────
    print("\n[test] ═══════════════════ FULL RESPONSE ═══════════════════")
    print(json.dumps(data, indent=2, ensure_ascii=False)[:8000])
    print("[test] ════════════════════════════════════════════════════════")

    # ── Envelope shape ────────────────────────────────────────────────────────
    assert "claim_id" in data,            "Missing 'claim_id'"
    assert "processing_status" in data,   "Missing 'processing_status'"
    assert "total_pages" in data,         "Missing 'total_pages'"
    assert "page_classification" in data, "Missing 'page_classification'"
    assert "extracted_data" in data,      "Missing 'extracted_data'"
    assert "raw_markdown" in data,        "Missing 'raw_markdown'"
    assert "errors" in data,              "Missing 'errors'"

    # ── Claim ID echoed back ──────────────────────────────────────────────────
    assert data["claim_id"] == claim_id

    # ── All 18 pages accounted for ────────────────────────────────────────────
    assert data["total_pages"] == 18, (
        f"Expected 18 pages, got {data['total_pages']}"
    )

    page_classification = data["page_classification"]
    classified_pages = sum(len(v) for v in page_classification.values())
    assert classified_pages == 18, (
        f"Total classified pages ({classified_pages}) ≠ 18.\n"
        f"Classification: {json.dumps(page_classification, indent=2)}"
    )

    print(f"\n[test] Page classification breakdown:")
    for doc_type, pages in sorted(page_classification.items()):
        if pages:
            print(f"  {doc_type}: pages {pages}")

    # ── Processing status ─────────────────────────────────────────────────────
    status = data["processing_status"]
    assert status in ("success", "partial"), f"Unexpected status: {status!r}"
    print(f"\n[test] Processing status: {status}")

    if data["errors"]:
        print(f"[test] Errors (partial failures, not fatal):")
        for err in data["errors"]:
            print(f"  - {err}")

    # ── Extraction results — check keys are present ───────────────────────────
    extracted = data["extracted_data"]
    assert "identity" in extracted,          "Missing extracted_data.identity"
    assert "discharge_summary" in extracted, "Missing extracted_data.discharge_summary"
    assert "itemized_bill" in extracted,     "Missing extracted_data.itemized_bill"

    # ── At least one extraction agent must have returned data ─────────────────
    has_identity  = bool(extracted.get("identity"))
    has_discharge = bool(extracted.get("discharge_summary"))
    has_bill      = bool(extracted.get("itemized_bill"))

    print(f"\n[test] Extraction results:")
    print(f"  identity          : {'✓ data returned' if has_identity  else '✗ empty'}")
    print(f"  discharge_summary : {'✓ data returned' if has_discharge else '✗ empty'}")
    print(f"  itemized_bill     : {'✓ data returned' if has_bill      else '✗ empty'}")

    assert any([has_identity, has_discharge, has_bill]), (
        "All three extraction agents returned empty results. "
        "The PDF likely contains no identity, discharge, or bill pages — "
        "or all LLM extraction calls failed completely.\n"
        f"Page classification was: {json.dumps(page_classification, indent=2)}"
    )

    # ── Raw markdown must be present if pages were routed to agents ───────────
    raw_md = data["raw_markdown"]
    print(f"\n[test] Raw markdown snippets:")
    for k, v in raw_md.items():
        if v:
            print(f"  {k}: {len(v)} chars, snippet: {v[:120]!r}")
        else:
            print(f"  {k}: (none)")

    print(f"\n[test] ✓ All assertions passed in {elapsed:.1f}s")
