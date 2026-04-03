"""
app/graph/nodes/segregator.py
------------------------------
Segregator Agent node.

ONE single Gemini vision call with ALL page images.
The model sees every page simultaneously and classifies them all at once.
This avoids per-page API calls (eliminating 429 rate limiting) and lets
the model use context from adjacent pages when classifying ambiguous ones.

Fail-fast policy: if the vision call fails for any reason, the error is
logged in full and immediately re-raised — stopping the entire request with
a clear 500 response. There is no silent fallback to "other" on LLM errors.

Returns page_assignments and a list of Send() objects — one per agent type
that has at least one page assigned to it.
"""
import json
import logging
from typing import Any

from langgraph.types import Send

from app.graph.state import (
    DOC_TYPES,
    AGENT_NODE_MAP,
    PageData,
    PipelineState,
)
from app.llm.client import LLMPipeline, LLMError

logger = logging.getLogger("uvicorn.error")


# ── Prompt for the single multi-image classification call ─────────────────────

CLASSIFY_PROMPT = """\
You are a medical insurance document analyst.

I am showing you ALL pages of a scanned insurance claim document packet.
Each page is labeled with its page number (e.g. "## Page 0", "## Page 1", etc.).

Classify EVERY page into EXACTLY ONE of the following document types:

  claim_forms             - Insurance claim forms, pre-authorization forms, TPA forms
  cheque_or_bank_details  - Cancelled cheques, bank statements, account/IFSC details
  identity_document       - Aadhaar, PAN, passport, driving licence, insurance/policy ID card
  itemized_bill           - Hospital/pharmacy bills showing individual line items with costs
  discharge_summary       - Clinical discharge summary, case summary, inpatient progress notes
  prescription            - Doctor's prescription, medication list, outpatient slip
  investigation_report    - Lab report, blood test, X-ray, MRI, CT scan, pathology report
  cash_receipt            - Payment receipt, cash memo, acknowledgement of payment
  other                   - Anything not fitting the above categories

Rules:
- Examine each page image carefully before classifying.
- If a page contains multiple types, pick the dominant one.
- You MUST classify every page shown — do not skip any.
- Return ONLY a JSON array like this (no markdown, no explanation):

[
  {"page_idx": 0, "doc_type": "claim_forms"},
  {"page_idx": 1, "doc_type": "identity_document"},
  ...
]
"""


async def segregator_node(
    state: PipelineState,
    *,
    llm: LLMPipeline,
) -> dict[str, Any]:
    """
    LangGraph node: classifies ALL pages in ONE Gemini vision call.

    Fail-fast: if the vision call fails, logs the full error reason and
    raises immediately — the request is stopped and returns HTTP 500.

    On successful vision call but bad JSON from model: falls back to
    classifying all pages as 'other' (model error, not infra error).
    """
    pages = state["pages"]
    errors: list[str] = []

    # ── Single Gemini call: all pages at once ─────────────────────────────────
    idx_to_type: dict[int, str] = {}
    try:
        logger.info(
            f"Segregator: sending {len(pages)} pages to Gemini vision model "
            f"({state.get('claim_id', '?')!r})"
        )
        raw = await llm.vision_pages_to_markdown(pages, CLASSIFY_PROMPT)

    except LLMError as exc:
        # Vision failed — log full reason, stop the request immediately
        logger.error(
            f"Segregator: Gemini vision call FAILED — stopping request. "
            f"Reason: {exc}"
        )
        raise  # propagates through LangGraph → caught in routes.py → HTTP 500

    # ── Parse classification JSON ─────────────────────────────────────────────
    try:
        cleaned = raw.strip()
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end + 1]

        classifications: list[dict] = json.loads(cleaned)

        for item in classifications:
            idx = int(item.get("page_idx", -1))
            doc_type = str(item.get("doc_type", "other")).strip().lower()
            if doc_type not in DOC_TYPES:
                logger.warning(
                    f"Page {idx}: unknown type '{doc_type}' from model → 'other'"
                )
                doc_type = "other"
            idx_to_type[idx] = doc_type
            logger.info(f"  Page {idx} → {doc_type}")

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        # Model returned bad JSON — log and fall back (this is a model quality
        # issue, not an infra failure, so we continue with 'other' fallback)
        logger.warning(
            f"Segregator: failed to parse classification JSON: {exc} — "
            "all pages marked as 'other'"
        )
        errors.append(f"Segregator classification parse error: {exc}")
        idx_to_type = {p["page_idx"]: "other" for p in pages}

    # ── Fill in any missing page indices ─────────────────────────────────────
    page_lookup: dict[int, PageData] = {p["page_idx"]: p for p in pages}
    for p in pages:
        if p["page_idx"] not in idx_to_type:
            logger.warning(f"Page {p['page_idx']} missing from model response → 'other'")
            idx_to_type[p["page_idx"]] = "other"

    # ── Build page_assignments (only real page indices) ───────────────────────
    page_assignments: dict[str, list[int]] = {t: [] for t in DOC_TYPES}
    for idx, doc_type in idx_to_type.items():
        if idx not in page_lookup:
            logger.warning(
                f"Segregator: model returned phantom page_idx={idx} "
                f"(PDF has {len(pages)} pages, valid range 0-{len(pages)-1}) — skipping"
            )
            continue
        page_assignments[doc_type].append(idx)

    for v in page_assignments.values():
        v.sort()

    # ── Build ONE Send per agent type (all pages of that type bundled) ────────
    sends: list[Send] = []
    for doc_type, node_name in AGENT_NODE_MAP.items():
        assigned_indices = page_assignments.get(doc_type, [])
        if not assigned_indices:
            continue
        agent_pages = [page_lookup[idx] for idx in assigned_indices]
        sends.append(
            Send(
                node_name,
                {
                    "pages": agent_pages,
                    "doc_type": doc_type,
                },
            )
        )
        logger.info(
            f"Routing {len(agent_pages)} page(s) → '{node_name}': "
            f"pages {assigned_indices}"
        )

    logger.info(
        f"Segregator done: {len(pages)} pages classified, "
        f"{len(sends)} agent(s) activated"
    )

    return {
        "page_assignments": page_assignments,
        "_sends": sends,
        "errors": errors,
    }
