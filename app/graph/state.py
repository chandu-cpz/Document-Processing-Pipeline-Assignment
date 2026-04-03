"""
app/graph/state.py
------------------
LangGraph shared state definitions.

Architecture (single-call-per-agent):
  - Segregator → ONE vision call with ALL page images → classifies all pages
  - Each agent → ONE vision call with ALL its assigned pages → transcribes all
                  ONE text call → extracts JSON from the combined markdown
  - Send() emits ONE message per agent TYPE (not per page)
"""
from typing import Annotated, Any
from typing_extensions import TypedDict
import operator


# ── Valid document types ──────────────────────────────────────────────────────

DOC_TYPES: list[str] = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

# Maps doc_type → LangGraph node name
AGENT_NODE_MAP: dict[str, str] = {
    "identity_document": "id_agent",
    "discharge_summary": "discharge_agent",
    "itemized_bill":     "bill_agent",
}


# ── Per-page image data ───────────────────────────────────────────────────────

class PageData(TypedDict):
    page_idx: int
    b64_image: str    # JPEG bytes encoded as base64 (no data-URL prefix)
    mime_type: str    # e.g. "image/jpeg" — always set by pdf_to_pages()


# ── Sub-state for each agent node (sent via Send()) ──────────────────────────

class AgentInput(TypedDict):
    """
    All pages of a single doc type, delivered to the extraction agent in one shot.
    Each agent receives this once (not once per page).
    """
    pages: list[PageData]   # all pages of this doc_type, sorted by page_idx
    doc_type: str


# ── Per-agent extraction result accumulated by the fan-in reducer ─────────────

class AgentResult(TypedDict):
    doc_type: str
    page_indices: list[int]       # which pages were processed
    markdown: str                 # full markdown from the single vision call
    extracted: dict[str, Any]     # free-form JSON from the text model


def _accumulate(
    existing: list[AgentResult],
    incoming: list[AgentResult],
) -> list[AgentResult]:
    """Reducer: safely accumulate results across parallel agent branches."""
    return existing + incoming


# ── Top-level pipeline state ──────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Input ───────────────────────────────────────────────────────────────
    claim_id: str
    pages: list[PageData]

    # ── Filled by segregator ────────────────────────────────────────────────
    page_assignments: dict[str, list[int]]   # doc_type → [page_idx, ...]

    # ── Filled by agent nodes (fan-in via reducer) ──────────────────────────
    agent_results: Annotated[list[AgentResult], _accumulate]

    # ── Filled by aggregator ────────────────────────────────────────────────
    final_output: dict[str, Any]

    # ── Accumulated errors ──────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
