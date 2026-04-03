"""
app/graph/nodes/page_agents.py
-------------------------------
Three extraction agent nodes:
  - id_agent        → all identity_document pages
  - discharge_agent → all discharge_summary pages
  - bill_agent      → all itemized_bill pages

Each agent receives ALL pages of its type in one AgentInput via Send().

Two-step pipeline (one Gemini call + one Qwen call each):
  1. Gemini vision call: ALL assigned pages in ONE call → rich Markdown
     The model transcribes all pages sequentially in a single response.
  2. Qwen text call: that Markdown → free-form JSON extraction
     No prescribed schema — the LLM decides what to extract.

Fail-fast policy on vision errors:
  If the Gemini vision call fails for any reason, the full error is logged
  and immediately re-raised — stopping that agent branch and propagating up
  to the routes layer as HTTP 500. Text extraction errors are logged and
  produce an empty extracted dict (non-fatal, agent continues).
"""
import logging
from typing import Any

from app.graph.state import AgentInput, AgentResult
from app.llm.client import LLMPipeline, LLMError

logger = logging.getLogger("uvicorn.error")


# ── Vision transcription prompt (all pages at once) ───────────────────────────

TRANSCRIBE_PROMPT = """\
You are a high-accuracy OCR engine for scanned medical and insurance documents.

I am showing you one or more scanned document pages (labeled ## Page N).
Transcribe ALL of them into clean, well-structured Markdown.

Rules:
1. Transcribe pages in order, keeping the ## Page N header for each.
2. Preserve ALL text exactly — no summarizing, no paraphrasing.
3. Render tables as Markdown tables with | column | separators |.
4. Use ## for section headings found on the page.
5. Use **Field Label:** value for key-value pairs.
6. Preserve all numbers, dates, amounts, codes exactly as printed.
7. Write [ILLEGIBLE] for genuinely unreadable text — do not guess.
8. Output ONLY the Markdown — nothing before or after it.
"""


# ── Extraction system prompts (per agent) ─────────────────────────────────────

ID_SYSTEM_PROMPT = """\
You are a data extractor for medical insurance claim processing.

You will receive Markdown transcriptions of one or more IDENTITY DOCUMENT pages
(Aadhaar, PAN card, passport, insurance ID card, policy document, etc.).

Extract every piece of identity and policy information present.
Return a single JSON object with all information found.
Use descriptive snake_case keys. Do not invent fields not in the text.
Do not omit any field that IS present.
Output ONLY valid JSON — no markdown fences, no explanation.
"""

DISCHARGE_SYSTEM_PROMPT = """\
You are a data extractor for medical insurance claim processing.

You will receive Markdown transcriptions of one or more DISCHARGE SUMMARY pages
(clinical discharge summary, case summary, inpatient progress notes, etc.).

Extract every piece of clinical, administrative, and patient information present.
Include patient details, hospital details, admission/discharge dates, diagnoses,
procedures, medications, physician names, department, discharge condition,
follow-up instructions — whatever is present in the document.
For list-type fields (diagnoses, procedures), use JSON arrays.
Return a single JSON object. Use descriptive snake_case keys.
Output ONLY valid JSON — no markdown fences, no explanation.
"""

BILL_SYSTEM_PROMPT = """\
You are a data extractor for medical insurance claim processing.

You will receive Markdown transcriptions of one or more ITEMIZED BILL pages
(hospital bills, pharmacy bills with individual line items and amounts).

Extract all billing details:
- Bill metadata (number, date, patient name, hospital/pharmacy name)
- ALL line items as a JSON array — each with description, quantity, rate, amount
- Subtotals, taxes, discounts, and grand total
- Payment details if present

Return a single JSON object. Use descriptive snake_case keys.
Preserve all numeric values and currency symbols exactly.
Output ONLY valid JSON — no markdown fences, no explanation.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "identity_document": ID_SYSTEM_PROMPT,
    "discharge_summary": DISCHARGE_SYSTEM_PROMPT,
    "itemized_bill":     BILL_SYSTEM_PROMPT,
}


# ── Shared two-step pipeline ──────────────────────────────────────────────────

async def _process_pages(
    state: AgentInput,
    llm: LLMPipeline,
) -> dict[str, Any]:
    """
    Processes ALL pages of a given doc_type in two steps:
      1. Single Gemini vision call → Markdown for all pages   [FAIL-FAST]
      2. Single Qwen text call    → free-form JSON extraction [logs on failure]

    Fail-fast on vision: if Gemini fails, logs reason and re-raises so the
    request stops immediately with HTTP 500. Text errors are non-fatal.

    Returns a state update with agent_results (one entry) and errors.
    """
    pages = state["pages"]
    doc_type = state["doc_type"]
    page_indices = [p["page_idx"] for p in sorted(pages, key=lambda x: x["page_idx"])]

    logger.info(
        f"[{doc_type}] Processing {len(pages)} page(s): {page_indices}"
    )

    # ── Step 1: All pages → single Gemini vision call → Markdown ─────────────
    try:
        markdown = await llm.vision_pages_to_markdown(pages, TRANSCRIBE_PROMPT)
        logger.info(
            f"[{doc_type}] Gemini transcription done: {len(markdown)} chars"
        )
    except LLMError as exc:
        # Vision failed — log full reason, stop this branch immediately
        logger.error(
            f"[{doc_type}] Gemini vision call FAILED — stopping request. "
            f"Pages: {page_indices}. Reason: {exc}"
        )
        raise  # propagates through LangGraph → routes.py → HTTP 500

    # ── Step 2: Markdown → single Qwen text call → JSON ──────────────────────
    system_prompt = SYSTEM_PROMPTS.get(
        doc_type,
        "Extract all information from this document as a JSON object with descriptive keys.",
    )
    extracted: dict[str, Any] = {}
    extraction_errors: list[str] = []

    try:
        extracted = await llm.markdown_to_json(system_prompt, markdown)
        logger.info(
            f"[{doc_type}] Qwen extraction done: {len(extracted)} top-level keys"
        )
    except LLMError as exc:
        # Text extraction failure is non-fatal — log and continue with empty dict
        logger.error(f"[{doc_type}] Qwen text extraction failed: {exc}")
        extraction_errors.append(
            f"Text LLM error for {doc_type} pages {page_indices}: {exc}"
        )

    result: AgentResult = {
        "doc_type": doc_type,
        "page_indices": page_indices,
        "markdown": markdown,
        "extracted": extracted,
    }

    return {
        "agent_results": [result],
        "errors": extraction_errors,
    }


# ── Agent node functions ──────────────────────────────────────────────────────

async def id_agent(
    state: AgentInput,
    *,
    llm: LLMPipeline,
) -> dict[str, Any]:
    """Extraction agent for identity_document pages."""
    return await _process_pages(state, llm)


async def discharge_agent(
    state: AgentInput,
    *,
    llm: LLMPipeline,
) -> dict[str, Any]:
    """Extraction agent for discharge_summary pages."""
    return await _process_pages(state, llm)


async def bill_agent(
    state: AgentInput,
    *,
    llm: LLMPipeline,
) -> dict[str, Any]:
    """Extraction agent for itemized_bill pages."""
    return await _process_pages(state, llm)
