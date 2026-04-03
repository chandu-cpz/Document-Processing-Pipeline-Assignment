"""
app/graph/nodes/aggregator.py
------------------------------
Aggregator node — merges all AgentResults into final_output.

Since each agent now processes all its pages in a single call and returns
one AgentResult, this is a simple key-based merge (no multi-page reducer
needed at this level).
"""
import logging
from typing import Any

from app.graph.state import PipelineState, AgentResult

logger = logging.getLogger("uvicorn.error")


async def aggregator_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: collapses all AgentResults into the final response payload.
    """
    all_results: list[AgentResult] = state.get("agent_results", [])

    # Key by doc_type (each type has exactly one result now)
    by_type: dict[str, AgentResult] = {r["doc_type"]: r for r in all_results}

    logger.info(
        f"Aggregator: received results from {len(all_results)} agent(s): "
        f"{list(by_type.keys())}"
    )

    extracted_data: dict[str, Any] = {
        "identity":          by_type.get("identity_document", {}).get("extracted", {}),
        "discharge_summary": by_type.get("discharge_summary", {}).get("extracted", {}),
        "itemized_bill":     by_type.get("itemized_bill", {}).get("extracted", {}),
    }

    raw_markdown: dict[str, str] = {
        "identity_pages":  by_type.get("identity_document", {}).get("markdown", ""),
        "discharge_pages": by_type.get("discharge_summary", {}).get("markdown", ""),
        "bill_pages":      by_type.get("itemized_bill", {}).get("markdown", ""),
    }

    final_output: dict[str, Any] = {
        "claim_id":            state["claim_id"],
        "processing_status":   "success" if not state.get("errors") else "partial",
        "total_pages":         len(state["pages"]),
        "page_classification": state.get("page_assignments", {}),
        "extracted_data":      extracted_data,
        "raw_markdown":        raw_markdown,
        "errors":              state.get("errors", []),
    }

    return {"final_output": final_output}
