"""
app/graph/workflow.py
---------------------
LangGraph StateGraph builder.

Call count (total LLM calls for an N-page PDF):
  1 call  — segregator (all N pages classified in one Gemini vision call)
  1 call  — id_agent vision          (Gemini, all identity pages at once)
  1 call  — id_agent text extraction (Qwen)
  1 call  — discharge_agent vision   (Gemini, all discharge pages at once)
  1 call  — discharge_agent text extraction (Qwen)
  1 call  — bill_agent vision        (Gemini, all bill pages at once)
  1 call  — bill_agent text extraction (Qwen)
  ─────────────────────────────────────────────────────────────────────────
  Max 7 LLM calls total, regardless of page count.
"""
import functools
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.graph.state import PipelineState
from app.graph.nodes.segregator import segregator_node
from app.graph.nodes.page_agents import id_agent, discharge_agent, bill_agent
from app.graph.nodes.aggregator import aggregator_node
from app.llm.client import LLMPipeline

logger = logging.getLogger("uvicorn.error")


def _fan_out(state: PipelineState) -> list[Send] | str:
    """
    Conditional edge: emit the Send objects built by the segregator.
    Each Send delivers all pages of one type to its agent node.

    Fallback: if no pages were routed to any extraction agent (e.g. the
    entire PDF classified as 'other', 'claim_forms', etc.) return the
    string "aggregator" so the graph still reaches the aggregator node
    and produces a valid (empty-extraction) response instead of silently
    terminating mid-graph.
    """
    sends = state.get("_sends", [])
    if not sends:
        logger.warning(
            "Fan-out: no pages routed to any extraction agent — "
            "skipping directly to aggregator"
        )
        return "aggregator"
    logger.info(f"Fan-out: {len(sends)} agent branch(es) activated")
    return sends


def compile_graph(llm: LLMPipeline):
    """
    Build and compile the LangGraph StateGraph.

    Args:
        llm: LLMPipeline facade (Gemini vision + OpenRouter text).
             Created once at app startup via lifespan and shared across requests.

    Returns:
        Compiled graph ready for ainvoke().
    """
    builder = StateGraph(PipelineState)

    # ── Nodes (LLM injected via functools.partial) ────────────────────────────
    builder.add_node("segregator",     functools.partial(segregator_node, llm=llm))
    builder.add_node("id_agent",       functools.partial(id_agent, llm=llm))
    builder.add_node("discharge_agent",functools.partial(discharge_agent, llm=llm))
    builder.add_node("bill_agent",     functools.partial(bill_agent, llm=llm))
    builder.add_node("aggregator",     aggregator_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "segregator")

    # Fan-out: segregator → [id_agent | discharge_agent | bill_agent]
    # "aggregator" is included as a fallback target for when _sends is empty
    # (all pages classified as non-routed types like 'other', 'claim_forms').
    builder.add_conditional_edges(
        "segregator",
        _fan_out,
        ["id_agent", "discharge_agent", "bill_agent", "aggregator"],
    )

    # Fan-in: all three agents → aggregator
    builder.add_edge("id_agent",        "aggregator")
    builder.add_edge("discharge_agent", "aggregator")
    builder.add_edge("bill_agent",      "aggregator")
    builder.add_edge("aggregator", END)

    graph = builder.compile()
    logger.info("LangGraph pipeline compiled successfully")
    return graph
