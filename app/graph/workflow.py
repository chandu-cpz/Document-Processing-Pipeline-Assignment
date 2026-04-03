import functools
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.graph.state import PipelineState, AGENT_NODE_MAP
from app.graph.nodes.segregator import segregator_node
from app.graph.nodes.page_agents import id_agent, discharge_agent, bill_agent
from app.graph.nodes.aggregator import aggregator_node
from app.llm.client import LLMPipeline

logger = logging.getLogger("uvicorn.error")


def _fan_out(state: PipelineState) -> list[Send] | str:
    """
    Conditional edge: emit the Send objects based on segregator's page_assignments.
    Each Send delivers all pages of one type to its agent node.

    Fallback: if no pages were routed to any extraction agent (e.g. the
    entire PDF classified as 'other', 'claim_forms', etc.) return the
    string "aggregator" so the graph still reaches the aggregator node
    and produces a valid (empty-extraction) response instead of silently
    terminating mid-graph.
    """
    page_assignments = state.get("page_assignments", {})
    pages = state.get("pages", [])
    sends: list[Send] = []

    for doc_type, node_name in AGENT_NODE_MAP.items():
        assigned_indices = page_assignments.get(doc_type, [])
        if not assigned_indices:
            continue
            
        # Filter the original pages list without building a full lookup dictionary
        assigned_set = set(assigned_indices)
        agent_pages = [p for p in pages if p["page_idx"] in assigned_set]
        
        if not agent_pages:
            continue
            
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
            f"Fan-out: routing {len(agent_pages)} page(s) → '{node_name}': "
            f"pages {assigned_indices}"
        )

    if not sends:
        logger.warning(
            "Fan-out: no pages routed to any extraction agent — "
            "skipping directly to aggregator"
        )
        return "aggregator"

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
