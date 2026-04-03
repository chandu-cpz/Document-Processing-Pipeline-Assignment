"""
app/schemas/models.py
---------------------
Minimal API envelope only.
We make ZERO assumptions about what fields the LLMs will extract —
that's entirely up to the models based on what they see in the document.
extracted_data is a free-form dict: whatever the LLM returns, we pass through.
"""
from typing import Any
from pydantic import BaseModel, Field


class ProcessingResponse(BaseModel):
    claim_id: str
    processing_status: str = Field(
        ...,
        description="'success' if no errors occurred, 'partial' if some pages failed",
    )
    total_pages: int
    page_classification: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Maps each doc type to its assigned page indices",
    )
    extracted_data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form extraction results keyed by agent type "
            "(identity, discharge_summary, itemized_bill). "
            "Structure is determined entirely by the LLM — not prescribed."
        ),
    )
    raw_markdown: dict[str, str] = Field(
        default_factory=dict,
        description="Raw markdown per doc type from the Gemini vision model, for debugging",
    )
    errors: list[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
