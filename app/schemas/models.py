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
