# Document Processing Pipeline

An AI-powered, multi-agent pipeline for extracting structured information from medical insurance claim PDFs. Built with **LangGraph**, **FastAPI**, **Google Gemini** (Vision), and **OpenRouter/Qwen** (Text).

## System Architecture

The pipeline processes documents in a single shot using a fan-out/fan-in Graph DAG approach to minimize LLM calls, reduce latency, and avoid rate limits:

```mermaid
graph TD
    START((START)) --> Segregator[Segregator Agent]
    
    %% Segregator groups pages and routes
    Segregator -- identity pages --> ID[Identity Agent]
    Segregator -- discharge pages --> Discharge[Discharge Agent]
    Segregator -- bill pages --> Bill[Itemized Bill Agent]
    
    %% Fallback if no pages routed
    Segregator -. empty sends .-> Aggregator
    
    %% Fan-in
    ID --> Aggregator[Aggregator]
    Discharge --> Aggregator
    Bill --> Aggregator
    
    Aggregator --> END((END))
```

1. **PDF Rendering**: Pages are converted to fast, lightweight JPEGs using PyMuPDF.
2. **Segregator**: A single Gemini vision call classifies *all pages simultaneously* into 9 medical document categories.
3. **Extraction Agents**: Parallel agents process their grouped pages. Each agent makes two calls:
   - Gemini Vision: Transcribes all its pages to clean Markdown.
   - Qwen Text: Extracts free-form structured JSON from the Markdown.
4. **Aggregator**: Consolidates results.

## Key Design Decisions

- **Single Vision Call for Segregation**: Instead of making $N$ API calls for an $N$-page document (which triggers 429 rate limits), all $N$ images are sent to the Vision model in a single prompt.
- **Free-form Text Extraction**: Instead of enforcing strict Pydantic schemas which often fail on highly variable medical documents, we ask the text LLM to return descriptive JSON mapping exactly what's printed on the page.
- **State Cleanliness**: The LangGraph state only holds core business data. Routing meta-data is dynamically computed inside conditional edges rather than polluting the shared pipeline state.

## Graph Internals

The graph relies on a shared `PipelineState` dictionary:
- `pages`: Base64 encoded JPEG pages.
- `page_assignments`: Populated by the **Segregator**, maps doc_type to page indices (e.g., `{"itemized_bill": [8, 9]}`).
- `agent_results`: Accumulated outputs from parallel agent runs.

**Dynamic Routing (`_fan_out`)**
The `Segregator` feeds into a conditional edge `_fan_out`. Instead of following hardcoded paths, `_fan_out` dynamically inspects the `page_assignments` and returns a list of `Send("agent_node", AgentInput)` objects. LangGraph natively spawns a parallel execution branch for every `Send` object emitted.

## Requirements

- Python 3.10+
- A Google AI Studio API Key
- An OpenRouter API Key

## Local Setup

1. **Clone & Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   Copy the example `.env` file and fill in your keys:
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

## Running the API

Start the FastAPI development server:

```bash
fastapi dev main.py
# Or using uvicorn directly:
# uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. You can view the interactive Swagger docs at `http://localhost:8000/docs`.

## API Usage Example

**Endpoint:** `POST /api/process`  
Accepts a generic `claim_id` and the `file` itself via multipart form data.

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-123456" \
  -F "file=@/path/to/your/document.pdf"
```

## Testing

The project includes an end-to-end testing suite that performs real LLM calls against the pipeline. 

> **Note:** These tests use real API calls. It will take a few minutes and consume a small amount of your API quota.

```bash
# Run unit & validation tests (no external LLM calls)
pytest tests/ -k "not real_pdf" -v

# Run the full end-to-end pipeline test
pytest tests/test_api.py::test_full_pipeline_real_pdf -v -s
```

## Error Handling Table

| Failure Point | Behavior | Result |
| --- | --- | --- |
| Invalid File / Empty File | FastAPI validation fails | **HTTP 400 Bad Request** |
| Segregator Vision Model Fails | Logged & immediately raised (Fail-Fast) | **HTTP 500 Server Error** |
| Segregator returns invalid JSON | Caught, all pages defaulted to `"other"` | Pipeline continues smoothly |
| Agent Vision Model Fails | Logged & raised for that branch (Fail-Fast) | **HTTP 500 Server Error** |
| Agent Text Model Fails | Logged heavily (Non-Fatal), returns empty `{}` | Pipeline continues smoothly |

