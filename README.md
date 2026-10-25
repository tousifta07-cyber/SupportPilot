# SupportPilot - AI Technical Support Assistant

SupportPilot is a Retrieval-Augmented Generation (RAG) application that answers technical support questions using IBM technical documentation. It combines hybrid information retrieval, reranking, a LangGraph workflow, and Gemini-based answer generation.

Users can ask questions through a Streamlit chat interface, inspect the retrieved technical passages, and submit support tickets for human follow-up.

## Features

- Search a corpus of IBM technical documents.
- Fixed-size text chunking with overlap.
- ChromaDB vector search.
- BM25 keyword retrieval.
- Reciprocal Rank Fusion (RRF) for combining search results.
- FlashRank reranking.
- LangGraph-based workflow orchestration.
- Gemini-based evidence-grounded answer generation.
- Citation-ID validation against retrieved source IDs.
- Abstention when retrieval returns no passages.
- FastAPI backend with health checks and API documentation.
- Streamlit chat interface with source passages.
- Human-support ticket creation with local JSONL storage.
- Automated tests for citation validation, ticket handling, API behavior, and graph execution.

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python 3.12 |
| API | FastAPI, Uvicorn |
| User interface | Streamlit |
| Workflow orchestration | LangGraph |
| Vector database | ChromaDB |
| Keyword retrieval | BM25, rank-bm25 |
| Ranking | Reciprocal Rank Fusion, FlashRank |
| Large language model | Google Gemini API |
| Data validation | Pydantic |
| Testing | Pytest |
| Configuration | python-dotenv |

## Architecture

SupportPilot separates the API, retrieval logic, answer-generation service, workflow orchestration, and ticket handling.

### Question-answering workflow

1. The user submits a technical question through Streamlit or the FastAPI endpoint.
2. The API passes the question to the LangGraph workflow.
3. The retrieval node searches the technical knowledge base using hybrid retrieval.
4. Reciprocal Rank Fusion combines vector-search and BM25 results.
5. FlashRank reranks candidate passages.
6. LangGraph checks whether passages were retrieved.
7. If passages are available, the Gemini service generates an evidence-grounded answer and validates its citation IDs.
8. If retrieval returns no passages, the workflow returns an abstention response.
9. The API returns the answer and retrieved source passages to the user.

The graph's abstention decision is based on whether retrieval returns passages. It does not independently establish that the retrieved passages contain sufficient evidence for every possible question.

### Human-support workflow

Users can submit a support ticket when they need additional assistance. Tickets are stored locally in JSONL format with a generated ticket ID, question, description, priority, timestamp, and status.

The local ticket store is intended for this project demonstration, not as a production ticket-management system.

## Dataset and Index

The project uses technical documents from the IBM TechQA dataset.

Recorded full-corpus preparation:

| Item | Value |
|---|---:|
| Source documents processed | 28,482 |
| Fixed-size chunks created | 56,393 |
| Chunk size | 300 words |
| Chunk overlap | 50 words |
| Vector database | ChromaDB |
| Collection name | `techqa_full_corpus_fixed` |

The prepared chunk file is:

`data/processed/full_corpus_fixed_chunks.jsonl`

The index is stored locally in:

`data/chroma_db_fixed`

The dataset, generated chunks, embeddings, and vector database are excluded from the GitHub repository because of their size. A fresh clone therefore needs the dataset and prepared index before the live application can perform retrieval.

The existing index is reused for ordinary questions. Embeddings are not regenerated for every API request.

## Retrieval Evaluation

The recorded benchmark contains 50 questions: 25 answerable and 25 unanswerable.

Hit Rate@5 and Mean Reciprocal Rank at 5 (MRR@5) are calculated over the 25 answerable questions. A retrieval hit means a gold-source document was found among the top five results; it does not necessarily mean the generated answer is correct.

| Retrieval method | Hit Rate@5 | MRR@5 |
|---|---:|---:|
| Full-corpus vector search | 68.00% (17/25) | 0.6033 |
| Hybrid BM25 + vector search (RRF) | 72.00% (18/25) | 0.6100 |
| Hybrid retrieval + FlashRank reranking | 76.00% (19/25) | 0.6600 |

The recorded hybrid candidate pool contained a gold-source document for 19 of the 25 answerable questions (76.00%).

Recorded mean latency for the evaluated hybrid configuration:

- Hybrid retrieval: approximately 1,685.68 ms.
- Reranking: approximately 262.99 ms.

These values are measurements from a particular evaluation run. Latency can vary with hardware, caching, and execution conditions.

**Evaluation limitations:** These are preliminary results from a small benchmark. Retrieval metrics do not establish answer factuality or citation entailment. Generated-answer correctness, citation support, abstention quality, and performance on a larger independently reviewed dataset require further evaluation.

## Requirements

- Python 3.12 or a compatible Python environment.
- A Gemini API key for answer generation.
- The IBM TechQA corpus and prepared ChromaDB index for live retrieval.
- Dependencies listed in `requirements.txt`.

## Setup on Windows

### 1. Clone the repository

After publishing the repository, clone it and open PowerShell in the project directory.

```powershell
git clone https://github.com/tousifta07-cyber/SupportPilot.git
cd SupportPilot
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure Gemini

Create a local `.env` file from the supplied template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set your actual API key and an available Gemini model:

```text
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=your_available_gemini_model
```

Replace both placeholders with values appropriate for your environment.

**Security:** Never commit `.env`, share your API key, or include credentials in screenshots or logs.

### 5. Prepare the knowledge base

The repository does not include the TechQA corpus, generated chunks, or prebuilt ChromaDB database.

Obtain the dataset separately and prepare the files and index expected by the ingestion and index-building scripts in `app/rag/`. The full-corpus configuration expects:

- Source document corpus: `data/techqa/extracted/training_and_dev/training_dev_technotes.json`
- Prepared chunks: `data/processed/full_corpus_fixed_chunks.jsonl`
- ChromaDB collection: `techqa_full_corpus_fixed`

Follow the ingestion and index-building scripts' supported entry points to create the local index. Do not rebuild an existing working index unnecessarily.

Verify that the index is available by running the health endpoint after starting the API.

## Running SupportPilot

Run the API and Streamlit interface in two separate PowerShell terminals. Keep both processes running while using the application.

### Terminal 1: FastAPI backend

```powershell
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Health check:

http://127.0.0.1:8000/health

Interactive API documentation:

http://127.0.0.1:8000/docs

### Terminal 2: Streamlit interface

```powershell
python -m streamlit run app/ui.py
```

Open the application:

http://localhost:8501

Enter a technical question to retrieve relevant passages and generate an answer. You can inspect the returned sources and submit a human-support ticket when needed.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome information |
| GET | `/health` | Check availability of the knowledge-base collection |
| POST | `/ask` | Run the LangGraph support workflow and return an answer with sources |
| POST | `/tickets` | Create a human-support ticket |

## Automated Tests

Run all tests from the project root:

```powershell
python -m pytest -q
```

The current recorded test suite contains **12 passing tests**.

The tests cover:

- Citation-ID validation.
- Support-ticket storage and validation.
- API response handling.
- LangGraph answer-generation routing.
- Abstention when retrieval returns no passages.
- Rejection of blank questions.

Retrieval and answer-generation functions are mocked in workflow tests to avoid external API requests. Passing these tests verifies the tested behavior, not the factual accuracy of every generated answer.

Additional useful checks:

```powershell
python -m pip check
git diff --check
```

## Project Structure

```text
SupportPilot/
├── app/
│   ├── agents/
│   │   └── support_graph.py
│   ├── rag/
│   │   ├── build_index.py
│   │   ├── ingest.py
│   │   └── retriever.py
│   ├── services/
│   │   ├── llm_service.py
│   │   └── ticket_service.py
│   ├── api.py
│   └── ui.py
├── evaluation/
├── tests/
│   ├── test_core.py
│   └── test_support_graph.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

The `data/`, `.venv/`, `backups/`, and `.env` paths are local resources and are intentionally excluded from Git tracking.

## Limitations and Future Improvements

- Evaluate answer correctness and citation support using an independently reviewed dataset.
- Improve evidence-sufficiency checks and abstention behavior.
- Continue evaluating retrieval quality with a consistent benchmark.
- Review the relevance of retrieved passages for specific technical questions.
- Improve answer readability and response formatting.
- Extend ticket handling for human-agent follow-up.
- Add authentication and authorization before exposing the API outside a trusted local environment.
- Add deployment monitoring, logging, and operational safeguards.

## Security

- Keep API keys and `.env` private.
- Do not commit virtual environments, datasets, vector databases, or sensitive ticket information.
- Support tickets are currently stored in a local JSONL file.
- The local ticket file is not a production ticket-management system.
- Verify important technical instructions against the supporting documentation.

## Project Status

The full-corpus index, hybrid retrieval, reranking, Gemini answer generation, LangGraph workflow, FastAPI endpoints, Streamlit interface, support-ticket creation, and automated tests have been exercised.

SupportPilot is a working project demonstration. Further evaluation of answer factuality, citation support, evidence sufficiency, security, and deployment behavior is required before describing it as production-ready.
