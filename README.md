---
title: Legal Financial CRAG
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ⚖️ Legal & Financial Document Intelligence System

> A production-grade **Corrective RAG (CRAG)** application for analyzing legal and financial documents with AI-powered question answering, cited responses, and structured clause extraction.

---

## 🏗️ Architecture

```
                              ┌──────────────────────────────────────────────────┐
                              │            STREAMLIT FRONTEND (ui/app.py)        │
                              │  ┌────────────┐  ┌──────────────┐               │
                              │  │  Doc Chat   │  │  Clause      │               │
                              │  │  Interface   │  │  Extraction  │               │
                              │  └─────┬───────┘  └──────┬───────┘               │
                              └────────┼─────────────────┼───────────────────────┘
                                       │   HTTP/REST     │
                              ┌────────▼─────────────────▼───────────────────────┐
                              │          FASTAPI BACKEND (api/main.py)           │
                              │  /api/v1/upload_document                        │
                              │  /api/v1/chat_query                             │
                              │  /api/v1/extract_clauses                        │
                              └────────┬─────────────────┬───────────────────────┘
                                       │                 │
                 ┌─────────────────────▼──────┐   ┌──────▼──────────────────────┐
                 │   CRAG STATE MACHINE        │   │  STRUCTURED EXTRACTION     │
                 │   (LangGraph — graph.py)    │   │  (clauses.py — JSON mode)  │
                 │                              │   └──────┬──────────────────────┘
                 │  ┌──────────┐               │          │
                 │  │ RETRIEVE │◄──────────┐   │   ┌──────▼──────┐
                 │  └────┬─────┘           │   │   │  Groq LLM   │
                 │       ▼                 │   │   │  70b-versatile│
                 │  ┌──────────┐           │   │   └─────────────┘
                 │  │  GRADE   │           │   │
                 │  │ (8b-inst)│           │   │
                 │  └────┬─────┘           │   │
                 │       ▼                 │   │
                 │  ┌─────────┐  YES  ┌────┤   │
                 │  │Relevant?├───────► GEN│   │
                 │  └────┬────┘       │ERATE│  │
                 │    NO │            └────┘   │
                 │       ▼                     │
                 │  ┌──────────┐               │
                 │  │ REWRITE  │───────────────┘
                 │  │(max 2x)  │
                 │  └──────────┘
                 └─────────────────────────────┘
                              │
                 ┌────────────▼────────────────┐
                 │      CHROMADB (Persistent)   │
                 │      ./chroma_db             │
                 │  ┌─────────────────────────┐ │
                 │  │ Collection:              │ │
                 │  │ legal_financial_docs     │ │
                 │  │  • Embeddings (MiniLM)   │ │
                 │  │  • Document chunks       │ │
                 │  │  • Rich metadata         │ │
                 │  └─────────────────────────┘ │
                 └─────────────────────────────┘

  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
  │   LlamaParse     │   │ SentenceTransf.  │   │   Groq API       │
  │   (PDF → MD)     │   │ all-MiniLM-L6-v2 │   │ llama-3.3-70b    │
  │                  │   │ (Local Embeddings)│   │ llama-3.1-8b     │
  └─────────────────┘   └──────────────────┘   └──────────────────┘
```

---

## 📁 Project Structure

```
legal_financial_rag/
├── .env.example                # API key template
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── uploaded_docs/          # Uploaded PDFs land here
├── chroma_db/                  # Auto-created by ChromaDB persistent client
├── core_logic/
│   ├── __init__.py
│   ├── ingestion.py            # Parse → Chunk → Embed → Store
│   ├── retriever.py            # ChromaDB semantic search
│   ├── grader.py               # LLM relevance grader
│   ├── rewriter.py             # Query rewriter
│   ├── generator.py            # Answer generator with citations
│   └── graph.py                # LangGraph CRAG state machine
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point
│   ├── models.py               # Pydantic v2 models
│   └── routes/
│       ├── __init__.py
│       ├── upload.py           # /upload_document
│       ├── chat.py             # /chat_query
│       └── clauses.py          # /extract_clauses
├── ui/
│   └── app.py                  # Streamlit frontend
└── evaluation/
    ├── eval_script.py          # Ragas evaluation runner
    └── golden_dataset.json     # 10 sample Q&A pairs
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd legal_financial_rag
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example env file and add your keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual keys:

```env
GROQ_API_KEY=gsk_your_actual_groq_key_here
LLAMA_CLOUD_API_KEY=llx-your_actual_llamaparse_key_here
```

**Get free API keys:**
- 🔑 **Groq**: [console.groq.com](https://console.groq.com) — Free tier available
- 🔑 **LlamaParse**: [cloud.llamaindex.ai](https://cloud.llamaindex.ai) — Free tier: 1,000 pages/day

### 3. Start the FastAPI Backend

```bash
uvicorn api.main:app --reload
```

The API will be available at:
- **API**: `http://localhost:8000`
- **Docs (Swagger)**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 4. Start the Streamlit Frontend

In a **new terminal**:

```bash
streamlit run ui/app.py
```

The UI will open at `http://localhost:8501`.

### 5. Run Evaluation (Optional)

```bash
python evaluation/eval_script.py
```

---

## 📡 API Endpoints

| Method | Endpoint                     | Description                              |
|--------|------------------------------|------------------------------------------|
| GET    | `/`                          | Health check                             |
| POST   | `/api/v1/upload_document`    | Upload and ingest a PDF document         |
| POST   | `/api/v1/chat_query`         | Ask a question (runs CRAG pipeline)      |
| POST   | `/api/v1/extract_clauses`    | Extract structured clauses from documents|

### Upload Document

```bash
curl -X POST "http://localhost:8000/api/v1/upload_document" \
  -F "file=@contract.pdf"
```

### Chat Query

```bash
curl -X POST "http://localhost:8000/api/v1/chat_query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the payment terms?", "top_k": 5}'
```

### Extract Clauses

```bash
curl -X POST "http://localhost:8000/api/v1/extract_clauses" \
  -H "Content-Type: application/json" \
  -d '{"query": "Extract all important clauses"}'
```

---

## 🧠 How the CRAG Pipeline Works

1. **Retrieve**: Embeds the query and finds the top-k similar chunks from ChromaDB
2. **Grade**: Uses a fast LLM (Llama 3.1 8B) to check if chunks are relevant
3. **Conditional Routing**:
   - If **relevant** → Generate answer with citations
   - If **not relevant** → Rewrite the query and retry (max 2 rewrites)
4. **Generate**: Uses a powerful LLM (Llama 3.3 70B) to produce a grounded, cited answer

---

## 🛡️ Design Principles

- **No hallucination**: Generator is strictly grounded in retrieved context
- **Full auditability**: Every answer includes source citations and chunk transparency
- **Loop protection**: Maximum 2 query rewrites prevent infinite loops
- **Persistent storage**: ChromaDB PersistentClient ensures data survives restarts
- **Local embeddings**: SentenceTransformer runs entirely locally — no external API needed
- **Structured extraction**: Clause extraction uses JSON mode for reliable parsing

---

## 📊 Evaluation

The evaluation suite uses [Ragas](https://docs.ragas.io/) with three metrics:

| Metric              | What it measures                                           |
|---------------------|------------------------------------------------------------|
| Context Precision   | Are the retrieved chunks relevant to the question?         |
| Faithfulness        | Is the answer faithful to the retrieved context?           |
| Answer Relevancy    | Is the answer relevant to the question asked?              |

Run evaluation:

```bash
python evaluation/eval_script.py
```

Results are saved to `evaluation/ragas_results.json`.

---

## 🔧 Tech Stack

| Component          | Technology                              |
|--------------------|-----------------------------------------|
| Document Parsing   | LlamaParse (cloud, markdown output)     |
| Chunking           | LlamaIndex HierarchicalNodeParser       |
| Embeddings         | SentenceTransformer (all-MiniLM-L6-v2)  |
| Vector Database    | ChromaDB (PersistentClient)             |
| RAG Orchestration  | LangGraph (Corrective RAG state machine)|
| Generation LLM     | Groq — llama-3.3-70b-versatile          |
| Grader / Rewriter  | Groq — llama-3.1-8b-instant             |
| Data Validation    | Pydantic v2                             |
| Backend API        | FastAPI                                 |
| Frontend           | Streamlit                               |
| Evaluation         | Ragas framework                         |

---

## 📝 License

This project is for educational and demonstration purposes.
