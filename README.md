# Groq LangChain RAG

A lightning-fast Retrieval Augmented Generation (RAG) application built with LangChain, Groq, and Streamlit. Ask questions against LangChain documentation and get answers in milliseconds powered by Groq's LPU inference engine.

## Stack

| Component | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [LangChain](https://langchain.com) | LLM orchestration & RAG pipeline |
| [Groq](https://groq.com) | Fast LLM inference (LPU) — `qwen-qwq-32b` |
| [HuggingFace](https://huggingface.co) | Local embeddings via `sentence-transformers/all-mpnet-base-v2` |
| [FAISS](https://faiss.ai) | Vector store for similarity search |
| [LangSmith](https://smith.langchain.com) | Observability & tracing |

## How It Works

1. **Embed** — Loads docs from `https://docs.langchain.com/`, splits into chunks, and builds a FAISS vector index (persisted to disk). Embeddings run locally via HuggingFace `sentence-transformers` — no external API needed.
2. **Retrieve** — On each query, finds the most semantically relevant chunks via FAISS similarity search
3. **Generate** — Passes retrieved context + question to Groq (`qwen-qwq-32b`) to produce a grounded answer
4. **Observe** — Every chain run is traced automatically in LangSmith

## LangChain Package Imports (v0.3+)

All imports use the latest LangChain split-package structure. The legacy `langchain.chains` namespace is **not used** — the RAG pipeline is composed using LCEL (LangChain Expression Language) runnables from `langchain-core`.

| Class / Function | Package |
|---|---|
| `ChatGroq` | `langchain-groq` |
| `WebBaseLoader` | `langchain-community` |
| `HuggingFaceEmbeddings` | `langchain-huggingface` |
| `FAISS` | `langchain-community` |
| `RecursiveCharacterTextSplitter` | `langchain-text-splitters` |
| `ChatPromptTemplate` | `langchain-core` |
| `RunnablePassthrough` | `langchain-core` |
| `StrOutputParser` | `langchain-core` |

## Setup

### Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com)
- A [LangSmith API key](https://smith.langchain.com) *(optional — only needed for tracing)*

No Ollama or external embedding service required. Embeddings run fully locally via `sentence-transformers`.

### Installation

```bash
git clone https://github.com/sushant144/groq-langchain-rag.git
cd groq-langchain-rag
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional — enables LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=groq-langchain-rag
```

### Run

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Click **"Document Embeddings"** to load and index the docs
   - First run: downloads the embedding model (~420MB) and builds the FAISS index — takes ~1–2 min
   - Subsequent runs: loads the persisted index from disk instantly
3. Type a question and hit Enter
4. View the answer and response time; expand **"Document Similarity Search"** to see the source chunks used

> **Note:** If you previously ran the app with Ollama embeddings, delete the `faiss_index/` directory before running again — the embedding dimensions are different and will cause a mismatch error.

## Reliability Safeguards

The app has two guards against silent failures:

1. **Model verification at startup** — calls `Groq.models.list()` and confirms the configured model (`qwen-qwq-32b`) is available on your account. If not, the app shows an error listing every model you can use and halts before any query.

2. **Out-of-scope question detection** — for every question, the app runs `similarity_search_with_relevance_scores` and checks the top score against `RELEVANCE_THRESHOLD` (default `0.3`). If no chunk passes the threshold, the LLM is **not called** — the user sees a "this question isn't covered by the indexed documents" message instead of a hallucinated answer. The top relevance score is also displayed alongside each successful answer.

   Tune `RELEVANCE_THRESHOLD` in `app.py` to be stricter (higher) or more permissive (lower).

## Observability with LangSmith

All RAG chain runs are automatically traced in LangSmith when `LANGCHAIN_TRACING_V2=true`. Visit your [LangSmith dashboard](https://smith.langchain.com) and open the `groq-langchain-rag` project to inspect:

- Input / output at each chain step
- Retrieved document chunks
- Latency breakdown
- Token usage

## Project Structure

```
groq-langchain-rag/
├── app.py            # Main Streamlit app
├── requirements.txt  # Python dependencies
├── .env.example      # Environment variable template
├── .gitignore        # Excludes .env and faiss_index from git
└── faiss_index/      # Persisted FAISS index (auto-created on first run)
```
