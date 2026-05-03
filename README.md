# Groq LangChain RAG

A lightning-fast Retrieval Augmented Generation (RAG) application built with LangChain, Groq, and Streamlit. Ask questions against LangChain documentation and get answers in milliseconds powered by Groq's LPU inference engine.

## Stack

| Component | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [LangChain](https://langchain.com) | LLM orchestration & RAG pipeline |
| [Groq](https://groq.com) | Fast LLM inference (LPU) |
| [FAISS](https://faiss.ai) | Vector store for similarity search |
| [Ollama Embeddings](https://ollama.com) | Text embeddings |
| [LangSmith](https://smith.langchain.com) | Observability & tracing |

## How It Works

1. **Embed** — Loads docs from `https://docs.smith.langchain.com/`, splits into chunks, and builds a FAISS vector index (persisted to disk)
2. **Retrieve** — On each query, finds the most semantically relevant chunks via similarity search
3. **Generate** — Passes retrieved context + question to Groq (`gemma2-9b-it`) to produce a grounded answer
4. **Observe** — Every chain run is traced automatically in LangSmith

## Setup

### Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) running locally (for embeddings)
- A [Groq API key](https://console.groq.com)
- A [LangSmith API key](https://smith.langchain.com)

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
2. Click **"Document Embeddings"** to load and index the docs (first run takes ~30s; subsequent runs load from disk instantly)
3. Type a question in the input box and hit Enter
4. View the answer and response time; expand **"Document Similarity Search"** to see the source chunks used

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
└── faiss_index/      # Persisted FAISS index (auto-created on first run)
```
