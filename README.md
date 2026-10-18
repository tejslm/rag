# RAG Pipeline

A minimal retrieval-augmented generation pipeline over local documents (`.txt` / `.pdf`),
using sentence-transformers embeddings, a Chroma vector store with cosine similarity,
and Gemini for answer generation.

## Structure

```
data/                     Source documents (txt, pdf)
ingestion/
  loader.py               Loads documents from data/
  chunker.py              Splits documents into chunks (size=1000, overlap=20)
  embedder.py             Embeds chunks with all-MiniLM-L6-v2
  vector_store.py         Stores/query vectors in Chroma (cosine space)
ingestion_pipeline.py     End-to-end ingestion: load → chunk → embed → store
retriever_pipeline.py     Retrieval + Gemini answer, interactive REPL
chroma_db/                Chroma persistence (generated, git-ignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install langchain-core langchain-community langchain-text-splitters \
            langchain-google-genai sentence-transformers chromadb python-dotenv
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your-key-here
```

## Usage

Ingest the documents (rebuilds the vector store):

```bash
python ingestion_pipeline.py
```

Ask questions in the interactive REPL:

```bash
python retriever_pipeline.py
Question: What does Tesla make?
...
Question: exit
```

Follow-up questions are supported — prior question/answer pairs are used to
rewrite the next question into a self-contained form before retrieval.

## Notes

- The Chroma collection is created with `hnsw:space: cosine`; retrieval scores
  are cosine similarities (`1 - distance`). Collections created without the
  cosine space are deleted and rebuilt automatically on first use — re-run
  `ingestion_pipeline.py` if that happens.
- Retrieval returns chunk texts with cosine scores; the answer step uses the
  top-1 chunk only.
