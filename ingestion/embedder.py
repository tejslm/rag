from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from .chunker import chunk_documents

MODEL_NAME = "all-MiniLM-L6-v2"

EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()


def embed_documents(documents: list[Document]) -> list[list[float]]:
    return embed_texts([document.page_content for document in documents])


if __name__ == "__main__":
    chunks = chunk_documents()
    embeddings = embed_documents(chunks)
    print(f"Embedded {len(embeddings)} chunk(s) with {MODEL_NAME} (dim={len(embeddings[0])})")
