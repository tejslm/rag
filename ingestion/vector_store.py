import hashlib
from pathlib import Path

import chromadb
from langchain_core.documents import Document

from .chunker import chunk_documents
from .embedder import embed_documents

PERSIST_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "documents"
BATCH_SIZE = 500


def get_client(persist_dir: Path = PERSIST_DIR) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(persist_dir))


COLLECTION_METADATA = {"hnsw:space": "cosine"}


def get_collection(
    client: chromadb.ClientAPI | None = None, name: str = COLLECTION_NAME
) -> chromadb.Collection:
    if client is None:
        client = get_client()
    collection = client.get_or_create_collection(
        name=name, embedding_function=None, metadata=COLLECTION_METADATA
    )
    if (collection.metadata or {}).get("hnsw:space") != "cosine":
        # Chroma cannot change the distance space of an existing collection, so
        # rebuild collections created before cosine space was configured.
        client.delete_collection(name)
        collection = client.get_or_create_collection(
            name=name, embedding_function=None, metadata=COLLECTION_METADATA
        )
    return collection


def _chunk_id(document: Document, index: int) -> str:
    source = document.metadata.get("source", "")
    page = document.metadata.get("page", "")
    digest = hashlib.sha1(document.page_content.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{page}:{index}:{digest}"


def _sanitize_metadata(metadata: dict) -> dict:
    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


def store_documents(
    chunks: list[Document],
    embeddings: list[list[float]],
    collection: chromadb.Collection | None = None,
) -> chromadb.Collection:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunk(s) but {len(embeddings)} embedding(s)"
        )

    if collection is None:
        collection = get_collection()

    sources = {doc.metadata.get("source") for doc in chunks if doc.metadata.get("source")}
    for source in sources:
        collection.delete(where={"source": source})

    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        collection.upsert(
            ids=[_chunk_id(doc, i) for i, doc in enumerate(chunks[start:end], start)],
            documents=[doc.page_content for doc in chunks[start:end]],
            embeddings=embeddings[start:end],
            metadatas=[_sanitize_metadata(doc.metadata) for doc in chunks[start:end]],
        )

    return collection


def query(
    query_embedding: list[float],
    n_results: int = 5,
    collection: chromadb.Collection | None = None,
) -> dict:
    if collection is None:
        collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


if __name__ == "__main__":
    chunks = chunk_documents()
    embeddings = embed_documents(chunks)

    collection = store_documents(chunks, embeddings)
    print(
        f"Stored {collection.count()} chunk(s) in '{COLLECTION_NAME}' at {PERSIST_DIR}"
    )
