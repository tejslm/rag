import chromadb

from ingestion.chunker import chunk_documents
from ingestion.embedder import embed_documents
from ingestion.loader import load_documents
from ingestion.vector_store import COLLECTION_NAME, PERSIST_DIR, store_documents


def main() -> chromadb.Collection:
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s)")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunk(s)")

    embeddings = embed_documents(chunks)
    print(f"Embedded {len(embeddings)} chunk(s)")

    collection = store_documents(chunks, embeddings)
    print(
        f"Stored {collection.count()} chunk(s) in '{COLLECTION_NAME}' at {PERSIST_DIR}"
    )

    return collection


if __name__ == "__main__":
    collection = main()
    print(f"Vector store now holds {collection.count()} chunk(s)")