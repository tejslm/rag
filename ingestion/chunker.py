from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .loader import DATA_DIR, load_documents

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def chunk_documents(
    documents: list[Document] | None = None, data_dir: Path = DATA_DIR
) -> list[Document]:
    if documents is None:
        documents = load_documents(data_dir)
    return SPLITTER.split_documents(documents)


if __name__ == "__main__":
    chunks = chunk_documents()
    print(f"Split {len(chunks)} chunk(s) (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
