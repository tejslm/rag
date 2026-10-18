from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LOADERS: dict[str, tuple[type, dict]] = {
    "**/*.pdf": (PyPDFLoader, {}),
    "**/*.txt": (TextLoader, {"encoding": "utf-8"}),
}


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    return [
        doc
        for glob, (loader_cls, loader_kwargs) in LOADERS.items()
        for doc in DirectoryLoader(
            str(data_dir),
            glob=glob,
            loader_cls=loader_cls,
            loader_kwargs=loader_kwargs,
            show_progress=False,
        ).load()
    ]


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document(s) from {DATA_DIR}")
