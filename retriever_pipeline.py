import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from ingestion.embedder import embed_texts
from ingestion.vector_store import get_collection, query

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set (check that .env exists)")

ANSWER_SYSTEM_PROMPT = (
    "You are an assistant that transforms and converts the given question and the "
    "given embeddings into a meaningful answer that is short and understandable to "
    "the user. If the question is not related to the given embeddings, respond with "
    "'I don't have the answer to this question.'."
)

CONDENSE_SYSTEM_PROMPT = (
    "You are an assistant that rewrites the user's question into a single, "
    "self-contained question using the previous question/answer pairs as context. "
    "Keep it short and understandable to the user."
)

LLM = ChatGoogleGenerativeAI(model=MODEL_NAME, api_key=GEMINI_API_KEY)


def _as_text(response) -> str:
    # Gemini responses can return str or a list of content parts (with
    # signature metadata); flatten to plain text either way.
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict)
    )


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_texts([question])[0]

    results = query(query_embedding, n_results=n_results, collection=collection)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Chroma's cosine space reports distance = 1 - cosine_similarity.
    return [
        {
            "document": document,
            "metadata": metadata or {},
            "score": 1.0 - distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def answer_question(question: str, n_results: int = 5) -> str:
    retrieved = retrieve(question, n_results=n_results)
    if not retrieved:
        return "No relevant documents found."
    return retrieved[0]["document"]


def format_answer(formatted_question: str) -> str:
    messages = [
        ("system", ANSWER_SYSTEM_PROMPT),
        ("human", formatted_question),
    ]
    response = LLM.invoke(messages)
    return _as_text(response)


def format_chain_question(history: list[tuple[str, str]], question: str) -> str:
    history_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    messages = [
        ("system", CONDENSE_SYSTEM_PROMPT),
        ("human", f"History:\n{history_text}\n\nQuestion: {question}"),
    ]
    response = LLM.invoke(messages)
    return _as_text(response)


if __name__ == "__main__":
    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question == "exit":
            break

        formatted_question = (
            question if not history else format_chain_question(history, question)
        )
        print("Formatted Question:", formatted_question)

        answer = answer_question(formatted_question)
        formatted_question_with_embeddings = (
            f"Question: {question}\n"
            f"Condensed question: {formatted_question}\n"
            f"Retrieved context: {answer}"
        )

        formatted_answer = format_answer(formatted_question_with_embeddings)
        print(f"\nAnswer:\n{formatted_answer}")

        history.append((question, formatted_answer))
