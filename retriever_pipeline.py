from ingestion.embedder import embed_texts
from ingestion.vector_store import get_collection, query
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


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


def format_answer(formatted_question):
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key=GEMINI_API_KEY)
    messages = [
    ("system", "You are an assistant that transforms and converts the given question and the given embeddings into a meaningful answer that is short and understandable to the user"),
    ("human", formatted_question),
    ]
    response = model.invoke(messages)
    return response.text


if __name__ == "__main__":
    question = input("Question: ").strip()
    answer = answer_question(question)
    formatted_question = f"""
    Question: {question}
    Embeddings: {answer}
    """
    formatted_answer = format_answer(formatted_question)
    print(f"\nAnswer:\n{formatted_answer}")
