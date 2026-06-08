from src.config import N_RESULTS
from src.llm_client import generate_answer
from src.vector_store import search_sources


def evaluate_retrieval(sources):
    if not sources:
        return {
            "label": "No sources found",
            "best_distance": None,
            "average_distance": None,
        }

    distances = [source["distance"] for source in sources]
    best_distance = min(distances)
    average_distance = sum(distances) / len(distances)

    if best_distance < 0.8:
        label = "Strong retrieval"
    elif best_distance < 1.3:
        label = "Medium retrieval"
    else:
        label = "Weak retrieval"

    return {
        "label": label,
        "best_distance": best_distance,
        "average_distance": average_distance,
    }


def build_context(sources):
    context_parts = []

    for index, source in enumerate(sources, start=1):
        context_parts.append(
            f"""Source {index}
Document: {source["document_name"]}
Page: {source["page"]}
Text:
{source["text"]}"""
        )

    return "\n\n".join(context_parts)


def answer_question(question, selected_documents):
    sources = search_sources(
        question=question,
        selected_documents=selected_documents,
        n_results=N_RESULTS,
    )

    evaluation = evaluate_retrieval(sources)
    context = build_context(sources)

    prompt = f"""
You are StudyMate AI, a friendly study tutor.

Use only the retrieved notes below to answer the user's question.
If the answer is not in the retrieved notes, say:
"I cannot find that in the uploaded notes."

When you use information from a source, cite it like this:
(Source 1, document name, page number)

Retrieved notes:
{context}

User question:
{question}
"""

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": sources,
        "evaluation": evaluation,
    }