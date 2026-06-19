from src.config import FINAL_CONTEXT_RESULTS, KEYWORD_RESULTS, VECTOR_RESULTS
from src.keyword_store import keyword_search_sources
from src.llm_client import generate_answer
from src.vector_store import search_sources


def source_key(source):
    return (
        source.get("document_name"),
        source.get("page"),
        source.get("chunk_index"),
    )


def combine_hybrid_sources(vector_sources, keyword_sources, final_limit):
    combined = {}
    ordered_sources = []

    for source in vector_sources:
        key = source_key(source)

        if key not in combined:
            source["hybrid_score"] = 1.0
            combined[key] = source
            ordered_sources.append(source)

    for source in keyword_sources:
        key = source_key(source)

        if key in combined:
            combined[key]["retrieval_method"] = "hybrid"
            combined[key]["keyword_score"] = source.get("keyword_score", 0)
            combined[key]["hybrid_score"] = combined[key].get("hybrid_score", 1.0) + 1.0
        else:
            source["hybrid_score"] = 1.0
            combined[key] = source
            ordered_sources.append(source)

    ordered_sources = sorted(
        ordered_sources,
        key=lambda source: source.get("hybrid_score", 0),
        reverse=True,
    )

    return ordered_sources[:final_limit]


def evaluate_retrieval(sources):
    if not sources:
        return {
            "label": "No sources found",
            "best_distance": None,
            "average_distance": None,
            "retrieval_methods": [],
        }

    distances = [
        source["distance"]
        for source in sources
        if source.get("distance") is not None
    ]

    retrieval_methods = sorted(
        set(source.get("retrieval_method", "unknown") for source in sources)
    )

    if not distances:
        label = "Keyword-only retrieval"
        best_distance = None
        average_distance = None
    else:
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
        "retrieval_methods": retrieval_methods,
    }


def build_context(sources):
    context_parts = []

    for index, source in enumerate(sources, start=1):
        context_parts.append(
            f"""Source {index}
Document: {source.get("document_name")}
Document type: {source.get("document_type", "unknown")}
Equipment ID: {source.get("equipment_id", "")}
Component: {source.get("component", "")}
Plant area: {source.get("plant_area", "")}
Fault code: {source.get("fault_code", "")}
Page: {source.get("page")}
Retrieval method: {source.get("retrieval_method", "unknown")}

Text:
{source.get("text")}"""
        )

    return "\n\n".join(context_parts)


def answer_question(question, selected_documents):
    vector_sources = search_sources(
        question=question,
        selected_documents=selected_documents,
        n_results=VECTOR_RESULTS,
    )

    keyword_sources = keyword_search_sources(
        question=question,
        selected_documents=selected_documents,
        n_results=KEYWORD_RESULTS,
    )

    sources = combine_hybrid_sources(
        vector_sources=vector_sources,
        keyword_sources=keyword_sources,
        final_limit=FINAL_CONTEXT_RESULTS,
    )

    evaluation = evaluate_retrieval(sources)
    context = build_context(sources)

    prompt = f"""
    You are an Industrial Maintenance RAG Assistant.

    Use only the retrieved maintenance documents below.
    Do not invent procedures, fault causes, alarm meanings, measurements, or safety instructions.

    If the retrieved documents do not contain enough information, say:
    "I cannot find enough information in the uploaded maintenance documents."

    Answer format:
    1. Direct answer
    2. Recommended checks or actions
    3. Safety notes, if relevant
    4. Evidence with source references

    Cite sources like this:
    (Source 1, document name, page number)

    Retrieved maintenance documents:
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