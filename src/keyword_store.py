import re

from rank_bm25 import BM25Okapi

from src.vector_store import get_all_sources


def tokenize(text):
    text = text.lower()
    return re.findall(r"[a-z0-9\-]+", text)


def build_search_text(source):
    metadata_parts = [
        source.get("document_name", ""),
        source.get("document_type", ""),
        source.get("equipment_id", ""),
        source.get("component", ""),
        source.get("plant_area", ""),
        source.get("fault_code", ""),
    ]

    return " ".join(metadata_parts + [source.get("text", "")])


def keyword_search_sources(question, selected_documents=None, n_results=5):
    sources = get_all_sources(selected_documents=selected_documents)

    if not sources:
        return []

    corpus = [build_search_text(source) for source in sources]
    tokenized_corpus = [tokenize(text) for text in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_question = tokenize(question)
    scores = bm25.get_scores(tokenized_question)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    keyword_sources = []

    for index in ranked_indexes[:n_results]:
        score = float(scores[index])

        if score <= 0:
            continue

        source = sources[index].copy()
        source["keyword_score"] = score
        source["retrieval_method"] = "keyword"
        keyword_sources.append(source)

    return keyword_sources