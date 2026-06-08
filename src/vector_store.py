import uuid

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def get_collection():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def document_exists(document_name):
    collection = get_collection()

    results = collection.get(
        where={"document_name": document_name},
        limit=1,
        include=["metadatas"],
    )

    return len(results["ids"]) > 0


def delete_document(document_name):
    collection = get_collection()

    if not document_exists(document_name):
        return 0

    existing = collection.get(
        where={"document_name": document_name},
        include=["metadatas"],
    )

    deleted_count = len(existing["ids"])

    collection.delete(
        where={"document_name": document_name},
    )

    return deleted_count


def add_document_to_vector_store(document_name, chunk_records, replace_existing=True):
    if replace_existing and document_exists(document_name):
        delete_document(document_name)

    collection = get_collection()
    embedding_model = load_embedding_model()

    document_id = str(uuid.uuid4())
    texts = [record["text"] for record in chunk_records]
    embeddings = embedding_model.encode(texts).tolist()

    ids = [f"{document_id}-{index}" for index in range(len(chunk_records))]

    metadatas = [
        {
            "document_name": document_name,
            "page": record["page"],
            "source_type": record.get("source_type", "unknown"),
            "chunk_index": record["chunk_index"],
        }
        for record in chunk_records
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunk_records)


def list_documents():
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    document_names = set()

    for metadata in results["metadatas"]:
        document_names.add(metadata.get("document_name", "Unknown document"))

    return sorted(document_names)


def get_document_stats():
    collection = get_collection()
    results = collection.get(include=["metadatas"])

    stats = {}

    for metadata in results["metadatas"]:
        document_name = metadata.get("document_name", "Unknown document")
        source_type = metadata.get("source_type", "unknown")

        if document_name not in stats:
            stats[document_name] = {
                "chunk_count": 0,
                "source_types": set(),
            }

        stats[document_name]["chunk_count"] += 1
        stats[document_name]["source_types"].add(source_type)

    clean_stats = {}

    for document_name, values in stats.items():
        clean_stats[document_name] = {
            "chunk_count": values["chunk_count"],
            "source_types": sorted(values["source_types"]),
        }

    return clean_stats


def search_sources(question, selected_documents, n_results):
    collection = get_collection()
    embedding_model = load_embedding_model()

    question_embedding = embedding_model.encode([question]).tolist()[0]

    query_args = {
        "query_embeddings": [question_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }

    if selected_documents:
        if len(selected_documents) == 1:
            query_args["where"] = {"document_name": selected_documents[0]}
        else:
            query_args["where"] = {"document_name": {"$in": selected_documents}}

    results = collection.query(**query_args)

    sources = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        sources.append(
            {
                "text": text,
                "document_name": metadata.get("document_name", "Unknown document"),
                "page": metadata.get("page", "Unknown"),
                "source_type": metadata.get("source_type", "unknown"),
                "chunk_index": metadata.get("chunk_index", "Unknown"),
                "distance": distance,
            }
        )

    return sources


def get_document_chunks(selected_documents, limit=12):
    collection = get_collection()

    get_args = {
        "include": ["documents", "metadatas"],
        "limit": limit,
    }

    if selected_documents:
        if len(selected_documents) == 1:
            get_args["where"] = {"document_name": selected_documents[0]}
        else:
            get_args["where"] = {"document_name": {"$in": selected_documents}}

    results = collection.get(**get_args)

    sources = []

    for text, metadata in zip(results["documents"], results["metadatas"]):
        sources.append(
            {
                "text": text,
                "document_name": metadata.get("document_name", "Unknown document"),
                "page": metadata.get("page", "Unknown"),
                "source_type": metadata.get("source_type", "unknown"),
                "chunk_index": metadata.get("chunk_index", "Unknown"),
                "distance": None,
            }
        )

    return sources