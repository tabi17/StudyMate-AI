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


def add_document_to_vector_store(document_name, chunk_records):
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
        document_names.add(metadata["document_name"])

    return sorted(document_names)


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
                "chunk_index": metadata.get("chunk_index", "Unknown"),
                "distance": distance,
            }
        )

    return sources