import os
import uuid

import chromadb
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


load_dotenv(dotenv_path=".env")

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="centered",
)

st.title("StudyMate AI")
st.write("Upload study notes, then ask questions about them.")

hf_token = os.getenv("HF_TOKEN")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def get_chroma_collection():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="study_notes")


def split_text(text, chunk_size=700, chunk_overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap

    return chunks


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8")


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages_text.append(f"Page {page_number}\n{text}")

    return "\n\n".join(pages_text)


def extract_text(uploaded_file):
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)

    return ""


def evaluate_retrieval(distances):
    if not distances:
        return {
            "label": "No sources found",
            "average_distance": None,
            "best_distance": None,
        }

    average_distance = sum(distances) / len(distances)
    best_distance = min(distances)

    if best_distance < 0.8:
        label = "Strong retrieval"
    elif best_distance < 1.3:
        label = "Medium retrieval"
    else:
        label = "Weak retrieval"

    return {
        "label": label,
        "average_distance": average_distance,
        "best_distance": best_distance,
    }


def reset_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

if "last_evaluation" not in st.session_state:
    st.session_state.last_evaluation = None


embedding_model = load_embedding_model()
collection = get_chroma_collection()

with st.sidebar:
    st.header("Document")

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["txt", "pdf"],
    )

    if st.session_state.document_uploaded:
        st.success("Document ready")
        st.write(st.session_state.document_name)

    show_sources = st.checkbox("Show retrieved sources", value=True)
    show_evaluation = st.checkbox("Show RAG evaluation", value=True)

    if st.button("Clear chat"):
        reset_chat()

if uploaded_file is not None and uploaded_file.name != st.session_state.document_name:
    document_text = extract_text(uploaded_file)

    if not document_text.strip():
        st.error("Could not extract text from this file.")
    else:
        chunks = split_text(document_text)
        document_id = str(uuid.uuid4())

        embeddings = embedding_model.encode(chunks).tolist()

        ids = [f"{document_id}-{index}" for index in range(len(chunks))]

        metadatas = [
            {
                "document_name": uploaded_file.name,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        st.session_state.document_uploaded = True
        st.session_state.document_name = uploaded_file.name
        st.session_state.messages = []
        st.session_state.last_sources = []
        st.session_state.last_evaluation = None

        st.success(f"Uploaded and indexed {uploaded_file.name}.")
        st.info(f"Created {len(chunks)} searchable chunks.")

if st.session_state.document_uploaded:
    st.caption(f"Current document: {st.session_state.document_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant" and "evaluation" in message:
                evaluation = message["evaluation"]

                with st.expander("RAG evaluation"):
                    st.write(f"Retrieval quality: {evaluation['label']}")

                    if evaluation["best_distance"] is not None:
                        st.write(f"Best distance: {evaluation['best_distance']:.3f}")
                        st.write(f"Average distance: {evaluation['average_distance']:.3f}")

            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources used"):
                    for index, source in enumerate(message["sources"], start=1):
                        st.write(f"Source {index}")
                        st.write(f"Distance: {source['distance']:.3f}")
                        st.write(source["text"])

    question = st.chat_input("Ask a question about your notes")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                question_embedding = embedding_model.encode([question]).tolist()[0]

                results = collection.query(
                    query_embeddings=[question_embedding],
                    n_results=3,
                    where={"document_name": st.session_state.document_name},
                    include=["documents", "metadatas", "distances"],
                )

                retrieved_chunks = results["documents"][0]
                retrieved_distances = results["distances"][0]
                retrieved_metadatas = results["metadatas"][0]

                sources = []

                for chunk, distance, metadata in zip(
                    retrieved_chunks,
                    retrieved_distances,
                    retrieved_metadatas,
                ):
                    sources.append(
                        {
                            "text": chunk,
                            "distance": distance,
                            "chunk_index": metadata["chunk_index"],
                        }
                    )

                evaluation = evaluate_retrieval(retrieved_distances)

                context_parts = []

                for index, source in enumerate(sources, start=1):
                    context_parts.append(
                        f"Source {index}:\n{source['text']}"
                    )

                context = "\n\n".join(context_parts)

                client = InferenceClient(token=hf_token)

                prompt = f"""
You are StudyMate AI, a friendly study tutor.

Use only the retrieved notes below to answer the user's question.
If the answer is not in the notes, say:
"I cannot find that in the uploaded notes."

At the end of your answer, mention which source number you used.

Retrieved notes:
{context}

User question:
{question}
"""

                response = client.chat.completions.create(
                    model="Qwen/Qwen2.5-7B-Instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are StudyMate AI. Answer clearly, simply, and only using the retrieved notes.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_tokens=400,
                )

                answer = response.choices[0].message.content

            st.write(answer)

            if show_evaluation:
                with st.expander("RAG evaluation"):
                    st.write(f"Retrieval quality: {evaluation['label']}")

                    if evaluation["best_distance"] is not None:
                        st.write(f"Best distance: {evaluation['best_distance']:.3f}")
                        st.write(f"Average distance: {evaluation['average_distance']:.3f}")

                    if evaluation["label"] == "Weak retrieval":
                        st.warning("The retrieved notes may not match the question well.")

            if show_sources:
                with st.expander("Sources used"):
                    for index, source in enumerate(sources, start=1):
                        st.write(f"Source {index}")
                        st.write(f"Distance: {source['distance']:.3f}")
                        st.write(source["text"])

            col1, col2 = st.columns(2)

            with col1:
                st.button("Helpful", key=f"helpful-{len(st.session_state.messages)}")

            with col2:
                st.button("Not helpful", key=f"not-helpful-{len(st.session_state.messages)}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "evaluation": evaluation,
            }
        )
else:
    st.info("Upload a .txt or .pdf file to start asking questions.")