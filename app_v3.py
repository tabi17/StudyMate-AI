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


def reset_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "last_retrieved_chunks" not in st.session_state:
    st.session_state.last_retrieved_chunks = []


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

    show_sources = st.checkbox("Show retrieved sources", value=False)

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
        st.session_state.last_retrieved_chunks = []

        st.success(f"Uploaded and indexed {uploaded_file.name}.")
        st.info(f"Created {len(chunks)} searchable chunks.")

if st.session_state.document_uploaded:
    st.caption(f"Current document: {st.session_state.document_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

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
                )

                retrieved_chunks = results["documents"][0]
                st.session_state.last_retrieved_chunks = retrieved_chunks

                context = "\n\n".join(retrieved_chunks)

                client = InferenceClient(token=hf_token)

                prompt = f"""
You are StudyMate AI, a friendly study tutor.

Use only the retrieved notes below to answer the user's question.
If the answer is not in the notes, say:
"I cannot find that in the uploaded notes."

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

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    if show_sources and st.session_state.last_retrieved_chunks:
        st.subheader("Retrieved Sources")

        for index, chunk in enumerate(st.session_state.last_retrieved_chunks, start=1):
            with st.expander(f"Source chunk {index}"):
                st.write(chunk)
else:
    st.info("Upload a .txt or .pdf file to start asking questions.")