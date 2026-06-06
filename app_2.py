import os
import uuid

import chromadb
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
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
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def reset_chat():
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "document_name" not in st.session_state:
    st.session_state.document_name = ""


embedding_model = load_embedding_model()
collection = get_chroma_collection()

uploaded_file = st.file_uploader(
    "Upload a text file",
    type=["txt"],
)

if uploaded_file is not None and uploaded_file.name != st.session_state.document_name:
    document_text = uploaded_file.read().decode("utf-8")
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

    st.success(f"Uploaded and indexed {uploaded_file.name}.")
    st.info(f"Created {len(chunks)} searchable chunks.")

if st.session_state.document_uploaded:
    st.caption(f"Current document: {st.session_state.document_name}")

    if st.button("Clear chat"):
        reset_chat()

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
else:
    st.info("Upload a .txt file to start asking questions.")