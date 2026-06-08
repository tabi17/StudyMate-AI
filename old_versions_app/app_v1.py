import os

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv(dotenv_path=".env")

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="centered",
)

st.title("StudyMate AI")
st.write("Upload study notes, then ask questions about them.")

hf_token = os.getenv("HF_TOKEN")

uploaded_file = st.file_uploader(
    "Upload a text file",
    type=["txt"],
)

document_text = ""

if uploaded_file is not None:
    document_bytes = uploaded_file.read()
    document_text = document_bytes.decode("utf-8")

    st.success("File uploaded successfully.")

    with st.expander("Preview uploaded document"):
        st.write(document_text[:2000])

question = st.text_area(
    "Your question",
    placeholder="Example: Summarize these notes in simple words.",
)

ask_button = st.button("Ask StudyMate")

if ask_button:
    if not hf_token:
        st.error("Missing HF_TOKEN. Add it to your .env file.")
    elif uploaded_file is None:
        st.warning("Please upload a .txt file first.")
    elif not question.strip():
        st.warning("Please write a question first.")
    else:
        with st.spinner("Reading your notes and thinking..."):
            client = InferenceClient(token=hf_token)

            prompt = f"""
You are StudyMate AI, a friendly study tutor.

Use the uploaded study notes to answer the user's question.
If the answer is not found in the notes, say:
"I cannot find that in the uploaded notes."

Uploaded notes:
{document_text}

User question:
{question}
"""

            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are StudyMate AI. Answer clearly, simply, and only using the uploaded notes.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=400,
            )

            answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)