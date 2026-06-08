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
st.write("Ask a study question and get a simple explanation.")

hf_token = os.getenv("HF_TOKEN")

st.write("Current folder:", os.getcwd())
st.write(".env exists:", os.path.exists(".env"))
st.write("Token loaded:", bool(hf_token))


question = st.text_area(
    "Your question",
    placeholder="Example: Explain LLMs in simple words.",
)

ask_button = st.button("Ask")

if ask_button:
    if not question.strip():
        st.warning("Please write a question first.")
    elif not hf_token:
        st.error("Missing HF_TOKEN. Add it to your .env file.")
    else:
        with st.spinner("Thinking..."):
            client = InferenceClient(token=hf_token)

            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are StudyMate AI, a friendly tutor. Explain clearly and simply.",
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_tokens=300,
            )

            answer = response.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)