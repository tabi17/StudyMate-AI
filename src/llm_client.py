import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from src.config import LLM_MODEL_NAME


load_dotenv(dotenv_path=".env")


def get_hf_token():
    return os.getenv("HF_TOKEN")


def generate_answer(prompt):
    hf_token = get_hf_token()

    if not hf_token:
        raise ValueError("Missing HF_TOKEN. Add it to your .env file.")

    client = InferenceClient(token=hf_token)

    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
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
        max_tokens=500,
    )

    return response.choices[0].message.content