from src.llm_client import generate_answer
from src.rag_pipeline import build_context
from src.vector_store import get_document_chunks


TOOL_INSTRUCTIONS = {
    "Summarize": """
Create a clear study summary from the retrieved notes.

Format:
- Main idea
- Key points
- Important terms
- What to remember
""",
    "Flashcards": """
Create flashcards from the retrieved notes.

Format each flashcard like:
Q: question
A: answer

Create 8 to 12 flashcards.
""",
    "Quiz": """
Create a short quiz from the retrieved notes.

Format:
1. Question
   A. option
   B. option
   C. option
   D. option
Answer: correct option

Create 5 questions.
""",
    "Explain Simply": """
Explain the retrieved notes in simple beginner-friendly language.

Use short paragraphs.
Use examples where helpful.
Avoid complicated words when possible.
""",
    "Study Plan": """
Create a practical study plan from the retrieved notes.

Format:
- What to study first
- What to study second
- Practice questions
- Review checklist
""",
}


def run_study_tool(tool_name, selected_documents):
    if tool_name not in TOOL_INSTRUCTIONS:
        raise ValueError(f"Unknown study tool: {tool_name}")

    sources = get_document_chunks(
        selected_documents=selected_documents,
        limit=12,
    )

    if not sources:
        return {
            "answer": "No document content found for this tool.",
            "sources": [],
        }

    context = build_context(sources)

    prompt = f"""
You are StudyMate AI, a helpful study assistant.

Use only the retrieved notes below.
If the notes do not contain enough information, say what is missing.

Task:
{TOOL_INSTRUCTIONS[tool_name]}

Retrieved notes:
{context}
"""

    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "sources": sources,
    }