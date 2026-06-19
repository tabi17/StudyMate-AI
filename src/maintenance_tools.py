from src.llm_client import generate_answer
from src.rag_pipeline import build_context
from src.vector_store import get_document_chunks


TOOL_INSTRUCTIONS = {
    "Troubleshooting Steps": """
Create troubleshooting steps from the retrieved maintenance documents.

Format:
- Observed problem
- Possible causes
- Inspection steps
- Recommended checks
- Safety precautions
- Sources
""",
    "Procedure Summary": """
Summarize the maintenance procedure from the retrieved documents.

Format:
- Purpose
- Required conditions
- Main steps
- Tools or materials mentioned
- Safety warnings
- Sources
""",
    "Safety Checklist": """
Create a safety checklist from the retrieved maintenance documents.

Format:
- Before work
- During work
- After work
- Warnings
- Sources
""",
    "Maintenance Plan": """
Create a maintenance plan from the retrieved documents.

Format:
- Inspection tasks
- Preventive maintenance tasks
- Frequency if mentioned
- Required checks
- Sources
""",
    "Fault Report Summary": """
Summarize fault or intervention information from the retrieved documents.

Format:
- Fault description
- Equipment involved
- Symptoms
- Actions taken
- Recommendations
- Sources
""",
    "Root Cause Hypotheses": """
List possible root cause hypotheses based only on retrieved documents.

Format:
- Hypothesis
- Supporting evidence
- What to inspect next
- Source
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