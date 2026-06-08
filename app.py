import streamlit as st

from src.document_loader import extract_document
from src.rag_pipeline import answer_question
from src.study_tools import run_study_tool
from src.text_splitter import split_document_pages
from src.vector_store import add_document_to_vector_store, list_documents


st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="centered",
)

st.title("StudyMate AI")
st.write("Upload study notes, choose sources, then ask questions or generate study tools.")


def reset_chat():
    st.session_state.messages = []


def add_assistant_message(content, sources=None, evaluation=None):
    message = {
        "role": "assistant",
        "content": content,
    }

    if sources is not None:
        message["sources"] = sources

    if evaluation is not None:
        message["evaluation"] = evaluation

    st.session_state.messages.append(message)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()


with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.indexed_files:
                pages = extract_document(uploaded_file)

                if not pages:
                    st.error(f"Could not extract text from {uploaded_file.name}.")
                    continue

                chunk_records = split_document_pages(pages)

                if not chunk_records:
                    st.error(f"No readable text found in {uploaded_file.name}.")
                    continue

                chunk_count = add_document_to_vector_store(
                    document_name=uploaded_file.name,
                    chunk_records=chunk_records,
                )

                st.session_state.indexed_files.add(uploaded_file.name)
                st.success(f"Indexed {uploaded_file.name}")
                st.caption(f"{chunk_count} chunks created")

    available_documents = list_documents()

    st.divider()

    st.subheader("Search Scope")

    search_all = st.checkbox("Search all documents", value=True)

    selected_documents = []

    if available_documents and not search_all:
        selected_documents = st.multiselect(
            "Choose documents",
            options=available_documents,
            default=available_documents[:1],
        )

    show_sources = st.checkbox("Show sources", value=True)
    show_evaluation = st.checkbox("Show evaluation", value=True)

    st.divider()

    st.subheader("Study Tools")

    study_tool = st.selectbox(
        "Choose tool",
        options=[
            "Summarize",
            "Flashcards",
            "Quiz",
            "Explain Simply",
            "Study Plan",
        ],
    )

    run_tool_button = st.button("Run study tool")

    st.divider()

    if st.button("Clear chat"):
        reset_chat()


if not available_documents:
    st.info("Upload one or more .txt, .pdf, .jpg, .jpeg, or .png files to start.")
else:
    if search_all:
        st.caption("Searching all uploaded documents.")
        search_documents = []
    else:
        st.caption(f"Searching {len(selected_documents)} selected document(s).")
        search_documents = selected_documents

    if run_tool_button:
        if not search_all and not selected_documents:
            st.warning("Choose at least one document or turn on Search all documents.")
        else:
            tool_prompt = f"Run study tool: {study_tool}"

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": tool_prompt,
                }
            )

            with st.chat_message("user"):
                st.write(tool_prompt)

            with st.chat_message("assistant"):
                with st.spinner(f"Running {study_tool}..."):
                    try:
                        result = run_study_tool(
                            tool_name=study_tool,
                            selected_documents=search_documents,
                        )

                        answer = result["answer"]
                        sources = result["sources"]

                        st.write(answer)

                        if show_sources and sources:
                            with st.expander("Sources used"):
                                for index, source in enumerate(sources, start=1):
                                    st.write(f"Source {index}")
                                    st.write(f"Document: {source['document_name']}")
                                    st.write(f"Page: {source['page']}")
                                    st.write(source["text"])

                        add_assistant_message(
                            content=answer,
                            sources=sources,
                        )

                    except ValueError as error:
                        st.error(str(error))

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            if message["role"] == "assistant":
                if show_evaluation and "evaluation" in message:
                    evaluation = message["evaluation"]

                    with st.expander("RAG evaluation"):
                        st.write(f"Retrieval quality: {evaluation['label']}")

                        if evaluation["best_distance"] is not None:
                            st.write(f"Best distance: {evaluation['best_distance']:.3f}")
                            st.write(f"Average distance: {evaluation['average_distance']:.3f}")

                        if evaluation["label"] == "Weak retrieval":
                            st.warning("The retrieved notes may not match the question well.")

                if show_sources and "sources" in message:
                    with st.expander("Sources used"):
                        for index, source in enumerate(message["sources"], start=1):
                            st.write(f"Source {index}")
                            st.write(f"Document: {source['document_name']}")
                            st.write(f"Page: {source['page']}")

                            if source["distance"] is not None:
                                st.write(f"Distance: {source['distance']:.3f}")

                            st.write(source["text"])

    question = st.chat_input("Ask a question about your selected notes")

    if question:
        if not search_all and not selected_documents:
            st.warning("Choose at least one document or turn on Search all documents.")
        else:
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
                    try:
                        result = answer_question(
                            question=question,
                            selected_documents=search_documents,
                        )

                        answer = result["answer"]
                        sources = result["sources"]
                        evaluation = result["evaluation"]

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
                                    st.write(f"Document: {source['document_name']}")
                                    st.write(f"Page: {source['page']}")
                                    st.write(f"Distance: {source['distance']:.3f}")
                                    st.write(source["text"])

                        add_assistant_message(
                            content=answer,
                            sources=sources,
                            evaluation=evaluation,
                        )

                    except ValueError as error:
                        st.error(str(error))