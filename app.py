import streamlit as st

from src.document_loader import extract_document
from src.rag_pipeline import answer_question
from src.maintenance_tools import run_study_tool
from src.text_splitter import split_document_pages
from src.vector_store import (
    add_document_to_vector_store,
    delete_document,
    get_document_stats,
    list_documents,
)


st.set_page_config(
    page_title="Industrial Maintenance RAG Assistant",
    page_icon="",
    layout="centered",
)

st.title("Industrial Maintenance RAG Assistant")
st.write(
    "Upload maintenance documents, then ask technical questions about equipment, "
    "faults, and procedures."
)


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


def show_source(source, index):
    st.write(f"Source {index}")
    st.write(f"Document: {source.get('document_name', 'Unknown document')}")
    st.write(f"Type: {source.get('source_type', 'unknown')}")
    st.write(f"Document type: {source.get('document_type', 'unknown')}")
    st.write(f"Equipment ID: {source.get('equipment_id', '')}")
    st.write(f"Component: {source.get('component', '')}")
    st.write(f"Plant area: {source.get('plant_area', '')}")
    st.write(f"Fault code: {source.get('fault_code', '')}")
    st.write(f"Page: {source.get('page', 'Unknown')}")
    st.write(f"Retrieval: {source.get('retrieval_method', 'unknown')}")

    if source.get("distance") is not None:
        st.write(f"Vector distance: {source['distance']:.3f}")

    if source.get("keyword_score") is not None:
        st.write(f"Keyword score: {source['keyword_score']:.3f}")

    st.write(source.get("text", ""))


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Documents")

    st.subheader("Document Metadata")

    document_type = st.selectbox(
        "Document type",
        options=[
            "manual",
            "procedure",
            "troubleshooting guide",
            "inspection report",
            "maintenance record",
            "alarm list",
            "other",
        ],
    )

    equipment_id = st.text_input("Equipment ID", placeholder="Example: P-204")
    component = st.text_input("Component", placeholder="Example: centrifugal pump")
    plant_area = st.text_input("Plant area", placeholder="Example: cooling water system")
    fault_code = st.text_input("Fault / alarm code", placeholder="Example: E-45")

    uploaded_files = st.file_uploader(
        "Upload manuals, procedures, reports, or images",
        type=["txt", "pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    replace_existing = st.checkbox("Replace existing files with same name", value=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
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
                replace_existing=replace_existing,
                base_metadata={
                    "document_type": document_type,
                    "equipment_id": equipment_id,
                    "component": component,
                    "plant_area": plant_area,
                    "fault_code": fault_code,
                },
            )

            st.success(f"Indexed {uploaded_file.name}")
            st.caption(f"{chunk_count} chunks created")

    available_documents = list_documents()
    document_stats = get_document_stats()

    st.divider()

    st.subheader("Library")

    if not available_documents:
        st.caption("No indexed documents yet.")
    else:
        for document_name in available_documents:
            stats = document_stats.get(
                document_name,
                {
                    "chunk_count": 0,
                    "source_types": ["unknown"],
                },
            )

            source_types = ", ".join(stats["source_types"])

            with st.expander(document_name):
                st.write(f"Chunks: {stats['chunk_count']}")
                st.write(f"Type: {source_types}")

        documents_to_delete = st.multiselect(
            "Delete documents",
            options=available_documents,
        )

        if st.button("Delete selected documents"):
            if not documents_to_delete:
                st.warning("Choose at least one document to delete.")
            else:
                total_deleted = 0

                for document_name in documents_to_delete:
                    total_deleted += delete_document(document_name)

                reset_chat()
                st.success(f"Deleted {total_deleted} chunks.")
                st.rerun()

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

    st.subheader("Maintenance Tools")

    maintenance_tool = st.selectbox(
        "Choose maintenance tool",
        options=[
            "Troubleshooting Steps",
            "Procedure Summary",
            "Safety Checklist",
            "Maintenance Plan",
            "Fault Report Summary",
            "Root Cause Hypotheses",
        ],
    )

    run_tool_button = st.button("Run maintenance tool")

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
            tool_prompt = f"Run maintenance tool: {maintenance_tool}"

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": tool_prompt,
                }
            )

            with st.chat_message("assistant"):
                with st.spinner(f"Running {maintenance_tool}..."):
                    try:
                        result = run_study_tool(
                            tool_name=maintenance_tool,
                            selected_documents=search_documents,
                        )

                        answer = result["answer"]
                        sources = result["sources"]

                        st.write(answer)

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

                        if evaluation.get("best_distance") is not None:
                            st.write(f"Best distance: {evaluation['best_distance']:.3f}")
                            st.write(f"Average distance: {evaluation['average_distance']:.3f}")

                        if evaluation["label"] == "Weak retrieval":
                            st.warning("The retrieved documents may not match the question well.")

                        if "retrieval_methods" in evaluation:
                            st.write(
                                f"Retrieval methods: {', '.join(evaluation['retrieval_methods'])}"
                            )

                if show_sources and "sources" in message:
                    with st.expander("Sources used"):
                        for index, source in enumerate(message["sources"], start=1):
                            show_source(source, index)

    question = st.chat_input("Ask a maintenance question about your selected documents")

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
                with st.spinner("Searching maintenance documents..."):
                    try:
                        result = answer_question(
                            question=question,
                            selected_documents=search_documents,
                        )

                        answer = result["answer"]
                        sources = result["sources"]
                        evaluation = result["evaluation"]

                        st.write(answer)

                        add_assistant_message(
                            content=answer,
                            sources=sources,
                            evaluation=evaluation,
                        )

                    except ValueError as error:
                        st.error(str(error))