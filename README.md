# StudyMate AI

StudyMate AI is a beginner-friendly AI study assistant.

The app lets a user upload study materials, ask questions about them, and generate study tools like summaries, quizzes, flashcards, simple explanations, and study plans.

This project was built step by step from a simple Streamlit app into a small RAG application with document management.

## What The App Can Do

- Upload `.txt`, `.pdf`, `.jpg`, `.jpeg`, and `.png` files
- Extract text from documents and images
- Split notes into smaller chunks
- Store chunks in a vector database
- Search the most relevant notes for a question
- Ask an LLM to answer using the retrieved notes
- Show sources used for each answer
- Show basic RAG evaluation
- Search all documents or only selected documents
- Generate study tools like summaries, quizzes, flashcards, simple explanations, and study plans
- Manage a document library and delete indexed documents

## Technologies Used

### Python

Python is the main programming language used for the project.

### Streamlit

Streamlit is used to build the web interface. It gives us buttons, upload boxes, chat messages, sidebars, checkboxes, and inputs without needing HTML/CSS/JavaScript.

### Hugging Face Inference API

Hugging Face is used to call an LLM. The LLM creates answers, summaries, quizzes, flashcards, and study plans.

### ChromaDB

ChromaDB is the vector database. It stores document chunks and their embeddings. When the user asks a question, ChromaDB finds the chunks that are closest in meaning.

### Sentence Transformers

Sentence Transformers creates embeddings. An embedding is a list of numbers that represents the meaning of text.

### RAG

RAG means Retrieval-Augmented Generation.

The app does not ask the LLM to answer from memory only. Instead, it first retrieves relevant notes from ChromaDB, then gives those notes to the LLM.

Flow:

```text
User question
-> create question embedding
-> search ChromaDB
-> retrieve relevant chunks
-> send chunks + question to LLM
-> show answer with sources
```

### pypdf

pypdf extracts text from normal PDF files.

### RapidOCR

RapidOCR extracts text from images like `.jpg`, `.jpeg`, and `.png`.

## Project Phases

## Phase 1: Basic AI Chat

In Phase 1, we created the first Streamlit app.

The app had:

- title
- question box
- ask button
- LLM answer

This phase taught:

- how to run Streamlit
- how to use a virtual environment
- how to call an LLM API
- how to store the Hugging Face token in `.env`

## Phase 2: Upload Text Notes

In Phase 2, we added `.txt` file upload.

The user could upload text notes and ask questions about the uploaded file.

This phase taught:

- `st.file_uploader`
- reading uploaded files
- sending document text to the LLM
- the first simple version of document question answering

## Phase 3: First Real RAG With ChromaDB

In Phase 3, we added real RAG.

The app started to:

- split documents into chunks
- create embeddings
- store chunks in ChromaDB
- search relevant chunks
- answer using only retrieved chunks

We also added chat history so previous answers stayed on screen.

This phase taught:

- chunks
- embeddings
- vector databases
- retrieval
- Streamlit session state

## Phase 4: PDF Support

In Phase 4, we added `.pdf` upload.

The app could extract text from PDFs and index it in ChromaDB.

We also added a source preview so we could see which chunks were retrieved.

This phase taught:

- PDF text extraction with `pypdf`
- page-based document reading
- debugging RAG sources

## Phase 5: Basic RAG Evaluation

In Phase 5, we added simple evaluation.

The app started showing:

- retrieved sources
- distance scores
- retrieval quality labels like strong, medium, or weak

This phase taught:

- how to inspect retrieval quality
- why RAG answers should not be trusted blindly
- how to see whether the right notes were retrieved

## Phase 6: Multi-Document RAG And Modular Code

In Phase 6, we added support for multiple documents.

The user could:

- upload many files
- search all documents
- select only specific documents
- ask questions using selected sources

We also split the code into separate files:

```text
src/config.py
src/document_loader.py
src/text_splitter.py
src/vector_store.py
src/llm_client.py
src/rag_pipeline.py
```

This phase taught:

- modular code structure
- document metadata
- filtering ChromaDB by document name
- making the project easier to grow

## Phase 7: Study Tools / Simple Agents

In Phase 7, we added study tools.

The app could generate:

- summaries
- flashcards
- quizzes
- simple explanations
- study plans

These are simple agents because each tool has a specific job and its own prompt.

This phase taught:

- task-specific prompts
- reusable study workflows
- building features on top of RAG

## Phase 8: Image OCR

In Phase 8, we added `.jpg`, `.jpeg`, and `.png` uploads.

The app uses OCR to extract text from images, then indexes that text like normal notes.

This phase taught:

- OCR
- reading text from images
- supporting screenshots or photos of printed notes

## Phase 9: Document Library Management

In Phase 9, we added a document library.

The app can now:

- show indexed documents
- show chunk counts
- show source types
- replace existing documents with the same name
- delete selected documents from ChromaDB

This phase taught:

- persistent vector database management
- deleting records from ChromaDB
- preventing duplicate chunks
- making the app easier to use over time

## Main App Flow

```text
Upload file
-> extract text
-> split text into chunks
-> create embeddings
-> store chunks in ChromaDB
-> user asks question
-> retrieve relevant chunks
-> send chunks to LLM
-> show answer, sources, and evaluation
```

## How To Run The Project

Open PowerShell:

```powershell
cd C:\Users\Tabita\Documents\StudyMate\StudyMate-AI
.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Important Files

### app.py

The Streamlit user interface.

### src/config.py

Project settings like model names, chunk size, and ChromaDB path.

### src/document_loader.py

Reads `.txt`, `.pdf`, `.jpg`, `.jpeg`, and `.png` files.

### src/text_splitter.py

Splits extracted text into chunks.

### src/vector_store.py

Handles ChromaDB: add, search, list, and delete documents.

### src/llm_client.py

Calls the Hugging Face LLM.

### src/rag_pipeline.py

Connects retrieval, prompt building, LLM answering, and evaluation.

### src/study_tools.py

Contains the study tools: summarize, flashcards, quiz, explain simply, and study plan.

## Files Not To Push To GitHub

These should be in `.gitignore`:

```text
.venv/
.env
__pycache__/
chroma_db/
```

`.env` contains the secret Hugging Face token, so it must stay private.

`chroma_db/` is local database storage and should not be pushed.

## Current Level

This is a junior-level AI project, but it includes real professional concepts:

- LLM API usage
- RAG
- embeddings
- vector databases
- OCR
- evaluation
- modular code
- document management

## Good Next Phases

Possible next improvements:

- deploy the app online
- add login or user accounts
- improve answer quality with reranking
- add better evaluation tests
- add scanned PDF OCR
- save user feedback
- export flashcards or quizzes
- improve the UI design
