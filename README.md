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



## Current Level

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
