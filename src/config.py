CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "study_notes"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
N_RESULTS = 5
VECTOR_RESULTS = 5
KEYWORD_RESULTS = 5
FINAL_CONTEXT_RESULTS = 6


# VECTOR_RESULTS = how many chunks from Chroma vector search
# KEYWORD_RESULTS = how many chunks from BM25 keyword search
# FINAL_CONTEXT_RESULTS = how many final chunks go to the LLM