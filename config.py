import os

MIN_TOKEN_LENGTH = 30
DISTANCE_THRESHOLD = 0.5
CHUNK_SIZE = 10
MODEL_PATH = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

state = {
    "nlp": None,
    "embedding_model": None,
    "llm_model": None,
    "chroma_client": None,
    "abstract_collection": None,
    "fulltext_collection": None,
}