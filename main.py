import chromadb
from contextlib import asynccontextmanager
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from config import state, MODEL_PATH
from database import _ensure_collections
from routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[Startup] Loading NLP pipeline...")
    nlp = English()
    nlp.add_pipe("sentencizer")
    state["nlp"] = nlp

    print("[Startup] Loading embedding model (all-MiniLM-L6-v2)...")
    state["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") 

    print(f"[Startup] Loading optimized GGUF model: {MODEL_PATH}...")
    llm_model = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=6,
        verbose=False
    )
    state["llm_model"] = llm_model

    print("[Startup] Initializing ChromaDB...")
    chroma_client = chromadb.Client()
    state["chroma_client"] = chroma_client
    _ensure_collections()

    print("[Startup] All systems ready!\n")
    yield

    print("[Shutdown] Cleaning up...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
