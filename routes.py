import os
import shutil

import requests
from fastapi import APIRouter, UploadFile, File, HTTPException

from config import state, UPLOAD_DIR
from database import _ensure_collections
from pdf_processing import ingest_pdf
from search import search, build_context_items
from llm import generate_answer
from schemas import QueryRequest, DownloadRequest, BulkDownloadRequest
from typing import List

router = APIRouter()

@router.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "RAG API is running"}

@router.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "models_loaded": state["llm_model"] is not None,
        "collections": {
            "abstracts": state["abstract_collection"].count() if state["abstract_collection"] else 0,
            "chunks": state["fulltext_collection"].count() if state["fulltext_collection"] else 0,
        }
    }

@router.post("/upload", tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file and ingest it into the vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Not a PDF file.")
 
    save_path = os.path.join(UPLOAD_DIR, file.filename)
 
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        doc_id = ingest_pdf(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
 
    return {
        "message": "PDF ingested successfully.",
        "filename": file.filename,
        "doc_id": doc_id,
        "chunks_total": state["fulltext_collection"].count()
    }

@router.post("/download-and-ingest", tags=["Ingestion"])
def download_and_ingest(req: BulkDownloadRequest):
    """Download one or multiple PDFs from URLs and ingest them."""
    results = []
    failed = []

    for item in req.pdfs:
        file_path = os.path.join(UPLOAD_DIR, item.filename + ".pdf")

        try:
            if not os.path.exists(file_path):
                response = requests.get(item.url)
                if response.status_code != 200:
                    failed.append({"filename": item.filename, "reason": f"Download failed: {response.status_code}"})
                    continue
                with open(file_path, "wb") as f:
                    f.write(response.content)

            doc_id = ingest_pdf(file_path)
            results.append({"filename": item.filename + ".pdf", "doc_id": doc_id})

        except Exception as e:
            failed.append({"filename": item.filename, "reason": str(e)})

    return {
        "message": f"{len(results)} PDF(s) ingested, {len(failed)} failed.",
        "success": results,
        "failed": failed,
        "chunks_total": state["fulltext_collection"].count()
    }

@router.post("/query", tags=["Query"])
def query(req: QueryRequest):
    """Send a query and get an LLM-generated answer from the ingested PDFs."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if state["abstract_collection"].count() == 0:
        raise HTTPException(status_code=400, detail="No PDFs have been ingested yet. Upload a PDF first.")

    try:
        _, child_results = search(req.query)
        context_items = build_context_items(child_results) # returns organized child_results data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    if not context_items:
        return {"answer": "No relevant context found for your query.", "sources": []}

    try:
        answer = generate_answer(
            req.query, context_items,
            temperature=req.temperature,
            max_new_tokens=req.max_new_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return {
        "answer": answer,
        "sources": [
            {
                "page": item["page_number"],
                "file": item["pdf_link"],
                "relevance_score": round(1 - item["distance"], 4),
                "excerpt": item["text"][:200] + "..."
            }
            for item in context_items
        ]
    }

@router.delete("/reset", tags=["Admin"])
def reset_collections():
    """Delete and recreate all collections (clears all ingested PDFs)."""
    client = state["chroma_client"]
    try:
        client.delete_collection("paper_abstracts")
        client.delete_collection("paper_fulltexts")
    except Exception:
        pass
    _ensure_collections()
    return {"message": "Collections reset successfully."}


@router.get("/collections/info", tags=["Admin"])
def collection_info():
    """Get counts for both collections."""
    return {
        "abstract_count": state["abstract_collection"].count(),
        "fulltext_chunk_count": state["fulltext_collection"].count()
    }