import os

from config import state, DISTANCE_THRESHOLD


# ─────────────────────────────────────────────
# SEARCH & CONTEXT BUILDING
# ─────────────────────────────────────────────
def search(query_text: str):
    embedding_model = state["embedding_model"]
    abstract_collection = state["abstract_collection"]
    fulltext_collection = state["fulltext_collection"]

    query_embedding = embedding_model.encode(query_text).tolist()

    parent_results = abstract_collection.query(
        query_embeddings=[query_embedding], n_results=3)
    relevant_doc_ids = [meta["doc_id"] for meta in parent_results["metadatas"][0]]

    child_results = fulltext_collection.query(
        query_embeddings=[query_embedding],
        n_results=5,   # raised from 5 — more chunks = more context for the LLM
        where={"doc_id": {"$in": relevant_doc_ids}}
    )
    return parent_results, child_results


def build_context_items(child_results) -> list[dict]:
    context_items = []
    for group_idx, result_group in enumerate(child_results["metadatas"]): # chroma return list of list (groups)
        for i, meta in enumerate(result_group):
            dist = child_results["distances"][group_idx][i]
            if dist > DISTANCE_THRESHOLD:
                continue
            context_items.append({
                "page_number": meta.get("page_number", "N/A"),
                "pdf_link": os.path.basename(meta.get("pdf_link", "Unknown")),
                "text": child_results["documents"][group_idx][i],
                "distance": round(dist, 4)
            })
    return context_items
