from config import state


# ─────────────────────────────────────────────
# COLLECTION HELPERS
# ─────────────────────────────────────────────
def _ensure_collections():
    client = state["chroma_client"]
    existing = {c.name for c in client.list_collections()}

    if "paper_abstracts" not in existing:
        state["abstract_collection"] = client.create_collection(
            name="paper_abstracts", metadata={"hnsw:space": "cosine"})
    else:
        state["abstract_collection"] = client.get_collection("paper_abstracts")

    if "paper_fulltexts" not in existing:
        state["fulltext_collection"] = client.create_collection(
            name="paper_fulltexts", metadata={"hnsw:space": "cosine"})
    else:
        state["fulltext_collection"] = client.get_collection("paper_fulltexts")
