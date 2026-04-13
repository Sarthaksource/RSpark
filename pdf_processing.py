import re
import uuid

import fitz
import pandas as pd

from config import state, CHUNK_SIZE, MIN_TOKEN_LENGTH


# ─────────────────────────────────────────────
# PDF PROCESSING HELPERS
# ─────────────────────────────────────────────
def open_read_pdf(path: str) -> list[dict]:
    pdf = fitz.open(path)
    pages_texts = []
    for page_number, page in enumerate(pdf):
        text = page.get_text().replace("\n", " ").strip()
        pages_texts.append({"page_number": page_number + 1, "text": text})
    return pages_texts


def extract_first_two_pages_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(min(2, len(doc))):
        text += doc[page_num].get_text("text") + "\n"
    doc.close()
    return text.strip()


def create_chunks(input_list: list, chunk_size: int) -> list[list]:
    return [input_list[i: i + chunk_size] for i in range(0, len(input_list), chunk_size)] # a list of list of sentences, each sublist contains CHUNK_SIZE=10 sentences


def preprocess(nlp, pages_texts: list[dict]) -> list[dict]:
    for item in pages_texts:
        item["sentences"] = [str(s) for s in nlp(item["text"]).sents]
    for item in pages_texts:
        item["sentences_chunks"] = create_chunks(item["sentences"], CHUNK_SIZE)

    pages_chunks = []
    for item in pages_texts:
        for sentence_chunk in item["sentences_chunks"]:
            joined = "".join(sentence_chunk).replace("  ", " ").strip()
            joined = re.sub(r"\.([A-Z])", r". \1", joined)
            pages_chunks.append({
                "page_number": item["page_number"],
                "sentence_chunk": joined, # stores a string
                "chunk_token_count": len(joined) / 4
            })

    df = pd.DataFrame(pages_chunks)
    return df[df["chunk_token_count"] > MIN_TOKEN_LENGTH].to_dict(orient="records")


def ingest_pdf(pdf_path: str) -> str:
    embedding_model = state["embedding_model"]
    abstract_collection = state["abstract_collection"]
    fulltext_collection = state["fulltext_collection"]
    nlp = state["nlp"]

    pages_texts = open_read_pdf(pdf_path)
    chunks = preprocess(nlp, pages_texts)

    # Abstract (parent)
    abstract_text = extract_first_two_pages_text(pdf_path)
    abstract_embedding = embedding_model.encode(abstract_text).tolist() # returns numpy array/vector, converted to list
    doc_id = str(uuid.uuid4())
    abstract_collection.add(
        ids=[doc_id],
        embeddings=[abstract_embedding],
        metadatas=[{"pdf_link": pdf_path, "doc_id": doc_id}],
        documents=[abstract_text]
    )

    # Chunks (child)
    for chunk in chunks:
        chunk_embedding = embedding_model.encode(chunk["sentence_chunk"]).tolist()
        fulltext_collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[chunk_embedding],
            metadatas=[{
                "pdf_link": pdf_path,
                "doc_id": doc_id, # linked using parent's doc_id
                "page_number": chunk["page_number"]
            }],
            documents=[chunk["sentence_chunk"]]
        )

    return doc_id # return parent's doc_id
