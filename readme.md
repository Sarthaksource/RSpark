# RSpark: RAG Based Research Assistant (Llama 3.2 GGUF + FastAPI)

This project allows you to upload PDFs, chunk and embed the text, and ask questions about your documents using a highly optimized, locally hosted Llama 3.2 model.

Everything runs entirely on your local machine (CPU-optimized) thanks to `llama.cpp` and GGUF models.

---

## 🚧 Frontend Status

**NOTE:** The frontend UI for this project is in progress!

For now, this repository is a pure FastAPI backend. You can interact with it using:

- Swagger UI → `http://127.0.0.1:8000/docs` (when server is running)

---

## 🚀 How to Run

Follow these steps to get your environment set up and the server running.

### 1️⃣ Set up Python Environment

```bash
python -m venv env
env\Scripts\activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

Install the optimized LLM backend (CPU wheel):

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --no-cache-dir --only-binary=llama-cpp-python
```

---

### 3️⃣ Download Model & NLP Tools

Download the GGUF model:

```bash
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q4_K_M.gguf --local-dir .
```

Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

### 4️⃣ Start the Server

```bash
uvicorn main:app --reload
```

---

## Testing the Backend

Once the server is running, test the API using sample payloads below.

---

### 📥 1. Ingest PDFs

**Endpoint:** `POST /download-and-ingest`

```json
{
  "pdfs": [
    {
      "url": "https://www.nber.org/system/files/working_papers/w28226/w28226.pdf",
      "filename": "climate_finance"
    },
    {
      "url": "https://www.nber.org/system/files/working_papers/w29421/w29421.pdf",
      "filename": "fintech_lending"
    }
  ]
}
```

---

### 💬 2. Query Documents

**Endpoint:** `POST /query`

```json
{
  "query": "What are the limitations of fintech lending?",
  "temperature": 0.7,
  "max_new_tokens": 512
}
```