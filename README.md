# Offline RAG Framework

A production-quality, fully offline Retrieval-Augmented Generation (RAG) framework built with Python, Ollama, ChromaDB, and Streamlit. 

This project is designed to be **domain-agnostic**. While it includes an agriculture demonstration dataset by default, the architecture ensures that switching to a new domain (military manuals, healthcare records, legal PDFs) requires changing only the raw dataset and configuration — the entire retrieval pipeline, LLM integration, and Streamlit UI remain untouched.

---

## 🌟 Features

- **100% Offline Inference:** After initial setup (downloading embeddings and LLM models), the system requires absolutely zero internet access. No external APIs are called.
- **Strict Factual Adherence:** Hallucination prevention is baked into the prompt engineering. The LLM is explicitly instructed to refuse to answer if the context is missing, using a strict fallback phrase.
- **Modular Architecture:** Complete separation of concerns between ingestion (`ingest.py`), retrieval (`retriever.py`), inference (`llm.py`), and the frontend (`app.py`).
- **Domain Agnostic:** All data is converted to plain text at the ingestion layer. The downstream pipeline does not know or care if the source was a CSV, PDF, or JSON file.
- **Batch Processing:** Memory-efficient, batch-based embedding generation using `SentenceTransformers`.
- **Stratified Sampling:** (For CSV demo) Ensures diverse representation of categorical columns rather than blindly taking the first N rows.
- **Rich Debug UI:** The Streamlit frontend exposes the exact retrieved context passages, similarity scores, source metadata, and the raw constructed prompt sent to the LLM.

---

## 🏗️ Architecture & Folder Structure

```text
offline-rag-framework/
│
├── app.py               # Streamlit web UI (Frontend)
├── config.py            # Centralised configuration and path resolution
├── ingest.py            # CLI script to populate the vector database
├── retriever.py         # Semantic search and ChromaDB interface
├── llm.py               # Local LLM wrapper (Ollama integration)
├── prompts.py           # Reusable prompt templates and instruction logic
├── utils.py             # Data loading, cleaning, and text extraction (Loaders)
├── logger.py            # Structured, rotating file and console logging
├── requirements.txt     # Pinned Python dependencies
├── README.md            # You are here
│
├── data/
│   ├── sample_dataset/  # Drop your raw files (CSV, PDF, etc.) here
│   └── vector_db/       # Persistent ChromaDB storage (auto-generated)
│
├── assets/              # Static assets for Streamlit (optional)
└── logs/                # Rotating application log files (auto-generated)
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- **Python 3.11+** installed.
- **Ollama** installed locally (download from [ollama.com](https://ollama.com/)).

### 2. Install Python Dependencies
It is highly recommended to use a virtual environment.
```bash
cd offline-rag-framework
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Pull the Local LLM Model
Ensure the Ollama daemon is running, then pull the required model (default is `qwen2.5:3b`, an excellent small model for factual Q&A).
```bash
ollama pull qwen2.5:3b
```
*Note: You can change the model in `config.py` (e.g., `llama3.2:3b`, `mistral:7b`).*

---

## 📊 Dataset Preparation (Agriculture Demo)

By default, the framework is configured to read the **Indian Agriculture Crop Production** dataset.

1. Download the dataset CSV (e.g., from Kaggle).
2. Rename it to `crop_production.csv` (or update `DATASET_FILENAME` in `config.py`).
3. Place it in `data/sample_dataset/crop_production.csv`.

### Switching Domains
To use your own data (e.g., PDFs, Word docs):
1. Place your files in `data/sample_dataset/`.
2. Update `DATASET_FORMAT` in `config.py` (e.g., `"pdf"` or `"docx"`).
3. Update `DATASET_FILENAME` if necessary.
*(Note: `utils.py` contains stubbed loader functions for PDF, DOCX, TXT, and JSON. You simply need to uncomment/implement the extraction logic for those formats inside `utils.py`.)*

---

## 🧠 Running the Ingestion Pipeline

Before you can ask questions, you must populate the vector database. Run the ingestion script once:

```bash
python ingest.py
```

- This script will read the raw data, clean it, split it into chunks, generate embeddings using `all-MiniLM-L6-v2`, and save them to ChromaDB.
- On the very first run, it will download the ~80MB embedding model. All subsequent runs are fully offline.
- If you change your dataset and need to rebuild the database, use:
  ```bash
  python ingest.py --force
  ```

---

## 💻 Launching the Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 💡 Example Queries

If using the agriculture demo dataset, try asking:
- *"What was the rice production in Punjab in 2015?"*
- *"Which districts cultivated wheat during the Rabi season?"*
- *"What is the typical yield for cotton in Maharashtra?"*

**Testing Hallucination Prevention:**
Ask a question completely outside the scope of the dataset (e.g., *"How do you build a rocket engine?"*). The system is strictly instructed to reply with:
> "I couldn't find sufficient information in the indexed dataset."

---

## 🔧 Future Enhancements

Because of the modular design, adding features is straightforward:
- **PDF/DOCX Support:** Simply complete the stubbed loader functions in `utils.py`. The ingestion and retrieval pipelines won't need to change.
- **Reranking:** To improve retrieval quality further, a cross-encoder reranker can be inserted inside `retriever.py` just before returning the results, with zero impact on the UI or LLM.
- **Hybrid Search:** ChromaDB supports keyword-based retrieval alongside semantic search. This can be enabled in `retriever.py` without touching the frontend.
