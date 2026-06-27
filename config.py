"""
config.py — Centralised Configuration for the Offline RAG Framework
=====================================================================
All tuneable parameters live here. No other module should hardcode values.

Design Philosophy:
    - Every constant that might change between domains, environments, or
      experiments is defined here.
    - Switching to a new domain (e.g. military manuals, legal documents)
      requires changing only DATASET_PATH, DATASET_FORMAT, and optionally
      COLLECTION_NAME — nothing else in the pipeline changes.
    - Path resolution is done once here using pathlib so all other modules
      receive absolute Path objects — no fragile string concatenation elsewhere.

Usage:
    from config import RAGConfig
    cfg = RAGConfig()
    print(cfg.CHROMA_DB_PATH)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load optional .env overrides (useful for deployment / CI environments)
# ---------------------------------------------------------------------------
# If a .env file is present in the project root, its values will override
# the defaults defined below. This keeps secrets and environment-specific
# paths out of source control without breaking offline-only setups.
load_dotenv()

# ---------------------------------------------------------------------------
# Resolve project root — everything is anchored to this single path
# ---------------------------------------------------------------------------
# __file__ is config.py itself, so .parent gives the project root directory.
# Using Path ensures cross-platform compatibility (Windows / Linux / macOS).
PROJECT_ROOT: Path = Path(__file__).resolve().parent


@dataclass
class RAGConfig:
    """
    Master configuration dataclass for the Offline RAG Framework.

    All attributes can be overridden at instantiation or via environment
    variables. Using a dataclass gives us:
        - IDE auto-complete support
        - Type hints enforced at the class level
        - A clean repr() for logging the active configuration
        - Easy extension — just add a new field

    Example (override at instantiation):
        cfg = RAGConfig(TOP_K=10, MAX_ROWS=5000)

    Example (override via environment):
        export RAG_TOP_K=10
        export RAG_MAX_ROWS=5000
    """

    # -----------------------------------------------------------------------
    # Paths — all resolved relative to PROJECT_ROOT
    # -----------------------------------------------------------------------

    DATA_DIR: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data"
    )
    """Root data directory. Contains sample_dataset/ and vector_db/."""

    DATASET_DIR: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "sample_dataset"
    )
    """
    Directory (or file path) containing the raw source data.

    DOMAIN SWITCH POINT #1:
        Agriculture CSV  → data/sample_dataset/crop_production.csv
        Military PDFs    → data/sample_dataset/military_manuals/
        Legal DOCX       → data/sample_dataset/legal_docs/
    Only this path needs updating when switching domains.
    """

    CHROMA_DB_PATH: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "vector_db"
    )
    """Persistent ChromaDB storage directory. Created automatically if absent."""

    LOGS_DIR: Path = field(
        default_factory=lambda: PROJECT_ROOT / "logs"
    )
    """Directory where rotating log files are written."""

    ASSETS_DIR: Path = field(
        default_factory=lambda: PROJECT_ROOT / "assets"
    )
    """Static assets (icons, images) used by the Streamlit UI."""

    # -----------------------------------------------------------------------
    # Dataset / Ingestion Settings
    # -----------------------------------------------------------------------

    DATASET_FORMAT: str = field(
        default_factory=lambda: os.getenv("RAG_DATASET_FORMAT", "csv")
    )
    """
    Format of the source data. Controls which loader is invoked in ingest.py.

    DOMAIN SWITCH POINT #2:
        Supported values: "csv" | "pdf" | "docx" | "txt" | "md" | "json"
    """

    DATASET_FILENAME: str = field(
        default_factory=lambda: os.getenv(
            "RAG_DATASET_FILENAME", "crop_production.csv"
        )
    )
    """
    Name of the primary dataset file inside DATASET_DIR.
    For directory-based loaders (pdf, docx), this field is ignored and
    all files inside DATASET_DIR are processed.
    """

    MAX_ROWS: int = field(
        default_factory=lambda: int(os.getenv("RAG_MAX_ROWS", "3000"))
    )
    """
    Hard cap on the number of rows / documents ingested.
    The ingestion pipeline performs stratified sampling BEFORE applying
    this cap, so diversity is always preserved regardless of the cap value.

    Embedding time benchmarks (all-MiniLM-L6-v2, CPU):
        1000 rows  →  ~45 seconds
        3000 rows  →  ~2–3 minutes
        5000 rows  →  ~4–5 minutes
    Keeping MAX_ROWS at 3000 gives rich, varied coverage well under the
    5-minute threshold, leaving headroom for other ingestion steps.
    """

    STRATIFY_COLUMNS: list = field(
        default_factory=lambda: ["State_Name", "Crop", "Season"]
    )
    """
    Columns used for stratified sampling during CSV ingestion.
    Ensures proportional representation across all categories.

    DOMAIN SWITCH POINT #3 (optional):
        For other domains, update these column names to match the new schema,
        or set to an empty list [] to disable stratification and use random
        sampling instead.
    """

    # -----------------------------------------------------------------------
    # Chunking Settings
    # -----------------------------------------------------------------------

    CHUNK_SIZE: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "512"))
    )
    """
    Maximum character length of each text chunk sent for embedding.
    For the agriculture CSV, each natural-language row is typically ~200–300
    characters, so chunking rarely fires. For PDF/DOCX domains with long
    paragraphs, this becomes critical.
    """

    CHUNK_OVERLAP: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    )
    """
    Number of characters shared between consecutive chunks.
    Overlap preserves context at chunk boundaries — important for documents
    where a sentence spans the boundary of two chunks.
    """

    # -----------------------------------------------------------------------
    # Embedding Model Settings
    # -----------------------------------------------------------------------

    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv(
            "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    """
    SentenceTransformer model name for generating embeddings.
    Downloaded once to the local HuggingFace cache; fully offline thereafter.

    all-MiniLM-L6-v2 is chosen because:
        - 22M parameters — fast on CPU
        - 384-dimensional embeddings — compact and accurate
        - Strong semantic similarity performance on short passages
        - MIT licensed
    """

    EMBEDDING_BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))
    )
    """
    Number of text chunks processed per embedding batch.
    Larger batches are faster but use more RAM.
    64 is a safe default for machines with 8GB+ RAM.
    """

    EMBEDDING_DEVICE: str = field(
        default_factory=lambda: os.getenv("RAG_EMBEDDING_DEVICE", "cpu")
    )
    """
    Device for running the embedding model.
    Options: "cpu" | "cuda" | "mps"
    Defaults to "cpu" for maximum compatibility in offline/university settings.
    If a CUDA GPU is available, set to "cuda" for 10x speed improvement.
    """

    # -----------------------------------------------------------------------
    # ChromaDB / Vector Store Settings
    # -----------------------------------------------------------------------

    COLLECTION_NAME: str = field(
        default_factory=lambda: os.getenv(
            "RAG_COLLECTION_NAME", "rag_knowledge_base"
        )
    )
    """
    ChromaDB collection name.
    Each domain can use a separate collection name to coexist in the same DB.

    DOMAIN SWITCH POINT #4 (optional):
        Agriculture  → "rag_knowledge_base"
        Military     → "military_docs"
        Healthcare   → "healthcare_records"
    """

    DISTANCE_METRIC: str = field(
        default_factory=lambda: os.getenv("RAG_DISTANCE_METRIC", "cosine")
    )
    """
    Distance metric used by ChromaDB for similarity search.
    Options: "cosine" | "l2" | "ip" (inner product)
    Cosine similarity is standard for semantic text search.
    """

    # -----------------------------------------------------------------------
    # Retrieval Settings
    # -----------------------------------------------------------------------

    TOP_K: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "5"))
    )
    """
    Number of top-matching documents retrieved per query.
    5 provides enough context for nuanced answers without exceeding
    the LLM's context window.
    """

    SIMILARITY_THRESHOLD: float = field(
        default_factory=lambda: float(
            os.getenv("RAG_SIMILARITY_THRESHOLD", "0.3")
        )
    )
    """
    Minimum cosine similarity score for a document to be included.
    Documents below this threshold are filtered out even if they are
    in the top-K results. This prevents the LLM from receiving
    completely irrelevant context.

    Cosine similarity range: 0.0 (unrelated) → 1.0 (identical)
    0.3 is a reasonable minimum for semantic relevance.
    """

    # -----------------------------------------------------------------------
    # Ollama / LLM Settings
    # -----------------------------------------------------------------------

    OLLAMA_MODEL: str = field(
        default_factory=lambda: os.getenv("RAG_OLLAMA_MODEL", "qwen2.5:3b")
    )
    """
    Name of the Ollama model to use for answer generation.
    Must be pulled locally before running: `ollama pull qwen2.5:3b`

    Alternatives (all offline-compatible):
        "llama3.2:3b"     — faster, slightly lower quality
        "mistral:7b"      — higher quality, needs more RAM
        "phi3:mini"       — very lightweight, good for low-RAM machines
    """

    OLLAMA_HOST: str = field(
        default_factory=lambda: os.getenv("RAG_OLLAMA_HOST", "http://localhost:11434")
    )
    """
    URL of the locally running Ollama server.
    Default port is 11434. Change only if you've configured Ollama differently.
    """

    OLLAMA_TIMEOUT: int = field(
        default_factory=lambda: int(os.getenv("RAG_OLLAMA_TIMEOUT", "120"))
    )
    """
    Seconds to wait for Ollama to respond before raising a timeout error.
    120 seconds is sufficient for qwen2.5:3b on most machines.
    Increase this for larger models (7b+) on slower hardware.
    """

    OLLAMA_TEMPERATURE: float = field(
        default_factory=lambda: float(
            os.getenv("RAG_OLLAMA_TEMPERATURE", "0.1")
        )
    )
    """
    Sampling temperature for the LLM.
    Range: 0.0 (deterministic) → 1.0 (creative/random)

    0.1 is intentionally low because:
        - This is a factual RAG system, not a creative chatbot
        - Low temperature reduces hallucination risk
        - Responses are grounded strictly in retrieved context
    """

    OLLAMA_NUM_CTX: int = field(
        default_factory=lambda: int(os.getenv("RAG_OLLAMA_NUM_CTX", "4096"))
    )
    """
    Context window size (tokens) passed to Ollama.
    4096 comfortably fits 5 retrieved passages + question + instructions.
    """

    # -----------------------------------------------------------------------
    # Application / UI Settings
    # -----------------------------------------------------------------------

    APP_TITLE: str = "Offline RAG Framework"
    """Title displayed in the Streamlit browser tab and header."""

    APP_DESCRIPTION: str = (
        "A domain-agnostic, fully offline Retrieval-Augmented Generation "
        "system. Ask questions and receive answers grounded exclusively "
        "in the locally indexed knowledge base."
    )
    """Subtitle description shown in the Streamlit UI."""

    KNOWLEDGE_BASE_LABEL: str = "Knowledge Base"
    """Generic label used in the UI — no domain-specific terminology."""

    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("RAG_LOG_LEVEL", "INFO")
    )
    """
    Logging verbosity level.
    Options: "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
    Use "DEBUG" during development to see every retrieval step.
    Use "INFO" for production / demo runs.
    """

    # -----------------------------------------------------------------------
    # Post-init: ensure all required directories exist
    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Automatically create required directories after instantiation.
        This means no other module needs to create directories manually.
        """
        dirs_to_create = [
            self.DATA_DIR,
            self.DATASET_DIR,
            self.CHROMA_DB_PATH,
            self.LOGS_DIR,
            self.ASSETS_DIR,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_file_path(self) -> Path:
        """
        Convenience property returning the full path to the primary dataset file.
        Combines DATASET_DIR and DATASET_FILENAME.
        """
        return self.DATASET_DIR / self.DATASET_FILENAME

    @property
    def chroma_db_path_str(self) -> str:
        """
        ChromaDB's Python client requires a string path, not a Path object.
        This property provides a safe cross-platform string conversion.
        """
        return str(self.CHROMA_DB_PATH)

    def summary(self) -> str:
        """
        Returns a human-readable summary of the active configuration.
        Useful for logging at application startup so the full config
        is always visible in the log file for reproducibility.
        """
        lines = [
            "=" * 60,
            "  Offline RAG Framework — Active Configuration",
            "=" * 60,
            f"  Dataset Path     : {self.dataset_file_path}",
            f"  Dataset Format   : {self.DATASET_FORMAT.upper()}",
            f"  Max Rows         : {self.MAX_ROWS:,}",
            f"  Embedding Model  : {self.EMBEDDING_MODEL}",
            f"  Embedding Device : {self.EMBEDDING_DEVICE.upper()}",
            f"  Batch Size       : {self.EMBEDDING_BATCH_SIZE}",
            f"  ChromaDB Path    : {self.CHROMA_DB_PATH}",
            f"  Collection Name  : {self.COLLECTION_NAME}",
            f"  Top-K Retrieval  : {self.TOP_K}",
            f"  Sim. Threshold   : {self.SIMILARITY_THRESHOLD}",
            f"  Ollama Model     : {self.OLLAMA_MODEL}",
            f"  Ollama Host      : {self.OLLAMA_HOST}",
            f"  Temperature      : {self.OLLAMA_TEMPERATURE}",
            f"  Context Window   : {self.OLLAMA_NUM_CTX} tokens",
            f"  Log Level        : {self.LOG_LEVEL}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton — import this directly in all other modules
# ---------------------------------------------------------------------------
# Using a module-level singleton avoids re-instantiation across imports and
# ensures all modules share the exact same configuration object.
#
# Usage in any other module:
#   from config import config
#   print(config.TOP_K)
#
config = RAGConfig()
