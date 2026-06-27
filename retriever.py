"""
retriever.py — Vector Database Interface for the Offline RAG Framework
=======================================================================
This module manages all interactions with the ChromaDB persistent vector
database. It is the bridge between the user's question and the indexed
knowledge base.

Responsibilities:
    - Load (or create) the ChromaDB collection.
    - Load the embedding model once and reuse it across queries.
    - Embed the user's question and search ChromaDB for the most
      semantically similar documents.
    - Filter results by the configured similarity threshold.
    - Return structured result dicts for use by llm.py and app.py.

This module has NO knowledge of:
    - Ollama or the LLM (llm.py handles that)
    - Streamlit or the UI (app.py handles that)
    - The source dataset format (utils.py handles that)
    - How the knowledge base was populated (ingest.py handles that)

Public API (the only two functions ingest.py / app.py need):
    load_database()  → RAGRetriever
    retriever.retrieve(question, top_k) → list[dict]

Usage:
    from retriever import load_database
    retriever = load_database()
    results = retriever.retrieve("What is the rice yield in Punjab?")
"""

import time
from dataclasses import dataclass, field
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import config
from logger import get_logger, log_elapsed, log_retrieval_result
from utils import cosine_similarity_from_distance

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Database status dataclass
# ---------------------------------------------------------------------------

@dataclass
class DatabaseInfo:
    """
    Snapshot of the current state of the ChromaDB collection.

    Returned by RAGRetriever.get_info() and displayed in the Streamlit
    sidebar to give users confidence that the knowledge base is loaded.

    Attributes:
        collection_name:  Name of the ChromaDB collection.
        document_count:   Number of documents (embeddings) stored.
        embedding_model:  Name of the SentenceTransformer model used.
        db_path:          Filesystem path to the ChromaDB storage directory.
        is_empty:         True if the collection contains zero documents.
    """
    collection_name: str
    document_count:  int
    embedding_model: str
    db_path:         str
    is_empty:        bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_empty = self.document_count == 0


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    Manages the ChromaDB collection and semantic search for the RAG pipeline.

    Lifecycle:
        1. Instantiated via load_database() — which handles all setup.
        2. Cached in Streamlit's session state (one instance per session).
        3. retrieve() is called for every user query.

    The embedding model is loaded ONCE at instantiation and reused for
    all subsequent queries — avoiding the costly model loading overhead
    on every call.
    """

    def __init__(self) -> None:
        """
        Initialise the retriever.

        Loads the SentenceTransformer embedding model and connects to the
        persistent ChromaDB collection. Does NOT embed any documents —
        that is done exclusively by ingest.py.

        Raises:
            RuntimeError: If ChromaDB cannot be initialised or the
                          collection does not exist.
        """
        log.info("Initialising RAGRetriever...")

        # Load embedding model
        log.info(
            "Loading embedding model: %s (device: %s)",
            config.EMBEDDING_MODEL,
            config.EMBEDDING_DEVICE,
        )
        t0 = time.perf_counter()
        self._embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )
        log_elapsed(log, "Embedding model load", time.perf_counter() - t0)

        # Connect to ChromaDB
        log.info("Connecting to ChromaDB at: %s", config.CHROMA_DB_PATH)
        self._client = chromadb.PersistentClient(
            path=config.chroma_db_path_str,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create the collection
        # Note: get_or_create_collection is used here so that the retriever
        # can connect even before ingestion has run — it will simply report
        # an empty collection rather than crashing.
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": config.DISTANCE_METRIC},
        )

        doc_count = self._collection.count()
        log.info(
            "Connected to collection '%s' — %d document(s) indexed.",
            config.COLLECTION_NAME,
            doc_count,
        )

        if doc_count == 0:
            log.warning(
                "Collection '%s' is empty. "
                "Run `python ingest.py` to populate the knowledge base.",
                config.COLLECTION_NAME,
            )

    def get_info(self) -> DatabaseInfo:
        """
        Return metadata about the current ChromaDB collection.

        Used by app.py to display knowledge base statistics in the sidebar.

        Returns:
            A DatabaseInfo dataclass with collection name, document count,
            model name, and database path.
        """
        return DatabaseInfo(
            collection_name=config.COLLECTION_NAME,
            document_count=self._collection.count(),
            embedding_model=config.EMBEDDING_MODEL,
            db_path=config.chroma_db_path_str,
        )

    def embed_query(self, question: str) -> list[float]:
        """
        Generate a dense vector embedding for the user's query.

        Uses the same model that was used during ingestion to ensure
        the query embedding lives in the same vector space as the
        stored document embeddings (essential for meaningful similarity).

        Args:
            question: The raw user question string.

        Returns:
            A list of floats representing the 384-dimensional embedding
            (for all-MiniLM-L6-v2).
        """
        return self._embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise for cosine similarity
        ).tolist()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically relevant documents for a given query.

        Pipeline:
            1. Embed the question using the same model used during ingestion.
            2. Query ChromaDB for the top-(top_k * 2) candidates.
            3. Convert raw distance scores to cosine similarity scores.
            4. Filter out documents below config.SIMILARITY_THRESHOLD.
            5. Return the top-k filtered results as structured dicts.

        Why retrieve top_k * 2 first?
            Because some candidates may be filtered by the similarity
            threshold. Fetching 2x ensures we have enough candidates to
            meet the top_k target after filtering without a second DB call.

        Args:
            question: The user's raw question string.
            top_k:    Number of results to return. Defaults to config.TOP_K.

        Returns:
            A list of result dicts, sorted by similarity (highest first):
                [
                    {
                        "text":       str,   # the document text
                        "metadata":   dict,  # source/domain metadata
                        "similarity": float, # cosine similarity [0.0, 1.0]
                        "distance":   float, # raw ChromaDB distance
                        "id":         str,   # ChromaDB document ID
                    },
                    ...
                ]
            Returns an empty list if no documents meet the threshold or
            if the collection is empty.

        Raises:
            ValueError: If the question string is empty.
            RuntimeError: If ChromaDB query fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k = top_k if top_k is not None else config.TOP_K

        # Check if collection has any documents
        if self._collection.count() == 0:
            log.warning("Retrieval attempted on empty collection.")
            return []

        t0 = time.perf_counter()

        # Generate query embedding
        query_embedding = self.embed_query(question.strip())

        # Query ChromaDB — fetch 2x candidates for threshold filtering
        fetch_k = min(k * 2, self._collection.count())
        raw_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        elapsed = time.perf_counter() - t0

        # Unpack ChromaDB response structure
        # ChromaDB wraps results in an extra list layer (one per query embedding)
        docs      = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        ids       = raw_results.get("ids", [[]])[0]

        # Build structured result list with similarity scores
        results: list[dict[str, Any]] = []

        for doc_text, metadata, distance, doc_id in zip(
            docs, metadatas, distances, ids
        ):
            similarity = cosine_similarity_from_distance(distance)

            # Filter: skip documents below similarity threshold
            if similarity < config.SIMILARITY_THRESHOLD:
                log.debug(
                    "Filtered doc '%s' — similarity %.3f below threshold %.3f",
                    doc_id,
                    similarity,
                    config.SIMILARITY_THRESHOLD,
                )
                continue

            results.append({
                "id":         doc_id,
                "text":       doc_text,
                "metadata":   metadata,
                "similarity": round(similarity, 4),
                "distance":   round(distance, 4),
            })

        # Sort by similarity descending (ChromaDB already returns sorted
        # by distance, but re-sorting after filtering is safe and explicit)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Hard cap at top_k
        results = results[:k]

        log_retrieval_result(log, question, len(results), elapsed)
        log_elapsed(log, "Retrieval", elapsed)

        if not results:
            log.warning(
                "No documents above similarity threshold (%.2f) for query: %s",
                config.SIMILARITY_THRESHOLD,
                question[:80],
            )

        return results

    def collection_exists_and_populated(self) -> bool:
        """
        Quick check: does the collection have at least one document?

        Used by ingest.py to determine whether ingestion should be skipped
        (when re-running the script on an already-populated database).

        Returns:
            True if the collection has at least one document.
            False if it is empty or does not exist.
        """
        try:
            return self._collection.count() > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Module-level factory function — the public entry point
# ---------------------------------------------------------------------------

def load_database() -> RAGRetriever:
    """
    Factory function that creates and returns a fully initialised RAGRetriever.

    This is the ONLY function that ingest.py and app.py should import
    from this module. It:
        1. Creates the RAGRetriever (loads model + connects to ChromaDB).
        2. Logs the database info summary.
        3. Returns the ready-to-use retriever instance.

    Returns:
        A fully initialised RAGRetriever instance.

    Raises:
        RuntimeError: If ChromaDB cannot be reached or the embedding
                      model cannot be loaded.

    Usage:
        from retriever import load_database
        retriever = load_database()
        results = retriever.retrieve("What crops grow in Punjab?")
    """
    log.info("Loading RAG knowledge base...")
    retriever = RAGRetriever()

    info = retriever.get_info()
    log.info(
        "Knowledge base ready — Collection: '%s' | Documents: %d | Model: %s",
        info.collection_name,
        info.document_count,
        info.embedding_model,
    )

    return retriever
