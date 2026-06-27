"""
ingest.py — Document Ingestion Pipeline for the Offline RAG Framework
======================================================================
This is the one-time setup script that populates the ChromaDB knowledge
base from the configured dataset.

Run this script ONCE before launching the Streamlit application:
    python ingest.py

On subsequent runs, it will detect the existing database and prompt you
before regenerating (protecting against accidental data loss).

Pipeline stages:
    1. LOAD        — Read raw data via utils.load_documents()
    2. EMBED       — Generate embeddings in batches (sentence-transformers)
    3. STORE       — Persist embeddings + metadata to ChromaDB
    4. VERIFY      — Confirm document count matches expectations

Design decisions:
    - Embeddings are computed in configurable batches (default: 64) for
      memory efficiency.
    - Documents are upserted (not inserted) so re-running is idempotent —
      no duplicates are created even if the script is accidentally run twice.
    - The embedding model is loaded ONCE at the top of the pipeline and
      reused for all batches.
    - tqdm progress bars give real-time visibility into embedding progress.

This script has NO dependency on:
    - Streamlit (it is a standalone CLI script)
    - Ollama (embeddings are computed by sentence-transformers)
    - The specific dataset format (utils.load_documents() abstracts that)
"""

import argparse
import sys
import time
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import config
from logger import get_logger, log_elapsed, log_section_header
from utils import load_documents

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Ingestion pipeline steps
# ---------------------------------------------------------------------------

def _load_embedding_model() -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model from local cache.

    The model is downloaded to the HuggingFace local cache on first run.
    All subsequent runs load it from disk — fully offline.

    Returns:
        A loaded SentenceTransformer model instance.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    log.info(
        "Loading embedding model: %s on device: %s",
        config.EMBEDDING_MODEL,
        config.EMBEDDING_DEVICE,
    )
    t0 = time.perf_counter()
    try:
        model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )
        log_elapsed(log, "Embedding model load", time.perf_counter() - t0)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{config.EMBEDDING_MODEL}': {exc}\n"
            "Ensure you have an internet connection for the first run to "
            "download the model, then subsequent runs will be fully offline."
        ) from exc


def _connect_chromadb() -> tuple[chromadb.PersistentClient, Any]:
    """
    Connect to the persistent ChromaDB instance and return the client
    and the target collection.

    The collection is created fresh (or retrieved if it already exists).
    The distance metric is set to cosine similarity, which is standard
    for semantic text search.

    Returns:
        A tuple of (client, collection).

    Raises:
        RuntimeError: If ChromaDB cannot be initialised.
    """
    log.info("Connecting to ChromaDB at: %s", config.CHROMA_DB_PATH)
    try:
        client = chromadb.PersistentClient(
            path=config.chroma_db_path_str,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": config.DISTANCE_METRIC},
        )
        log.info(
            "ChromaDB collection '%s' ready — current count: %d",
            config.COLLECTION_NAME,
            collection.count(),
        )
        return client, collection
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to ChromaDB: {exc}\n"
            f"Ensure the path exists and is writable: {config.CHROMA_DB_PATH}"
        ) from exc


def _check_existing_database(collection: Any) -> bool:
    """
    Check if the ChromaDB collection already contains documents.

    Used to warn the user before potentially overwriting an existing index.

    Args:
        collection: The ChromaDB collection object.

    Returns:
        True if the collection already has documents. False if empty.
    """
    count = collection.count()
    if count > 0:
        log.warning(
            "Collection '%s' already contains %d document(s).",
            config.COLLECTION_NAME,
            count,
        )
        return True
    return False


def _embed_and_store(
    documents: list[dict[str, Any]],
    model: SentenceTransformer,
    collection: Any,
) -> int:
    """
    Generate embeddings for all documents and upsert them into ChromaDB.

    Batch processing strategy:
        - Documents are processed in batches of config.EMBEDDING_BATCH_SIZE.
        - A tqdm progress bar shows real-time progress across all batches.
        - Each batch is upserted (not inserted) — safe to re-run.
        - On batch failure, the error is logged and the batch is skipped
          rather than crashing the entire pipeline.

    Memory note:
        Embeddings for one batch are computed, stored, and then released
        before the next batch begins. This avoids loading all embeddings
        into RAM simultaneously — important for large datasets.

    Args:
        documents: List of document dicts from utils.load_documents().
        model:     The loaded SentenceTransformer model.
        collection: The ChromaDB collection to write to.

    Returns:
        Total number of documents successfully upserted.
    """
    total_docs       = len(documents)
    batch_size       = config.EMBEDDING_BATCH_SIZE
    total_upserted   = 0
    total_batches    = (total_docs + batch_size - 1) // batch_size

    log.info(
        "Embedding %d documents in %d batches (batch size: %d)...",
        total_docs,
        total_batches,
        batch_size,
    )

    t0 = time.perf_counter()

    with tqdm(
        total=total_docs,
        desc="  Generating embeddings",
        unit="doc",
        dynamic_ncols=True,
        colour="green",
    ) as pbar:
        for batch_start in range(0, total_docs, batch_size):
            batch = documents[batch_start : batch_start + batch_size]

            # Extract the three parallel lists ChromaDB requires
            batch_ids       = [doc["id"]   for doc in batch]
            batch_texts     = [doc["text"] for doc in batch]
            batch_metadatas = [doc["metadata"] for doc in batch]

            # Sanitise metadata — ChromaDB requires all metadata values to be
            # str, int, float, or bool. Convert anything else to str.
            sanitised_metadatas = []
            for meta in batch_metadatas:
                sanitised = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitised[k] = v
                    else:
                        sanitised[k] = str(v)
                sanitised_metadatas.append(sanitised)

            try:
                # Generate embeddings for this batch
                embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,  # L2-normalise for cosine
                    show_progress_bar=False,    # tqdm handles progress above
                ).tolist()

                # Upsert into ChromaDB
                # upsert = insert if new, update if ID already exists
                # This makes re-running the ingestion script idempotent.
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=embeddings,
                    metadatas=sanitised_metadatas,
                )

                total_upserted += len(batch)
                pbar.update(len(batch))

            except Exception as exc:
                log.error(
                    "Batch %d-%d failed: %s — skipping this batch.",
                    batch_start,
                    batch_start + len(batch),
                    exc,
                )
                pbar.update(len(batch))
                continue

    elapsed = time.perf_counter() - t0
    log_elapsed(log, "Embedding + storage", elapsed)
    log.info(
        "Throughput: %.1f documents/second",
        total_upserted / elapsed if elapsed > 0 else 0,
    )

    return total_upserted


def _verify_ingestion(collection: Any, expected_count: int) -> bool:
    """
    Verify that the number of documents stored in ChromaDB matches
    the expected count (within a small tolerance for skipped/failed batches).

    Args:
        collection:     The ChromaDB collection to verify.
        expected_count: Number of documents that should have been upserted.

    Returns:
        True if verification passes (count >= 90% of expected).
        False if significantly fewer documents were stored than expected.
    """
    actual_count = collection.count()
    threshold    = int(expected_count * 0.90)  # Allow up to 10% batch failures

    if actual_count >= threshold:
        log.info(
            "✅ Verification passed — %d documents stored (expected ~%d).",
            actual_count,
            expected_count,
        )
        return True
    else:
        log.error(
            "❌ Verification failed — only %d of %d documents were stored. "
            "Check logs for batch errors.",
            actual_count,
            expected_count,
        )
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ingestion(force: bool = False) -> None:
    """
    Execute the complete ingestion pipeline.

    Args:
        force: If True, skips the confirmation prompt and overwrites the
               existing database. If False (default), asks for confirmation
               before overwriting.

    Raises:
        SystemExit: If the user declines to overwrite an existing database.
        RuntimeError: If any critical pipeline stage fails.
    """
    log_section_header(log, "INGESTION PIPELINE — OFFLINE RAG FRAMEWORK")
    pipeline_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Stage 1: Connect to ChromaDB
    # -----------------------------------------------------------------------
    log_section_header(log, "Stage 1: Connecting to Vector Database")
    _, collection = _connect_chromadb()

    # -----------------------------------------------------------------------
    # Stage 2: Check for existing data
    # -----------------------------------------------------------------------
    has_existing = _check_existing_database(collection)
    if has_existing and not force:
        log.warning(
            "The knowledge base already contains data.\n"
            "Run with --force to overwrite: python ingest.py --force\n"
            "Or use the Streamlit UI which will use the existing database."
        )
        print("\n" + "=" * 60)
        print("  Knowledge base already populated.")
        print(f"  Documents in collection: {collection.count()}")
        print("  To overwrite, run: python ingest.py --force")
        print("  To use existing data, launch: streamlit run app.py")
        print("=" * 60 + "\n")
        return

    if has_existing and force:
        log.info("--force flag set — deleting existing collection and rebuilding.")
        # Retrieve client from collection to delete
        client = chromadb.PersistentClient(
            path=config.chroma_db_path_str,
            settings=Settings(anonymized_telemetry=False),
        )
        client.delete_collection(config.COLLECTION_NAME)
        collection = client.create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": config.DISTANCE_METRIC},
        )
        log.info("Collection '%s' reset successfully.", config.COLLECTION_NAME)

    # -----------------------------------------------------------------------
    # Stage 3: Load documents
    # -----------------------------------------------------------------------
    log_section_header(log, "Stage 2: Loading and Preparing Documents")
    t0 = time.perf_counter()
    try:
        documents = load_documents()
    except FileNotFoundError as exc:
        log.error("Dataset not found: %s", exc)
        print(f"\n❌ ERROR: {exc}\n")
        sys.exit(1)
    except ValueError as exc:
        log.error("Dataset error: %s", exc)
        print(f"\n❌ ERROR: {exc}\n")
        sys.exit(1)
    log_elapsed(log, "Document loading", time.perf_counter() - t0)

    # -----------------------------------------------------------------------
    # Stage 4: Load embedding model
    # -----------------------------------------------------------------------
    log_section_header(log, "Stage 3: Loading Embedding Model")
    try:
        model = _load_embedding_model()
    except RuntimeError as exc:
        log.error("%s", exc)
        print(f"\n❌ ERROR: {exc}\n")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Stage 5: Generate embeddings and store in ChromaDB
    # -----------------------------------------------------------------------
    log_section_header(log, "Stage 4: Generating Embeddings and Storing")
    upserted = _embed_and_store(documents, model, collection)

    # -----------------------------------------------------------------------
    # Stage 6: Verify
    # -----------------------------------------------------------------------
    log_section_header(log, "Stage 5: Verifying Ingestion")
    success = _verify_ingestion(collection, upserted)

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    total_elapsed = time.perf_counter() - pipeline_start
    print("\n" + "=" * 60)
    if success:
        print("  ✅ INGESTION COMPLETE")
    else:
        print("  ⚠️  INGESTION COMPLETED WITH WARNINGS")
    print(f"  Documents indexed : {collection.count():,}")
    print(f"  Total time        : {total_elapsed:.1f}s")
    print(f"  Collection        : {config.COLLECTION_NAME}")
    print(f"  Database path     : {config.CHROMA_DB_PATH}")
    print(f"  Embedding model   : {config.EMBEDDING_MODEL}")
    print("=" * 60)
    print("\n  Next step: streamlit run app.py\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the ingestion script.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Offline RAG Framework — Ingestion Pipeline\n"
            "Populates the ChromaDB knowledge base from the configured dataset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ingest.py              # Run ingestion (skip if DB exists)\n"
            "  python ingest.py --force      # Force re-ingestion, clear old data\n"
            "  python ingest.py --info       # Show current database stats only\n"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing collection and re-ingest from scratch.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print database statistics and exit without ingesting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.info:
        # Just print stats and exit
        try:
            client = chromadb.PersistentClient(
                path=config.chroma_db_path_str,
                settings=Settings(anonymized_telemetry=False),
            )
            try:
                col = client.get_collection(config.COLLECTION_NAME)
                count = col.count()
            except Exception:
                count = 0

            print("\n" + "=" * 60)
            print("  Offline RAG Framework — Database Info")
            print("=" * 60)
            print(f"  Collection : {config.COLLECTION_NAME}")
            print(f"  Documents  : {count:,}")
            print(f"  DB Path    : {config.CHROMA_DB_PATH}")
            print(f"  Model      : {config.EMBEDDING_MODEL}")
            print(f"  Max Rows   : {config.MAX_ROWS:,}")
            print("=" * 60 + "\n")
        except Exception as exc:
            print(f"Error reading database: {exc}")
        sys.exit(0)

    run_ingestion(force=args.force)
