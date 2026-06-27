"""
utils.py — Data Loading, Cleaning, and Text Conversion Utilities
================================================================
This module is the ONLY place in the framework that contains domain-aware
logic for the agriculture demonstration dataset.

Architecture contract:
    - Every function in this file takes raw data as input and returns
      clean, plain-text Python strings as output.
    - Once text leaves this module, every downstream component
      (ingest.py, retriever.py, llm.py, app.py) is 100% domain-agnostic.
    - Adding a new domain means adding a new loader function here —
      NOTHING else in the pipeline changes.

Supported loaders (current):
    ┌─────────────┬────────────────────────────────────────────────────┐
    │ Format      │ Function                                           │
    ├─────────────┼────────────────────────────────────────────────────┤
    │ CSV         │ load_csv_documents()                               │
    │ PDF         │ load_pdf_documents()          [future-ready]       │
    │ DOCX        │ load_docx_documents()         [future-ready]       │
    │ TXT         │ load_txt_documents()          [future-ready]       │
    │ Markdown    │ load_markdown_documents()     [future-ready]       │
    │ JSON        │ load_json_documents()         [future-ready]       │
    └─────────────┴────────────────────────────────────────────────────┘

The unified entry point `load_documents()` inspects `config.DATASET_FORMAT`
and dispatches to the correct loader automatically. ingest.py calls only
this single function — it never needs to know the format.
"""

import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

from config import config
from logger import get_logger

log = get_logger(__name__)


# ===========================================================================
# Section 1 — Text Cleaning Utilities
# ===========================================================================

def clean_text(text: str) -> str:
    """
    Normalise and clean a raw text string for embedding.

    Steps applied (in order):
        1. Unicode normalisation (NFKC) — converts ligatures, fullwidth
           characters, and compatibility forms to standard equivalents.
        2. Strip leading/trailing whitespace.
        3. Collapse internal whitespace — multiple spaces/tabs/newlines
           become a single space.
        4. Remove non-printable control characters (except newline).
        5. Strip residual HTML-like tags (safety net for scraped data).

    This function is intentionally conservative — it does NOT:
        - Remove punctuation (important for semantic meaning)
        - Convert to lowercase (embedding models handle case internally)
        - Perform stemming or lemmatisation (unnecessary for dense retrieval)

    Args:
        text: Raw input string from any loader.

    Returns:
        Cleaned string ready for embedding. Returns empty string if input
        is empty or contains only whitespace after cleaning.

    Example:
        >>> clean_text("  Rice  \n\n  production   was  good.  ")
        "Rice production was good."
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Unicode normalisation
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Strip HTML-like tags (safety net — not needed for CSV but
    # critical for scraped or markdown content)
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 3: Remove non-printable control characters (keep \n for now)
    text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)

    # Step 4: Collapse all whitespace (spaces, tabs, newlines) to single space
    text = re.sub(r"\s+", " ", text)

    # Step 5: Final strip
    text = text.strip()

    return text


def is_valid_document(text: str, min_length: int = 20) -> bool:
    """
    Check whether a text string is worth embedding.

    Filters out:
        - Empty strings
        - Strings shorter than min_length characters (likely garbage rows)
        - Strings that are purely numeric (no semantic value)
        - Strings composed only of punctuation/symbols

    Args:
        text:       The cleaned document text to validate.
        min_length: Minimum character count to consider a document valid.
                    Default is 20 — anything shorter is almost certainly
                    an incomplete or corrupted record.

    Returns:
        True if the document should be included in the knowledge base.
        False if it should be discarded.
    """
    if not text or len(text) < min_length:
        return False
    # Check that the string contains at least some alphabetic characters
    if not any(c.isalpha() for c in text):
        return False
    return True


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split a long text string into overlapping chunks of at most `chunk_size`
    characters.

    This is primarily needed for PDF/DOCX domains where a single document
    can be thousands of characters long. For the agriculture CSV demo,
    each natural-language row is ~200–300 characters and will almost never
    be chunked — but the function is called on every document for consistency.

    Chunking strategy:
        - Splits on word boundaries (never mid-word) to preserve readability.
        - Each chunk overlaps the previous by `overlap` characters to avoid
          losing context at chunk boundaries.
        - If the entire text fits within chunk_size, it is returned as-is
          in a single-element list.

    Args:
        text:       The cleaned document text to split.
        chunk_size: Maximum character length of each chunk (from config).
        overlap:    Character overlap between consecutive chunks (from config).

    Returns:
        A list of non-empty string chunks. Always contains at least one
        element if `text` is non-empty.

    Example:
        chunk_text("A B C D E", chunk_size=4, overlap=1)
        → ["A B C", "C D E", "E"]
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start: int = 0

    while start < len(text):
        end: int = start + chunk_size

        if end >= len(text):
            # Last chunk — take everything remaining
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Walk backwards from `end` to find the nearest word boundary
        # (i.e. the last space before position `end`)
        boundary = text.rfind(" ", start, end)
        if boundary == -1:
            # No space found — hard split at chunk_size (rare edge case)
            boundary = end

        chunk = text[start:boundary].strip()
        if chunk:
            chunks.append(chunk)

        # Advance start by (chunk_size - overlap) to create the overlap window
        start = boundary - overlap if boundary - overlap > start else boundary

    return chunks


# ===========================================================================
# Section 2 — Agriculture CSV Loader (Domain-Specific, Demo Dataset)
# ===========================================================================

def _row_to_natural_language(row: pd.Series) -> str:
    """
    Convert a single row of the Indian Agriculture Crop Production CSV
    into a rich, descriptive natural-language sentence.

    This is the ONLY agriculture-specific function in the entire framework.
    For a new domain, you would write an equivalent converter for that
    domain's data structure.

    Expected CSV columns (Kaggle dataset):
        State_Name, District_Name, Crop_Year, Season, Crop,
        Area, Production

    Derived field:
        Yield = Production / Area  (quintals per hectare)

    The output sentence is designed to be semantically rich so that
    the embedding model can capture relationships between:
        - Crop types and seasons
        - Geographic regions and production volumes
        - Year-over-year trends (when multiple rows exist for same crop/region)

    Args:
        row: A pandas Series representing one row of the agriculture CSV.

    Returns:
        A natural-language string describing this agricultural record.
        Returns empty string if critical fields are missing.

    Example output:
        "In 2011, Wheat was cultivated in Amritsar district of Punjab
         during the Rabi season. The area under cultivation was 45,200.0
         hectares. Total production was 1,24,300.0 quintals. The yield
         was approximately 2.75 quintals per hectare."
    """
    try:
        # Extract and sanitise each field with safe fallbacks
        state    = str(row.get("State_Name", "")).strip().title()
        district = str(row.get("District_Name", "")).strip().title()
        year     = str(row.get("Crop_Year", "")).strip()
        season   = str(row.get("Season", "")).strip().strip()
        crop     = str(row.get("Crop", "")).strip().title()

        # Area and production may be NaN — coerce safely
        area_raw  = row.get("Area", None)
        prod_raw  = row.get("Production", None)

        area       = float(area_raw)  if pd.notna(area_raw)  else None
        production = float(prod_raw)  if pd.notna(prod_raw)  else None

        # Require at minimum: state, crop, year — discard rows missing these
        if not state or not crop or not year:
            return ""

        # Build the natural language sentence piece by piece
        parts: list[str] = []

        # Core geographic + temporal context
        if district:
            parts.append(
                f"In {year}, {crop} was cultivated in {district} district "
                f"of {state}"
            )
        else:
            parts.append(f"In {year}, {crop} was cultivated in {state}")

        # Season context
        if season:
            parts[0] += f" during the {season} season."
        else:
            parts[0] += "."

        # Area under cultivation
        if area is not None:
            parts.append(
                f"The area under cultivation was {area:,.1f} hectares."
            )

        # Production volume
        if production is not None:
            parts.append(
                f"Total production was {production:,.1f} quintals."
            )

        # Yield (derived) — only compute if both area and production are valid
        if area is not None and production is not None and area > 0:
            yield_val = production / area
            parts.append(
                f"The yield was approximately {yield_val:.2f} quintals "
                f"per hectare."
            )

        return " ".join(parts)

    except (ValueError, TypeError, KeyError) as exc:
        log.debug("Skipping row due to conversion error: %s", exc)
        return ""


def _stratified_sample(
    df: pd.DataFrame,
    stratify_cols: list[str],
    max_rows: int,
) -> pd.DataFrame:
    """
    Select a maximally diverse subset of `max_rows` rows from `df` using
    stratified sampling across the specified columns.

    Algorithm:
        1. Create a composite stratum key by concatenating the stratify_cols.
        2. Count how many rows each stratum contributes.
        3. Allocate each stratum a proportional share of max_rows,
           with a minimum of 1 row per stratum.
        4. Sample from each stratum (with replacement if the stratum is
           smaller than its allocation).
        5. Concatenate and shuffle the final sample.

    Why custom stratification instead of sklearn's StratifiedShuffleSplit?
        sklearn's implementation requires every stratum to appear at least
        twice, which fails on rare State+Crop+Season combinations that have
        only a single row. Our implementation handles singletons gracefully.

    Args:
        df:             Full DataFrame loaded from the CSV.
        stratify_cols:  Column names to stratify across (from config).
        max_rows:       Target sample size (hard cap).

    Returns:
        A shuffled DataFrame of at most max_rows rows with diverse coverage
        across all strata.
    """
    # Filter to only columns that actually exist in this CSV
    valid_cols = [c for c in stratify_cols if c in df.columns]

    if not valid_cols:
        log.warning(
            "None of the configured STRATIFY_COLUMNS %s found in CSV. "
            "Falling back to random sample.",
            stratify_cols,
        )
        return df.sample(n=min(max_rows, len(df)), random_state=42)

    log.info("Stratifying across columns: %s", valid_cols)

    # Build composite stratum key
    df = df.copy()
    df["_stratum"] = df[valid_cols].astype(str).agg(" | ".join, axis=1)

    stratum_counts = df["_stratum"].value_counts()
    num_strata     = len(stratum_counts)
    log.info("Unique strata found: %d", num_strata)

    sampled_frames: list[pd.DataFrame] = []
    total_allocated: int = 0

    for stratum, count in stratum_counts.items():
        # Proportional allocation: give each stratum a fair share
        proportion    = count / len(df)
        allocation    = max(1, round(proportion * max_rows))
        # Cap allocation to avoid over-sampling a single stratum
        allocation    = min(allocation, max_rows // max(1, num_strata // 2))

        stratum_df    = df[df["_stratum"] == stratum]
        replace_flag  = len(stratum_df) < allocation

        sampled = stratum_df.sample(
            n=min(allocation, len(stratum_df)) if not replace_flag else allocation,
            replace=replace_flag,
            random_state=42,
        )
        sampled_frames.append(sampled)
        total_allocated += len(sampled)

        if total_allocated >= max_rows:
            break

    result = (
        pd.concat(sampled_frames, ignore_index=True)
        .drop(columns=["_stratum"])
        .sample(frac=1, random_state=42)   # shuffle the final result
        .reset_index(drop=True)
    )

    # Hard cap — should be very close to max_rows but protect against
    # small over-allocation from rounding
    result = result.iloc[:max_rows]
    log.info(
        "Stratified sample: %d rows selected from %d total (%.1f%% coverage)",
        len(result),
        len(df),
        100.0 * len(result) / len(df),
    )
    return result


def load_csv_documents() -> list[dict[str, Any]]:
    """
    Load the agriculture CSV, apply stratified sampling, clean each row,
    and convert it to a natural-language document dictionary.

    Returns a list of document dicts, each with:
        {
            "text":     str,   # clean natural-language paragraph
            "metadata": dict,  # structured metadata for ChromaDB
            "id":       str,   # unique document identifier
        }

    The `metadata` dict is stored alongside the embedding in ChromaDB and
    is returned with every retrieval result. It enables the UI to display
    source information without re-reading the CSV.

    Returns:
        List of document dicts ready for ingestion into ChromaDB.
        Empty documents (failed conversion) are silently filtered out.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the configured path.
        ValueError:        If the CSV is empty after loading.
    """
    csv_path: Path = config.dataset_file_path

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {csv_path}\n"
            f"Please place your CSV file at that location and re-run ingestion."
        )

    log.info("Loading CSV from: %s", csv_path)

    # Load full CSV — we only read, so memory usage is manageable even for
    # very large files (pandas lazy evaluation handles this efficiently)
    df = pd.read_csv(csv_path, low_memory=False)

    if df.empty:
        raise ValueError(f"CSV file is empty: {csv_path}")

    log.info("Full dataset: %d rows × %d columns", len(df), len(df.columns))
    log.info("Columns detected: %s", list(df.columns))

    # Drop rows where the most critical columns are entirely missing
    critical_cols = [c for c in ["State_Name", "Crop", "Crop_Year"] if c in df.columns]
    before = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    after  = len(df)
    if before != after:
        log.warning("Dropped %d rows with missing critical fields.", before - after)

    # Stratified sampling — ensures geographic and crop diversity
    df = _stratified_sample(
        df=df,
        stratify_cols=config.STRATIFY_COLUMNS,
        max_rows=config.MAX_ROWS,
    )

    # Convert each row to a natural-language document
    documents: list[dict[str, Any]] = []
    skipped: int = 0

    for idx, row in df.iterrows():
        raw_text = _row_to_natural_language(row)
        cleaned  = clean_text(raw_text)

        if not is_valid_document(cleaned):
            skipped += 1
            continue

        # Apply chunking (for this CSV demo, most texts will not be chunked)
        chunks = chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        for chunk_idx, chunk in enumerate(chunks):
            # Build metadata — everything stored here is searchable/displayable
            metadata: dict[str, Any] = {
                "source":      str(csv_path.name),
                "format":      "csv",
                "row_index":   int(idx),
                "chunk_index": chunk_idx,
                "state":       str(row.get("State_Name", "")).strip(),
                "district":    str(row.get("District_Name", "")).strip(),
                "crop":        str(row.get("Crop", "")).strip(),
                "season":      str(row.get("Season", "")).strip(),
                "year":        str(row.get("Crop_Year", "")).strip(),
            }

            # Unique document ID — deterministic so re-ingesting the same
            # data never creates duplicate entries in ChromaDB
            doc_id = f"doc_{idx}_chunk_{chunk_idx}"

            documents.append({
                "id":       doc_id,
                "text":     chunk,
                "metadata": metadata,
            })

    log.info(
        "Documents prepared: %d valid, %d skipped",
        len(documents),
        skipped,
    )
    return documents


# ===========================================================================
# Section 3 — Future Loaders (Stubbed — ready to implement)
# ===========================================================================

def load_pdf_documents() -> list[dict[str, Any]]:
    """
    Load and extract text from all PDF files in DATASET_DIR.

    Uses PyMuPDF (fitz) for extraction. Each page becomes one document.
    Chunking is applied via chunk_text() for pages exceeding CHUNK_SIZE.

    To activate: set DATASET_FORMAT = "pdf" in config.py and place
    PDF files in data/sample_dataset/.

    Returns:
        List of document dicts in the same format as load_csv_documents().

    Raises:
        ImportError:       If PyMuPDF is not installed.
        FileNotFoundError: If no PDF files are found in DATASET_DIR.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF loading. "
            "Install it with: pip install pymupdf"
        ) from exc

    pdf_dir: Path = config.DATASET_DIR
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {pdf_dir}\n"
            "Place your PDF files there and re-run ingestion."
        )

    log.info("Found %d PDF file(s) in: %s", len(pdf_files), pdf_dir)
    documents: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        log.info("Processing PDF: %s", pdf_path.name)
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                raw_text = page.get_text("text")
                cleaned  = clean_text(raw_text)

                if not is_valid_document(cleaned):
                    continue

                chunks = chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{pdf_path.stem}_p{page_num}_c{chunk_idx}"
                    documents.append({
                        "id":   doc_id,
                        "text": chunk,
                        "metadata": {
                            "source":      pdf_path.name,
                            "format":      "pdf",
                            "page":        page_num + 1,
                            "chunk_index": chunk_idx,
                        },
                    })
            doc.close()
        except Exception as exc:
            log.error("Failed to process PDF %s: %s", pdf_path.name, exc)

    log.info("Total PDF documents prepared: %d", len(documents))
    return documents


def load_docx_documents() -> list[dict[str, Any]]:
    """
    Load and extract text from all DOCX files in DATASET_DIR.

    Uses python-docx for extraction. Each paragraph becomes one document
    (or is chunked if it exceeds CHUNK_SIZE).

    To activate: set DATASET_FORMAT = "docx" in config.py and place
    DOCX files in data/sample_dataset/.

    Returns:
        List of document dicts in the same format as load_csv_documents().

    Raises:
        ImportError:       If python-docx is not installed.
        FileNotFoundError: If no DOCX files are found in DATASET_DIR.
    """
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise ImportError(
            "python-docx is required for DOCX loading. "
            "Install it with: pip install python-docx"
        ) from exc

    docx_dir: Path = config.DATASET_DIR
    docx_files = list(docx_dir.glob("*.docx"))

    if not docx_files:
        raise FileNotFoundError(
            f"No DOCX files found in: {docx_dir}\n"
            "Place your DOCX files there and re-run ingestion."
        )

    log.info("Found %d DOCX file(s) in: %s", len(docx_files), docx_dir)
    documents: list[dict[str, Any]] = []

    for docx_path in docx_files:
        log.info("Processing DOCX: %s", docx_path.name)
        try:
            doc = DocxDocument(str(docx_path))
            para_idx = 0
            for para in doc.paragraphs:
                cleaned = clean_text(para.text)
                if not is_valid_document(cleaned):
                    continue
                chunks = chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{docx_path.stem}_p{para_idx}_c{chunk_idx}"
                    documents.append({
                        "id":   doc_id,
                        "text": chunk,
                        "metadata": {
                            "source":      docx_path.name,
                            "format":      "docx",
                            "paragraph":   para_idx,
                            "chunk_index": chunk_idx,
                        },
                    })
                para_idx += 1
        except Exception as exc:
            log.error("Failed to process DOCX %s: %s", docx_path.name, exc)

    log.info("Total DOCX documents prepared: %d", len(documents))
    return documents


def load_txt_documents() -> list[dict[str, Any]]:
    """
    Load and extract text from all plain-text (.txt) files in DATASET_DIR.

    Each file is read line-by-line. Non-empty lines are chunked and embedded.

    To activate: set DATASET_FORMAT = "txt" in config.py and place
    .txt files in data/sample_dataset/.

    Returns:
        List of document dicts in the same format as load_csv_documents().
    """
    txt_dir: Path = config.DATASET_DIR
    txt_files = list(txt_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in: {txt_dir}\n"
            "Place your text files there and re-run ingestion."
        )

    log.info("Found %d TXT file(s) in: %s", len(txt_files), txt_dir)
    documents: list[dict[str, Any]] = []

    for txt_path in txt_files:
        log.info("Processing TXT: %s", txt_path.name)
        try:
            raw = txt_path.read_text(encoding="utf-8", errors="replace")
            # Treat paragraphs (double newlines) as document boundaries
            paragraphs = re.split(r"\n{2,}", raw)
            for para_idx, para in enumerate(paragraphs):
                cleaned = clean_text(para)
                if not is_valid_document(cleaned):
                    continue
                chunks = chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{txt_path.stem}_p{para_idx}_c{chunk_idx}"
                    documents.append({
                        "id":   doc_id,
                        "text": chunk,
                        "metadata": {
                            "source":      txt_path.name,
                            "format":      "txt",
                            "paragraph":   para_idx,
                            "chunk_index": chunk_idx,
                        },
                    })
        except Exception as exc:
            log.error("Failed to process TXT %s: %s", txt_path.name, exc)

    log.info("Total TXT documents prepared: %d", len(documents))
    return documents


def load_json_documents() -> list[dict[str, Any]]:
    """
    Load and extract text from a JSON or JSONL file in DATASET_DIR.

    Expects either:
        - A JSON array of objects: [{"text": "...", ...}, ...]
        - A JSONL file (one JSON object per line): {"text": "..."}\n...

    The "text" key is used as the document content. All other keys
    are stored as metadata.

    To activate: set DATASET_FORMAT = "json" in config.py and place
    your JSON/JSONL file in data/sample_dataset/.

    Returns:
        List of document dicts in the same format as load_csv_documents().
    """
    import json

    json_dir   = config.DATASET_DIR
    json_files = list(json_dir.glob("*.json")) + list(json_dir.glob("*.jsonl"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON/JSONL files found in: {json_dir}\n"
            "Place your JSON file there and re-run ingestion."
        )

    log.info("Found %d JSON file(s) in: %s", len(json_files), json_dir)
    documents: list[dict[str, Any]] = []

    for json_path in json_files:
        log.info("Processing JSON: %s", json_path.name)
        try:
            raw = json_path.read_text(encoding="utf-8")
            # Try JSON array first, fall back to JSONL
            try:
                records = json.loads(raw)
                if isinstance(records, dict):
                    records = [records]
            except json.JSONDecodeError:
                records = [json.loads(line) for line in raw.splitlines() if line.strip()]

            for idx, record in enumerate(records):
                text_val = record.get("text", "") or record.get("content", "")
                cleaned  = clean_text(str(text_val))
                if not is_valid_document(cleaned):
                    continue
                metadata = {
                    k: str(v) for k, v in record.items()
                    if k not in ("text", "content")
                }
                metadata["source"] = json_path.name
                metadata["format"] = "json"
                chunks = chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
                for chunk_idx, chunk in enumerate(chunks):
                    documents.append({
                        "id":       f"{json_path.stem}_{idx}_c{chunk_idx}",
                        "text":     chunk,
                        "metadata": {**metadata, "chunk_index": chunk_idx},
                    })
        except Exception as exc:
            log.error("Failed to process JSON %s: %s", json_path.name, exc)

    log.info("Total JSON documents prepared: %d", len(documents))
    return documents


# ===========================================================================
# Section 4 — Unified Entry Point
# ===========================================================================

# Dispatch table — maps format string to its loader function.
# Adding support for a new format requires:
#   1. Writing a new load_xxx_documents() function above.
#   2. Adding one line here.
#   3. Nothing else changes anywhere in the project.
_LOADER_REGISTRY: dict[str, Any] = {
    "csv":  load_csv_documents,
    "pdf":  load_pdf_documents,
    "docx": load_docx_documents,
    "txt":  load_txt_documents,
    "md":   load_txt_documents,   # Markdown treated as plain text for now
    "json": load_json_documents,
}


def load_documents() -> list[dict[str, Any]]:
    """
    Unified document loading entry point called by ingest.py.

    Reads `config.DATASET_FORMAT` and dispatches to the appropriate
    domain-specific loader function. The caller (ingest.py) never needs
    to know which format is being used.

    Returns:
        List of document dicts:
            [
                {
                    "id":       "doc_0_chunk_0",
                    "text":     "In 2011, Wheat was cultivated in ...",
                    "metadata": {"source": "crop_production.csv", ...}
                },
                ...
            ]

    Raises:
        ValueError:        If DATASET_FORMAT is not recognised.
        FileNotFoundError: If the dataset file/directory does not exist.
        ValueError:        If the loader returns zero valid documents.
    """
    fmt = config.DATASET_FORMAT.lower().strip()

    if fmt not in _LOADER_REGISTRY:
        supported = list(_LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported dataset format: '{fmt}'. "
            f"Supported formats: {supported}"
        )

    log.info("Loading documents using '%s' loader...", fmt.upper())
    loader_fn = _LOADER_REGISTRY[fmt]
    documents = loader_fn()

    if not documents:
        raise ValueError(
            "No valid documents were loaded. "
            "Check that your dataset file is correctly formatted and "
            "contains sufficient data."
        )

    log.info("Total documents ready for ingestion: %d", len(documents))
    return documents


# ===========================================================================
# Section 5 — Miscellaneous Helpers
# ===========================================================================

def format_score(score: float) -> str:
    """
    Format a raw ChromaDB distance score as a human-readable similarity
    percentage string.

    ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
    We convert it to a cosine *similarity* (1 = identical, -1 = opposite)
    and then clamp to [0, 1] for display purposes.

    Args:
        score: Raw ChromaDB distance value (float, typically 0.0–2.0).

    Returns:
        Formatted string like "87.3%" or "N/A" if score is invalid.
    """
    try:
        similarity = 1.0 - float(score)
        similarity = max(0.0, min(1.0, similarity))
        return f"{similarity * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def cosine_similarity_from_distance(distance: float) -> float:
    """
    Convert a ChromaDB cosine distance to a cosine similarity score.

    ChromaDB stores cosine distance = 1 - cosine_similarity when the
    collection is created with metric="cosine".

    Args:
        distance: Raw ChromaDB distance (0.0 = identical, 1.0 = orthogonal).

    Returns:
        Cosine similarity in range [0.0, 1.0].
    """
    return max(0.0, min(1.0, 1.0 - float(distance)))


def truncate(text: str, max_chars: int = 300) -> str:
    """
    Truncate a string to max_chars characters, appending "..." if truncated.

    Used in the Streamlit UI to display previews of long documents without
    overwhelming the interface.

    Args:
        text:      The string to truncate.
        max_chars: Maximum character count before truncation.

    Returns:
        Original string if shorter than max_chars, otherwise truncated + "..."
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
