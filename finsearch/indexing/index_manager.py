"""
Incremental Index Manager
─────────────────────────
build()  → First time: chunk + embed all PDFs, save FAISS + corpus + manifest
update() → New PDFs only: chunk + embed only new files, add to existing index
load()   → App start: load FAISS index from disk in seconds (no recompute)
"""
import os, json, hashlib
import numpy as np
import pandas as pd
import faiss

from config import (
    FAISS_INDEX_PATH, CORPUS_PATH, MANIFEST_PATH,
    KNOWLEDGE_BASE_DIR, INDEX_DIR
)
from indexing.chunker  import chunk_pdf
from indexing.embedder import embed_chunks


# ── Helpers ────────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    """MD5 hash of file — detects duplicate uploads."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _scan_knowledge_base() -> list[tuple[str, str]]:
    """
    Scan knowledge_base/ folder.
    Returns list of (pdf_path, category) for all PDFs found.
    """
    results = []
    for category in os.listdir(KNOWLEDGE_BASE_DIR):
        cat_path = os.path.join(KNOWLEDGE_BASE_DIR, category)
        if not os.path.isdir(cat_path):
            continue
        for fname in os.listdir(cat_path):
            if fname.lower().endswith(".pdf"):
                results.append((os.path.join(cat_path, fname), category))
    return results


# ── Core Functions ─────────────────────────────────────────────────────────────

def build():
    """
    Full build from scratch.
    Chunks + embeds ALL PDFs in knowledge_base/, saves everything to index/.
    Run once — after that use load() or update().
    """
    os.makedirs(INDEX_DIR, exist_ok=True)
    pdf_files = _scan_knowledge_base()

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {KNOWLEDGE_BASE_DIR}. "
            "Copy your PDFs into knowledge_base/[Category]/ folders first."
        )

    print(f"Building index from scratch — {len(pdf_files)} PDFs found.")
    all_chunks   = []
    manifest     = {}

    for pdf_path, category in pdf_files:
        fname    = os.path.basename(pdf_path)
        doc_name = os.path.splitext(fname)[0]
        fhash    = _file_hash(pdf_path)
        print(f"  Chunking: {fname}")
        chunks = chunk_pdf(pdf_path, category)
        all_chunks.extend(chunks)
        manifest[fname] = {
            "doc_name"  : doc_name,
            "category"  : category,
            "file_hash" : fhash,
            "num_chunks": len(chunks),
            "chunk_ids" : [c["chunk_id"] for c in chunks],
        }

    # Embed all chunks
    print(f"\nEmbedding {len(all_chunks):,} chunks...")
    texts      = [c["text"] for c in all_chunks]
    embeddings = embed_chunks(texts)

    # Build FAISS index
    dim         = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)

    # Save everything
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    pd.DataFrame(all_chunks).to_csv(CORPUS_PATH, index=False)
    _save_manifest(manifest)

    print(f"\nIndex built: {faiss_index.ntotal:,} vectors")
    print(f"Saved to: {INDEX_DIR}")
    return faiss_index, all_chunks


def update():
    """
    Incremental update — only indexes NEW PDFs not in manifest.
    Adds new embeddings to existing FAISS index without rebuilding.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        print("No existing index found. Running full build instead.")
        return build()

    manifest    = _load_manifest()
    pdf_files   = _scan_knowledge_base()
    corpus_df   = pd.read_csv(CORPUS_PATH)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    new_chunks  = []
    updated     = False

    for pdf_path, category in pdf_files:
        fname  = os.path.basename(pdf_path)
        fhash  = _file_hash(pdf_path)

        # Already indexed with same content → skip
        if fname in manifest and manifest[fname]["file_hash"] == fhash:
            print(f"  Skipping (already indexed): {fname}")
            continue

        print(f"  Indexing new file: {fname}")
        chunks = chunk_pdf(pdf_path, category)
        doc_name = os.path.splitext(fname)[0]
        new_chunks.extend(chunks)
        manifest[fname] = {
            "doc_name"  : doc_name,
            "category"  : category,
            "file_hash" : fhash,
            "num_chunks": len(chunks),
            "chunk_ids" : [c["chunk_id"] for c in chunks],
        }
        updated = True

    if not updated:
        print("All PDFs already indexed. Nothing to update.")
        return faiss_index, corpus_df.to_dict("records")

    # Embed only new chunks
    print(f"\nEmbedding {len(new_chunks):,} new chunks...")
    new_texts      = [c["text"] for c in new_chunks]
    new_embeddings = embed_chunks(new_texts)

    # Add to existing FAISS index
    faiss_index.add(new_embeddings)

    # Append to corpus CSV
    new_df    = pd.DataFrame(new_chunks)
    corpus_df = pd.concat([corpus_df, new_df], ignore_index=True)
    corpus_df.to_csv(CORPUS_PATH, index=False)

    # Save updated index + manifest
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    _save_manifest(manifest)

    print(f"\nIndex updated: {faiss_index.ntotal:,} total vectors")
    print(f"New chunks added: {len(new_chunks):,}")
    return faiss_index, corpus_df.to_dict("records")


def load():
    """
    Load existing FAISS index + corpus from disk.
    Fast — called on every app start instead of rebuilding.
    Returns (faiss_index, corpus_texts, corpus_ids).
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(
            "No FAISS index found. Run index_manager.build() first.\n"
            "From finsearch/: python -m indexing.index_manager"
        )

    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    corpus_df   = pd.read_csv(CORPUS_PATH)

    corpus_texts = corpus_df["text"].tolist()
    corpus_ids   = corpus_df["chunk_id"].tolist()
    corpus_cats  = corpus_df["category"].tolist()

    return faiss_index, corpus_texts, corpus_ids, corpus_cats


def index_is_ready() -> bool:
    return (
        os.path.exists(FAISS_INDEX_PATH) and
        os.path.exists(CORPUS_PATH) and
        os.path.exists(MANIFEST_PATH)
    )


def get_manifest_summary() -> dict:
    manifest = _load_manifest()
    return {
        "total_pdfs"  : len(manifest),
        "total_chunks": sum(v["num_chunks"] for v in manifest.values()),
        "categories"  : list({v["category"] for v in manifest.values()}),
        "files"       : list(manifest.keys()),
    }


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "update"
    if cmd == "build":
        build()
    elif cmd == "update":
        update()
    else:
        print("Usage: python -m indexing.index_manager [build|update]")
