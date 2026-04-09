"""
S4 Token-Exact Chunking — same strategy used in pdf_chunking experiments.
Splits PDF text into chunks of fixed token size with overlap.
"""
import re
from transformers import AutoTokenizer
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOKENIZER_NAME

_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc  = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from {pdf_path}: {e}")


def chunk_text(text: str, doc_name: str, category: str) -> list[dict]:
    """
    Split text into token-exact chunks (S4 strategy).
    Returns list of dicts: {chunk_id, text, doc_name, category}
    """
    tokenizer = get_tokenizer()
    tokens    = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start  = 0
    idx    = 0

    while start < len(tokens):
        end        = min(start + CHUNK_SIZE, len(tokens))
        chunk_tok  = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tok, skip_special_tokens=True).strip()

        if chunk_text:
            chunk_id = f"{doc_name}_chunk_{idx}"
            chunks.append({
                "chunk_id" : chunk_id,
                "text"     : chunk_text,
                "doc_name" : doc_name,
                "category" : category,
            })
            idx += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP
        if end == len(tokens):
            break

    return chunks


def chunk_pdf(pdf_path: str, category: str) -> list[dict]:
    """Full pipeline: PDF path → list of chunks."""
    import os
    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text     = extract_text_from_pdf(pdf_path)
    return chunk_text(text, doc_name, category)
