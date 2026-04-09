import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ── Force CPU + disable all parallelism (fixes Mac segfault) ──────────────────
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]        = ""
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ["OMP_NUM_THREADS"]             = "1"
os.environ["MKL_NUM_THREADS"]             = "1"
os.environ["OPENBLAS_NUM_THREADS"]        = "1"

# ── Base Paths ─────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR         = os.path.join(BASE_DIR, "index")
KNOWLEDGE_BASE_DIR= os.path.join(BASE_DIR, "knowledge_base")

# Persistent index files
FAISS_INDEX_PATH  = os.path.join(INDEX_DIR, "faiss_index.bin")
CORPUS_PATH       = os.path.join(INDEX_DIR, "corpus.csv")
MANIFEST_PATH     = os.path.join(INDEX_DIR, "manifest.json")

# ── Model Paths ────────────────────────────────────────────────────────────────
# Copy these from your project:
#   fine_tuning/minilm_finetuned/         → finsearch/models/minilm_finetuned/
#   intent_classification/minilm_intent_classifier/ → finsearch/models/minilm_intent_classifier/
MODELS_DIR            = os.path.join(BASE_DIR, "models")
FT_MODEL_PATH         = os.path.join(MODELS_DIR, "minilm_finetuned")
INTENT_MODEL_PATH     = os.path.join(MODELS_DIR, "minilm_intent_classifier")

# ── External Model Names ───────────────────────────────────────────────────────
CE_MODEL_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL_NAME  = "cross-encoder/nli-deberta-v3-small"
LLAMA_MODEL     = "meta-llama/llama-3.3-70b-instruct"

# ── API ────────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Retrieval Settings ─────────────────────────────────────────────────────────
RETRIEVAL_TOP_K = 20
RERANK_TOP_K    = 3

# ── Confidence Thresholds ──────────────────────────────────────────────────────
OOS_CONFIDENCE_THRESHOLD = 0.30
CONF_HIGH   = 0.70
CONF_MEDIUM = 0.40

# ── Intent Categories ──────────────────────────────────────────────────────────
CATEGORIES = ["Regulatory", "Consumer_Protection", "Payment_Industry", "Synthetic_Policies"]
LABEL2ID   = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL   = {i: c for c, i in LABEL2ID.items()}

# ── Chunking Settings (S4 Token-Exact) ────────────────────────────────────────
CHUNK_SIZE    = 400   # tokens
CHUNK_OVERLAP = 100   # tokens
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Synthetic Data Paths ───────────────────────────────────────────────────────
# These point to your existing Dataset/ folder
DATASET_DIR      = os.path.join(BASE_DIR, "..", "Dataset", "Synthetic")
USERS_DB_PATH    = os.path.join(DATASET_DIR, "Users", "users.db")
TRANSACTIONS_PATH= os.path.join(DATASET_DIR, "Transactions", "transactions.csv")
DISPUTES_PATH    = os.path.join(DATASET_DIR, "Disputes", "disputes.csv")
