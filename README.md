# FinSearch: Intent-Aware Financial Document Intelligence

An end-to-end retrieval-augmented generation (RAG) system for financial documents. FinSearch routes user queries by intent, retrieves relevant passages from 41 financial PDFs across 4 regulatory categories, reranks with an LLM, generates a grounded answer, and scores it for faithfulness.

---

## Project Structure

```
FinSearch/
├── baseline/                   # Week 1 — BM25 Baseline
├── dense_retrieval/            # Week 2 — Dense Retrieval (4 models compared)
├── hybrid/                     # Week 2 — Hybrid Retrieval (BM25 + Dense)
├── finrerank/                  # Week 3 — LLM Reranking + Pipeline Comparison
├── pdf_chunking/               # Chunking Strategy Evaluation (foundation)
├── intent_classification/      # Intent Classifier (4-category routing)
├── Dataset/                    # FiQA corpus + PDF knowledge base
├── config.py                   # Central path config
├── requirements.txt
├── poster_plots.py             # Generates all comparison charts → visualization/
└── visualization/              # All experiment comparison charts
```

---

## Knowledge Base

41 financial PDFs across 4 categories:

| Category | Description |
|----------|-------------|
| `Regulatory` | Central bank and securities regulation documents |
| `Consumer_Protection` | Financial consumer protection guidelines |
| `Payment_Industry` | Payment systems and card network standards |
| `Synthetic_Policies` | Complaint procedures and internal policy documents |

---

## PDF Chunking Strategy — Foundation

**Goal:** Find the best way to split financial PDFs into retrieval-ready chunks before any retrieval experiment.  
**Files:** `pdf_chunking/PDF_Chunking.ipynb`  
**Corpus:** 4 representative PDFs (1 per category), evaluated on 20 synthetic QA pairs.

5 strategies evaluated with MiniLM and BGE-Large:

| Strategy | Description | MiniLM Recall@10 | BGE-Large Recall@10 |
|----------|-------------|:---:|:---:|
| S1 | Sliding window — 512 words | 0.35 | 0.35 |
| S2 | Sliding window — 256 words | 0.40 | 0.40 |
| S3 | Paragraph-based | 0.30 | 0.35 |
| **S4** | **Token-Exact 200/400 tokens** | **0.75** | **0.80 ✓ Winner** |
| S5 | Section-aware 200/400 | 0.60 | 0.75 |

**Winner: S4 Token-Exact (400 tokens, 100 overlap) — Recall@10 = 0.80**

> Token-exact chunking uses a HuggingFace tokenizer to split on exact token boundaries with zero truncation. Word-count methods silently cut mid-sentence; token-exact preserves full semantic units in formal regulatory language.

---

## Week 1 — BM25 Baseline

**Goal:** Establish a keyword-search baseline on the FiQA financial QA dataset.  
**Files:** `baseline/Baseline_model.ipynb`

| Model | NDCG@10 | MRR | Recall@10 | Queries |
|-------|:-------:|:---:|:---------:|:-------:|
| BM25 (k1=1.2, b=0.75) | 0.2169 | 0.2706 | 0.2784 | 648 |

BM25 struggles with synonym mismatch — user says "returns", document says "yield". This established the floor to beat.

---

## Week 2 — Dense Retrieval (4 Models Compared)

**Goal:** Replace keyword search with semantic vector search and find the best dense encoder.  
**Files:** `dense_retrieval/Dense_Retrieval.ipynb`, `finrerank/FinDomain_ModelComparison.ipynb`

4 dense models were evaluated on a stratified FiQA sub-corpus (194 queries, same random seed):

| Model | Params | Dim | NDCG@10 | MRR | Recall@10 |
|-------|:------:|:---:|:-------:|:---:|:---------:|
| `all-MiniLM-L6-v2` | 22M | 384 | 0.5821 | 0.6721 | 0.6468 |
| `BAAI/bge-base-en-v1.5` | 109M | 768 | 0.5817 | 0.6549 | 0.6570 |
| `intfloat/e5-small-v2` | 33M | 384 | 0.5634 | 0.6394 | 0.6323 |
| **`BAAI/bge-large-en-v1.5`** | **335M** | **1024** | **0.6355** | **0.6990** | **0.7258** |

**Winner: BGE-Large-EN-v1.5**

> BGE-Large outperforms others due to its larger capacity (1024-dim vs 384-dim) and training on MS-MARCO + financial-style corpora. E5-Small was the weakest despite having the same dimension as MiniLM — the quality of pre-training data matters more than parameter count alone.

**Reference — MiniLM on full corpus (648 queries):**

| Model | NDCG@10 | MRR | Recall@10 |
|-------|:-------:|:---:|:---------:|
| BM25 Baseline | 0.2169 | 0.2706 | 0.2784 |
| MiniLM Dense (full 648 q) | 0.3687 | 0.4451 | 0.4413 |

Dense retrieval alone gives **+70% NDCG@10** over BM25.

---

## Week 2 — Hybrid Retrieval

**Goal:** Combine BM25 (lexical) and dense (semantic) retrieval for best of both.  
**Files:** `hybrid/Hybrid_RRF.ipynb`

Two fusion methods compared — alpha-weighted interpolation and Reciprocal Rank Fusion (RRF):

| Method | NDCG@10 | MRR | Recall@10 |
|--------|:-------:|:---:|:---------:|
| BM25 Baseline | 0.2169 | 0.2706 | 0.2784 |
| MiniLM Dense | 0.3687 | 0.4451 | 0.4413 |
| Hybrid RRF (k=60) | 0.3519 | 0.4171 | 0.4396 |
| **Hybrid Alpha (α=0.7)** | **0.3791** | **0.4606** | **0.4473** |

Alpha sweep result — α=0.7 (70% dense + 30% BM25) is the optimal balance:

| α | NDCG@10 | | α | NDCG@10 |
|---|:-------:|-|---|:-------:|
| 0.1 | 0.2656 | | 0.6 | 0.3748 |
| 0.3 | 0.3084 | | **0.7** | **0.3791** |
| 0.5 | 0.3593 | | 0.9 | 0.3735 |

> RRF underperforms alpha-interpolation here because RRF ignores score magnitudes — it only uses rank positions, which discards confidence information that the dense model captures well.

---

## Week 3 — LLM Reranking + Query Expansion

**Goal:** Add query expansion and LLM-based reranking on top of the best dense retriever (BGE-Large).  
**Files:** `finrerank/FinChatbot.ipynb`, `finrerank/FinPipeline_Comparison.ipynb`

### What Was Built

```
User Query
    │
    ▼
[1] Query Expansion ──────── Groq LLaMA 3.3 70B
    │  Appends 8–12 financial synonyms/related terms to the query
    │  e.g. "returns" → "returns yield dividend payout equity income"
    │
    ▼
[2] Dense Retrieval ─────── BGE-Large-EN-v1.5 (1024-dim) FAISS
    │  Top-50 candidate passages retrieved
    │
    ▼
[3] LLM Reranking ──────── [compared Groq LLaMA vs Mistral Large]
    │  Sends top-20 passages to LLM with prompt:
    │  "Return a JSON array of passage numbers sorted most to least relevant"
    │  Takes top-10 after reranking
    │
    ▼
[4] Answer Generation ───── Groq LLaMA 3.3 70B
    │  "Answer using ONLY the provided documents"
    │
    ▼
[5] Confidence Score ────── DeBERTa-v3-small NLI CrossEncoder
       Retrieval confidence  = mean(normalized retrieval scores)       → weight 40%
       Faithfulness          = NLI entailment(document, answer)        → weight 60%
       Final label           = HIGH (≥0.7) / MED (≥0.4) / LOW (<0.4)
```

### Reranker Comparison: Groq vs Mistral

Two rerankers were tried head-to-head:

| Reranker | NDCG@10 | MRR | Recall@10 |
|----------|:-------:|:---:|:---------:|
| Groq LLaMA 3.3 70B | 0.3791* | 0.4606 | 0.4473 |
| **Mistral Large 2411** | **0.3885** | **0.4775** | **0.4485** |

> *Groq reranker was the same model used for answer generation — convenient but not specialized. Mistral Large 2411 performed better as a reranker because it is more instruction-following and precise in returning structured JSON rankings.

**Winner: Mistral Large 2411 as reranker**

### 4-Pipeline Comparison (194 queries, with QE + Mistral rerank)

| Pipeline | Retrieval Model | Strategy | NDCG@10 | MRR | Recall@10 |
|----------|----------------|----------|:-------:|:---:|:---------:|
| A1 | MiniLM (384-dim) | Dense | 0.5917 | 0.6607 | 0.6724 |
| A2 | MiniLM (384-dim) | Hybrid α=0.7 | 0.5813 | 0.6685 | 0.6513 |
| **B1** | **BGE-Large (1024-dim)** | **Dense** | **0.6056** | **0.6679** | **0.6917** |
| B2 | BGE-Large (1024-dim) | Hybrid α=0.7 | 0.5381 | 0.6243 | 0.5984 |

**Best pipeline: B1 — BGE-Large Dense + Query Expansion + Mistral Rerank + LLaMA Answer**

> Hybrid retrieval (B2) actually hurts BGE-Large — the model is strong enough on its own that adding BM25 introduces noise. Hybrid helps weaker models (A1 vs A2 is closer) but not a well-trained large encoder.

### Full Progression Across All Weeks

| Stage | Model | NDCG@10 | MRR | Recall@10 |
|-------|-------|:-------:|:---:|:---------:|
| Week 1 | BM25 Baseline | 0.2169 | 0.2706 | 0.2784 |
| Week 2 | MiniLM Dense | 0.3687 | 0.4451 | 0.4413 |
| Week 3 | Hybrid α=0.7 | 0.3791 | 0.4606 | 0.4473 |
| Week 4 | Hybrid + Mistral Rerank | 0.3885 | 0.4775 | 0.4485 |
| **Week 4** | **B1 Full Pipeline (sub-corpus)** | **0.6056** | **0.6679** | **0.6917** |

> NDCG@10 improvement from BM25 baseline to B1 full pipeline: **+179%**

---

## Intent Classification

**Goal:** Route user queries to the correct knowledge-base category before retrieval.  
**Files:** `intent_classification/FinIntent_Classifier.ipynb`, `intent_classification/FinIntent_DataPrep.ipynb`

3 classifiers compared on 4 categories (Regulatory, Consumer Protection, Payment Industry, Synthetic Policies):

| Model | Training Data | Banking77 Acc | QA Eval Acc (120 q) |
|-------|--------------|:-------------:|:-------------------:|
| Zero-Shot DeBERTa NLI | None | 25.5% | 5.0% |
| Fine-Tuned MiniLM — PDF Only | 600 Groq PDF questions | 73.5% | 75.8% |
| **Fine-Tuned MiniLM — Full** | **Banking77 + Groq PDF** | **93.0%** | **90.0%** |

**Winner: Fine-Tuned MiniLM on full dataset (Banking77 + PDF domain questions)**

- Evaluation set: 120 held-out Groq questions (30 per category)
- Saved to: `intent_classification/minilm_intent_classifier/`

> Training on diverse data (Banking77 conversational + PDF-domain formal questions) generalizes better than PDF-only. Zero-Shot DeBERTa fails almost entirely on domain-specific routing — general NLI models are not suited for fine-grained financial category classification.

---

## What's Next — Remaining Work

| Step | Task | Description |
|------|------|-------------|
| 1 | Retrieval Fine-Tuning | Fine-tune the BGE-Large or MiniLM encoder on (question, chunk) pairs generated directly from the 41 PDFs for better domain adaptation |
| 2 | Chatbot UI | Build a Streamlit interface connecting: Intent Classifier → BGE-Large FAISS (S4 chunks) → QE → Mistral Rerank → LLaMA Answer → DeBERTa Confidence Score |
| 3 | End-to-End PDF Evaluation | Run the full B1 pipeline on the 41 PDF knowledge base (not FiQA) and evaluate with domain-specific QA pairs |
| 4 | Answer Quality Evaluation | Beyond retrieval metrics — evaluate answer correctness, faithfulness rate, and confidence calibration on held-out questions |

---

## Models Used Across the Project

| Role | Model | Where |
|------|-------|--------|
| Dense retrieval — baseline | `sentence-transformers/all-MiniLM-L6-v2` | Local |
| Dense retrieval — compared | `BAAI/bge-base-en-v1.5` | Local |
| Dense retrieval — compared | `intfloat/e5-small-v2` | Local |
| Dense retrieval — **best** | `BAAI/bge-large-en-v1.5` | Local |
| Query expansion | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| LLM reranker — compared | `meta-llama/llama-3.3-70b-instruct` (Groq) | OpenRouter |
| LLM reranker — **best** | `mistralai/mistral-large-2411` | OpenRouter |
| Answer generation | `meta-llama/llama-3.3-70b-instruct` | OpenRouter |
| NLI confidence scorer | `cross-encoder/nli-deberta-v3-small` | Local |
| Intent classifier | Fine-tuned `all-MiniLM-L6-v2` | Local (saved) |
| Question generation (data) | `llama3-8b-8192` | Groq API |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
Create a `.env` file in the repo root:
```
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

### 3. Run experiments in order

| Step | Notebook | Purpose |
|------|----------|---------|
| 0 | `pdf_chunking/PDF_Chunking.ipynb` | Chunking strategy evaluation |
| 1 | `baseline/Baseline_model.ipynb` | BM25 baseline |
| 2 | `dense_retrieval/Dense_Retrieval.ipynb` | MiniLM dense retrieval |
| 2b | `finrerank/FinDomain_ModelComparison.ipynb` | Compare 4 dense models |
| 3 | `hybrid/Hybrid_RRF.ipynb` | Hybrid BM25 + Dense |
| 4 | `finrerank/FinChatbot.ipynb` | QE + reranker + answer + confidence |
| 4b | `finrerank/FinPipeline_Comparison.ipynb` | Compare all 4 pipelines |
| 5a | `intent_classification/FinIntent_DataPrep.ipynb` | Generate training data |
| 5b | `intent_classification/FinIntent_Classifier.ipynb` | Train intent classifier |

### 4. Generate comparison charts
```bash
python3 poster_plots.py
# Saves 4 charts to: visualization/
```

---

## Visualizations

All experiment comparison charts are in [`visualization/`](visualization/):

| Chart | Description |
|-------|-------------|
| [`01_all_experiments_comparison.png`](visualization/01_all_experiments_comparison.png) | 4-panel master chart — retrieval stages, full pipelines, chunking, intent classifier |
| [`02_metrics_table.png`](visualization/02_metrics_table.png) | All experiments in one metrics table, color-coded by week |
| [`03_ndcg_progression.png`](visualization/03_ndcg_progression.png) | NDCG@10 improvement from BM25 → full pipeline |
| [`04_recall_all_experiments.png`](visualization/04_recall_all_experiments.png) | Recall@10 across all experiments side by side |

---

## Dataset

- **FiQA** (Financial Question Answering): 57K passages, 648 test queries with relevance judgements — used for all retrieval evaluation.
- **Banking77**: 10K labeled banking intent queries across 77 classes — mapped to 4 categories for intent classifier training.
- **41 Financial PDFs**: Internal knowledge base across 4 regulatory/industry categories — chunked with S4 token-exact strategy.
- **Groq-generated questions**: ~2,400 PDF-domain questions generated by LLaMA-3 from the 41 PDFs — used for intent classifier training.
