# config.py — single source of truth for all experiment settings.
# The baseline and the ontology-guided method both use THIS file,
# so models, rate limits and LightRAG parameters are guaranteed to be identical.
from pathlib import Path

# ---------- Paths (relative to the repository root) ----------
ROOT = Path(__file__).resolve().parent.parent
RAW_PDF = ROOT / "data" / "raw" / "Lay.pdf"                 # full textbook (not committed)
PDF_DIR = ROOT / "data" / "pdf"                             # cut corpus and per-chapter PDFs
PARSED_DIR = ROOT / "data" / "parsed" / "hybrid_full"       # MinerU output from Colab
STORAGE_DIR = ROOT / "rag_storage"                          # LightRAG knowledge bases
RESULTS_DIR = ROOT / "results"                              # metrics and reports

# ---------- Corpus: PDF pages (1-based as shown in a PDF viewer, inclusive) ----------
# Chapters 1–5 without applied sections 1.6, 1.10, 2.6, 2.7, 4.8, 4.9, 5.6–5.8,
# introductory examples and Supplementary Exercises. Offset from book page numbers: +17.
CHAPTER_RANGES = {
    "ch1": [(19, 66), (73, 97)],     # 1.1–1.5, 1.7–1.9
    "ch2": [(111, 150), (165, 178)], # 2.1–2.5, 2.8–2.9
    "ch3": [(183, 204)],             # 3.1–3.3
    "ch4": [(209, 262)],             # 4.1–4.7
    "ch5": [(285, 319)],             # 5.1–5.5
}
CHAPTERS = list(CHAPTER_RANGES)

# ---------- Parsing (pinned for reproducibility) ----------
MINERU_VERSION = "3.4.5"
MINERU_BACKEND = "hybrid-auto-engine"

# ---------- Models ----------
LLM_MODEL = "gemini-3.5-flash-lite"      # entity extraction, equation descriptions, answers
VISION_MODEL = "gemini-3.5-flash-lite"   # figure descriptions (full multimodal mode only)
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768

# Model "thinking". None = model default.
# Checked 2026-11-19: Flash-Lite spends 0 thinking tokens by default.
THINKING = None

# ---------- API requests ----------
LLM_MAX_ASYNC = 8      # concurrent requests to Gemini
MAX_RETRIES = 6        # retries on 503/429

# Tier 1 rate limits (AI Studio -> Rate limits, 2026-11-19), with a safety margin
LLM_RPM = 1000         # Gemini 3.5 Flash Lite: 4K RPM, 4M TPM, 150K RPD
EMBED_RPM = 2000       # Gemini Embedding 1: 3K RPM (each text in a batch counts as a request)
EMBED_TPM = 800000     # Gemini Embedding 1: 1M TPM

# Prices for cost estimates (USD per 1M tokens, Gemini 3.5 Flash-Lite, paid tier)
PRICE_INPUT = 0.30
PRICE_OUTPUT = 2.50    # output tokens, including thinking tokens

# LightRAG technical parameters: they do not affect graph content, only how requests are sent
LIGHTRAG_KWARGS = {
    "llm_model_max_async": LLM_MAX_ASYNC,
    "embedding_func_max_async": 4,
    "default_embedding_timeout": 300,
    "default_llm_timeout": 600,
}

# Retrieval parameters (identical for both configurations)
QUERY_MODE = "hybrid"
ENABLE_RERANK = False
