# ── torch + device must be defined FIRST so _load_desklib() can always
# reference the global `device`, even if a later import fails. ──────
import torch
import os
import logging

# Inference-only service — disable autograd globally to eliminate gradient
# tensor allocation overhead on every forward pass (~50-150 MB per worker).
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)

# [C-12 FIX] Cap torch intra-op threads to avoid CPU over-subscription.
# Concurrency already comes from gunicorn gthread workers × the plugin
# ThreadPoolExecutor (plugin_registry.py) × per-document batching. Letting torch
# also spin up one BLAS thread per core multiplies into cores² runnable threads,
# causing context-switch thrashing and p99 latency blow-ups on CPU. Default 1;
# override with TORCH_NUM_THREADS when running a single-request, latency-bound box.
if device.type == "cpu":
    try:
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    except (ValueError, RuntimeError) as _thr_err:
        logger.warning("Could not set torch num_threads: %s", _thr_err)

import re
import hashlib as _hashlib
import threading as _threading
import time as _time
from collections import OrderedDict as _OrderedDict
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# Bare import: SegmentadorSemantico lives in this same directory. The alias
# finder installed by app/engine/__init__.py canonicalizes it to
# app.engine.segmentador so both import spellings share one module object.
from segmentador import SegmentadorSemantico


# ── Core model: single Desklib binary AI-text detector ───────────────────────
# [2026.07 swap] Replaces the previous 3-seed ModernBERT 41-class ensemble
# (model_1/model_2/model_3 + label_mapping) with the single DeBERTa-v3-based
# Desklib detector — mean-pooled hidden states -> one linear head -> sigmoid
# AI-probability. There is no longer a per-family "detected_model" signal or
# an inter-seed disagreement score; both are reported as None / 0.0 below so
# downstream plugins (which already read them defensively) degrade cleanly.
#
# DESKLIB_MODEL_NAME: Hugging Face Hub repo id, fetched into the standard HF
#   cache on first run — the same "download directly on the server" pattern
#   already used for the old ModernBERT weights (see .gitignore), just via the
#   Hub instead of a manually placed .bin file.
# DESKLIB_LOCAL_PATH / XPLAGIAX_EVAL_WEIGHTS: point at a local snapshot dir to
#   load fully offline (local_files_only=True) — mirrors the old candidate-
#   weights override used by scripts/retrain_pipeline.py evaluate.
# MODEL_FALLBACK_DIR: tried last if the primary source fails to load, so a
#   corrupt/unreachable primary doesn't take the whole engine down.
MODEL_NAME = os.getenv("DESKLIB_MODEL_NAME", "desklib/ai-text-detector-v1.01")
_LOCAL_MODEL_PATH = os.getenv("XPLAGIAX_EVAL_WEIGHTS") or os.getenv("DESKLIB_LOCAL_PATH")
MAX_LEN = int(os.getenv("DESKLIB_MAX_LEN", "768"))


class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}


# ── Model load bookkeeping (source / fallback) ─────────
# Populated by _load_desklib(); surfaced via get_model_info() and /api/drift-status
# so operators can see exactly which source each worker loaded the model from.
_MODEL_LOAD_INFO: List[Dict[str, object]] = []


def _load_desklib():
    """Load tokenizer + model, trying (in order) a local override, the Hub id,
    then MODEL_FALLBACK_DIR. Records provenance in _MODEL_LOAD_INFO."""
    fallback_dir = os.getenv("MODEL_FALLBACK_DIR", "")
    candidates: List[Tuple[str, bool]] = []
    if _LOCAL_MODEL_PATH:
        candidates.append((_LOCAL_MODEL_PATH, True))
    candidates.append((MODEL_NAME, False))
    if fallback_dir:
        candidates.append((fallback_dir, True))

    last_err: Optional[Exception] = None
    for i, (name_or_path, local_only) in enumerate(candidates):
        try:
            tok = AutoTokenizer.from_pretrained(name_or_path, local_files_only=local_only)
            mdl = DesklibAIDetectionModel.from_pretrained(name_or_path, local_files_only=local_only)
        except Exception as exc:
            last_err = exc
            logger.error("Desklib load failed from %s: %s", name_or_path, exc)
            continue
        _MODEL_LOAD_INFO.append({
            "requested": MODEL_NAME,
            "loaded_from": name_or_path,
            "fallback": i > 0,
        })
        return tok, mdl
    raise RuntimeError(
        f"Could not load Desklib AI detector ({MODEL_NAME}) from any candidate source"
    ) from last_err


tokenizer, model = _load_desklib()
model.to(device).eval()
# Pin tensors in POSIX shared memory so forked Gunicorn/Celery workers read
# the same physical pages without triggering Copy-on-Write faults.
if device.type == "cpu":
    try:
        model.share_memory()
    except Exception as _shm_err:
        # /dev/shm too small (Docker default 64 MB). Model stays in anon
        # CoW memory — perfectly fine for gthread workers and preload_app.
        logger.debug("share_memory() skipped: %s", _shm_err)


def get_model_info() -> Dict[str, object]:
    """Weight provenance + version for /api/drift-status and diagnostics."""
    return {
        "version": os.getenv("MODEL_VERSION", "2026.06"),
        "device": str(device),
        "model_name": MODEL_NAME,
        "weights": list(_MODEL_LOAD_INFO),
        "fallbacks_used": [
            i["loaded_from"] for i in _MODEL_LOAD_INFO if i.get("fallback")
        ],
    }


# Kept for backward compatibility with any external caller that inspected the
# old 41-class mapping; the Desklib head is a single binary logit, so this is
# now purely descriptive and never drives "detected_model" resolution.
label_mapping = {0: "Not AI Generated", 1: "AI Generated"}


@dataclass
class DetectionResult:
    prediction:           str
    confidence:           float
    human_percentage:     float
    ai_percentage:        float
    detected_model:       Optional[str]
    raw_scores:           Dict[str, float]
    statistical_features: Dict[str, float] = field(default_factory=dict)
    uncertainty_zone:     bool = False
    ensemble_disagreement: float = 0.0   # always 0.0 — single model, no seed disagreement to measure


def clean_text(text: str) -> str:
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\s+([,.;:?!])', r'\1', text)
    return text


def classify_text(text, generate_plot: bool = False):
    """
    Classifies the text and (optionally) generates a plot of human vs AI probability.
    Returns (result_message, fig, DetectionResult).

    generate_plot defaults to False. The pyplot global state is NOT thread-safe,
    and this function is reached from the ThreadPoolExecutor that runs plugins
    (plugin_registry.py). Set generate_plot=True only from single-threaded
    callers (e.g. Gradio).
    """
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        empty_result = DetectionResult(
            prediction="Unknown",
            confidence=0,
            human_percentage=50,
            ai_percentage=50,
            detected_model=None,
            raw_scores={"human": 0.0, "ai": 0.0},
            uncertainty_zone=True,
        )
        return "", None, empty_result

    encoded = tokenizer(
        cleaned_text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded)["logits"]
        ai_prob = torch.sigmoid(logits).item()

    human_percentage = (1.0 - ai_prob) * 100
    ai_percentage = ai_prob * 100

    if human_percentage > ai_percentage:
        result_message = (
            f"**The text is** <span class='highlight-human'>**{human_percentage:.2f}%** likely <b>Human written</b>.</span>"
        )
    else:
        result_message = (
            f"**The text is** <span class='highlight-ai'>**{ai_percentage:.2f}%** likely <b>AI generated</b>.</span>\n\n"
        )

    # Precise percentages BEFORE rounding the display values — human+ai == 100
    # (sigmoid complement), so these are the real model scores on a [0,100] scale.
    raw_scores = {
        "human": round(human_percentage, 2),
        "ai": round(ai_percentage, 2),
    }

    fig = None
    if generate_plot and plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))

        categories = ['Human', 'AI']
        probabilities_for_plot = [human_percentage, ai_percentage]

        bars = ax.bar(categories, probabilities_for_plot, color=['#4CAF50', '#FF5733'], alpha=0.8)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title('Human vs AI Probability', fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{height:.2f}%', ha='center')

        ax.set_ylim(0, 100)
        plt.tight_layout()

    human_percentage = round(human_percentage)
    ai_percentage    = round(ai_percentage)

    det_result = DetectionResult(
        prediction="Human" if human_percentage > ai_percentage else "AI",
        confidence=round(max(human_percentage, ai_percentage)),
        human_percentage=human_percentage,
        ai_percentage=ai_percentage,
        detected_model=None,
        raw_scores=raw_scores,
    )

    if fig is not None:
        plt.close(fig)
    return result_message, fig, det_result


def _gradio_classify(text: str):
    # Gradio runs single-threaded for this call → safe to build the pyplot figure.
    msg, fig, _ = classify_text(text, generate_plot=True)
    return msg, fig


# ── [C2] Segment-level inference cache ────────────────────────────
# Shared by classify_batch() (hybrid window classifier, classify_segment(), any
# chunked caller) AND analyze_fast() (document segmentation). Previously each
# had its own cache, so the SAME segment/window text scored via analyze_fast
# (ai_detection) and via classify_batch (segment_analysis's hybrid detector, or
# full_analysis running both) paid for the forward pass twice in one request —
# the two caches never shared a hit. Storing the raw (ai_prob, tok_len) pair
# here — instead of pre-rounded percentages — lets both call sites read from
# and write to the SAME entries while each still applies its own public
# rounding contract (classify_batch: int %, analyze_fast: 2-decimal %) exactly
# as before. Each entry is a float + an int, so the cache tops out at a few
# hundred KB. Namespaced by _CACHE_NS: a model swap invalidates it.
_SEG_CACHE: "_OrderedDict[str, Tuple[float, int]]" = _OrderedDict()
_SEG_CACHE_LOCK = _threading.Lock()
_SEG_CACHE_MAX = int(os.getenv("SEGMENT_CACHE_MAX", "2048"))


def _seg_cache_lookup(cleaned_texts: List[str]) -> Tuple[List[Optional[Tuple[float, int]]], List[str]]:
    """Shared LRU lookup: keys are _CACHE_NS + sha1(cleaned_text). Returns the
    per-text (ai_prob, tok_len) hit (or None on miss) plus the computed keys,
    so callers can write misses back under the same keys after inference."""
    keys = [
        _CACHE_NS + ":" + _hashlib.sha1(t.encode("utf-8")).hexdigest()
        for t in cleaned_texts
    ]
    hits: List[Optional[Tuple[float, int]]] = [None] * len(cleaned_texts)
    with _SEG_CACHE_LOCK:
        for i, key in enumerate(keys):
            hit = _SEG_CACHE.get(key)
            if hit is not None:
                _SEG_CACHE.move_to_end(key)
                hits[i] = hit
    return hits, keys


def _seg_cache_store(keys: List[str], values: List[Tuple[float, int]]) -> None:
    """Write (ai_prob, tok_len) misses back into the shared LRU, keyed as above."""
    with _SEG_CACHE_LOCK:
        for key, value in zip(keys, values):
            _SEG_CACHE[key] = value
            _SEG_CACHE.move_to_end(key)
        while len(_SEG_CACHE) > _SEG_CACHE_MAX:
            _SEG_CACHE.popitem(last=False)


def _cache_namespace() -> str:
    """Namespace the result cache by model identity + version so a model/weights
    swap (or MODEL_VERSION bump) never serves stale verdicts keyed only by text hash."""
    try:
        ident = f"{getattr(model.config, '_name_or_path', MODEL_NAME)}"
    except Exception:
        ident = "default"
    ident += ":" + os.environ.get("MODEL_VERSION", "")
    return _hashlib.sha1(ident.encode()).hexdigest()[:10]


_CACHE_NS: str = _cache_namespace()


@torch.inference_mode()
def classify_batch(texts: List[str]) -> List[Tuple[float, float]]:
    """
    Clasifica una lista de segmentos en un solo lote (batch).
    Es mucho más rápido que procesar uno por uno.

    Segment results are memoised in the shared LRU (also consulted by
    analyze_fast — see _SEG_CACHE above), keyed by the cleaned text, so only
    cache MISSES reach the model. Scores for misses are numerically identical
    to the uncached path (same tokenization, same sigmoid). Return shape
    (rounded int percentages) is unchanged from before the cache was unified.
    """
    if not texts:
        return []

    # Cooperative checkpoint: if the caller's timeout already expired (registry
    # reported the error), stop before paying for another forward pass.
    from exec_context import check_deadline
    check_deadline()

    cleaned_texts = [clean_text(t) for t in texts]
    hits, keys = _seg_cache_lookup(cleaned_texts)

    results: List[Optional[Tuple[float, float]]] = [None] * len(texts)
    for i, hit in enumerate(hits):
        if hit is not None:
            ai_prob, _tok_len = hit
            results[i] = (round((1.0 - ai_prob) * 100), round(ai_prob * 100))

    miss_idx = [i for i, r in enumerate(results) if r is None]
    if miss_idx:
        miss_texts = [cleaned_texts[i] for i in miss_idx]

        inputs = tokenizer(
            miss_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        ).to(device)

        logits = model(**inputs)["logits"]
        ai_probs = torch.sigmoid(logits).squeeze(-1)
        tok_lens = inputs["attention_mask"].sum(dim=1)

        miss_values: List[Tuple[float, int]] = []
        for j, i in enumerate(miss_idx):
            ai_prob = ai_probs[j].item()
            results[i] = (round((1.0 - ai_prob) * 100), round(ai_prob * 100))
            miss_values.append((ai_prob, int(tok_lens[j].item())))
        _seg_cache_store([keys[i] for i in miss_idx], miss_values)

    return results  # every slot is filled: hit above or miss inference here


@torch.inference_mode()
def classify_segment(text: str) -> Tuple[float, float]:
    """Clasifica un único segmento. Ahora usa classify_batch internamente."""
    results = classify_batch([text])
    return results[0] if results else (50.0, 50.0)


# ── Embedding / inference result cache ───────────────────────────────────────
# Keyed by sha256(text). Prevents re-running the model when the same text is
# analyzed by multiple plugins in the same request (e.g. both ai_detection and
# full_analysis requested together) or repeated shortly after.
_FAST_CACHE: dict = {}
_FAST_CACHE_LOCK = _threading.Lock()
_FAST_CACHE_TTL: float = 300.0   # 5 minutes — covers same-request multi-plugin calls
_FAST_CACHE_MAX: int = 20        # keep memory bounded; LRU eviction


# ── Document segmentation ────────────────────────────────────────────────────
# analyze_fast() now delegates segmentation to SegmentadorSemantico
# (app/engine/segmentador.py): it normalizes the text to a continuous sentence
# stream and cuts on topic change (lexical/LSA cohesion), so a document
# segments identically whether it arrives with line breaks or all run
# together — a PDF-extracted document and the same text pasted by hand no
# longer produce different verdicts depending on how newlines happened to
# fall. SEGMENT_MIN_CHARS / SEGMENT_MAX_CHARS bound segment size: too small
# carries no statistical signal for a verdict, too large risks silent
# tokenizer truncation past MAX_LEN.
_segmentador = SegmentadorSemantico(
    ventana=2,
    sensibilidad=0.0,
    min_oraciones=2,
    min_palabras_oracion=5,
    min_caracteres=int(os.getenv("SEGMENT_MIN_CHARS", "200")),
    max_caracteres=int(os.getenv("SEGMENT_MAX_CHARS", "3000")),
)


@torch.inference_mode()
def analyze_fast(text: str) -> dict:
    """
    Semantic-segment-aware document analysis matching the reference
    classify_text() pipeline.

    Each semantic segment (SegmentadorSemantico — topic-change boundaries, not
    raw newlines) is classified as an independent unit:
      1. clean_text() normalization
      2. tokenizer(segment, truncation=True) — one forward pass per batch
      3. sigmoid AI-probability -> human_pct / ai_pct

    Result cache: TTL=5min, 20-entry LRU — same-text repeated calls cost 0ms.
    """
    if not text.strip():
        return {"error": "El documento está vacío."}

    # Cache on raw text (before cleaning) to preserve hit rate across callers,
    # namespaced by model version so a model swap invalidates stale entries.
    _text_hash = _CACHE_NS + ":" + _hashlib.sha256(text.encode()).hexdigest()
    _now = _time.monotonic()
    with _FAST_CACHE_LOCK:
        _entry = _FAST_CACHE.get(_text_hash)
        if _entry is not None and _now - _entry[1] < _FAST_CACHE_TTL:
            return _entry[0]
        if len(_FAST_CACHE) >= _FAST_CACHE_MAX:
            _oldest = min(_FAST_CACHE, key=lambda k: _FAST_CACHE[k][1])
            del _FAST_CACHE[_oldest]

    segments_text: List[str] = [s.texto for s in _segmentador.segmentar(text)]
    if not segments_text:
        segments_text = [clean_text(text).strip()]

    BATCH_SIZE = int(os.getenv("SEGMENT_BATCH_SIZE", "12"))

    # Deduplicate identical cleaned segments. A thesis repeats boilerplate —
    # running headers, chapter labels, the "References" heading, recurring
    # citations — and inference is deterministic, so an identical segment
    # always yields an identical verdict. Classify each UNIQUE segment once and
    # fan the result back out to every position that shares it.
    cleaned_segments = [clean_text(s) for s in segments_text]
    unique_index: Dict[str, int] = {}
    unique_texts: List[str] = []
    seg_to_unique: List[int] = []
    for s in cleaned_segments:
        u = unique_index.get(s)
        if u is None:
            u = len(unique_texts)
            unique_index[s] = u
            unique_texts.append(s)
        seg_to_unique.append(u)

    # [C2] Shared-cache lookup FIRST: a segment already scored via classify_batch()
    # in this same request (e.g. segment_analysis's hybrid detector, or another
    # plugin run alongside ai_detection) skips tokenization AND the forward pass
    # entirely — not just the forward pass — since (ai_prob, tok_len) is cached
    # together. This is the same _SEG_CACHE classify_batch() reads/writes, so
    # the two call paths finally share hits instead of each paying separately.
    unique_hits, unique_keys = _seg_cache_lookup(unique_texts)
    unique_pcts: List[Optional[Tuple[float, float]]] = [None] * len(unique_texts)
    unique_tok_len: List[int] = [0] * len(unique_texts)
    for i, hit in enumerate(unique_hits):
        if hit is not None:
            ai_prob, tok_len = hit
            unique_pcts[i] = (round((1.0 - ai_prob) * 100, 2), round(ai_prob * 100, 2))
            unique_tok_len[i] = tok_len

    # Length-bucketed batching over cache MISSES only: segments are classified in
    # ascending-length order so each batch pads to a near-uniform length instead
    # of the batch max. Mixed-length documents (short + long paragraphs) waste
    # 30-50% of forward-pass FLOPs on pad tokens otherwise.
    miss_idx = [i for i, v in enumerate(unique_pcts) if v is None]
    order = sorted(miss_idx, key=lambda i: len(unique_texts[i]))

    from exec_context import check_deadline
    for i in range(0, len(order), BATCH_SIZE):
        check_deadline()  # abort orphaned threads at batch boundaries
        bucket = order[i:i + BATCH_SIZE]
        batch_texts = [unique_texts[j] for j in bucket]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
        ).to(device)
        logits = model(**inputs)["logits"]
        ai_probs = torch.sigmoid(logits).squeeze(-1)
        tok_lens = inputs["attention_mask"].sum(dim=1)
        batch_cache_values: List[Tuple[float, int]] = []
        for k, uniq_idx in enumerate(bucket):
            ai_prob = ai_probs[k].item()
            tok_len = int(tok_lens[k].item())
            unique_pcts[uniq_idx] = (
                round((1.0 - ai_prob) * 100, 2),
                round(ai_prob * 100, 2),
            )
            unique_tok_len[uniq_idx] = tok_len
            batch_cache_values.append((ai_prob, tok_len))
        _seg_cache_store([unique_keys[j] for j in bucket], batch_cache_values)

    # Fan the unique results back out to every original segment position.
    all_pcts = [unique_pcts[seg_to_unique[i]] for i in range(len(segments_text))]
    all_tok_len = [unique_tok_len[seg_to_unique[i]] for i in range(len(segments_text))]

    # Per-segment results + token-weighted aggregate
    segments = []
    total_human_w = total_ai_w = total_len = 0

    for idx, ((human_pct, ai_pct), tok_len, seg_text) in enumerate(
        zip(all_pcts, all_tok_len, segments_text)
    ):
        segments.append({
            "segment_id": idx + 1,
            "text": seg_text,
            "dominant_label": "AI" if ai_pct > human_pct else "Human",
            "score": max(ai_pct, human_pct),
            "ensemble_disagreement": 0.0,
        })
        total_human_w += human_pct * tok_len
        total_ai_w += ai_pct * tok_len
        total_len += tok_len

    if total_len == 0:
        return {"error": "No se pudieron procesar tokens del documento."}

    overall_human = round(total_human_w / total_len, 2)
    overall_ai = round(total_ai_w / total_len, 2)

    _result = {
        "overall_summary": {
            "total_human_percentage": overall_human,
            "total_ai_percentage": overall_ai,
            "overall_prediction": "AI" if overall_ai > overall_human else "Human",
            "detected_model": None,
            "ensemble_disagreement": 0.0,
        },
        "segments": segments,
    }
    with _FAST_CACHE_LOCK:
        _FAST_CACHE[_text_hash] = (_result, _time.monotonic())
    return _result


def classify_text_aggregate(text: str) -> DetectionResult:
    """
    Document-level DetectionResult covering the FULL text.

    classify_text() tokenizes with truncation=True, so for documents longer
    than MAX_LEN it only classifies the first chunk — the forensic verdict
    would then ignore the bulk of a long document. This helper instead reuses
    analyze_fast(), whose per-segment, token-weighted aggregate spans the
    whole text, and packages it as a DetectionResult so PluginOrchestrator
    produces a verdict representative of the entire document.

    Falls back to a neutral Unknown result on empty/error input.
    """
    doc = analyze_fast(text)
    if not isinstance(doc, dict) or "error" in doc:
        return DetectionResult(
            prediction="Unknown", confidence=0,
            human_percentage=50, ai_percentage=50,
            detected_model=None, raw_scores={"human": 0.0, "ai": 0.0},
            uncertainty_zone=True,
        )
    s = doc.get("overall_summary", {})
    human = s.get("total_human_percentage", 50)
    ai = s.get("total_ai_percentage", 50)
    prediction = s.get("overall_prediction", "Human")
    disagree = float(s.get("ensemble_disagreement", 0.0))
    # Uncertain when the margin is thin. disagree is always 0.0 for a single
    # model but the formula is kept so a future multi-model reintroduction
    # only needs to populate ensemble_disagreement to re-activate this leg.
    uncertain = abs(ai - human) < 15 or disagree >= 12.0
    return DetectionResult(
        prediction=prediction,
        confidence=round(max(human, ai), 1),
        human_percentage=human,
        ai_percentage=ai,
        detected_model=s.get("detected_model") if prediction == "AI" else None,
        raw_scores={"human": float(human), "ai": float(ai)},
        uncertainty_zone=uncertain,
        ensemble_disagreement=disagree,
    )
