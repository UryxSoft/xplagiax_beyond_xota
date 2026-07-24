"""
Stylometric Profiler Module  v2.1.1 (Refactored)
==================================================
Creates unique writing fingerprints for individual authors.
Detects when AI has "taken over" a human's writing.

Refactor changelog (v2.1 -> v2.1.1)
-------------------------------------
  [FIX]     Exported VECTOR_DIM constant — eliminates magic-number mismatch.
  [FIX]     spaCy doc parsed ONCE per text, passed to POS + syntactic extractors.
  [FIX]     MATTR uses adaptive stride for O(n) on long texts.
  [FIX]     StyleProfile.schema_version field for forward-compatible serialisation.
  [FIX]     to_dict/from_dict use dataclasses.fields() — auto-picks up new fields.
  [FIX]     StylometricProfiler accepts injectable nlp (testability).
  [API]     Public vectorize(text) method replaces private _build_temp_profile access.
  [STYLE]   PEP 8 strict; narrowed exception handlers; English identifiers.

All v2.1 public APIs remain fully compatible.

Requires
--------
  numpy, scipy              (required)
  spacy                     (optional — POS bigrams + syntactic features)
    pip install spacy && python -m spacy download en_core_web_sm

Do NOT call logging.basicConfig() here — let the application configure logging.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional spaCy — module-level fallback (overridable via DI)
# ---------------------------------------------------------------------------
# [C7 FIX] Share the single spaCy pipeline (see app/engine/_nlp.py). Previously this
# module and hallucination_profile.py each loaded their own en_core_web_sm copy.
from app.engine._nlp import get_nlp as _get_nlp, spacy_available as _spacy_available
_NLP = _get_nlp()
_SPACY_AVAILABLE = _spacy_available()
if _SPACY_AVAILABLE:
    logger.debug("spaCy loaded (shared) — POS tagger + dependency parser active.")
else:
    logger.debug("spaCy unavailable — regex fallbacks active.")


# ---------------------------------------------------------------------------
# Pluggable protocols
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """
    Protocol for any dense-embedding model.

    Accepted by the internal _compute_profile_data() helper's optional
    embedding_provider parameter, to fuse semantic embeddings with the
    statistical fingerprint. Not currently wired to any public method —
    vectorize()/compute_stats() call _compute_profile_data() without one.
    Kept as an extension point for a future caller that wants to inject
    embeddings (see app/engine/author_embedding.py for the embedding-based
    authorship approach this project uses today instead).

    Example with sentence-transformers
    ------------------------------------
    from sentence_transformers import SentenceTransformer

    class STEmbedder:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self._model = SentenceTransformer(model_name)

        def embed(self, text: str) -> np.ndarray:
            return self._model.encode(text, normalize_embeddings=True)

        @property
        def dim(self) -> int:
            return self._model.get_sentence_embedding_dimension()
    """

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float64 unit vector for the given text."""
        ...

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vector."""
        ...


# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

FUNCTION_WORDS: frozenset = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "that", "this", "these", "those", "it", "its", "i", "you",
        "he", "she", "we", "they", "me", "him", "her", "us", "them",
    }
)

FILLER_WORDS: frozenset = frozenset(
    {
        "like", "basically", "literally", "actually", "honestly",
        "seriously", "anyway", "whatever", "stuff", "things", "right",
        "kind of", "sort of", "you know", "i mean",
    }
)

TRANSITION_WORDS: frozenset = frozenset(
    {
        "however", "therefore", "furthermore", "moreover", "additionally",
        "consequently", "nevertheless", "nonetheless", "thus", "hence",
        "although", "despite", "meanwhile", "subsequently",
    }
)

_MULTI_WORD_FILLERS = frozenset(w for w in FILLER_WORDS if " " in w)
_SINGLE_WORD_FILLERS = FILLER_WORDS - _MULTI_WORD_FILLERS

# Pre-compiled regex for multi-word filler detection (one scan instead of N).
# Longest alternatives first to prevent partial matches ("sort" matching inside "sort of").
_MULTI_WORD_FILLER_RE: Optional[re.Pattern] = (
    re.compile(
        "|".join(re.escape(e) for e in sorted(_MULTI_WORD_FILLERS, key=len, reverse=True))
    )
    if _MULTI_WORD_FILLERS
    else None
)

# Unified stop-word set used by _find_signature_words().
# Merges FUNCTION_WORDS with high-frequency English words not already covered.
_STOP_WORDS: frozenset = FUNCTION_WORDS | frozenset(
    {
        "be", "to", "that", "so", "up", "out", "if", "about",
        "who", "get", "which", "go", "when", "make",
        # additional common words excluded from signature detection
        "not", "his", "say", "my", "one", "all", "there", "their", "what",
    }
)

# ── Sentence splitting ────────────────────────────────────────────────────────
# Guards against false splits on abbreviations (Dr., Mr., etc.) and decimal
# numbers (3.14, 99.5) that the naive [.!?]+ regex would split incorrectly.

_ABBREV_SENTINEL = "\x02"   # single control char — not present in normal text
_DECIMAL_SENTINEL = "\x03"
_ACRONYM_SENTINEL = "\x04"  # EC-01: protects ALL-CAPS acronyms (NASA., CIA., EU.)

_ABBREV_PATTERN = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|al|Fig|eq|Vol|No|pp)\."
    r"(?=\s)",
    re.IGNORECASE,
)
_DECIMAL_PATTERN = re.compile(r"(\d)\.(\d)")
# 2+ uppercase letters before a period — almost never a sentence boundary.
# Handles 100%-uppercase text where _ABBREV_PATTERN's title-case assumptions break.
# Protect "NASA. and" / "CIA. work" (lowercase continues the sentence) but NOT
# "MIT. He" — an all-caps acronym followed by a capitalized word is a genuine
# sentence boundary and must stay splittable (EC-01 covers both directions).
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,})\.(?!\s+[A-Z][a-z])")
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?=\s|$)")


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences with basic abbreviation and decimal guards.

    Improvements over the naive re.split(r'[.!?]+', text):
      - Decimal numbers (3.14, 99.5%) are not split mid-number.
      - Common abbreviations (Dr., Mr., Prof., etc.) are not split.
      - ALL-CAPS acronyms (NASA., CIA., EU.) are not split (EC-01).
      - Only splits when punctuation is followed by whitespace or end-of-string,
        preventing splits inside URLs and compound words.
    """
    # Protect ALL-CAPS acronyms first so they survive the ABBREV and DECIMAL passes
    protected = _ACRONYM_PATTERN.sub(
        lambda m: m.group().replace(".", _ACRONYM_SENTINEL), text
    )
    protected = _ABBREV_PATTERN.sub(
        lambda m: m.group().replace(".", _ABBREV_SENTINEL), protected
    )
    protected = _DECIMAL_PATTERN.sub(
        lambda m: m.group().replace(".", _DECIMAL_SENTINEL), protected
    )
    parts = _SENT_SPLIT_RE.split(protected)
    return [
        p.replace(_ABBREV_SENTINEL, ".")
         .replace(_DECIMAL_SENTINEL, ".")
         .replace(_ACRONYM_SENTINEL, ".")
         .strip()
        for p in parts
        if p.strip()
    ]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StyleProfile:
    """
    Complete stylometric fingerprint for one author.

    Scalar fields store normalised rates or means.
    Dict fields store relative-frequency distributions.
    The feature_vector is z-score normalised (Delta-of-Burrows).
    embedding_vector is populated only when an EmbeddingProvider is used.
    """

    # -- Schema version (NEW v2.1.1) ------------------------------------
    schema_version: str = "2.1.1"

    # -- Metadata -------------------------------------------------------
    author_id: str = ""
    sample_count: int = 0
    total_words: int = 0
    created_at: str = ""

    # -- Lexical --------------------------------------------------------
    vocabulary_richness: float = 0.0
    avg_word_length: float = 0.0
    rare_word_ratio: float = 0.0
    hapax_legomena_ratio: float = 0.0

    # -- Punctuation ----------------------------------------------------
    comma_rate: float = 0.0
    semicolon_rate: float = 0.0
    exclamation_rate: float = 0.0
    question_rate: float = 0.0
    ellipsis_rate: float = 0.0
    dash_rate: float = 0.0

    # -- Sentence structure ---------------------------------------------
    avg_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    avg_paragraph_length: float = 0.0

    # -- Syntactic depth (v2.1) -----------------------------------------
    avg_dep_distance: float = 0.0
    max_dep_distance: float = 0.0
    avg_tree_depth: float = 0.0
    complex_sentence_ratio: float = 0.0

    # -- Burstiness (v2.1) ----------------------------------------------
    burstiness_score: float = 0.0

    # -- Word n-grams ---------------------------------------------------
    function_word_frequencies: Dict[str, float] = field(default_factory=dict)
    bigram_frequencies: Dict[str, float] = field(default_factory=dict)
    trigram_frequencies: Dict[str, float] = field(default_factory=dict)
    filler_word_frequencies: Dict[str, float] = field(default_factory=dict)
    transition_frequencies: Dict[str, float] = field(default_factory=dict)

    # -- Char n-grams ---------------------------------------------------
    char_4gram_frequencies: Dict[str, float] = field(default_factory=dict)
    char_5gram_frequencies: Dict[str, float] = field(default_factory=dict)

    # -- POS bigrams ----------------------------------------------------
    pos_bigram_frequencies: Dict[str, float] = field(default_factory=dict)
    pos_bigrams_source: str = "none"

    # -- Signature words ------------------------------------------------
    signature_words: List[str] = field(default_factory=list)

    # -- Statistical feature vector + normalisation ---------------------
    # NOTE: size is set lazily; VECTOR_DIM is the canonical source.
    feature_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    feature_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    feature_std: np.ndarray = field(
        default_factory=lambda: np.ones(0)
    )

    # -- Semantic embedding (populated by EmbeddingProvider) -------------
    embedding_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )

    # -- Adaptive threshold ---------------------------------------------
    adaptive_threshold: Optional[float] = None

def _extract_lexical_features(text: str) -> Dict[str, float]:
    """Extract vocabulary richness, word length, and rarity metrics."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {
            "vocabulary_richness": 0.0,
            "avg_word_length": 0.0,
            "rare_word_ratio": 0.0,
            "hapax_legomena_ratio": 0.0,
        }

    counts = Counter(words)
    total = len(words)
    window = 100

    if total >= window:
        # Adaptive stride: O(n) for large texts instead of O(n^2)
        if total < 2000:
            stride = 1
        elif total < 10_000:
            stride = 5
        else:
            stride = 10

        ttrs = [
            len(set(words[i : i + window])) / window
            for i in range(0, total - window + 1, stride)
        ]
        mattr = float(np.mean(ttrs))
    else:
        mattr = len(set(words)) / total

    hapax = sum(1 for v in counts.values() if v == 1)
    rare = sum(1 for v in counts.values() if v < 3)

    # Hapax ratio is a sample-size artifact below _HAPAX_MIN_WORDS: in a 13-word
    # fragment almost every word appears exactly once simply because there is no
    # room for repetition, regardless of who wrote it — the ratio sits near 1.0
    # for any author. The fusion weights it -0.25 (pro-human, see
    # engine/fusion.py _HEURISTIC_HUMAN_WEIGHTS), so on tiny inputs this noise
    # was diluting an otherwise confident neural verdict (e.g. 97% -> 91%).
    # Zeroed out below the floor instead of omitted: fusion multiplies
    # weight * value, so 0.0 contributes nothing in either direction — the same
    # "missing evidence is neutral, not pro-human" contract author_signature and
    # semantic_consistency already use for their own inconclusive cases.
    hapax_ratio = (hapax / total) if total >= _HAPAX_MIN_WORDS else 0.0

    return {
        "vocabulary_richness": mattr,
        "avg_word_length": float(np.mean([len(w) for w in words])),
        "rare_word_ratio": rare / total,
        "hapax_legomena_ratio": hapax_ratio,
    }


def _extract_punctuation_features(
    text: str, num_sentences: int
) -> Dict[str, float]:
    """Extract punctuation rate features normalised by sentence count."""
    d = max(num_sentences, 1)
    return {
        "comma_rate": text.count(",") / d,
        "semicolon_rate": text.count(";") / d,
        "exclamation_rate": text.count("!") / d,
        "question_rate": text.count("?") / d,
        "ellipsis_rate": text.count("...") / d,
        "dash_rate": (
            text.count("\u2014") + text.count("\u2013") + text.count(" - ")
        )
        / d,
    }


def _extract_sentence_features(text: str) -> Dict[str, float]:
    """Extract sentence-level length statistics."""
    sentences = _split_sentences(text)
    if not sentences:
        return {"avg_sentence_length": 0.0, "sentence_length_variance": 0.0}
    lengths = [len(s.split()) for s in sentences]
    return {
        "avg_sentence_length": float(np.mean(lengths)),
        "sentence_length_variance": (
            float(np.var(lengths, ddof=1)) if len(lengths) > 1 else 0.0
        ),
    }


def _compute_burstiness(text: str) -> float:
    """
    Burstiness score of sentence-length distribution.

    Formula:  B = (sigma - mu) / (sigma + mu)   range: [-1, +1]

    Human writing is characteristically bursty (B > 0).
    LLMs produce more uniform sentence lengths (B near 0 or negative).
    """
    sentences = _split_sentences(text)
    lengths = np.array([len(s.split()) for s in sentences], dtype=float)

    if len(lengths) < 3:
        return 0.0

    mu = float(np.mean(lengths))
    sigma = float(np.std(lengths, ddof=1))

    if mu + sigma < 1e-9:
        return 0.0

    return float((sigma - mu) / (sigma + mu))


# ---------------------------------------------------------------------------
# Dependency tree helpers
# ---------------------------------------------------------------------------


def _token_depth(tok: Any) -> int:
    """Walk up the dependency tree to compute depth. Guard against cycles."""
    d = 0
    cur = tok
    while cur.head != cur:
        cur = cur.head
        d += 1
        if d > 50:
            break
    return d


def _extract_syntactic_features(
    doc: Any = None,
) -> Dict[str, float]:
    """
    Dependency-tree metrics from a pre-parsed spaCy Doc.

    When doc is None (spaCy unavailable), returns zeros.
    """
    zeros: Dict[str, float] = {
        "avg_dep_distance": 0.0,
        "max_dep_distance": 0.0,
        "avg_tree_depth": 0.0,
        "complex_sentence_ratio": 0.0,
    }

    if doc is None:
        return zeros

    per_sent_avg_dist: List[float] = []
    per_sent_max_dist: List[float] = []
    per_sent_avg_depth: List[float] = []
    complex_count = 0
    sent_count = 0
    _depth_cache: Dict[int, int] = {}

    def _cached_depth(tok: Any) -> int:
        if tok.i not in _depth_cache:
            _depth_cache[tok.i] = _token_depth(tok)
        return _depth_cache[tok.i]

    for sent in doc.sents:
        sent_count += 1
        tokens = list(sent)
        if not tokens:
            continue

        dists = [abs(tok.i - tok.head.i) for tok in tokens if tok.head != tok]
        if dists:
            per_sent_avg_dist.append(float(np.mean(dists)))
            per_sent_max_dist.append(float(max(dists)))

        depths = [_cached_depth(tok) for tok in tokens]
        per_sent_avg_depth.append(float(np.mean(depths)))

        dep_labels = {tok.dep_ for tok in tokens}
        if dep_labels & {"advcl", "relcl", "acl", "csubj", "ccomp"}:
            complex_count += 1

    return {
        "avg_dep_distance": (
            float(np.mean(per_sent_avg_dist)) if per_sent_avg_dist else 0.0
        ),
        "max_dep_distance": (
            float(np.mean(per_sent_max_dist)) if per_sent_max_dist else 0.0
        ),
        "avg_tree_depth": (
            float(np.mean(per_sent_avg_depth)) if per_sent_avg_depth else 0.0
        ),
        "complex_sentence_ratio": complex_count / max(sent_count, 1),
    }


# ---------------------------------------------------------------------------
# Word / char n-gram helpers
# ---------------------------------------------------------------------------


def _word_frequencies(text: str, word_set: frozenset) -> Dict[str, float]:
    """Compute relative frequency of single-token words from a given set."""
    words = re.findall(r"\b\w+\b", text.lower())
    total = max(len(words), 1)
    counts = Counter(words)
    return {
        w: counts[w] / total
        for w in word_set
        if w in counts and " " not in w
    }


def _mixed_frequencies(text: str, word_set: frozenset) -> Dict[str, float]:
    """Handle both single-token and multi-word entries (e.g. 'kind of')."""
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    total = max(len(words), 1)
    counts = Counter(words)
    result: Dict[str, float] = {}

    # One regex scan for all multi-word entries (avoids N separate findall calls)
    if _MULTI_WORD_FILLER_RE is not None:
        mc: Counter = Counter()
        for m in _MULTI_WORD_FILLER_RE.finditer(lower):
            entry = m.group()
            if entry in word_set:
                mc[entry] += 1
        for entry, n in mc.items():
            result[entry] = n / total
    else:
        for entry in word_set:
            if " " in entry:
                n = len(re.findall(re.escape(entry), lower))
                if n:
                    result[entry] = n / total

    for entry in word_set:
        if " " not in entry and entry in counts:
            result[entry] = counts[entry] / total

    return result


def _build_word_ngrams(words: List[str], n: int) -> Counter:
    """Build a Counter of word n-grams."""
    return Counter(
        " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
    )


def _extract_char_ngrams(
    text: str, n: int, top_k: int = 80
) -> Dict[str, float]:
    """Extract character n-gram relative frequencies."""
    if len(text) < n:
        return {}
    ngrams = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    total = max(sum(ngrams.values()), 1)
    return {ng: cnt / total for ng, cnt in ngrams.most_common(top_k)}


# ---------------------------------------------------------------------------
# POS bigrams (spaCy or regex fallback)
# ---------------------------------------------------------------------------

_POS_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(am|is|are|was|were|be|been|being)$"), "AUX"),
    (re.compile(r"^(will|would|could|should|may|might|shall|can|must)$"), "AUX"),
    (
        re.compile(
            r"^(the|a|an|this|that|these|those|my|your|his|her|its|our|their)$"
        ),
        "DET",
    ),
    (re.compile(r"^(and|but|or|nor|for|yet|so)$"), "CCONJ"),
    (
        re.compile(r"^(although|because|since|while|unless|if|when|as|than)$"),
        "SCONJ",
    ),
    (
        re.compile(
            r"^(in|on|at|to|for|of|with|by|from|about|into|through"
            r"|during|before|after)$"
        ),
        "ADP",
    ),
    (
        re.compile(
            r"^(i|you|he|she|it|we|they|me|him|her|us|them"
            r"|myself|yourself|himself|herself|itself)$"
        ),
        "PRON",
    ),
    (re.compile(r"\w+ly$"), "ADV"),
    (re.compile(r"\w+(ing|ed)$"), "VERB"),
    (re.compile(r"\w+(tion|ness|ment|ity|ism|age|ance|ence|ship)$"), "NOUN"),
    (re.compile(r"\w+(ful|less|ous|ive|able|ible|al|ic)$"), "ADJ"),
    (re.compile(r"^\d+$"), "NUM"),
    (re.compile(r"^[^\w\s]+$"), "PUNCT"),
]


def _pos_tag_fallback(text: str) -> List[str]:
    """Regex-based POS tagging when spaCy is unavailable."""
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
    tags: List[str] = []
    for tok in tokens:
        matched = False
        for pattern, tag in _POS_RULES:
            if pattern.fullmatch(tok):
                tags.append(tag)
                matched = True
                break
        if not matched:
            tags.append("NOUN")
    return tags


def _extract_pos_bigrams(
    doc_or_text: Any = None,
    raw_text: Optional[str] = None,
    top_k: int = 40,
) -> Tuple[Dict[str, float], str]:
    """
    Extract POS-tag bigram frequencies.

    Accepts either a pre-parsed spaCy Doc (preferred — avoids double parsing)
    or falls back to regex-based POS tagging on raw_text.
    """
    if doc_or_text is not None and hasattr(doc_or_text, "sents"):
        # spaCy Doc
        tags = [tok.pos_ for tok in doc_or_text if not tok.is_space]
        source = "spacy"
    elif raw_text is not None:
        tags = _pos_tag_fallback(raw_text)
        source = "regex_fallback"
    else:
        return {}, "none"

    if len(tags) < 2:
        return {}, source

    total = max(len(tags) - 1, 1)
    bigrams = Counter(
        f"{tags[i]} {tags[i + 1]}" for i in range(len(tags) - 1)
    )
    return {bg: cnt / total for bg, cnt in bigrams.most_common(top_k)}, source


# ---------------------------------------------------------------------------
# Feature vector layout
# ---------------------------------------------------------------------------

_SCALAR_FEATURES: Tuple[str, ...] = (
    # Lexical (4)
    "vocabulary_richness",
    "avg_word_length",
    "rare_word_ratio",
    "hapax_legomena_ratio",
    # Punctuation (6)
    "comma_rate",
    "semicolon_rate",
    "exclamation_rate",
    "question_rate",
    "ellipsis_rate",
    "dash_rate",
    # Sentence structure (3)
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_paragraph_length",
    # Syntactic — v2.1 (4)
    "avg_dep_distance",
    "max_dep_distance",
    "avg_tree_depth",
    "complex_sentence_ratio",
    # Burstiness — v2.1 (1)
    "burstiness_score",
)

_TOP_FUNCTION_WORDS: Tuple[str, ...] = tuple(sorted(FUNCTION_WORDS))
_TOP_FILLER_WORDS: Tuple[str, ...] = tuple(sorted(FILLER_WORDS))
_TOP_TRANSITIONS: Tuple[str, ...] = tuple(sorted(TRANSITION_WORDS))

_N_BIGRAMS = 30
_N_TRIGRAMS = 20
_N_CHAR4 = 30
_N_CHAR5 = 20
_N_POS_BG = 15

_STAT_VECTOR_SIZE: int = (
    len(_SCALAR_FEATURES)
    + len(_TOP_FUNCTION_WORDS)
    + len(_TOP_FILLER_WORDS)
    + len(_TOP_TRANSITIONS)
    + _N_BIGRAMS
    + _N_TRIGRAMS
    + _N_CHAR4
    + _N_CHAR5
    + _N_POS_BG
)

# ===== CANONICAL EXPORT — use this everywhere, never a magic number =====
VECTOR_DIM: int = _STAT_VECTOR_SIZE

# Features used for per-feature explainability in compare()
_EXPLAINABLE: Tuple[str, ...] = (
    "vocabulary_richness",
    "avg_word_length",
    "rare_word_ratio",
    "hapax_legomena_ratio",
    "comma_rate",
    "semicolon_rate",
    "exclamation_rate",
    "question_rate",
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_dep_distance",
    "avg_tree_depth",
    "complex_sentence_ratio",
    "burstiness_score",
)


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

# Declarative schema: (source_attr, ordered_keys | None, count)
_VECTOR_SCHEMA: List[Tuple[str, Optional[Tuple[str, ...]], int]] = [
    # fixed-key segments
    ("function_word_frequencies", _TOP_FUNCTION_WORDS, len(_TOP_FUNCTION_WORDS)),
    ("filler_word_frequencies", _TOP_FILLER_WORDS, len(_TOP_FILLER_WORDS)),
    ("transition_frequencies", _TOP_TRANSITIONS, len(_TOP_TRANSITIONS)),
    # variable-key segments (top-N by insertion order)
    ("bigram_frequencies", None, _N_BIGRAMS),
    ("trigram_frequencies", None, _N_TRIGRAMS),
    ("char_4gram_frequencies", None, _N_CHAR4),
    ("char_5gram_frequencies", None, _N_CHAR5),
    ("pos_bigram_frequencies", None, _N_POS_BG),
]


def _build_raw_stat_vector(profile: StyleProfile) -> np.ndarray:
    """
    Build the un-normalised statistical feature vector.

    Returns a numpy array of exactly VECTOR_DIM elements.
    """
    v: List[float] = []

    # Scalars
    for feat in _SCALAR_FEATURES:
        v.append(float(getattr(profile, feat, 0.0)))

    # Dict segments
    for attr_name, keys, count in _VECTOR_SCHEMA:
        freq_dict: Dict[str, float] = getattr(profile, attr_name, {})
        if keys is not None:
            # Fixed-key: look up each canonical key
            for k in keys:
                v.append(freq_dict.get(k, 0.0))
        else:
            # Variable-key: take first `count` values, pad remainder
            vals = list(freq_dict.values())[:count]
            v.extend(vals)
            v.extend([0.0] * (count - len(vals)))

    arr = np.array(v[:VECTOR_DIM], dtype=np.float64)
    if arr.size < VECTOR_DIM:
        arr = np.pad(arr, (0, VECTOR_DIM - arr.size))

    assert len(arr) == VECTOR_DIM, (
        f"Vector length {len(arr)} != VECTOR_DIM {VECTOR_DIM}"
    )
    return arr



# ---------------------------------------------------------------------------
# Core profile builder (pure function)
# ---------------------------------------------------------------------------


def _compute_profile_data(
    author_id: str,
    texts: List[str],
    embedding_provider: Optional[EmbeddingProvider] = None,
    nlp: Any = None,
) -> StyleProfile:
    """
    Pure computation of a StyleProfile.

    Does NOT register anywhere — safe to call from compare() / sliding window.
    The `nlp` parameter receives a spaCy Language instance; when provided,
    the text is parsed ONCE and the Doc is shared by both POS-bigram and
    syntactic-feature extractors.
    """
    combined = " ".join(texts)
    sentences = _split_sentences(combined)
    paragraphs = [p.strip() for p in combined.split("\n\n") if p.strip()]
    words = re.findall(r"\b\w+\b", combined.lower())
    total_words = max(len(words), 1)

    # Parse spaCy doc ONCE
    doc = nlp(combined) if nlp is not None else None

    lex = _extract_lexical_features(combined)
    punct = _extract_punctuation_features(combined, len(sentences))
    sent = _extract_sentence_features(combined)
    syn = _extract_syntactic_features(doc)
    burst = _compute_burstiness(combined)
    pos_bg, pos_src = _extract_pos_bigrams(
        doc_or_text=doc, raw_text=combined
    )

    avg_paragraph_length = (
        float(np.mean([len(p.split()) for p in paragraphs]))
        if paragraphs
        else 0.0
    )

    fw_freq = _word_frequencies(combined, FUNCTION_WORDS)
    filler_freq = _mixed_frequencies(combined, FILLER_WORDS)
    trans_freq = _word_frequencies(combined, TRANSITION_WORDS)

    # Single pass for both bigrams and trigrams (avoids two O(n) sweeps)
    bigrams: Counter = Counter()
    trigrams: Counter = Counter()
    _wn = len(words)
    for _i in range(_wn):
        if _i + 1 < _wn:
            bigrams[words[_i] + " " + words[_i + 1]] += 1
        if _i + 2 < _wn:
            trigrams[words[_i] + " " + words[_i + 1] + " " + words[_i + 2]] += 1
    char4 = _extract_char_ngrams(combined, n=4, top_k=80)
    char5 = _extract_char_ngrams(combined, n=5, top_k=60)

    # Optional embedding
    emb_vec = np.zeros(0)
    if embedding_provider is not None:
        try:
            emb_vec = np.array(
                embedding_provider.embed(combined), dtype=np.float64
            )
        except Exception as exc:
            logger.warning("EmbeddingProvider.embed() failed: %s", exc)

    profile = StyleProfile(
        author_id=author_id,
        sample_count=len(texts),
        total_words=len(words),
        created_at=datetime.now().isoformat(),
        vocabulary_richness=lex["vocabulary_richness"],
        avg_word_length=lex["avg_word_length"],
        rare_word_ratio=lex["rare_word_ratio"],
        hapax_legomena_ratio=lex["hapax_legomena_ratio"],
        comma_rate=punct["comma_rate"],
        semicolon_rate=punct["semicolon_rate"],
        exclamation_rate=punct["exclamation_rate"],
        question_rate=punct["question_rate"],
        ellipsis_rate=punct["ellipsis_rate"],
        dash_rate=punct["dash_rate"],
        avg_sentence_length=sent["avg_sentence_length"],
        sentence_length_variance=sent["sentence_length_variance"],
        avg_paragraph_length=avg_paragraph_length,
        avg_dep_distance=syn["avg_dep_distance"],
        max_dep_distance=syn["max_dep_distance"],
        avg_tree_depth=syn["avg_tree_depth"],
        complex_sentence_ratio=syn["complex_sentence_ratio"],
        burstiness_score=burst,
        function_word_frequencies=fw_freq,
        bigram_frequencies={
            bg: cnt / total_words for bg, cnt in bigrams.most_common(50)
        },
        trigram_frequencies={
            tg: cnt / total_words for tg, cnt in trigrams.most_common(30)
        },
        filler_word_frequencies=filler_freq,
        transition_frequencies=trans_freq,
        char_4gram_frequencies=char4,
        char_5gram_frequencies=char5,
        pos_bigram_frequencies=pos_bg,
        pos_bigrams_source=pos_src,
        signature_words=_find_signature_words(words),
        embedding_vector=emb_vec,
    )
    profile.feature_vector = _build_raw_stat_vector(profile)
    return profile


def _find_signature_words(
    words: List[str], top_n: int = 20, min_freq: int = 3
) -> List[str]:
    """Identify distinctive vocabulary for the author."""
    counts = Counter(words)
    candidates = [
        (w, c)
        for w, c in counts.items()
        if w not in _STOP_WORDS and c >= min_freq and len(w) > 3
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in candidates[:top_n]]


# ---------------------------------------------------------------------------
# Syntactic distance helper
# ---------------------------------------------------------------------------

_SYNTACTIC_FEATURES: Tuple[str, ...] = (
    "avg_dep_distance",
    "max_dep_distance",
    "avg_tree_depth",
    "complex_sentence_ratio",
)


def _syntactic_distance(
    profile: StyleProfile, candidate: StyleProfile
) -> float:
    """Normalised L1 distance on syntactic scalar features (0=same, 1=max)."""
    pv = np.array(
        [getattr(profile, f, 0.0) for f in _SYNTACTIC_FEATURES], dtype=float
    )
    cv = np.array(
        [getattr(candidate, f, 0.0) for f in _SYNTACTIC_FEATURES], dtype=float
    )
    # Use max of both to avoid near-zero denominator inflating distance
    # when spaCy features are missing on one side.
    denom = np.maximum(np.abs(pv), np.abs(cv)) + 1e-9
    return float(np.mean(np.abs(pv - cv) / denom))


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class StylometricProfiler:
    """
    Builds and compares writing-style fingerprints for individual authors.

    Parameters
    ----------
    nlp : optional spaCy Language model
        Injected NLP pipeline. When None, the module-level _NLP fallback
        is used (or None if spaCy is unavailable, activating regex mode).
        Pass a mock or alternative model for testing.
    """

    def __init__(self, nlp: Any = None) -> None:
        self._profiles: Dict[str, StyleProfile] = {}

        if nlp is not None:
            self._nlp = nlp
        elif _SPACY_AVAILABLE and _NLP is not None:
            self._nlp = _NLP
        else:
            self._nlp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vectorize(self, text: str) -> np.ndarray:
        """
        Return the raw statistical feature vector for a single text.

        This is the **public API** for external vectorization (e.g.
        mass_vectorizer). Returns a numpy array of exactly VECTOR_DIM
        elements. No normalisation is applied (raw frequencies / rates).

        Parameters
        ----------
        text : str
            Input text to profile.

        Returns
        -------
        np.ndarray
            1-D float64 array, length == VECTOR_DIM.
        """
        if not text or len(text.strip()) < 20:
            return np.zeros(VECTOR_DIM, dtype=np.float64)

        profile = _compute_profile_data(
            "_vectorize_", [text], nlp=self._nlp
        )
        vec = profile.feature_vector
        assert len(vec) == VECTOR_DIM, (
            f"vectorize produced {len(vec)} dims, expected {VECTOR_DIM}"
        )
        return vec

    def compute_stats(self, text: str) -> Dict[str, float]:
        """
        Return human-readable statistical features for a single text.

        This replaces the old distilgpt2-based StatisticalFeatureExtractor.
        All metrics are computed on CPU with zero GPU overhead.

        The returned dict is compatible with ForensicReportGenerator's
        ``_bridge_stats_from_result`` and the evaluator's stat display.

        Keys returned
        -------------
        burstiness, lexical_diversity, avg_sentence_length,
        sentence_length_variance, avg_word_length, vocabulary_richness,
        hapax_legomena_ratio, rare_word_ratio, comma_rate,
        avg_dep_distance, complex_sentence_ratio, burstiness_score.
        """
        if not text or len(text.strip()) < 20:
            return {
                "burstiness": 0.0,
                "lexical_diversity": 0.0,
                "avg_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "avg_word_length": 0.0,
                "vocabulary_richness": 0.0,
                "hapax_legomena_ratio": 0.0,
                "rare_word_ratio": 0.0,
                "comma_rate": 0.0,
                "avg_dep_distance": 0.0,
                "complex_sentence_ratio": 0.0,
            }

        profile = _compute_profile_data(
            "_stats_", [text], nlp=self._nlp
        )
        # lexical_diversity = simple TTR (unique/total); vocabulary_richness = MATTR
        _words = re.findall(r"\b\w+\b", text.lower())
        simple_ttr = len(set(_words)) / max(len(_words), 1)
        return {
            "burstiness": profile.burstiness_score,
            "lexical_diversity": simple_ttr,
            "avg_sentence_length": profile.avg_sentence_length,
            "sentence_length_variance": profile.sentence_length_variance,
            "avg_word_length": profile.avg_word_length,
            "vocabulary_richness": profile.vocabulary_richness,
            "hapax_legomena_ratio": profile.hapax_legomena_ratio,
            "rare_word_ratio": profile.rare_word_ratio,
            "comma_rate": profile.comma_rate,
            "avg_dep_distance": profile.avg_dep_distance,
            "complex_sentence_ratio": profile.complex_sentence_ratio,
        }

