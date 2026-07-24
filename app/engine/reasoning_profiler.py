"""
reasoning_profiler.py
=====================
Zero-Resource Feature Extractor for Reasoning-Model Detection (v5.0)

Extracts a fixed-dimension numerical vector φ(x) from raw text for
downstream classification (XGBoost, PyTorch MLP, ModernBERT head, etc.).
Contains **NO** classification logic, thresholds, or heuristic scores —
pure statistical feature extraction following Late Fusion architecture.

Architecture:
    ParsedText (immutable DTO, computed once)
      └─► StylometricExtractor   → indices 0–5
      └─► DiscourseExtractor     → indices 6–9
      └─► ReasoningMarkerExtractor → indices 10–12
      └─► StructuralExtractor    → indices 13–14

Target: PyTorch / Polars pipelines processing 4M+ documents.
Complexity: O(n) per document where n = token count.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ============================================================================
# VECTOR SCHEMA — single source of truth for feature → tensor index mapping
# ============================================================================

REASONING_VECTOR_DIM: int = 15

_VECTOR_SCHEMA: tuple[tuple[str, int], ...] = (
    # ── Stylometric / Lexical (0–5) ───────────────────────────────────────
    ("type_token_ratio",            0),   # TTR = |V| / N
    ("mean_sentence_length",        1),   # μ(sentence word counts)
    ("std_sentence_length",         2),   # σ(sentence word counts), ddof=1
    ("mean_word_length",            3),   # Σ len(token) / N
    ("punctuation_ratio",           4),   # |punct chars| / |all chars|
    ("stopword_ratio",              5),   # |stopword tokens| / N
    # ── Discourse Connector Densities (6–9) ───────────────────────────────
    ("consequence_density",         6),   # matches / sentence_count
    ("causal_density",              7),
    ("contrast_density",            8),
    ("sequence_density",            9),
    # ── Reasoning-Specific Markers (10–12) ────────────────────────────────
    ("backtracking_density",       10),   # self-correction / sentence_count
    ("cot_scaffold_density",       11),   # CoT scaffolding / sentence_count
    ("intuition_leap_density",     12),   # heuristic leaps / sentence_count
    # ── Structural / Information-Theoretic (13–14) ────────────────────────
    ("paragraph_length_cv",        13),   # CV = σ/μ of paragraph word counts
    ("word_entropy_normalised",    14),   # H(words) / log₂(|V|)
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in _VECTOR_SCHEMA)

# Compile-time assertion: schema covers every index exactly once
assert len(_VECTOR_SCHEMA) == REASONING_VECTOR_DIM
assert sorted(idx for _, idx in _VECTOR_SCHEMA) == list(range(REASONING_VECTOR_DIM))


# ============================================================================
# MODULE-LEVEL COMPILED REGEXES  (word-boundary guarded, re.VERBOSE)
# ============================================================================

# Sentence boundary: split after terminal punctuation + whitespace.
# Avoids splitting on decimals (3.14), abbreviations (U.S.), ellipses.
_SENTENCE_RE: re.Pattern[str] = re.compile(
    r"""
    (?<= [.!?] )   # lookbehind: sentence-ending punctuation
    \s+             # one or more whitespace chars
    (?= [A-Z\d"'] )  # lookahead: likely sentence start
    """,
    re.VERBOSE,
)

# ── Discourse connectors ──────────────────────────────────────────────────
# Multi-word patterns listed FIRST so they match before single-word fallbacks.
# Every alternative is \b-guarded to prevent the substring bug
# (e.g. "also" matching "so", "asset" matching "as").

_CONSEQUENCE_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: as \s+ a \s+ result
         | consequently
         | accordingly
         | therefore
         | thus
         | hence
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CAUSAL_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: due \s+ to
         | owing \s+ to
         | given \s+ that
         | because
         | since
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CONTRAST_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: on \s+ the \s+ other \s+ hand
         | nevertheless
         | nonetheless
         | however
         | although
         | despite
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEQUENCE_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: first   (?:ly)?
         | second  (?:ly)?
         | third   (?:ly)?
         | finally
         | subsequently
         | initially
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Reasoning markers ─────────────────────────────────────────────────────
# Backtracking / self-correction signatures from o1, DeepSeek-R1, QwQ traces

_BACKTRACK_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: wait \s* [,.\-!…]+ \s* wait          # "wait...wait"
         | but  \s+ wait
         | let  (?:'s | \s+ us | \s+ me) \s+ re-?evaluate
         | on   \s+ (?:the \s+)? second \s+ thought
         | (?:no | actually) [,\s]+ that (?:'s | \s+ is) \s+
           (?:wrong | incorrect | not \s+ (?:right | correct))
         | i \s+ made \s+ (?:an? \s+)? (?:error | mistake)
         | let  \s+ me \s+ reconsider
         | this \s+ (?:reasoning | approach) \s+ is \s+
           (?:not \s+)? correct
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# CoT scaffolding: structural framing typical of step-by-step models

_COT_SCAFFOLD_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: let (?:'s | \s+ me) \s+ think
         | step \s+ by \s+ step
         | let (?:'s | \s+ me) \s+ break \s+ (?:this | it) \s+ down
         | (?:first | now) \s+ (?:i | we) \s+ need \s+ to
         | working  \s+ through
         | to  \s+ solve  \s+ this
         | the \s+ key    \s+ insight
         | step \s+ \d+
         | from \s+ this \s+ we \s+ can \s+ conclude
         | it \s+ follows \s+ that
         | reasoning \s+ through
         | analyzing \s+ this
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Intuition/heuristic leap markers (human-like shortcuts)

_INTUITION_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: obviously
         | clearly
         | of    \s+ course
         | naturally
         | it    \s+ goes \s+ without \s+ saying
         | needless \s+ to \s+ say
         | surely
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Non-word characters (punctuation counter)
_PUNCTUATION_RE: re.Pattern[str] = re.compile(r"[^\w\s]")


# ============================================================================
# ENGLISH STOPWORDS  (compact inline set — zero external dependency)
# ============================================================================

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "not", "no", "nor", "so", "as", "it", "its", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "when", "where", "how", "why", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "about", "above", "after", "again", "also", "any", "because",
    "before", "between", "during", "into", "only", "over", "same", "then",
    "there", "through", "under", "until", "up", "while",
})

# Division guard constant
_EPS: float = 1e-9


# ============================================================================
# PARSED TEXT DTO  (immutable, computed EXACTLY ONCE, injected everywhere)
# ============================================================================

@dataclass(frozen=True, slots=True)
class ParsedText:
    """
    Immutable pre-processed representation of a document.

    Every downstream extractor receives this object rather than raw text,
    guaranteeing that tokenisation, lowercasing, and set construction
    happen at most once per document.
    """

    raw: str
    lower: str
    tokens: tuple[str, ...]
    token_count: int
    char_count: int
    sentences: tuple[str, ...]
    sentence_count: int
    sentence_word_counts: np.ndarray          # dtype=float64
    paragraphs: tuple[str, ...]
    paragraph_count: int
    paragraph_word_counts: np.ndarray          # dtype=float64
    word_freq: Counter[str]
    unique_token_count: int
    stopword_count: int
    punctuation_count: int

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_raw(cls, text: str) -> ParsedText:
        raw = text.strip()
        lower = raw.lower()

        # Tokens — single whitespace split on lowered text
        tokens = tuple(lower.split())
        token_count = len(tokens)
        char_count = max(len(raw), 1)  # guard against empty

        # Sentences — split on terminal punctuation + capital lookahead
        raw_sents = _SENTENCE_RE.split(raw)
        sentences = tuple(s.strip() for s in raw_sents if len(s.split()) >= 3)
        sentence_count = max(len(sentences), 1)
        sentence_word_counts = np.array(
            [len(s.split()) for s in sentences], dtype=np.float64
        ) if sentences else np.zeros(1, dtype=np.float64)

        # Paragraphs — double newline split
        raw_paras = raw.split("\n\n")
        paragraphs = tuple(p.strip() for p in raw_paras if p.strip())
        paragraph_count = len(paragraphs)
        paragraph_word_counts = np.array(
            [len(p.split()) for p in paragraphs], dtype=np.float64
        ) if paragraphs else np.zeros(1, dtype=np.float64)

        # Word frequency distribution
        word_freq: Counter[str] = Counter(tokens)
        unique_token_count = len(word_freq)

        # Stopwords — O(|stopwords|) lookup against the freq counter
        stopword_count = sum(word_freq[w] for w in _STOPWORDS if w in word_freq)

        # Punctuation characters
        punctuation_count = len(_PUNCTUATION_RE.findall(raw))

        return cls(
            raw=raw,
            lower=lower,
            tokens=tokens,
            token_count=token_count,
            char_count=char_count,
            sentences=sentences,
            sentence_count=sentence_count,
            sentence_word_counts=sentence_word_counts,
            paragraphs=paragraphs,
            paragraph_count=paragraph_count,
            paragraph_word_counts=paragraph_word_counts,
            word_freq=word_freq,
            unique_token_count=unique_token_count,
            stopword_count=stopword_count,
            punctuation_count=punctuation_count,
        )


# ============================================================================
# SUB-EXTRACTORS  (__slots__, stateless, write directly into the output vector)
# ============================================================================

class StylometricExtractor:
    """
    Indices 0–5: TTR, sentence length μ/σ, word length μ,
    punctuation ratio, stopword ratio.

    All formulas from Stylometric Detectability (SD) literature.
    """

    __slots__ = ()

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        tc = max(pt.token_count, 1)

        # [0] Type-Token Ratio: TTR = |V| / N
        vec[0] = pt.unique_token_count / tc

        # [1] Mean sentence length: μ
        vec[1] = float(pt.sentence_word_counts.mean())

        # [2] Sentence length standard deviation: σ (sample, ddof=1)
        if len(pt.sentence_word_counts) >= 2:
            vec[2] = float(pt.sentence_word_counts.std(ddof=1))
        else:
            vec[2] = 0.0

        # [3] Mean word length (chars per token)
        total_chars = sum(len(t) for t in pt.tokens)
        vec[3] = total_chars / tc

        # [4] Punctuation ratio: |punct| / |chars|
        vec[4] = pt.punctuation_count / pt.char_count

        # [5] Stopword ratio: |stopwords| / N
        vec[5] = pt.stopword_count / tc


class DiscourseExtractor:
    """
    Indices 6–9: consequence / causal / contrast / sequence connector
    densities normalised by sentence count.

    Uses compiled word-boundary regexes to prevent the substring bug
    (e.g. "also" ≠ "so", "asset" ≠ "as").
    """

    __slots__ = ()

    _PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
        (6, _CONSEQUENCE_RE),
        (7, _CAUSAL_RE),
        (8, _CONTRAST_RE),
        (9, _SEQUENCE_RE),
    )

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        sc = pt.sentence_count  # already ≥ 1 from ParsedText
        for idx, pattern in DiscourseExtractor._PATTERNS:
            vec[idx] = len(pattern.findall(pt.lower)) / sc


class ReasoningMarkerExtractor:
    """
    Indices 10–12: backtracking density, CoT scaffolding density,
    intuition leap density.

    Captures the three dominant reasoning signatures documented in
    o1 / DeepSeek-R1 / QwQ trace analysis.
    """

    __slots__ = ()

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        sc = pt.sentence_count
        vec[10] = len(_BACKTRACK_RE.findall(pt.lower)) / sc
        vec[11] = len(_COT_SCAFFOLD_RE.findall(pt.lower)) / sc
        vec[12] = len(_INTUITION_RE.findall(pt.lower)) / sc


class StructuralExtractor:
    """
    Indices 13–14: paragraph length coefficient of variation (CV = σ/μ),
    normalised Shannon entropy of word distribution.

    CV replaces the original's ``1 - var/500`` magic-number formula.
    Shannon entropy replaces all-pairs Jaccard (O(n²)) with O(n)
    information-theoretic redundancy measurement.
    """

    __slots__ = ()

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        # [13] Paragraph-length Coefficient of Variation: CV = σ / μ
        if pt.paragraph_count >= 2:
            mu = pt.paragraph_word_counts.mean()
            sigma = pt.paragraph_word_counts.std(ddof=1)
            vec[13] = sigma / max(mu, _EPS)
        else:
            vec[13] = 0.0

        # [14] Normalised Shannon entropy of the word distribution
        #      H_norm = H(words) / log₂(|V|)
        #      where H = −Σ p(t) log₂ p(t)
        #
        #      Value near 1.0 → high lexical diversity (human-like)
        #      Value near 0.0 → repetitive / low diversity (AI loop)
        if pt.unique_token_count >= 2:
            counts = np.fromiter(
                pt.word_freq.values(), dtype=np.float64, count=pt.unique_token_count,
            )
            probs = counts / counts.sum()
            entropy = -float(np.sum(probs * np.log2(probs + _EPS)))
            max_entropy = math.log2(pt.unique_token_count)
            vec[14] = entropy / max(max_entropy, _EPS)
        else:
            vec[14] = 0.0


# ============================================================================
# MAIN PROFILER — PUBLIC API
# ============================================================================

class ReasoningProfiler:
    """
    Zero-resource feature extractor for reasoning-model detection.

    Produces a fixed-dimension numpy vector φ(x) suitable for any
    downstream classifier. Contains NO classification or thresholding
    logic — pure Late Fusion feature extraction.

    Usage::

        profiler = ReasoningProfiler()

        # Single document
        vec = profiler.vectorize("Some text to analyse...")
        assert vec.shape == (15,)

        # Batch (for Polars / DataLoader integration)
        matrix = profiler.vectorize_batch(list_of_texts)
        assert matrix.shape == (len(list_of_texts), 15)

    Integration with forensic_reporter v3.2::

        vec = profiler.vectorize(text)
        feature_dict = dict(zip(FEATURE_NAMES, vec.tolist()))
        reporter.generate_report(
            text,
            detection_result=result,
            additional_analyses={"reasoning_profile": feature_dict},
        )
    """

    __slots__ = ("_extractors", "_min_tokens")

    def __init__(self, min_tokens: int = 20) -> None:
        self._min_tokens = min_tokens
        self._extractors: tuple[type, ...] = (
            StylometricExtractor,
            DiscourseExtractor,
            ReasoningMarkerExtractor,
            StructuralExtractor,
        )

    def vectorize(self, text: str) -> np.ndarray:
        """
        Extract a feature vector of shape ``(REASONING_VECTOR_DIM,)``
        from raw text.

        Returns a zero vector for empty / near-empty inputs
        (fewer than ``min_tokens`` whitespace-delimited tokens).

        All indices in the output are guaranteed to be written;
        allocation via ``np.empty`` is safe.
        """
        vec = np.empty(REASONING_VECTOR_DIM, dtype=np.float64)

        if not text or len(text.split()) < self._min_tokens:
            vec[:] = 0.0
            return vec

        pt = ParsedText.from_raw(text)

        for extractor_cls in self._extractors:
            extractor_cls.extract(pt, vec)

        return vec

    def vectorize_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Vectorize a batch of texts.

        Returns:
            np.ndarray of shape ``(len(texts), REASONING_VECTOR_DIM)``.

        Note:
            For maximum throughput in production, parallelise at the
            Polars / DataLoader level rather than here. This method
            is a convenience wrapper for sequential processing.
        """
        n = len(texts)
        out = np.empty((n, REASONING_VECTOR_DIM), dtype=np.float64)
        for i in range(n):
            out[i] = self.vectorize(texts[i])
        return out

    @staticmethod
    def schema() -> tuple[tuple[str, int], ...]:
        """Return the (feature_name, index) mapping."""
        return _VECTOR_SCHEMA

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        """Return ordered feature names matching tensor indices."""
        return FEATURE_NAMES

    @staticmethod
    def dim() -> int:
        """Return the dimensionality of the output vector."""
        return REASONING_VECTOR_DIM


# ============================================================================
# REASONING RISK CLASSIFIER — display/reporting layer
# ============================================================================
# [Relocated from forensic_reports.py] This heuristic classifier consumes the
# 15-dim vector from ReasoningProfiler above — it belongs beside its own
# extractor, matching the pattern PerplexityRiskClassifier/perplexity_profiler
# and HallucinationRiskClassifier/hallucination_profile already follow.
# forensic_reports.py re-exports this name for backward compatibility with
# any code still doing `from forensic_reports import ReasoningRiskClassifier`.

class ReasoningRiskClassifier:
    """
    Heuristic classifier for the 15-dim vector from ReasoningProfiler.
    Display/reporting layer — does NOT modify ReasoningProfiler itself.
    """

    _WEIGHTS = {
        "backtracking_density":    0.26,
        "cot_scaffold_density":    0.23,
        "consequence_density":     0.09,
        "causal_density":          0.07,
        "sequence_density":        0.07,
        "contrast_density":        0.05,
        "word_entropy_normalised": 0.07,
        "type_token_ratio":        0.05,
        "paragraph_length_cv":     0.03,
    }
    _INVERSE_WEIGHT = 0.08

    # EC-04 / 4.2-Bias-2: causal/consequence/contrast/sequence thresholds raised to
    # reduce false positives on formal academic text, which is inherently dense in
    # logical connectors. backtracking + cot_scaffold thresholds are unchanged —
    # those markers are unique to reasoning models, not academic prose.
    _THR = {
        "type_token_ratio":         (0.35, 0.72),
        "mean_sentence_length":     (12.0, 26.0),
        "std_sentence_length":      (4.0,  14.0),
        "mean_word_length":         (3.8,  5.8),
        "punctuation_ratio":        (0.02, 0.06),
        "stopword_ratio":           (0.30, 0.52),
        "consequence_density":      (0.03, 0.10),  # was (0.02, 0.06)
        "causal_density":           (0.03, 0.12),  # was (0.02, 0.07)
        "contrast_density":         (0.03, 0.09),  # was (0.02, 0.06)
        "sequence_density":         (0.02, 0.09),  # was (0.01, 0.05)
        "backtracking_density":     (0.01, 0.07),  # unchanged — reasoning-model unique
        "cot_scaffold_density":     (0.02, 0.10),  # unchanged — reasoning-model unique
        "intuition_leap_density":   (0.01, 0.04),
        "paragraph_length_cv":      (0.18, 0.55),
        "word_entropy_normalised":  (0.68, 0.90),
    }

    _HIGH   = 0.55
    _MEDIUM = 0.28

    _EXPL = {
        "backtracking_density": {
            "display": "Self-Correction Density", "group": "CoT & Self-Correction",
            "high":   "Very high self-correction density (value: {v:.6f}). Phrases such as 'wait', 'let me reconsider', 'actually that is incorrect', 'I made an error' appear with significant frequency. Strongest single marker of a reasoning-optimised model (o1, o3, DeepSeek-R1, QwQ). Standard models rarely produce this at detectable density.",
            "medium": "Moderate self-correction language (value: {v:.6f}). Could indicate a reasoning model on a moderate task, a standard model with 'think step by step' instructions, or a careful human author revising mid-composition.",
            "low":    "Minimal self-correction language (value: {v:.6f}). Consistent with standard autoregressive models (GPT-4o, Claude 3.x, Gemini 1.5) or typical human prose.",
        },
        "cot_scaffold_density": {
            "display": "Chain-of-Thought Scaffolding Density", "group": "CoT & Self-Correction",
            "high":   "Dense CoT scaffolding (value: {v:.6f}). 'step by step', 'let me think', 'working through this', 'step N:', 'from this we can conclude'. Characteristic of models trained with extended thinking budgets.",
            "medium": "Moderate CoT scaffolding (value: {v:.6f}). May reflect a reasoning model, a prompted standard model, or a methodical human author.",
            "low":    "Negligible CoT scaffolding (value: {v:.6f}). Typical of conversational AI or informal human text.",
        },
        "consequence_density": {
            "display": "Logical Consequence Connector Density", "group": "Logical Connectors",
            "high":   "Dense logical consequence connectors (value: {v:.6f}). 'therefore', 'thus', 'consequently', 'hence', 'accordingly'. Signals deductive reasoning chains.",
            "medium": "Moderate consequence language (value: {v:.6f}).",
            "low":    "Sparse consequence connectors (value: {v:.6f}). Narrative or descriptive prose.",
        },
        "causal_density": {
            "display": "Causal Connector Density", "group": "Logical Connectors",
            "high":   "High causal language density (value: {v:.6f}). 'because', 'due to', 'since', 'owing to', 'given that'. Reasoning models produce dense causal chains when constructing derivations.",
            "medium": "Moderate causal language (value: {v:.6f}).",
            "low":    "Low causal density (value: {v:.6f}). Narrative style predominates.",
        },
        "contrast_density": {
            "display": "Contrast Connector Density", "group": "Logical Connectors",
            "high":   "High contrast language (value: {v:.6f}). 'however', 'nevertheless', 'despite', 'although'. Signals dialectical reasoning.",
            "medium": "Moderate contrastive language (value: {v:.6f}).",
            "low":    "Sparse contrast markers (value: {v:.6f}). Monological style.",
        },
        "sequence_density": {
            "display": "Sequential Structure Density", "group": "Logical Connectors",
            "high":   "Heavy sequential framing (value: {v:.6f}). 'first', 'second', 'third', 'finally', 'subsequently'. Strongly characteristic of step-by-step reasoning model output.",
            "medium": "Moderate sequential structure (value: {v:.6f}).",
            "low":    "Non-sequential prose (value: {v:.6f}). No explicit step enumeration.",
        },
        "intuition_leap_density": {
            "display": "Intuitive Assertion Density [INVERSE SIGNAL]", "group": "Style Markers",
            "high":   "Frequent intuitive assertions (value: {v:.6f}). 'obviously', 'clearly', 'of course', 'naturally'. INVERSE signal: high density here is more consistent with human writing or standard AI — reasoning models prefer explicit derivation.",
            "medium": "Moderate intuitive language (value: {v:.6f}). Does not strongly indicate or rule out a reasoning model.",
            "low":    "Minimal intuitive leaps (value: {v:.6f}). Expected profile for reasoning models (o1, DeepSeek-R1, QwQ) that prefer explicit derivation over bare assertion.",
        },
        "type_token_ratio": {
            "display": "Vocabulary Diversity (TTR = |V|/N)", "group": "Lexical Quality",
            "high": "High lexical diversity TTR={v:.4f}. Rich, varied vocabulary.",
            "medium": "Moderate vocabulary diversity TTR={v:.4f}.",
            "low": "Low lexical diversity TTR={v:.4f}. Vocabulary repetition detected.",
        },
        "word_entropy_normalised": {
            "display": "Normalised Word Entropy H(words)/log₂(|V|)", "group": "Lexical Quality",
            "high":   "High normalised word entropy H_norm={v:.4f}. Word distribution spread broadly — rich, varied text.",
            "medium": "Moderate word entropy H_norm={v:.4f}.",
            "low":    "Low normalised entropy H_norm={v:.4f}. Concentrated, repetitive word distribution.",
        },
        "paragraph_length_cv": {
            "display": "Paragraph Length CV (σ/μ)", "group": "Structural Variety",
            "high":   "High paragraph length variability CV={v:.4f}. Reasoning models often produce structurally heterogeneous paragraphs — brief assertions alternating with extended derivations.",
            "medium": "Moderate paragraph length variation CV={v:.4f}.",
            "low":    "Highly uniform paragraph lengths CV={v:.4f}. Common in templated AI output.",
        },
        "mean_sentence_length": {
            "display": "Mean Sentence Length (words/sentence)", "group": "Stylometric",
            "high": "Long mean sentence length μ={v:.1f}. Complex, multi-clause constructions.",
            "medium": "Moderate sentence length μ={v:.1f}.",
            "low": "Short mean sentence length μ={v:.1f}. Terse, direct prose.",
        },
        "std_sentence_length": {
            "display": "Sentence Length Std. Deviation σ", "group": "Stylometric",
            "high": "High sentence length variance σ={v:.1f}.",
            "medium": "Moderate sentence length variation σ={v:.1f}.",
            "low": "Uniform sentence lengths σ={v:.1f}. Highly regular pattern.",
        },
        "mean_word_length": {
            "display": "Mean Word Length (chars/token)", "group": "Stylometric",
            "high": "Long average word length μ={v:.2f}. Dense technical vocabulary.",
            "medium": "Moderate word length μ={v:.2f}.",
            "low": "Short average word length μ={v:.2f}. Informal register.",
        },
        "punctuation_ratio": {
            "display": "Punctuation Density (punct/chars)", "group": "Stylometric",
            "high": "High punctuation density r={v:.4f}. Complex sentence structure.",
            "medium": "Moderate punctuation r={v:.4f}.",
            "low": "Sparse punctuation r={v:.4f}. Linear sentence structure.",
        },
        "stopword_ratio": {
            "display": "Stopword Ratio (stopwords/tokens)", "group": "Stylometric",
            "high": "High stopword density r={v:.4f}. Functional language dominates.",
            "medium": "Moderate stopword ratio r={v:.4f}.",
            "low": "Low stopword density r={v:.4f}. Content-dense, technical writing.",
        },
    }

    def classify(self, vec, feature_names):
        features = dict(zip(feature_names, vec.tolist()))
        score  = self._score(features)
        level  = self._level(score)
        return {
            "ai_score":        score,
            "risk_level":      level,
            "feature_details": self._feature_details(features),
            "group_scores":    self._group_scores(features),
            "top_signals":     self._top_signals(features),
            "interpretation":  self._interpretation(score, features),
        }

    def _norm(self, feat, val):
        thr = self._THR.get(feat, (0.0, 1.0))
        return min(1.0, val / max(thr[1], 1e-9))

    def _score(self, features):
        s = sum(w * self._norm(f, features.get(f, 0.0)) for f, w in self._WEIGHTS.items())
        inv = self._norm("intuition_leap_density", features.get("intuition_leap_density", 0.0))
        s += self._INVERSE_WEIGHT * max(0.0, 1.0 - inv)
        return round(min(1.0, max(0.0, s)), 4)

    def _level(self, score):
        if score >= self._HIGH:   return "HIGH — Reasoning Model"
        if score >= self._MEDIUM: return "MEDIUM — Possible Reasoning Model"
        return "LOW — Standard Model or Human"

    def _feat_level(self, feat, val):
        thr = self._THR.get(feat, (0.0, 1.0))
        if val >= thr[1]: return "high"
        if val >= thr[0]: return "medium"
        return "low"

    def _feature_details(self, features):
        details = {}
        for feat, val in features.items():
            em = self._EXPL.get(feat)
            if em is None: continue
            thr = self._THR.get(feat, (0.0, 1.0))
            lev = self._feat_level(feat, val)
            et = em.get(lev, "")
            details[feat] = {
                "display_name": em["display"], "group": em.get("group", "Other"),
                "value": round(val, 6), "level": lev,
                "explanation": et.format(v=val) if "{v" in et else et,
                "threshold_low": thr[0], "threshold_high": thr[1],
            }
        return details

    def _top_signals(self, features, k=5):
        scored = []
        for feat, w in self._WEIGHTS.items():
            val = features.get(feat, 0.0); norm = self._norm(feat, val)
            lev = self._feat_level(feat, val); em = self._EXPL.get(feat, {})
            et = em.get(lev, "")
            scored.append({"feature": feat, "display_name": em.get("display", feat),
                "group": em.get("group", ""), "raw_value": round(val, 6),
                "normalised": round(norm, 4), "weight": w, "level": lev,
                "explanation": (et.format(v=val) if "{v" in et else et)[:280]})
        iv = features.get("intuition_leap_density", 0.0)
        norm = self._norm("intuition_leap_density", iv); lev = self._feat_level("intuition_leap_density", iv)
        ie = self._EXPL.get("intuition_leap_density", {}); et = ie.get(lev, "")
        scored.append({"feature": "intuition_leap_density",
            "display_name": ie.get("display", "Intuitive Assertion Density"),
            "group": ie.get("group", "Style Markers"), "raw_value": round(iv, 6),
            "normalised": round(norm, 4), "weight": self._INVERSE_WEIGHT, "level": lev,
            "explanation": (et.format(v=iv) if "{v" in et else et)[:280]})
        scored.sort(key=lambda x: x["normalised"], reverse=True)
        return scored[:k]

    def _group_scores(self, features):
        n = lambda f: self._norm(f, features.get(f, 0.0))
        return {
            "CoT & Self-Correction": round(n("backtracking_density")*0.55 + n("cot_scaffold_density")*0.45, 4),
            "Logical Connectors":   round(n("consequence_density")*0.30 + n("causal_density")*0.25 + n("contrast_density")*0.20 + n("sequence_density")*0.25, 4),
            "Lexical Richness":     round(n("type_token_ratio")*0.50 + n("word_entropy_normalised")*0.50, 4),
            "Structural Variety":   round(n("paragraph_length_cv"), 4),
            "Intuitive Assertions (inverse)": round(max(0.0, 1.0 - n("intuition_leap_density")), 4),
        }

    def _interpretation(self, score, features):
        bt  = features.get("backtracking_density", 0.0)
        cot = features.get("cot_scaffold_density", 0.0)
        seq = features.get("sequence_density", 0.0)
        con = features.get("consequence_density", 0.0)
        ent = features.get("word_entropy_normalised", 0.0)
        inv = features.get("intuition_leap_density", 0.0)
        if score >= self._HIGH:
            parts = []
            if bt  >= self._THR["backtracking_density"][1]:  parts.append(f"self-correction (density={bt:.4f})")
            if cot >= self._THR["cot_scaffold_density"][1]:  parts.append(f"CoT scaffolding (density={cot:.4f})")
            if seq >= self._THR["sequence_density"][1]:       parts.append(f"step enumeration (density={seq:.4f})")
            sig = "; ".join(parts) if parts else f"combined score={score:.2f}"
            return (f"Strong reasoning-model signature (overall score={score:.2f}). Dominant signals: {sig}. "
                    f"Characteristic of o1, o3-mini, DeepSeek-R1, QwQ — trained via process reward models or "
                    f"MCTS-style search for explicit multi-step deliberation.")
        if score >= self._MEDIUM:
            return (f"Moderate reasoning-model indicators (overall score={score:.2f}). "
                    f"CoT scaffolding={cot:.4f}, consequence connectors={con:.4f}, word entropy={ent:.4f}. "
                    f"Compatible with a reasoning-capable model, a standard model with step-by-step "
                    f"system-prompt instructions, or a methodical human author.")
        return (f"Low reasoning-model indicators (overall score={score:.2f}). "
                f"Self-correction={bt:.4f}, CoT scaffolding={cot:.4f}, intuitive assertions={inv:.4f}. "
                f"Consistent with a standard autoregressive model without extended chain-of-thought "
                f"inference, or with natural human prose.")
