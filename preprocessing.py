"""
src/preprocessing.py
────────────────────────────────────────────────────────────────────
BrandPulse AI – Twitter Sentiment Analysis
Text Preprocessing Pipeline v2.0

Author  : BrandPulse AI – Senior NLP Engineering Team
Version : 2.0.0

Pipeline stages (in order)
──────────────────────────
 1. Input coercion & NaN handling
 2. HTML entity decoding        (&amp; → &, &lt; → <, &#39; → ')
 3. Unicode normalisation       (NFKD → ASCII)
 4. Lowercasing
 5. Contraction expansion       (preserves negations: don't → do not)
 6. URL removal                 (http / https / www)
 7. Handle removal              (@username)
 8. Hashtag retention           (#delay → delay)
 9. Number removal
10. Special-character removal   (retains [a-z] and space)
11. Whitespace normalisation
12. Stop-word removal (with negation retention) + lemmatisation
      ↳ primary  : spaCy  en_core_web_sm
      ↳ fallback : NLTK   WordNetLemmatizer + stopwords

Key improvements over v1.0
───────────────────────────
• Contraction expansion preserves critical negation semantics
  ("can't" → "can not", "won't" → "will not", "n't" → " not")
• Negation words (not, no, never, nor, neither, cannot, without) are
  explicitly kept even when they appear in the stop-word list, preventing
  the loss of sentiment-reversing tokens.
• NFKD normalisation resolves typographic ligatures (ﬁ → fi) that NFD misses.
• transform_batch() uses spaCy nlp.pipe() for vectorised throughput.

Usage (single string)
──────────────────────
    from src.preprocessing import TextPreprocessor
    tp = TextPreprocessor()
    tp.clean("@AmericanAir can't believe you #cancelled my flight! http://t.co/x")
    # → 'american air not believe cancel flight'

Usage (pandas DataFrame / Series)
───────────────────────────────────
    import pandas as pd
    from src.preprocessing import preprocess_dataframe

    df = pd.read_csv("data/Tweets.csv")
    df = preprocess_dataframe(df, text_col="text", out_col="clean_text")
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 1. COMPILED REGEX PATTERNS  (module-level, compiled once)
# ══════════════════════════════════════════════════════════════════

_RE_URL         = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_HANDLE      = re.compile(r"@\w+")
_RE_HASHTAG     = re.compile(r"#(\w+)")           # capture group → keep word
_RE_NUMBERS     = re.compile(r"\d+")
_RE_SPECIAL     = re.compile(r"[^a-z\s]")         # retain only lowercase letters + space
_RE_WHITESPACE  = re.compile(r"\s+")
# Residual HTML entity fragments that survive html.unescape (e.g. malformed &entity)
_RE_HTML_ENTITY = re.compile(r"&[#a-zA-Z0-9]+;")


# ══════════════════════════════════════════════════════════════════
# 2. CONTRACTION & NEGATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════

# Common contraction → expanded form.  Applied post-lowercase.
# Expansion is ordered longest-match first to avoid partial replacements.
_CONTRACTIONS: dict[str, str] = {
    "won't":   "will not",
    "can't":   "can not",
    "cannot":  "can not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "hadn't":  "had not",
    "hasn't":  "has not",
    "haven't": "have not",
    "didn't":  "did not",
    "doesn't": "does not",
    "don't":   "do not",
    "isn't":   "is not",
    "aren't":  "are not",
    "wasn't":  "was not",
    "weren't": "were not",
    "mustn't": "must not",
    "needn't": "need not",
    "shan't":  "shall not",
    "n't":     " not",         # generic suffix catch-all (applied last)
    "i'm":     "i am",
    "i've":    "i have",
    "i'll":    "i will",
    "i'd":     "i would",
    "it's":    "it is",
    "he's":    "he is",
    "she's":   "she is",
    "there's": "there is",
    "that's":  "that is",
    "they're": "they are",
    "we're":   "we are",
    "you're":  "you are",
    "they've": "they have",
    "we've":   "we have",
    "you've":  "you have",
    "they'll": "they will",
    "we'll":   "we will",
    "you'll":  "you will",
    "they'd":  "they would",
    "we'd":    "we would",
    "you'd":   "you would",
    "let's":   "let us",
    "that'll": "that will",
}
# Pre-sort by length descending for greedy longest-match substitution
_CONTRACTIONS_SORTED = sorted(_CONTRACTIONS.items(), key=lambda x: -len(x[0]))

# Sentiment-critical negation words that must NEVER be dropped as stop words
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "nor", "neither", "cannot", "without",
    "nobody", "nothing", "nowhere", "hardly", "barely", "scarcely",
})


# ══════════════════════════════════════════════════════════════════
# 3. PURE NORMALIZATION UTILITIES  (stateless functions)
# ══════════════════════════════════════════════════════════════════

def decode_html_entities(text: str) -> str:
    """Decode standard and numeric HTML entities.

    Uses the stdlib ``html.unescape()`` then strips any residual entity
    fragments that could not be decoded (e.g. malformed ``&entity``).

    Parameters
    ----------
    text : str
        Raw input string potentially containing HTML entities.

    Returns
    -------
    str
        Text with entities decoded or stripped.

    Examples
    --------
    >>> decode_html_entities("I love you &amp; hate delays &lt;3")
    'I love you & hate delays <3'
    >>> decode_html_entities("&quot;amazing&quot; service &#39;today&#39;")
    '"amazing" service \'today\''
    """
    text = html.unescape(text)
    return _RE_HTML_ENTITY.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Decompose unicode characters (NFKD) and drop non-ASCII bytes.

    NFKD is preferred over NFD because it also resolves typographic
    ligatures (e.g. ``ﬁ`` → ``fi``) and compatibility forms.

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Pure-ASCII string with accents and non-representable characters
        removed.

    Examples
    --------
    >>> normalize_unicode("café déjà vu ﬁne")
    'cafe deja vu fine'
    """
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("utf-8", "ignore")


def expand_contractions(text: str) -> str:
    """Expand English contractions to preserve negation semantics.

    Negations are the most important class of contractions for sentiment
    analysis: "can't" → "can not", "don't" → "do not".  Expansion is
    applied longest-match-first to avoid partial substitutions.

    Parameters
    ----------
    text : str
        Lowercased input string.

    Returns
    -------
    str
        String with contractions expanded.

    Examples
    --------
    >>> expand_contractions("i can't believe they won't refund me")
    'i can not believe they will not refund me'
    >>> expand_contractions("it's not bad, don't you think?")
    'it is not bad, do not you think?'
    """
    for contraction, expansion in _CONTRACTIONS_SORTED:
        text = text.replace(contraction, expansion)
    return text


def clean_text_regex(text: Optional[object]) -> str:
    """Apply all regex-based cleaning stages to a single value.

    This is a pure, stateless function covering stages 1–11 of the
    pipeline.  It does NOT perform lemmatisation or stop-word removal
    (those require a loaded NLP backend — see :class:`TextPreprocessor`).

    Parameters
    ----------
    text : str | None | float | object
        Raw text value.  ``None``, ``float('nan')``, and non-string
        values are safely coerced to ``''``.

    Returns
    -------
    str
        Cleaned, lowercased text ready for NLP processing.

    Examples
    --------
    >>> clean_text_regex("@Delta can't believe #cancelled! http://t.co/x &amp; no reply 😤")
    'delta can not believe cancelled no reply'
    """
    # ── Stage 1: coerce to str ─────────────────────────────────────
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            if np.isnan(float(text)):  # type: ignore[arg-type]
                return ""
        except (TypeError, ValueError):
            pass
        text = str(text)
    if not text.strip():
        return ""

    # ── Stage 2: HTML entity decoding ─────────────────────────────
    text = decode_html_entities(text)

    # ── Stage 3: Unicode normalisation (NFKD → ASCII) ─────────────
    text = normalize_unicode(text)

    # ── Stage 4: Lowercase ────────────────────────────────────────
    text = text.lower()

    # ── Stage 5: Contraction expansion ────────────────────────────
    text = expand_contractions(text)

    # ── Stage 6: URL removal ──────────────────────────────────────
    text = _RE_URL.sub(" ", text)

    # ── Stage 7: Handle removal (@username) ───────────────────────
    text = _RE_HANDLE.sub(" ", text)

    # ── Stage 8: Hashtag retention (#word → word) ─────────────────
    text = _RE_HASHTAG.sub(r"\1", text)

    # ── Stage 9: Number removal ───────────────────────────────────
    text = _RE_NUMBERS.sub(" ", text)

    # ── Stage 10: Special-character removal ───────────────────────
    text = _RE_SPECIAL.sub(" ", text)

    # ── Stage 11: Whitespace normalisation ────────────────────────
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


# ══════════════════════════════════════════════════════════════════
# 4. NLP BACKEND LOADERS
# ══════════════════════════════════════════════════════════════════

def _load_spacy(model: str = "en_core_web_sm"):
    """Load a spaCy model with ``parser`` and ``ner`` disabled.

    Returns ``None`` if spaCy or the model is unavailable so the NLTK
    fallback can be activated automatically.
    """
    try:
        import spacy  # noqa: PLC0415
        nlp = spacy.load(model, disable=["parser", "ner"])
        # Un-mark negation words so they are never discarded as stop words
        for word in _NEGATION_WORDS:
            if word in nlp.vocab:
                nlp.vocab[word].is_stop = False
        logger.info("Loaded spaCy model: %s", model)
        return nlp
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy model '%s' unavailable (%s). Using NLTK fallback.", model, exc)
        return None


def _load_nltk_fallback() -> tuple:
    """Download required NLTK corpora and return ``(lemmatizer, stop_words)``."""
    import nltk  # noqa: PLC0415
    from nltk.stem import WordNetLemmatizer  # noqa: PLC0415

    for corpus in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, quiet=True)

    from nltk.corpus import stopwords  # noqa: PLC0415

    lemmatizer = WordNetLemmatizer()
    # Exclude negation words so they survive stop-word filtering
    stop_words = frozenset(stopwords.words("english")) - _NEGATION_WORDS
    logger.info("NLTK fallback lemmatizer initialised.")
    return lemmatizer, stop_words


# ══════════════════════════════════════════════════════════════════
# 5. TweetPreprocessor  (stateful — holds NLP backend references)
# ══════════════════════════════════════════════════════════════════

class TweetPreprocessor:
    """Robust tweet text preprocessor using spaCy (primary) with NLTK fallback.

    Incorporates contraction expansion and negation-word retention to
    preserve sentiment-critical tokens that naive stop-word removal
    would discard (e.g. "not", "never", "no").

    Parameters
    ----------
    use_spacy : bool
        Attempt to load the spaCy ``en_core_web_sm`` model.
        Falls back to NLTK automatically if the model is missing.
    spacy_model : str
        spaCy model name.  Default ``'en_core_web_sm'``.
    min_token_length : int
        Discard lemmatised tokens shorter than this.  Default ``2``.
    extra_stopwords : list[str] | None
        Additional domain-specific stop words to add on top of the
        default English list.

    Attributes
    ----------
    backend : str
        ``'spacy'`` or ``'nltk'``.

    Examples
    --------
    >>> tp = TweetPreprocessor()
    >>> tp.clean("@Delta can't believe #cancelled my flight! http://t.co/x &amp; no reply")
    'delta not believe cancel flight no reply'
    """

    def __init__(
        self,
        use_spacy: bool = True,
        spacy_model: str = "en_core_web_sm",
        min_token_length: int = 2,
        extra_stopwords: Optional[List[str]] = None,
    ) -> None:
        self._min_length = min_token_length
        self._extra_sw: frozenset[str] = (
            frozenset(extra_stopwords) - _NEGATION_WORDS
            if extra_stopwords
            else frozenset()
        )

        self._nlp = _load_spacy(spacy_model) if use_spacy else None
        self._lemmatizer = None
        self._stop_words: frozenset[str] = frozenset()

        if self._nlp is not None:
            self.backend = "spacy"
            for word in self._extra_sw:
                if word in self._nlp.vocab:
                    self._nlp.vocab[word].is_stop = True
        else:
            self.backend = "nltk"
            self._lemmatizer, base_sw = _load_nltk_fallback()
            self._stop_words = (base_sw | self._extra_sw) - _NEGATION_WORDS

        logger.info("TweetPreprocessor initialised (backend=%s).", self.backend)

    # ── Lemmatisation helpers ─────────────────────────────────────

    def _lemmatize_spacy(self, doc) -> str:
        """Extract lemmas from a spaCy Doc, retaining negation words."""
        tokens = [
            token.lemma_.strip()
            for token in doc
            if (
                (not token.is_stop or token.text in _NEGATION_WORDS)
                and not token.is_punct
                and not token.is_space
                and len(token.lemma_.strip()) >= self._min_length
            )
        ]
        return " ".join(tokens)

    def _lemmatize_nltk(self, text: str) -> str:
        """Lemmatise with NLTK WordNetLemmatizer, retaining negation words."""
        tokens = [
            self._lemmatizer.lemmatize(w)
            for w in text.split()
            if (w not in self._stop_words or w in _NEGATION_WORDS)
            and len(w) >= self._min_length
        ]
        return " ".join(tokens)

    # ── Public API ────────────────────────────────────────────────

    def clean(self, text: object) -> str:
        """Clean and lemmatise a single tweet string.

        Parameters
        ----------
        text : object
            Raw tweet text.  ``None`` / ``NaN`` / non-string values are
            safely handled.

        Returns
        -------
        str
            Cleaned, lower-cased, lemmatised text with stop words removed
            (negation words retained).  Returns ``''`` for empty/missing input.

        Examples
        --------
        >>> tp = TweetPreprocessor()
        >>> tp.clean("@SouthwestAir I can't believe my flight is #cancelled again!")
        'southwest air not believe flight cancel'
        >>> tp.clean(None)
        ''
        """
        cleaned = clean_text_regex(text)
        if not cleaned:
            return ""

        if self._nlp is not None:
            doc = self._nlp(cleaned)
            return self._lemmatize_spacy(doc)

        return self._lemmatize_nltk(cleaned)

    # Alias for backward compatibility with v1.0 callers
    transform_single = clean

    def transform_batch(
        self,
        texts: Iterable[Optional[str]],
        batch_size: int = 500,
    ) -> List[str]:
        """Preprocess an iterable of tweet strings efficiently.

        Uses ``spaCy nlp.pipe()`` for vectorised batch throughput when
        the spaCy backend is active.  Falls back to sequential
        ``transform_single()`` calls for the NLTK backend.

        Parameters
        ----------
        texts : iterable of str | None
            Raw tweet strings (may include ``None`` / ``NaN``).
        batch_size : int
            Number of documents per spaCy pipe batch.  Default ``500``.

        Returns
        -------
        list[str]
            List of cleaned strings aligned to the input order.

        Examples
        --------
        >>> tp = TweetPreprocessor()
        >>> tp.transform_batch(["Great flight!", "Worst airline ever.", None])
        ['great flight', 'bad airline', '']
        """
        # Stage 1–11: regex cleaning for all inputs (fast, vectorisable)
        logger.info("Starting batch regex cleaning...")
        cleaned_texts: List[str] = [clean_text_regex(t) for t in texts]

        if self._nlp is not None:
            # spaCy batch path — nlp.pipe() processes in C-speed batches
            logger.info("Running batched spaCy lemmatization (batch_size=%d)...", batch_size)
            results: List[str] = []
            for doc in self._nlp.pipe(cleaned_texts, batch_size=batch_size):
                results.append(self._lemmatize_spacy(doc))
            return results

        # NLTK fallback — sequential per-token processing
        logger.info("Running NLTK lemmatization (sequential fallback)...")
        return [self._lemmatize_nltk(t) for t in cleaned_texts]

    def transform(self, series: pd.Series, batch_size: int = 500) -> pd.Series:
        """Apply the pipeline to a pandas ``Series`` and return a new ``Series``.

        Parameters
        ----------
        series : pd.Series
            Series of raw tweet text strings.
        batch_size : int
            spaCy pipe batch size.  Default ``500``.

        Returns
        -------
        pd.Series
            Cleaned Series aligned to the original index.

        Examples
        --------
        >>> import pandas as pd
        >>> s = pd.Series(["@Delta #worst flight", None, "I love this airline"])
        >>> tp.transform(s)
        0       worst flight
        1
        2      love airline
        dtype: object
        """
        results = self.transform_batch(list(series), batch_size=batch_size)
        return pd.Series(results, index=series.index, dtype="object")


# ── Backward-compatibility alias ──────────────────────────────────
#    v1.0 code used `TextPreprocessor`; v2.0 uses `TweetPreprocessor`.
#    Both names are supported indefinitely.
TextPreprocessor = TweetPreprocessor


# ══════════════════════════════════════════════════════════════════
# 6. MODULE-LEVEL CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def preprocess_series(
    series: pd.Series,
    preprocessor: Optional[TweetPreprocessor] = None,
    **preprocessor_kwargs,
) -> pd.Series:
    """Convenience wrapper: clean a ``pd.Series`` of tweet texts.

    Parameters
    ----------
    series : pd.Series
        Series of raw tweet strings.
    preprocessor : TweetPreprocessor | None
        Pre-initialised preprocessor.  A new one is created with default
        settings if ``None``.
    **preprocessor_kwargs
        Forwarded to ``TweetPreprocessor.__init__`` when creating a new instance.

    Returns
    -------
    pd.Series
        Cleaned Series with the same index as *series*.

    Examples
    --------
    >>> cleaned = preprocess_series(df["text"])
    """
    tp = preprocessor or TweetPreprocessor(**preprocessor_kwargs)
    return tp.transform(series)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    out_col: str = "clean_text",
    preprocessor: Optional[TweetPreprocessor] = None,
    **preprocessor_kwargs,
) -> pd.DataFrame:
    """Apply the preprocessing pipeline to a DataFrame column in-place copy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (copied, not mutated in-place).
    text_col : str
        Column containing raw tweet text.  Default ``'text'``.
    out_col : str
        Output column name for cleaned text.  Default ``'clean_text'``.
    preprocessor : TweetPreprocessor | None
        Pre-initialised preprocessor.  Created with defaults if ``None``.
    **preprocessor_kwargs
        Forwarded to ``TweetPreprocessor.__init__``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a new *out_col* column containing cleaned text.

    Raises
    ------
    KeyError
        If *text_col* is not in *df*.

    Examples
    --------
    >>> df = pd.read_csv("data/Tweets.csv")
    >>> df = preprocess_dataframe(df, text_col="text", out_col="clean_text")
    >>> df[["text", "clean_text"]].head()
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found in DataFrame. "
            f"Available: {list(df.columns)}"
        )

    tp = preprocessor or TweetPreprocessor(**preprocessor_kwargs)
    result = df.copy()
    logger.info(
        "Preprocessing column '%s' -> '%s' for %d rows ...",
        text_col, out_col, len(result),
    )
    result[out_col] = tp.transform(result[text_col])
    logger.info("Preprocessing complete.")
    return result
