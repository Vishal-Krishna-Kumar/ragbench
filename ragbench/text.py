"""Text normalisation shared by every retriever, so scores are comparable."""

import re

# A compact English stop-word list. Kept short on purpose: aggressive stop-word
# removal hurts short technical queries ("how to scale", "what is rag").
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lower-case, split on non-alphanumerics, optionally drop stop-words.

    Hyphenated / dotted terms such as ``gpt-4`` or ``node.js`` are kept as one token.
    """
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
