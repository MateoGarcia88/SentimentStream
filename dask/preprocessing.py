"""Shared text preprocessing for training, prediction, and the API."""
import re

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOPWORDS_ES = set(stopwords.words("spanish"))

_NON_LETTERS = re.compile(r"[^a-záéíóúüñ\s]")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = _NON_LETTERS.sub("", text)
    return " ".join(t for t in text.split() if t not in STOPWORDS_ES and len(t) > 2)
