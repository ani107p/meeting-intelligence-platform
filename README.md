"""
Lightweight local embedding backend.

In production this module would call OpenAI/Cohere/HF embedding endpoints.
For this offline demo we build a deterministic dense embedding using
TF-IDF + truncated SVD (LSA) fitted on the corpus -- this gives genuine
semantic clustering behavior (synonymy/co-occurrence) without any network
calls or model downloads, so hybrid search is exercising real vector
similarity, not a stub.

Swap `LocalEmbedder` for `OpenAIEmbeddings` / `AnthropicEmbeddings` /
whatever your infra uses -- the interface (`embed(texts) -> np.ndarray`)
is what the rest of the pipeline depends on.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class LocalEmbedder:
    def __init__(self, n_components: int = 128):
        self.vectorizer = TfidfVectorizer(
            max_features=20000, stop_words="english", ngram_range=(1, 2)
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, corpus: list[str]) -> "LocalEmbedder":
        tfidf = self.vectorizer.fit_transform(corpus)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LocalEmbedder must be fit() on a corpus before embed()")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        # L2 normalize for cosine similarity via dot product
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        return dense / norms
