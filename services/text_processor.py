"""Japanese-first term extraction, embeddings, clusters and topics."""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np

from core.config import AnalysisSettings
from core.models import TranscriptSegment


@lru_cache(maxsize=1)
def load_embedding_model():
    """Load the multilingual embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def extract_terms(text: str, stop_words: set[str] | frozenset[str]) -> list[str]:
    """Extract content words with Janome, falling back to Unicode tokens."""
    try:
        from janome.tokenizer import Tokenizer

        terms: list[str] = []
        for token in Tokenizer().tokenize(text):
            surface = token.surface.strip()
            base = token.base_form if token.base_form != "*" else surface
            pos = token.part_of_speech.split(",")[0]
            if (
                pos in {"名詞", "動詞", "形容詞", "副詞"}
                and len(surface) > 1
                and surface not in stop_words
                and base not in stop_words
                and not re.fullmatch(r"[\W_]+|\d+", surface)
            ):
                terms.append(base)
        return terms
    except ImportError:
        return [
            token.lower() for token in re.findall(r"[A-Za-z]{2,}|[一-龥ぁ-んァ-ヶー]{2,}", text)
            if token.lower() not in stop_words
        ]


def analyze_text(
    segments: list[TranscriptSegment],
    settings: AnalysisSettings,
) -> tuple[dict[str, int], list[dict], np.ndarray, np.ndarray, dict[int, list[str]]]:
    """Create frequencies, keyword timestamps, shared embeddings and clusters."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    full_text = "\n".join(segment.text for segment in segments)
    frequencies = dict(Counter(extract_terms(full_text, settings.stop_words)).most_common())
    keyword_rows: list[dict] = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                normalized = extract_terms(word.text, settings.stop_words)
                if normalized:
                    word.normalized = normalized[0]
                    keyword_rows.append({
                        "word": word.text, "normalized_word": normalized[0],
                        "start": word.start, "end": word.end, "segment_id": segment.id,
                        "speaker_id": segment.speaker_id, "confidence": word.probability,
                    })
        else:
            for term in extract_terms(segment.text, settings.stop_words):
                keyword_rows.append({
                    "word": term, "normalized_word": term, "start": segment.start,
                    "end": segment.end, "segment_id": segment.id,
                    "speaker_id": segment.speaker_id, "confidence": None,
                })

    texts = [segment.text for segment in segments]
    try:
        model = load_embedding_model()
        embeddings = np.asarray(model.encode(texts, normalize_embeddings=True))
    except Exception:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=1024
        )
        embeddings = normalize(vectorizer.fit_transform(texts)).toarray()

    count = len(texts)
    if count < 3:
        cluster_count = 1
    elif settings.requested_cluster_count:
        cluster_count = max(2, min(settings.requested_cluster_count, count - 1))
    else:
        scored = []
        for k in range(2, min(settings.max_cluster_count, count - 1) + 1):
            labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
            scored.append((silhouette_score(embeddings, labels, metric="cosine"), k))
        cluster_count = max(scored)[1] if scored else 1
    labels = (
        np.zeros(count, dtype=int) if cluster_count == 1
        else KMeans(n_clusters=cluster_count, random_state=42, n_init=10).fit_predict(embeddings)
    )
    keywords: dict[int, list[str]] = {}
    for cluster_id in sorted(set(labels)):
        cluster_text = " ".join(text for text, label in zip(texts, labels) if label == cluster_id)
        keywords[int(cluster_id)] = [
            word for word, _ in Counter(
                extract_terms(cluster_text, settings.stop_words)
            ).most_common(5)
        ]
    for segment, label in zip(segments, labels):
        segment.cluster_id = int(label)
        segment.topic_id = int(label)
    if count >= 2 and embeddings.shape[1] >= 2:
        points = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        points = np.column_stack((np.arange(count), np.zeros(count)))
    return frequencies, keyword_rows, embeddings, points, keywords
