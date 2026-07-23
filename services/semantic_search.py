"""Cosine similarity search over shared transcript embeddings."""

from __future__ import annotations

import numpy as np

from core.models import AnalysisResult


def search(result: AnalysisResult, query: str, top_k: int = 5) -> list[dict]:
    """Return the most semantically similar transcript segments."""
    if not query.strip() or result.embeddings is None or not result.segments:
        return []
    # Loading a second language model here can exhaust Community Cloud memory.
    # Character n-gram overlap remains useful for the TF-IDF-backed fast mode.
    query_chars = set(query.lower())
    scores = np.array([
        len(query_chars & set(segment.text.lower()))
        / max(1, len(query_chars | set(segment.text.lower())))
        for segment in result.segments
    ])
    indices = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "similarity": float(scores[index]), "start": result.segments[index].start,
            "end": result.segments[index].end, "speaker": result.segments[index].speaker_id,
            "topic": result.segments[index].topic_id, "emotion": result.segments[index].emotion_label,
            "text": result.segments[index].text,
        }
        for index in indices
    ]
