import numpy as np

from core.models import TranscriptSegment
from services.insight_analyzers import build_cooccurrence, create_chapters, detect_highlights


def test_cooccurrence_counts_pairs() -> None:
    rows = [{"normalized_word": word} for word in ["音声", "分析", "音声", "分析"]]
    nodes, edges = build_cooccurrence(rows, window_size=2, minimum=1)
    assert {node["word"] for node in nodes} == {"音声", "分析"}
    assert edges[0]["cooccurrence_count"] >= 1


def test_chapters_with_empty_input() -> None:
    assert create_chapters([], {}) == ("", [], [])


def test_highlights_empty_input() -> None:
    assert detect_highlights([], None) == []
