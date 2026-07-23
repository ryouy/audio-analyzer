import numpy as np

from core.models import AnalysisResult, TranscriptSegment
from services.semantic_search import search


def test_empty_query_returns_no_results() -> None:
    result = AnalysisResult("job", "source", "audio.wav", "now")
    result.segments = [TranscriptSegment(0, 0, 1, "テスト")]
    result.embeddings = np.array([[1.0, 0.0]])
    assert search(result, "") == []
