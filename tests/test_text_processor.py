from core.config import AnalysisSettings, DEFAULT_STOP_WORDS
from core.models import TranscriptSegment
from services.text_processor import analyze_text, extract_terms


def test_extract_terms_removes_stop_words() -> None:
    terms = extract_terms("これは 音声 分析 をする アプリです", DEFAULT_STOP_WORDS)
    assert "する" not in terms
    assert "音声" in terms


def test_extract_terms_empty() -> None:
    assert extract_terms("", DEFAULT_STOP_WORDS) == []


def test_fast_mode_does_not_load_sentence_transformer(monkeypatch) -> None:
    segments = [
        TranscriptSegment(id=0, start=0, end=1, text="音声解析のテストです"),
        TranscriptSegment(id=1, start=1, end=2, text="会話内容を分類します"),
    ]

    def unexpected_model_load():
        raise AssertionError("fast mode must not load the embedding model")

    monkeypatch.setattr(
        "services.text_processor.load_embedding_model", unexpected_model_load
    )
    _, _, embeddings, _, _ = analyze_text(
        segments, AnalysisSettings(fast_mode=True)
    )

    assert embeddings.shape[0] == len(segments)
