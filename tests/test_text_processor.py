from core.config import DEFAULT_STOP_WORDS
from services.text_processor import extract_terms


def test_extract_terms_removes_stop_words() -> None:
    terms = extract_terms("これは 音声 分析 をする アプリです", DEFAULT_STOP_WORDS)
    assert "する" not in terms
    assert "音声" in terms


def test_extract_terms_empty() -> None:
    assert extract_terms("", DEFAULT_STOP_WORDS) == []
