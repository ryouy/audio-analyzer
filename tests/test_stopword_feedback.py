from types import SimpleNamespace

import pytest

from core.models import AnalysisResult, TranscriptSegment
from services.stopword_feedback import (
    apply_stopword_feedback,
    restore_stopword_feedback,
    review_stopwords,
)


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        payload = kwargs["text_format"]
        if self.calls == 1:
            parsed = payload(
                suggestions=[{
                    "word": "入力にない語", "category": "フィラー",
                    "reason": "テスト", "confidence": 0.9,
                }],
                summary="invalid",
            )
        else:
            parsed = payload(
                suggestions=[{
                    "word": "やっぱり", "category": "談話標識",
                    "reason": "一般会話で多い", "confidence": 0.8,
                }],
                summary="再試行成功",
            )
        return SimpleNamespace(output_parsed=parsed, _request_id="req_test")


def test_review_retries_invalid_natural_language_result(monkeypatch) -> None:
    monkeypatch.setattr("services.stopword_feedback.time.sleep", lambda _: None)
    responses = FakeResponses()
    review = review_stopwords(
        {"やっぱり": 10, "音響": 4},
        client=SimpleNamespace(responses=responses),
        max_attempts=3,
    )
    assert responses.calls == 2
    assert review.suggestions[0].word == "やっぱり"
    assert review.attempts == 2


def test_apply_and_restore_feedback() -> None:
    result = AnalysisResult("job", "source", "audio.wav", "now")
    result.segments = [TranscriptSegment(0, 0, 1, "音響 やっぱり", cluster_id=0)]
    result.keyword_rows = [
        {"normalized_word": "音響", "segment_id": 0, "start": 0},
        {"normalized_word": "やっぱり", "segment_id": 0, "start": 0.5},
    ]
    result.word_frequencies = {"音響": 1, "やっぱり": 1}
    apply_stopword_feedback(result, {"やっぱり"}, cooccurrence_minimum=1)
    assert result.word_frequencies == {"音響": 1}
    restore_stopword_feedback(result, cooccurrence_minimum=1)
    assert result.word_frequencies == {"音響": 1, "やっぱり": 1}


def test_authentication_error_is_not_retried() -> None:
    class AuthenticationError(Exception):
        status_code = 401

    class FailingResponses:
        calls = 0

        def parse(self, **_kwargs):
            self.calls += 1
            raise AuthenticationError("invalid key")

    responses = FailingResponses()
    with pytest.raises(RuntimeError, match="1回"):
        review_stopwords(
            {"音響": 1},
            client=SimpleNamespace(responses=responses),
            max_attempts=3,
        )
    assert responses.calls == 1
