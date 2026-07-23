"""OpenAI-assisted conversational stop-word review and local feedback application."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

from core.models import AnalysisResult
from services.insight_analyzers import build_cooccurrence


DEFAULT_MODEL = "gpt-5.6-sol"
SYSTEM_INSTRUCTIONS = """\
あなたは日本語の音声分析用語彙を整理する編集者です。
入力は、1本の会話・会議・動画から抽出した全単語と出現回数です。

目的:
- 普通の会話で非常によく現れ、内容分析・トピック分析・ワードクラウドの識別力を
  下げる語だけを除外候補として提案する。
- フィラー、相づち、談話標識、汎用的すぎる動詞・形容詞・副詞を主に検討する。

厳守:
- 固有名詞、製品名、人物名、地名、組織名、専門用語、話題を特徴づける語は除外しない。
- 高頻度という理由だけでは除外しない。
- 文脈がなく判断できない語は残す。
- 入力に存在しない語を提案しない。
- 日本語以外の語も同じ基準で評価する。
- 理由は日本語で簡潔に説明する。
"""


@dataclass(frozen=True)
class StopwordSuggestion:
    word: str
    category: str
    reason: str
    confidence: float


@dataclass(frozen=True)
class StopwordReview:
    suggestions: list[StopwordSuggestion]
    summary: str
    request_id: str | None
    model: str
    attempts: int


def vocabulary_fingerprint(
    frequencies: dict[str, int], model: str, user_guidance: str
) -> str:
    """Create a stable cache key without exposing vocabulary contents."""
    payload = json.dumps(
        {"frequencies": frequencies, "model": model, "guidance": user_guidance},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_models():
    """Declare structured-output models lazily so local analysis does not require Pydantic."""
    from pydantic import Field, create_model

    suggestion_payload = create_model(
        "StopwordSuggestionPayload",
        word=(str, ...),
        category=(str, ...),
        reason=(str, ...),
        confidence=(float, Field(ge=0.0, le=1.0)),
    )
    ReviewPayload = create_model(
        "StopwordReviewPayload",
        suggestions=(list[suggestion_payload], ...),
        summary=(str, ...),
    )
    return ReviewPayload


def review_stopwords(
    frequencies: dict[str, int],
    *,
    api_key: str | None = None,
    model: str | None = None,
    user_guidance: str = "",
    max_attempts: int = 3,
    timeout: float = 60.0,
    on_retry: Callable[[int, str], None] | None = None,
    client: Any = None,
) -> StopwordReview:
    """Ask OpenAI for structured stop-word candidates and retry invalid results.

    The OpenAI SDK performs transport retries. This function additionally retries
    schema/semantic validation failures, such as an empty response or words that
    were not present in the supplied vocabulary.
    """
    if not frequencies:
        raise ValueError("評価する単語がありません。")
    if max_attempts < 1:
        raise ValueError("max_attemptsは1以上で指定してください。")

    selected_model = model or os.getenv("OPENAI_STOPWORD_MODEL", DEFAULT_MODEL)
    if client is None:
        from openai import OpenAI

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEYが未設定です。Streamlit Secretsまたは環境変数に設定してください。"
            )
        client = OpenAI(api_key=resolved_key, max_retries=2, timeout=timeout)

    payload_model = _response_models()
    vocabulary = set(frequencies)
    vocabulary_rows = [
        {"word": word, "count": count} for word, count in frequencies.items()
    ]
    input_text = json.dumps(
        {
            "vocabulary": vocabulary_rows,
            "user_guidance": user_guidance.strip() or "追加指示なし",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    last_error: Exception | None = None
    attempts_made = 0
    for attempt in range(1, max_attempts + 1):
        attempts_made = attempt
        try:
            attempt_input = input_text
            if last_error is not None:
                attempt_input += (
                    "\n\n前回の結果はアプリ検証に失敗しました。次の問題を修正して、"
                    "元の語彙だけから改めて回答してください: "
                    f"{type(last_error).__name__}: {str(last_error)[:400]}"
                )
            response = client.responses.parse(
                model=selected_model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=attempt_input,
                text_format=payload_model,
                reasoning={"effort": "low"},
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("構造化された結果が返されませんでした。")
            valid = []
            seen = set()
            for item in parsed.suggestions:
                word = item.word.strip()
                if word not in vocabulary:
                    raise ValueError(f"入力にない単語が返されました: {word}")
                if word in seen:
                    continue
                seen.add(word)
                valid.append(StopwordSuggestion(
                    word=word,
                    category=item.category.strip() or "その他",
                    reason=item.reason.strip(),
                    confidence=float(item.confidence),
                ))
            if not valid:
                raise ValueError("除外候補が空でした。")
            return StopwordReview(
                suggestions=valid,
                summary=parsed.summary.strip(),
                request_id=getattr(response, "_request_id", None),
                model=selected_model,
                attempts=attempt,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts or not _is_retryable(exc):
                break
            if on_retry:
                on_retry(attempt + 1, f"{type(exc).__name__}: {exc}")
            time.sleep(min(4.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(
        f"OpenAIによる除外候補生成に{attempts_made}回で成功しませんでした: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _is_retryable(exc: Exception) -> bool:
    """Retry transient SDK failures and natural-language/schema validation errors."""
    if isinstance(exc, (ValueError, TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429} or (isinstance(status, int) and status >= 500):
        return True
    return exc.__class__.__name__ in {
        "APIConnectionError", "APITimeoutError", "RateLimitError",
        "InternalServerError", "ValidationError",
    }


def apply_stopword_feedback(
    result: AnalysisResult,
    selected_words: set[str],
    *,
    cooccurrence_window: int = 5,
    cooccurrence_minimum: int = 2,
) -> None:
    """Apply reviewed words locally without rerunning audio or embedding models."""
    clean_words = {word.strip() for word in selected_words if word.strip()}
    result.applied_stop_words = sorted(set(result.applied_stop_words) | clean_words)
    removed = [
        row for row in result.keyword_rows
        if row.get("normalized_word") in clean_words
    ]
    result.removed_keyword_rows.extend(removed)
    result.keyword_rows = [
        row for row in result.keyword_rows
        if row.get("normalized_word") not in clean_words
    ]
    _refresh_text_derivatives(
        result,
        cooccurrence_window=cooccurrence_window,
        cooccurrence_minimum=cooccurrence_minimum,
    )


def restore_stopword_feedback(
    result: AnalysisResult,
    *,
    cooccurrence_window: int = 5,
    cooccurrence_minimum: int = 2,
) -> None:
    """Undo all user-applied AI suggestions and rebuild text derivatives."""
    result.keyword_rows.extend(result.removed_keyword_rows)
    result.keyword_rows.sort(key=lambda row: (
        float(row.get("start") or 0.0),
        int(row.get("segment_id") or 0),
    ))
    result.removed_keyword_rows = []
    result.applied_stop_words = []
    _refresh_text_derivatives(
        result,
        cooccurrence_window=cooccurrence_window,
        cooccurrence_minimum=cooccurrence_minimum,
    )


def _refresh_text_derivatives(
    result: AnalysisResult,
    *,
    cooccurrence_window: int,
    cooccurrence_minimum: int,
) -> None:
    """Rebuild values derived from the filtered keyword timeline."""
    counts = Counter(row["normalized_word"] for row in result.keyword_rows)
    result.word_frequencies = dict(counts.most_common())
    result.cooccurrence_nodes, result.cooccurrence_edges = build_cooccurrence(
        result.keyword_rows,
        window_size=cooccurrence_window,
        minimum=cooccurrence_minimum,
    )

    cluster_terms: dict[int, Counter[str]] = {}
    segment_clusters = {segment.id: segment.cluster_id for segment in result.segments}
    for row in result.keyword_rows:
        cluster_id = segment_clusters.get(row.get("segment_id"))
        if cluster_id is None:
            continue
        cluster_terms.setdefault(cluster_id, Counter())[row["normalized_word"]] += 1
    result.cluster_keywords = {
        cluster_id: [word for word, _ in counter.most_common(5)]
        for cluster_id, counter in cluster_terms.items()
    }
    result.topic_keywords = result.cluster_keywords.copy()
