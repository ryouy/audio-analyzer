"""Fault-tolerant orchestration for all analysis services."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from core.config import AnalysisSettings
from core.models import AnalysisResult
from services.acoustic_analyzer import analyze_audio
from services.diarizer import diarize_audio
from services.insight_analyzers import (
    analyze_emotions,
    build_cooccurrence,
    create_chapters,
    detect_highlights,
)
from services.text_processor import analyze_text
from services.transcriber import transcribe_audio

ProgressCallback = Callable[[float, str], None]


def run_pipeline(
    wav_path: Path,
    output_dir: Path,
    source_name: str,
    job_id: str,
    settings: AnalysisSettings,
    progress: ProgressCallback | None = None,
    hf_token: str | None = None,
) -> AnalysisResult:
    """Run enabled stages independently and retain partial results on failure."""
    callback = progress or (lambda _value, _message: None)
    result = AnalysisResult(
        job_id=job_id, source_name=source_name, audio_path=str(wav_path),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    def stage(name: str, value: float, message: str, action) -> None:
        callback(value, message)
        try:
            action()
            result.completed_stages.append(name)
        except Exception as exc:
            result.errors[name] = f"{type(exc).__name__}: {exc}"

    if "acoustic" in settings.enabled:
        def acoustic() -> None:
            features, harmonic, percussive = analyze_audio(wav_path, output_dir, settings)
            result.features, result.harmonic_path, result.percussive_path = (
                features, str(harmonic), str(percussive)
            )
        stage("音響分析", 0.16, "音響特徴を抽出しています", acoustic)

    if "transcription" in settings.enabled:
        def transcription() -> None:
            segments, language, probability = transcribe_audio(wav_path, settings)
            result.segments, result.language, result.language_probability = (
                segments, language, probability
            )
        stage("文字起こし", 0.36, "Whisperで文字起こししています", transcription)

    if result.segments and "diarization" in settings.enabled:
        stage(
            "話者分離", 0.45, "話者を推定しています",
            lambda: diarize_audio(wav_path, result.segments, hf_token),
        )

    if result.segments and "emotion" in settings.enabled:
        stage(
            "感情分析", 0.52, "感情とarousalを推定しています",
            lambda: analyze_emotions(result.segments, result.features),
        )

    text_consumers = {
        "wordcloud", "clustering", "keywords", "cooccurrence", "topics",
        "summary", "highlights", "semantic_search",
    }
    if result.segments and settings.enabled.intersection(text_consumers):
        def text_analysis() -> None:
            frequencies, rows, embeddings, points, keywords = analyze_text(result.segments, settings)
            result.word_frequencies = frequencies
            result.keyword_rows = rows
            result.embeddings = embeddings
            result.points_2d = points
            result.cluster_keywords = keywords
            result.topic_keywords = keywords.copy()
        stage("テキスト分析", 0.66, "単語・埋め込み・クラスタを分析しています", text_analysis)

    if result.segments and "cooccurrence" in settings.enabled:
        def cooccurrence() -> None:
            result.cooccurrence_nodes, result.cooccurrence_edges = build_cooccurrence(
                result.keyword_rows, settings.cooccurrence_window,
                settings.cooccurrence_minimum,
            )
        stage("共起分析", 0.75, "共起ネットワークを構築しています", cooccurrence)

    if result.segments and "summary" in settings.enabled:
        def chapters() -> None:
            result.summary, result.key_points, result.chapters = create_chapters(
                result.segments, result.topic_keywords, settings.min_chapter_sec
            )
        stage("要約・章立て", 0.83, "要約とチャプターを生成しています", chapters)

    if result.segments and "highlights" in settings.enabled:
        def highlights() -> None:
            result.highlights = detect_highlights(
                result.segments, result.features, settings.highlight_threshold
            )
        stage("ハイライト", 0.91, "盛り上がり候補を検出しています", highlights)

    callback(1.0, "解析が完了しました")
    return result
