"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_STOP_WORDS = {
    "する", "ある", "いる", "なる", "れる", "られる", "こと", "もの",
    "これ", "それ", "あれ", "ここ", "そこ", "ため", "よう", "そう",
    "さん", "ん", "の", "です", "ます", "ない", "という", "思う",
    "the", "and", "that", "this", "with", "from", "have", "you", "are",
}


@dataclass(frozen=True)
class AnalysisSettings:
    sample_rate: int = 22050
    frame_length: int = 2048
    hop_length: int = 512
    peak_threshold: float = 0.05
    minimum_peak_interval_sec: float = 0.20
    language: str | None = "ja"
    whisper_model: str = "tiny"
    beam_size: int = 5
    requested_cluster_count: int | None = None
    max_cluster_count: int = 8
    max_words: int = 200
    cooccurrence_window: int = 5
    cooccurrence_minimum: int = 2
    highlight_threshold: float = 62.0
    highlight_window_sec: float = 10.0
    highlight_step_sec: float = 2.0
    min_chapter_sec: float = 60.0
    fast_mode: bool = True
    enabled: frozenset[str] = field(default_factory=lambda: frozenset({
        "acoustic", "transcription", "emotion", "summary", "wordcloud",
        "clustering", "keywords", "cooccurrence", "topics", "highlights",
        "semantic_search",
    }))
    stop_words: frozenset[str] = field(
        default_factory=lambda: frozenset(DEFAULT_STOP_WORDS)
    )


MAX_UPLOAD_MB = 500
PIPELINE_VERSION = "1.0.0"
