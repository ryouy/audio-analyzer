"""Shared, serializable data models for the analysis pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TranscriptWord:
    start: float | None
    end: float | None
    text: str
    normalized: str | None = None
    probability: float | None = None


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    speaker_id: str | None = None
    emotion_label: str | None = None
    emotion_score: float | None = None
    valence: float | None = None
    arousal: float | None = None
    cluster_id: int | None = None
    topic_id: int | None = None
    highlight_score: float | None = None


@dataclass
class AudioFeatureSet:
    sample_rate: int
    duration: float
    tempo: float | None
    waveform: Any
    waveform_times: Any
    rms: Any
    rms_times: Any
    pitch: Any
    pitch_times: Any
    entropy: Any
    entropy_times: Any
    beat_times: Any
    peak_times: Any
    spectrogram_db: Any
    spectrogram_times: Any
    spectrogram_frequencies: Any
    fft_frequencies: Any
    fft_values: Any
    onset: Any = None
    onset_times: Any = None


@dataclass
class Chapter:
    id: int
    title: str
    summary: str
    start: float
    end: float
    segment_ids: list[int]


@dataclass
class Highlight:
    id: int
    start: float
    end: float
    score: float
    score_components: dict[str, float]
    title: str
    transcript: str


@dataclass
class AnalysisResult:
    job_id: str
    source_name: str
    audio_path: str
    created_at: str
    features: AudioFeatureSet | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str | None = None
    language_probability: float | None = None
    word_frequencies: dict[str, int] = field(default_factory=dict)
    keyword_rows: list[dict[str, Any]] = field(default_factory=list)
    removed_keyword_rows: list[dict[str, Any]] = field(default_factory=list)
    stopword_suggestions: list[dict[str, Any]] = field(default_factory=list)
    applied_stop_words: list[str] = field(default_factory=list)
    stopword_review_metadata: dict[str, Any] = field(default_factory=dict)
    embeddings: Any = None
    points_2d: Any = None
    cluster_keywords: dict[int, list[str]] = field(default_factory=dict)
    topic_keywords: dict[int, list[str]] = field(default_factory=dict)
    chapters: list[Chapter] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    cooccurrence_nodes: list[dict[str, Any]] = field(default_factory=list)
    cooccurrence_edges: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    harmonic_path: str | None = None
    percussive_path: str | None = None
    errors: dict[str, str] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)

    def metadata(self) -> dict[str, Any]:
        """Return lightweight JSON-safe metadata."""
        return {
            "job_id": self.job_id,
            "source_name": self.source_name,
            "created_at": self.created_at,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.features.duration if self.features else None,
            "sample_rate": self.features.sample_rate if self.features else None,
            "tempo": self.features.tempo if self.features else None,
            "completed_stages": self.completed_stages,
            "errors": self.errors,
            "applied_stop_words": self.applied_stop_words,
        }

    def segments_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(segment) for segment in self.segments]
