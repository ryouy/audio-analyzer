"""Lightweight local insight analyzers used when large models are unavailable."""

from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np

from core.models import AudioFeatureSet, Chapter, Highlight, TranscriptSegment


EMOTION_WORDS = {
    "joy": {"嬉しい", "楽しい", "最高", "好き", "ありがとう", "happy", "great", "love"},
    "sadness": {"悲しい", "残念", "つらい", "寂しい", "sad", "sorry"},
    "anger": {"怒る", "嫌い", "最悪", "許せない", "angry", "hate"},
    "fear": {"怖い", "不安", "心配", "危険", "fear", "afraid"},
    "surprise": {"驚く", "すごい", "まさか", "びっくり", "wow", "surprise"},
}


def analyze_emotions(
    segments: list[TranscriptSegment], features: AudioFeatureSet | None
) -> None:
    """Attach transparent lexical emotion estimates and acoustic arousal."""
    rms_max = float(np.max(features.rms)) if features is not None and len(features.rms) else 1.0
    for segment in segments:
        scores = {
            label: sum(word in segment.text.lower() for word in words)
            for label, words in EMOTION_WORDS.items()
        }
        label, hits = max(scores.items(), key=lambda item: item[1])
        if hits == 0:
            label = "neutral"
        midpoint = (segment.start + segment.end) / 2
        if features is not None and len(features.rms):
            index = int(np.argmin(np.abs(np.asarray(features.rms_times) - midpoint)))
            arousal = float(np.clip(features.rms[index] / max(rms_max, 1e-9), 0, 1))
        else:
            arousal = 0.3
        valence = {"joy": 0.8, "surprise": 0.3, "neutral": 0.0,
                   "sadness": -0.6, "anger": -0.7, "fear": -0.6}[label]
        segment.emotion_label = label
        segment.emotion_score = min(1.0, 0.55 + 0.12 * hits) if hits else 0.5
        segment.valence = valence
        segment.arousal = arousal


def create_chapters(
    segments: list[TranscriptSegment],
    topic_keywords: dict[int, list[str]],
    min_duration: float = 60.0,
) -> tuple[str, list[str], list[Chapter]]:
    """Create extractive summary and chapters at topic/time boundaries."""
    if not segments:
        return "", [], []
    ranked = sorted(
        segments, key=lambda s: (len(s.text), s.end - s.start), reverse=True
    )
    key_points = [segment.text for segment in ranked[:5]]
    summary = " ".join(key_points[:3])
    groups: list[list[TranscriptSegment]] = [[segments[0]]]
    for segment in segments[1:]:
        current = groups[-1]
        elapsed = current[-1].end - current[0].start
        topic_changed = segment.topic_id != current[-1].topic_id
        if elapsed >= min_duration and (topic_changed or segment.start - current[-1].end > 3):
            groups.append([segment])
        elif elapsed >= min_duration * 2:
            groups.append([segment])
        else:
            current.append(segment)
    chapters = []
    for index, group in enumerate(groups):
        topic_id = group[0].topic_id or 0
        words = topic_keywords.get(topic_id, [])
        title = " / ".join(words[:3]) or f"チャプター {index + 1}"
        representative = max(group, key=lambda item: len(item.text))
        chapters.append(Chapter(
            id=index, title=title, summary=representative.text,
            start=group[0].start, end=group[-1].end,
            segment_ids=[item.id for item in group],
        ))
    return summary, key_points, chapters


def build_cooccurrence(
    keyword_rows: list[dict],
    window_size: int = 5,
    minimum: int = 2,
    max_nodes: int = 50,
) -> tuple[list[dict], list[dict]]:
    """Build a normalized sliding-window word co-occurrence graph."""
    words = [row["normalized_word"] for row in keyword_rows]
    frequency = Counter(words)
    allowed = {word for word, _ in frequency.most_common(max_nodes)}
    edges: Counter[tuple[str, str]] = Counter()
    for index, word in enumerate(words):
        if word not in allowed:
            continue
        neighborhood = words[index + 1:index + 1 + window_size]
        for other in set(neighborhood):
            if other in allowed and other != word:
                edges[tuple(sorted((word, other)))] += 1
    edge_rows = [
        {
            "source": source, "target": target, "cooccurrence_count": count,
            "normalized_weight": count / np.sqrt(frequency[source] * frequency[target]),
        }
        for (source, target), count in edges.most_common()
        if count >= minimum
    ]
    degree = defaultdict(float)
    for edge in edge_rows:
        degree[edge["source"]] += edge["normalized_weight"]
        degree[edge["target"]] += edge["normalized_weight"]
    nodes = [
        {"word": word, "frequency": count, "centrality": float(degree[word])}
        for word, count in frequency.most_common(max_nodes)
    ]
    return nodes, edge_rows


def detect_highlights(
    segments: list[TranscriptSegment],
    features: AudioFeatureSet | None,
    threshold: float = 62.0,
) -> list[Highlight]:
    """Score transcript intervals from volume, arousal and speech density."""
    if not segments:
        return []
    rms_max = float(np.max(features.rms)) if features is not None and len(features.rms) else 1.0
    candidates = []
    for segment_index, segment in enumerate(segments):
        duration = max(0.1, segment.end - segment.start)
        speech = min(1.0, len(segment.text) / duration / 12.0)
        emotion = float(segment.arousal or 0.0)
        if features is not None and len(features.rms):
            mask = (features.rms_times >= segment.start) & (features.rms_times <= segment.end)
            volume = float(np.mean(features.rms[mask]) / max(rms_max, 1e-9)) if mask.any() else 0.0
        else:
            volume = 0.0
        topic_change = (
            1.0
            if segment_index > 0 and segment.topic_id != segments[segment_index - 1].topic_id
            else 0.0
        )
        score = float(np.clip(100 * (0.4 * volume + 0.3 * emotion + 0.2 * speech + 0.1 * topic_change), 0, 100))
        segment.highlight_score = score
        if score >= threshold:
            candidates.append((segment, score, {
                "rms": volume, "emotion": emotion, "speech": speech, "topic": topic_change
            }))
    return [
        Highlight(
            id=index, start=max(0, segment.start - 3), end=segment.end + 3,
            score=score, score_components=components,
            title=f"ハイライト {index + 1}", transcript=segment.text,
        )
        for index, (segment, score, components) in enumerate(
            sorted(candidates, key=lambda item: item[1], reverse=True)
        )
    ]
