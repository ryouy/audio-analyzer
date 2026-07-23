"""Create portable result bundles without persisting uploaded media."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import asdict

from core.models import AnalysisResult
from utils.timecode import youtube_timecode


def _csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def build_export_zip(result: AnalysisResult) -> bytes:
    """Build the complete in-memory ZIP export."""
    buffer = io.BytesIO()
    segments = result.segments_as_dicts()
    flat_segments = [
        {key: value for key, value in row.items() if key != "words"} for row in segments
    ]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(result.metadata(), ensure_ascii=False, indent=2))
        archive.writestr("summary.json", json.dumps({
            "summary": result.summary, "key_points": result.key_points,
            "chapters": [asdict(item) for item in result.chapters],
        }, ensure_ascii=False, indent=2))
        archive.writestr("transcript.txt", "\n".join(item.text for item in result.segments))
        archive.writestr("transcript_segments.json", json.dumps(segments, ensure_ascii=False, indent=2))
        archive.writestr("transcript_segments.csv", _csv_bytes(flat_segments))
        archive.writestr("word_frequencies.csv", _csv_bytes([
            {"word": word, "count": count} for word, count in result.word_frequencies.items()
        ]))
        archive.writestr("stopword_feedback.json", json.dumps({
            "applied_stop_words": result.applied_stop_words,
            "review_metadata": result.stopword_review_metadata,
            "remaining_suggestions": result.stopword_suggestions,
        }, ensure_ascii=False, indent=2))
        archive.writestr("clusters.csv", _csv_bytes([
            {"segment_id": s.id, "cluster": s.cluster_id, "text": s.text} for s in result.segments
        ]))
        archive.writestr("topics.csv", _csv_bytes([
            {"segment_id": s.id, "topic": s.topic_id, "text": s.text} for s in result.segments
        ]))
        archive.writestr("chapters.csv", _csv_bytes([asdict(item) for item in result.chapters]))
        archive.writestr("chapters_youtube.txt", "\n".join(
            f"{youtube_timecode(item.start)} {item.title}" for item in result.chapters
        ))
        archive.writestr("emotions.csv", _csv_bytes([
            {"segment_id": s.id, "emotion": s.emotion_label, "valence": s.valence, "arousal": s.arousal}
            for s in result.segments
        ]))
        archive.writestr("keyword_timeline.csv", _csv_bytes(result.keyword_rows))
        archive.writestr("cooccurrence_nodes.csv", _csv_bytes(result.cooccurrence_nodes))
        archive.writestr("cooccurrence_edges.csv", _csv_bytes(result.cooccurrence_edges))
        archive.writestr("highlights.json", json.dumps(
            [asdict(item) for item in result.highlights], ensure_ascii=False, indent=2
        ))
        archive.writestr("highlights.csv", _csv_bytes([asdict(item) for item in result.highlights]))
        for source, name in [
            (result.harmonic_path, "harmonic.wav"),
            (result.percussive_path, "percussive.wav"),
        ]:
            if source:
                from pathlib import Path
                source_path = Path(source)
                if source_path.is_file():
                    archive.write(source_path, name)
    return buffer.getvalue()
