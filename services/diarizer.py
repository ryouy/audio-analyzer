"""Optional pyannote speaker diarization."""

from __future__ import annotations

import os
from pathlib import Path

from core.models import TranscriptSegment


def diarize_audio(wav_path: Path, segments: list[TranscriptSegment], token: str | None = None) -> int:
    """Assign speakers by maximum temporal overlap; requires a Hugging Face token."""
    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKENが未設定のため話者分離をスキップしました。")
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    output = pipeline(str(wav_path))
    turns = [
        (float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in output.itertracks(yield_label=True)
    ]
    speakers = set()
    for segment in segments:
        overlaps = [
            (max(0.0, min(segment.end, end) - max(segment.start, start)), speaker)
            for start, end, speaker in turns
        ]
        if overlaps:
            overlap, speaker = max(overlaps)
            if overlap > 0:
                segment.speaker_id = speaker
                speakers.add(speaker)
    return len(speakers)
