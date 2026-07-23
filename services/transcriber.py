"""Whisper transcription service with GPU auto-detection."""

from __future__ import annotations

import gc
from functools import lru_cache
from pathlib import Path

from core.config import AnalysisSettings
from core.models import TranscriptSegment, TranscriptWord


@lru_cache(maxsize=6)
def _load_model(model_size: str, device: str, compute_type: str):
    """Load each Whisper model/device combination once per process."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)


def clear_model_cache() -> None:
    """Release cached Whisper weights after a memory-constrained job."""
    _load_model.cache_clear()
    gc.collect()


def transcribe_audio(
    wav_path: Path,
    settings: AnalysisSettings,
) -> tuple[list[TranscriptSegment], str, float]:
    """Transcribe audio with faster-whisper and word timestamps."""
    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False
    device, compute_type = ("cuda", "float16") if use_cuda else ("cpu", "int8")
    model = _load_model(settings.whisper_model, device, compute_type)
    generated, info = model.transcribe(
        str(wav_path), language=settings.language, beam_size=settings.beam_size,
        vad_filter=True, word_timestamps=True,
    )
    result: list[TranscriptSegment] = []
    for item in generated:
        text = item.text.strip()
        if not text:
            continue
        words = [
            TranscriptWord(
                start=None if word.start is None else float(word.start),
                end=None if word.end is None else float(word.end),
                text=word.word.strip(),
                probability=float(word.probability),
            )
            for word in (item.words or [])
        ]
        result.append(TranscriptSegment(
            id=int(item.id), start=float(item.start), end=float(item.end),
            text=text, words=words, speaker_id="SPEAKER_00",
        ))
    if not result:
        raise ValueError("文字起こし結果が空です。明瞭な発話があるか確認してください。")
    return result, str(info.language), float(info.language_probability)
