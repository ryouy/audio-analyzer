from pathlib import Path

import numpy as np
import soundfile as sf

from core.config import AnalysisSettings
from services.acoustic_analyzer import analyze_audio


def test_fast_acoustic_analysis_keeps_chart_data_bounded(tmp_path: Path) -> None:
    sample_rate = 16_000
    times = np.arange(sample_rate * 2) / sample_rate
    audio = (0.1 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    wav_path = tmp_path / "input.wav"
    sf.write(wav_path, audio, sample_rate)

    features, harmonic_path, percussive_path = analyze_audio(
        wav_path,
        tmp_path,
        AnalysisSettings(sample_rate=sample_rate, fast_mode=True),
    )

    assert features.duration == 2.0
    assert features.spectrogram_db.shape[0] <= 512
    assert features.spectrogram_db.shape[1] <= 1001
    assert len(features.spectrogram_times) == features.spectrogram_db.shape[1]
    assert harmonic_path.exists()
    assert percussive_path.exists()
