"""Notebook-compatible acoustic feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import AnalysisSettings
from core.models import AudioFeatureSet


def analyze_audio(
    wav_path: Path,
    output_dir: Path,
    settings: AnalysisSettings,
) -> tuple[AudioFeatureSet, Path, Path]:
    """Extract waveform, RMS, STFT, pitch, entropy, HPSS, tempo and FFT."""
    import librosa
    import soundfile as sf
    from scipy.signal import find_peaks

    y, sr = librosa.load(wav_path, sr=settings.sample_rate, mono=True)
    if y.size == 0:
        raise ValueError("読み込んだ音声データが空です。")

    n_fft, hop = settings.frame_length, settings.hop_length
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    power = np.abs(stft_complex) ** 2
    spectrogram_db = librosa.power_to_db(power, ref=np.max)
    power_sum = np.sum(power, axis=0, keepdims=True)
    probability = power / np.maximum(power_sum, np.finfo(float).eps)
    entropy = -np.sum(probability * np.log2(probability + np.finfo(float).eps), axis=0)
    normalized_entropy = entropy / np.log2(power.shape[0])

    try:
        pitch, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=n_fft, hop_length=hop,
        )
    except Exception:
        pitch = np.full(power.shape[1], np.nan)

    harmonic, percussive = librosa.effects.hpss(y)
    harmonic_path, percussive_path = output_dir / "harmonic.wav", output_dir / "percussive.wav"
    sf.write(harmonic_path, harmonic, sr)
    sf.write(percussive_path, percussive, sr)

    tempo_result, beats = librosa.beat.beat_track(y=percussive, sr=sr, hop_length=hop)
    tempo = float(np.asarray(tempo_result).reshape(-1)[0])
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    peak_distance = max(1, int(settings.minimum_peak_interval_sec * sr / hop))
    peak_indices, _ = find_peaks(
        rms, height=settings.peak_threshold, distance=peak_distance
    )

    window = np.hanning(len(y))
    fft_values = np.abs(np.fft.rfft(y * window)) / max(1, len(y))
    fft_frequencies = np.fft.rfftfreq(len(y), d=1 / sr)
    frame_times = librosa.frames_to_time(np.arange(power.shape[1]), sr=sr, hop_length=hop)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # Keep interactive charts responsive for long recordings.
    stride = max(1, len(y) // 100_000)
    waveform = y[::stride]
    waveform_times = np.arange(len(waveform)) * stride / sr
    spec_stride = max(1, power.shape[1] // 2000)

    features = AudioFeatureSet(
        sample_rate=sr,
        duration=float(librosa.get_duration(y=y, sr=sr)),
        tempo=tempo,
        waveform=waveform,
        waveform_times=waveform_times,
        rms=rms,
        rms_times=rms_times,
        pitch=pitch,
        pitch_times=frame_times[: len(pitch)],
        entropy=normalized_entropy,
        entropy_times=frame_times,
        beat_times=librosa.frames_to_time(beats, sr=sr, hop_length=hop),
        peak_times=rms_times[peak_indices],
        spectrogram_db=spectrogram_db[:, ::spec_stride],
        spectrogram_times=frame_times[::spec_stride],
        spectrogram_frequencies=frequencies,
        fft_frequencies=fft_frequencies,
        fft_values=fft_values,
        onset=onset,
        onset_times=librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop),
    )
    return features, harmonic_path, percussive_path
