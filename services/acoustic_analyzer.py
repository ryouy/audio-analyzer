"""Notebook-compatible acoustic feature extraction."""

from __future__ import annotations

import gc
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
    frequency_count = n_fft // 2 + 1
    estimated_frames = max(1, int(np.ceil(len(y) / hop)))
    spec_time_stride = max(1, estimated_frames // 1000)
    spec_frequency_stride = max(1, int(np.ceil(frequency_count / 512)))
    spectrogram_parts: list[np.ndarray] = []
    spectrogram_frame_parts: list[np.ndarray] = []
    entropy_parts: list[np.ndarray] = []
    frame_offset = 0

    # A full STFT of a long recording can occupy hundreds of MB. Build it in
    # bounded chunks and retain only the resolution needed by the charts.
    chunk_samples = max(n_fft, sr * (60 if settings.fast_mode else 300))
    for start in range(0, len(y), chunk_samples):
        chunk = y[start : start + chunk_samples]
        stft_chunk = librosa.stft(chunk, n_fft=n_fft, hop_length=hop)
        power = np.abs(stft_chunk)
        del stft_chunk
        np.square(power, out=power)

        power_sum = np.maximum(
            np.sum(power, axis=0, dtype=np.float64), np.finfo(np.float32).eps
        )
        entropy = np.zeros(power.shape[1], dtype=np.float32)
        for frequency_start in range(0, power.shape[0], 64):
            probability = (
                power[frequency_start : frequency_start + 64] / power_sum
            )
            entropy -= np.sum(
                probability * np.log2(probability + np.finfo(np.float32).eps),
                axis=0,
            )
        entropy_parts.append(entropy / np.log2(power.shape[0]))

        local_indices = np.arange(power.shape[1])
        keep = (local_indices + frame_offset) % spec_time_stride == 0
        spectrogram_frame_parts.append(local_indices[keep] + frame_offset)
        reduced_power = power[::spec_frequency_stride, keep]
        spectrogram_parts.append(
            librosa.power_to_db(reduced_power, ref=float(np.max(power))).astype(
                np.float32, copy=False
            )
        )
        frame_offset += power.shape[1]
        del power, power_sum, reduced_power

    normalized_entropy = np.concatenate(entropy_parts)
    spectrogram_db = np.concatenate(spectrogram_parts, axis=1)
    spectrogram_frames = np.concatenate(spectrogram_frame_parts)
    if spectrogram_db.shape[1] > 1000:
        chart_indices = np.linspace(
            0, spectrogram_db.shape[1] - 1, 1000, dtype=int
        )
        spectrogram_db = spectrogram_db[:, chart_indices]
        spectrogram_frames = spectrogram_frames[chart_indices]
    frame_count = len(normalized_entropy)
    del entropy_parts, spectrogram_parts, spectrogram_frame_parts
    gc.collect()

    try:
        if settings.fast_mode:
            pitch = librosa.yin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sr, frame_length=n_fft, hop_length=hop,
            )
        else:
            pitch, _, _ = librosa.pyin(
                y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                sr=sr, frame_length=n_fft, hop_length=hop,
            )
    except Exception:
        pitch = np.full(frame_count, np.nan)

    harmonic_path, percussive_path = output_dir / "harmonic.wav", output_dir / "percussive.wav"
    if settings.fast_mode:
        with (
            sf.SoundFile(harmonic_path, "w", sr, 1, subtype="PCM_16") as harmonic_file,
            sf.SoundFile(percussive_path, "w", sr, 1, subtype="PCM_16") as percussive_file,
        ):
            for start in range(0, len(y), chunk_samples):
                harmonic, percussive = librosa.effects.hpss(
                    y[start : start + chunk_samples]
                )
                harmonic_file.write(harmonic)
                percussive_file.write(percussive)
                del harmonic, percussive
        tempo_source = y
    else:
        harmonic, percussive = librosa.effects.hpss(y)
        sf.write(harmonic_path, harmonic, sr, subtype="PCM_16")
        sf.write(percussive_path, percussive, sr, subtype="PCM_16")
        tempo_source = percussive

    tempo_result, beats = librosa.beat.beat_track(
        y=tempo_source, sr=sr, hop_length=hop
    )
    tempo = float(np.asarray(tempo_result).reshape(-1)[0])
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    peak_distance = max(1, int(settings.minimum_peak_interval_sec * sr / hop))
    peak_indices, _ = find_peaks(
        rms, height=settings.peak_threshold, distance=peak_distance
    )

    fft_sample_count = min(len(y), sr * 300) if settings.fast_mode else len(y)
    fft_source = y[:fft_sample_count]
    window = np.hanning(fft_sample_count).astype(np.float32)
    fft_values = np.abs(np.fft.rfft(fft_source * window)) / max(1, fft_sample_count)
    fft_frequencies = np.fft.rfftfreq(fft_sample_count, d=1 / sr)
    frame_times = librosa.frames_to_time(np.arange(frame_count), sr=sr, hop_length=hop)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[
        ::spec_frequency_stride
    ]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    # Keep interactive charts responsive for long recordings.
    stride = max(1, len(y) // 100_000)
    waveform = y[::stride]
    waveform_times = np.arange(len(waveform)) * stride / sr
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
        spectrogram_db=spectrogram_db,
        spectrogram_times=librosa.frames_to_time(
            spectrogram_frames, sr=sr, hop_length=hop
        ),
        spectrogram_frequencies=frequencies,
        fft_frequencies=fft_frequencies,
        fft_values=fft_values,
        onset=onset,
        onset_times=librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop),
    )
    return features, harmonic_path, percussive_path
