"""Safe media file helpers."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


ALLOWED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".webm", ".mkv"}


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("http または https の有効なURLを入力してください。")


def convert_to_wav(source: Path, destination: Path, sample_rate: int) -> Path:
    """Normalize arbitrary supported media to mono WAV using ffmpeg."""
    command = [
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar",
        str(sample_rate), "-c:a", "pcm_s16le", str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=3600)
    if not destination.exists():
        raise RuntimeError("ffmpegによるWAV変換に失敗しました。")
    return destination


def download_youtube(url: str, destination: Path) -> Path:
    """Download public media as WAV through yt-dlp."""
    validate_url(url)
    command = [
        sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "wav",
        "--audio-quality", "0", "--force-overwrites", "--no-playlist",
        "--socket-timeout", "30", "-o", str(destination), url,
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=1800)
    if not destination.exists():
        raise RuntimeError("YouTube音声を取得できませんでした。")
    return destination
