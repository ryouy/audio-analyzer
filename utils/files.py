"""Safe media file helpers."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
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


def normalize_url_input(value: str) -> str:
    """Normalize a raw URL or a URL copied as a Markdown/autolink.

    Chat and document tools often copy links as ``[label](https://...)``.
    Passing that wrapper to yt-dlp makes the otherwise valid URL fail.
    """
    value = html.unescape(value.strip())
    markdown_link = re.fullmatch(r"\[[^\]]*]\((https?://[^)\s]+)\)", value)
    if markdown_link:
        value = markdown_link.group(1)
    elif value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    validate_url(value)
    return value


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


def _deno_executable() -> str | None:
    """Locate Deno installed directly or through the Python deno package."""
    executable = shutil.which("deno")
    if executable:
        return executable
    try:
        import deno

        return str(deno.find_deno_bin())
    except (ImportError, OSError, AttributeError, RuntimeError):
        return None


def build_youtube_command(
    url: str,
    destination: Path,
    *,
    impersonate: bool = False,
) -> list[str]:
    """Build a yt-dlp command with the current YouTube EJS requirements."""
    command = [
        sys.executable, "-m", "yt_dlp", "-x", "--audio-format", "wav",
        "--audio-quality", "0", "--force-overwrites", "--no-playlist",
        "--socket-timeout", "30",
    ]
    deno_path = _deno_executable()
    if deno_path:
        command.extend(["--js-runtimes", f"deno:{deno_path}"])
    if impersonate:
        command.extend(["--impersonate", "chrome"])
    command.extend(["-o", str(destination), normalize_url_input(url)])
    return command


def _run_youtube_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bounded yt-dlp attempt."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def download_youtube(url: str, destination: Path) -> Path:
    """Download public media as WAV through yt-dlp with one 403 fallback."""
    normalized_url = normalize_url_input(url)
    command = build_youtube_command(normalized_url, destination)
    completed = _run_youtube_command(command)
    first_details = (completed.stderr or completed.stdout or "").strip()
    if completed.returncode != 0 and (
        "HTTP Error 403" in first_details or "Forbidden" in first_details
    ):
        completed = _run_youtube_command(
            build_youtube_command(normalized_url, destination, impersonate=True)
        )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "").strip()
        details = "\n".join(details.splitlines()[-8:])
        if "No supported JavaScript runtime" in details:
            details += (
                "\nDenoが見つかりません。requirements.txtからdenoを"
                "インストールして再デプロイしてください。"
            )
        if "HTTP Error 403" in details or "Sign in to confirm" in details:
            details += (
                "\nStreamlit Cloudの共有IPがYouTubeに拒否されている可能性があります。"
                "この場合はファイルアップロードを利用してください。"
            )
        raise RuntimeError(
            "YouTube音声の取得に失敗しました。"
            + (f"\nyt-dlp: {details}" if details else "")
        )
    if not destination.exists():
        raise RuntimeError("YouTube音声を取得できませんでした。")
    return destination
