import subprocess
from pathlib import Path

import pytest

from utils.files import (
    build_youtube_command,
    download_youtube,
    normalize_url_input,
)
from utils.timecode import format_timecode, youtube_timecode


def test_format_timecode() -> None:
    assert format_timecode(3661.9) == "01:01:01"
    assert format_timecode(-1) == "00:00:00"
    assert format_timecode(1.25, milliseconds=True) == "00:00:01.250"


def test_youtube_timecode() -> None:
    assert youtube_timecode(65) == "1:05"
    assert youtube_timecode(3665) == "1:01:05"


def test_normalize_markdown_url() -> None:
    url = "https://www.youtube.com/watch?v=bch25Ieqxwk&pp=test"
    assert normalize_url_input(f"[{url}]({url})") == url
    assert normalize_url_input(f"<{url}>") == url
    assert normalize_url_input(url) == url


def test_youtube_command_uses_deno(monkeypatch) -> None:
    monkeypatch.setattr("utils.files._deno_executable", lambda: "/venv/bin/deno")
    command = build_youtube_command(
        "https://www.youtube.com/watch?v=test",
        Path("/tmp/audio.wav"),
    )
    assert command[command.index("--js-runtimes") + 1] == "deno:/venv/bin/deno"


def test_youtube_403_fallback_can_impersonate(monkeypatch) -> None:
    monkeypatch.setattr("utils.files._deno_executable", lambda: "/venv/bin/deno")
    command = build_youtube_command(
        "https://www.youtube.com/watch?v=test",
        Path("/tmp/audio.wav"),
        impersonate=True,
    )
    assert command[command.index("--impersonate") + 1] == "chrome"


def test_youtube_download_retries_403_with_impersonation(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "audio.wav"
    calls = []

    def fake_run(command):
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 1, "", "HTTP Error 403: Forbidden")
        destination.write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("utils.files._deno_executable", lambda: "/venv/bin/deno")
    monkeypatch.setattr("utils.files._run_youtube_command", fake_run)
    assert download_youtube(
        "https://www.youtube.com/watch?v=test", destination
    ) == destination
    assert "--impersonate" not in calls[0]
    assert "--impersonate" in calls[1]


def test_youtube_download_explains_persistent_cloud_403(
    monkeypatch, tmp_path: Path
) -> None:
    def always_forbidden(command):
        return subprocess.CompletedProcess(
            command, 1, "", "HTTP Error 403: Forbidden"
        )

    monkeypatch.setattr("utils.files._deno_executable", lambda: "/venv/bin/deno")
    monkeypatch.setattr("utils.files._run_youtube_command", always_forbidden)
    with pytest.raises(RuntimeError, match="共有IP"):
        download_youtube(
            "https://www.youtube.com/watch?v=test", tmp_path / "audio.wav"
        )
