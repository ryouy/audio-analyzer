from utils.timecode import format_timecode, youtube_timecode


def test_format_timecode() -> None:
    assert format_timecode(3661.9) == "01:01:01"
    assert format_timecode(-1) == "00:00:00"
    assert format_timecode(1.25, milliseconds=True) == "00:00:01.250"


def test_youtube_timecode() -> None:
    assert youtube_timecode(65) == "1:05"
    assert youtube_timecode(3665) == "1:01:05"
