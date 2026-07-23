"""Time formatting utilities."""


def format_timecode(seconds: float, milliseconds: bool = False) -> str:
    """Convert seconds to HH:MM:SS (or HH:MM:SS.mmm)."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def youtube_timecode(seconds: float) -> str:
    """Format a compact timestamp for YouTube descriptions."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
