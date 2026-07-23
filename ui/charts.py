"""Interactive Plotly charts for synchronized analysis results."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import AnalysisResult


LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=45, r=20, t=55, b=40),
    hovermode="x unified",
)


def acoustic_overview(result: AnalysisResult) -> go.Figure:
    features = result.features
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    figure.add_trace(go.Scatter(
        x=features.waveform_times, y=features.waveform, name="波形",
        line=dict(color="#367BF5", width=0.7),
    ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=features.rms_times, y=features.rms, name="RMS",
        line=dict(color="#E8793E"),
    ), row=2, col=1)
    for peak in features.peak_times[:1000]:
        figure.add_vline(x=float(peak), line_width=0.5, line_color="#D64550", opacity=0.2, row=2, col=1)
    figure.add_trace(go.Scatter(
        x=features.pitch_times, y=features.pitch, name="ピッチ",
        line=dict(color="#6B5DD3"),
    ), row=3, col=1)
    figure.add_trace(go.Scatter(
        x=features.entropy_times, y=features.entropy, name="スペクトルエントロピー",
        line=dict(color="#25A18E"), yaxis="y4",
    ), row=3, col=1)
    figure.update_yaxes(title_text="振幅", row=1, col=1)
    figure.update_yaxes(title_text="RMS", row=2, col=1)
    figure.update_yaxes(title_text="Hz / entropy", row=3, col=1)
    figure.update_xaxes(title_text="時間（秒）", rangeslider_visible=True, row=3, col=1)
    figure.update_layout(height=720, title="共通時間軸：波形・音量・ピッチ・エントロピー", **LAYOUT)
    return figure


def spectrogram(result: AnalysisResult) -> go.Figure:
    f = result.features
    mask = np.asarray(f.spectrogram_frequencies) >= 20
    figure = go.Figure(go.Heatmap(
        x=f.spectrogram_times, y=np.asarray(f.spectrogram_frequencies)[mask],
        z=np.asarray(f.spectrogram_db)[mask], colorscale="Viridis",
        colorbar=dict(title="dB"),
    ))
    figure.update_yaxes(type="log", title="周波数（Hz）")
    figure.update_xaxes(title="時間（秒）")
    figure.update_layout(height=480, title="スペクトログラム", **LAYOUT)
    return figure


def fft_chart(result: AnalysisResult) -> go.Figure:
    f = result.features
    max_frequency = min(20_000, f.sample_rate / 2)
    mask = np.asarray(f.fft_frequencies) <= max_frequency
    figure = go.Figure(go.Scatter(
        x=np.asarray(f.fft_frequencies)[mask], y=np.asarray(f.fft_values)[mask],
        line=dict(width=1, color="#367BF5"),
    ))
    figure.update_layout(title="FFT周波数スペクトル", **LAYOUT)
    figure.update_xaxes(title="周波数（Hz）")
    figure.update_yaxes(title="正規化振幅")
    return figure


def emotion_timeline(result: AnalysisResult) -> go.Figure:
    rows = [{
        "time": (s.start + s.end) / 2, "valence": s.valence or 0,
        "arousal": s.arousal or 0, "emotion": s.emotion_label or "unknown",
        "speaker": s.speaker_id or "UNKNOWN",
    } for s in result.segments]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True)
    figure.add_trace(go.Scatter(
        x=[r["time"] for r in rows], y=[r["valence"] for r in rows],
        mode="lines+markers", name="valence",
    ), row=1, col=1)
    figure.add_trace(go.Scatter(
        x=[r["time"] for r in rows], y=[r["arousal"] for r in rows],
        mode="lines+markers", name="arousal",
    ), row=2, col=1)
    figure.update_xaxes(title="時間（秒）", rangeslider_visible=True, row=2, col=1)
    figure.update_layout(height=520, title="感情推定の時系列（モデルによる推定）", **LAYOUT)
    return figure


def cluster_chart(result: AnalysisResult) -> go.Figure:
    points = np.asarray(result.points_2d)
    rows = [{
        "x": points[i, 0], "y": points[i, 1], "cluster": str(segment.cluster_id),
        "text": segment.text, "time": segment.start,
    } for i, segment in enumerate(result.segments)]
    figure = px.scatter(
        rows, x="x", y="y", color="cluster", hover_data=["text", "time"],
        title="発話の意味クラスタ（PCA投影）",
    )
    figure.update_layout(**LAYOUT)
    return figure


def highlight_chart(result: AnalysisResult) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=[(s.start + s.end) / 2 for s in result.segments],
        y=[s.highlight_score or 0 for s in result.segments],
        mode="lines+markers", name="スコア", line=dict(color="#D64550"),
    ))
    for item in result.highlights:
        figure.add_vrect(x0=item.start, x1=item.end, fillcolor="#FFB000", opacity=0.18, line_width=0)
    figure.update_layout(title="盛り上がりスコア", **LAYOUT)
    figure.update_xaxes(title="時間（秒）", rangeslider_visible=True)
    figure.update_yaxes(title="スコア（0–100）", range=[0, 100])
    return figure
