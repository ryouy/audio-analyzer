"""Streamlit Audio Intelligence application."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.config import AnalysisSettings, MAX_UPLOAD_MB
from core.pipeline import run_pipeline
from services.exporter import build_export_zip
from services.semantic_search import search
from services.stopword_feedback import (
    DEFAULT_MODEL,
    apply_stopword_feedback,
    restore_stopword_feedback,
    review_stopwords,
    vocabulary_fingerprint,
)
from ui.charts import (
    acoustic_overview,
    cluster_chart,
    emotion_timeline,
    fft_chart,
    highlight_chart,
    spectrogram,
)
from utils.files import ALLOWED_SUFFIXES, convert_to_wav, download_youtube, file_digest
from utils.timecode import format_timecode, youtube_timecode


st.set_page_config(
    page_title="Audio Intelligence", page_icon="🎧", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1500px}
  [data-testid="stMetric"] {background:#f7f9fc;border:1px solid #e5eaf1;padding:12px;border-radius:12px}
  h1 {letter-spacing:-.03em}
</style>
""", unsafe_allow_html=True)


def sidebar_settings() -> tuple[str, object, AnalysisSettings, str | None]:
    st.sidebar.header("入力")
    input_mode = st.sidebar.radio("入力方式", ["ファイルアップロード", "YouTube URL"])
    if input_mode == "ファイルアップロード":
        source = st.sidebar.file_uploader(
            "音声または動画", type=[suffix[1:] for suffix in sorted(ALLOWED_SUFFIXES)]
        )
    else:
        source = st.sidebar.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")

    st.sidebar.header("解析設定")
    language_label = st.sidebar.selectbox("言語", ["日本語", "自動判定", "英語"])
    language = {"日本語": "ja", "自動判定": None, "英語": "en"}[language_label]
    model = st.sidebar.selectbox("Whisperモデル", ["tiny", "base", "small", "medium", "large-v3"])
    sample_rate = st.sidebar.selectbox("サンプルレート", [16000, 22050, 44100], index=1)
    mode = st.sidebar.radio("処理モード", ["高速", "高精度"], horizontal=True)

    with st.sidebar.expander("有効にする機能", expanded=False):
        labels = {
            "acoustic": "音響分析", "transcription": "文字起こし",
            "diarization": "話者分離（要HF Token）", "emotion": "感情分析",
            "summary": "要約・章立て", "wordcloud": "ワードクラウド",
            "clustering": "意味クラスタリング", "keywords": "キーワード時間分析",
            "cooccurrence": "共起ネットワーク", "topics": "トピック推移",
            "highlights": "盛り上がり検出", "semantic_search": "類似発話検索",
        }
        enabled = {
            key for key, label in labels.items()
            if st.checkbox(label, value=key != "diarization", key=f"feature_{key}")
        }
    with st.sidebar.expander("詳細設定"):
        peak = st.number_input("RMSピーク閾値", 0.001, 1.0, 0.05, 0.005)
        peak_interval = st.number_input("最小ピーク間隔（秒）", 0.05, 10.0, 0.20, 0.05)
        auto_cluster = st.checkbox("クラスタ数を自動決定", True)
        cluster_count = None if auto_cluster else st.slider("クラスタ数", 2, 20, 5)
        max_clusters = st.slider("最大クラスタ数", 2, 20, 8)
        co_window = st.slider("共起ウィンドウ", 2, 20, 5)
        co_minimum = st.slider("最小共起回数", 1, 20, 2)
        highlight_threshold = st.slider("ハイライト閾値", 0, 100, 62)
        stop_text = st.text_area("追加ストップワード（改行区切り）")
    try:
        hf_token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        hf_token = None
    defaults = AnalysisSettings()
    learned_stop_words = frozenset(st.session_state.get("learned_stop_words", set()))
    settings = AnalysisSettings(
        sample_rate=sample_rate, peak_threshold=float(peak),
        minimum_peak_interval_sec=float(peak_interval), language=language,
        whisper_model=model, requested_cluster_count=cluster_count,
        max_cluster_count=max_clusters, cooccurrence_window=co_window,
        cooccurrence_minimum=co_minimum, highlight_threshold=float(highlight_threshold),
        fast_mode=mode == "高速", beam_size=3 if mode == "高速" else 5,
        enabled=frozenset(enabled),
        stop_words=(
            defaults.stop_words
            | learned_stop_words
            | frozenset(line.strip() for line in stop_text.splitlines() if line.strip())
        ),
    )
    return input_mode, source, settings, hf_token


def execute(input_mode: str, source: object, settings: AnalysisSettings, hf_token: str | None) -> None:
    if not source:
        st.warning("解析するファイルまたはURLを指定してください。")
        return
    job_id = uuid.uuid4().hex
    job_dir = Path(tempfile.mkdtemp(prefix=f"audio-intel-{job_id[:8]}-"))
    try:
        if input_mode == "ファイルアップロード":
            data = source.getvalue()
            if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"ファイルサイズは{MAX_UPLOAD_MB} MB以下にしてください。")
            suffix = Path(source.name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise ValueError("対応していないファイル形式です。")
            source_path = job_dir / f"source{suffix}"
            source_path.write_bytes(data)
            digest = file_digest(data)
            source_name = source.name
        else:
            source_name = "YouTube"
            digest = file_digest(str(source).encode())
            source_path = download_youtube(str(source), job_dir / "source.wav")
        cache_key = f"{digest}:{hash(settings)}"
        cached = st.session_state.setdefault("analysis_cache", {}).get(cache_key)
        if cached is not None:
            st.session_state.result = cached
            st.session_state.analysis_settings = settings
            st.success("同一入力・同一設定のセッションキャッシュを利用しました。")
            return
        wav_path = source_path if source_path.suffix.lower() == ".wav" else job_dir / "normalized.wav"
        if wav_path != source_path:
            with st.spinner("音声をWAVへ変換しています"):
                convert_to_wav(source_path, wav_path, settings.sample_rate)
        progress_bar, status = st.progress(0.0), st.empty()
        result = run_pipeline(
            wav_path, job_dir, source_name, job_id, settings,
            progress=lambda value, message: (progress_bar.progress(value), status.info(message)),
            hf_token=hf_token,
        )
        st.session_state.result = result
        st.session_state.analysis_settings = settings
        # Analysis results contain audio features and embeddings. Retaining several
        # jobs can exceed the Community Cloud memory limit.
        st.session_state.analysis_cache = {cache_key: result}
        status.success("解析が完了しました")
    except Exception as exc:
        st.error(f"入力処理に失敗しました: {type(exc).__name__}: {exc}")


def transcript_frame(result) -> pd.DataFrame:
    return pd.DataFrame([{
        "開始": format_timecode(s.start, True), "終了": format_timecode(s.end, True),
        "start_sec": s.start, "end_sec": s.end, "話者": s.speaker_id or "UNKNOWN",
        "発話": s.text, "感情": s.emotion_label, "トピック": s.topic_id,
        "クラスタ": s.cluster_id,
        "Whisper信頼度": (
            sum(word.probability for word in s.words if word.probability is not None)
            / max(1, sum(word.probability is not None for word in s.words))
            if s.words else None
        ),
        "ハイライト": s.highlight_score,
    } for s in result.segments])


def overview_tab(result) -> None:
    f = result.features
    cols = st.columns(8)
    metrics = [
        ("音声時間", format_timecode(f.duration) if f else "—"),
        ("テンポ", f"{f.tempo:.1f} BPM" if f and f.tempo is not None else "—"),
        ("話者数", len({s.speaker_id for s in result.segments if s.speaker_id})),
        ("発話区間", len(result.segments)),
        ("総文字数", sum(len(s.text) for s in result.segments)),
        ("トピック数", len({s.topic_id for s in result.segments if s.topic_id is not None})),
        ("ハイライト", len(result.highlights)),
        ("言語", f"{result.language or '—'} {result.language_probability or 0:.0%}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    st.audio(result.audio_path)
    if result.errors:
        with st.expander(f"一部機能でエラー（{len(result.errors)}件）"):
            for name, error in result.errors.items():
                st.warning(f"{name}: {error}")
    left, right = st.columns([2, 1])
    with left:
        st.subheader("自動要約")
        st.write(result.summary or "文字起こし完了後に表示されます。")
        st.subheader("自動チャプター")
        for chapter in result.chapters:
            st.markdown(f"**{youtube_timecode(chapter.start)}　{chapter.title}**  \n{chapter.summary}")
    with right:
        st.subheader("上位キーワード")
        st.dataframe(pd.DataFrame(
            list(result.word_frequencies.items())[:15], columns=["単語", "回数"]
        ), hide_index=True, use_container_width=True)
        if result.segments:
            counts = Counter(s.emotion_label or "unknown" for s in result.segments)
            st.plotly_chart(px.pie(
                names=list(counts), values=list(counts.values()), title="感情構成比（推定）"
            ), use_container_width=True)


def audio_tab(result) -> None:
    if not result.features:
        st.info("音響分析結果はありません。")
        return
    st.plotly_chart(acoustic_overview(result), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(spectrogram(result), use_container_width=True)
    right.plotly_chart(fft_chart(result), use_container_width=True)
    left, right = st.columns(2)
    if result.harmonic_path:
        left.caption("ハーモニック成分")
        left.audio(result.harmonic_path)
    if result.percussive_path:
        right.caption("パーカッシブ成分")
        right.audio(result.percussive_path)


def transcript_tab(result) -> None:
    if not result.segments:
        st.info("文字起こし結果はありません。Whisperの導入状況を確認してください。")
        return
    frame = transcript_frame(result)
    query = st.text_input("発話を検索", key="transcript_query")
    speakers = st.multiselect("話者フィルタ", sorted(frame["話者"].unique()))
    filtered = frame
    if query:
        filtered = filtered[filtered["発話"].str.contains(query, case=False, na=False)]
    if speakers:
        filtered = filtered[filtered["話者"].isin(speakers)]
    st.dataframe(filtered, hide_index=True, use_container_width=True, height=480)
    st.download_button("CSVをダウンロード", filtered.to_csv(index=False).encode("utf-8-sig"),
                       "transcript.csv", "text/csv")
    download_cols = st.columns(2)
    download_cols[0].download_button(
        "JSONをダウンロード",
        json.dumps(result.segments_as_dicts(), ensure_ascii=False, indent=2),
        "transcript.json", "application/json",
    )
    download_cols[1].download_button(
        "TXTをダウンロード",
        "\n".join(
            f"[{format_timecode(s.start)}] {s.speaker_id or 'UNKNOWN'}: {s.text}"
            for s in result.segments
        ),
        "transcript.txt", "text/plain",
    )
    selected = st.selectbox(
        "発話位置から再生", range(len(result.segments)),
        format_func=lambda i: f"{format_timecode(result.segments[i].start)} {result.segments[i].text[:60]}",
    )
    st.audio(result.audio_path, start_time=int(result.segments[selected].start))


def speakers_emotions_tab(result) -> None:
    if not result.segments:
        st.info("発話データがありません。")
        return
    names = sorted({s.speaker_id or "UNKNOWN" for s in result.segments})
    rename = {}
    columns = st.columns(min(4, len(names)))
    for index, name in enumerate(names):
        rename[name] = columns[index % len(columns)].text_input(name, value=f"話者{index + 1}", key=f"name_{name}")
    durations = Counter()
    for segment in result.segments:
        durations[rename[segment.speaker_id or "UNKNOWN"]] += segment.end - segment.start
    st.plotly_chart(px.bar(
        x=list(durations), y=list(durations.values()), labels={"x": "話者", "y": "発話秒数"},
        title="話者別発話時間",
    ), use_container_width=True)
    st.plotly_chart(emotion_timeline(result), use_container_width=True)
    st.caption("感情ラベル、valence、arousalはモデルによる推定であり、事実を断定するものではありません。")


def keywords_tab(result) -> None:
    if not result.keyword_rows:
        st.info("表示できるキーワードがありません。")
        if result.applied_stop_words:
            st.write("適用済み除外語: " + "、".join(result.applied_stop_words))
            if st.button("AI除外語を元に戻す", key="restore_empty_keywords"):
                analysis_settings = st.session_state.get("analysis_settings", AnalysisSettings())
                restore_stopword_feedback(
                    result,
                    cooccurrence_window=analysis_settings.cooccurrence_window,
                    cooccurrence_minimum=analysis_settings.cooccurrence_minimum,
                )
                st.rerun()
        return
    st.subheader("AIによる一般会話語フィードバック")
    st.write(
        "現在の全抽出語と出現回数をOpenAI APIで評価し、分析上の情報量が少ない"
        "一般会話語を候補として提示します。確認するまで単語は削除されません。"
    )
    st.caption("APIの利用量に応じてOpenAI API料金が発生します。")
    with st.expander("OpenAI APIによる除外候補を作成", expanded=not result.stopword_suggestions):
        api_key = _openai_api_key()
        if api_key:
            st.success("OPENAI_API_KEYを検出しました。値は表示・保存しません。")
        else:
            st.warning("OPENAI_API_KEYが未設定です。Streamlit Secretsまたは環境変数へ設定してください。")
        consent = st.checkbox(
            "全抽出語と出現回数をOpenAI APIへ送信することに同意します",
            key="stopword_api_consent",
        )
        model = st.text_input(
            "モデル", value=_openai_stopword_model(),
            help="OPENAI_STOPWORD_MODEL環境変数でも変更できます。",
        )
        guidance = st.text_area(
            "追加フィードバック（任意）",
            placeholder="例：技術用語と商品名は必ず残し、相づちだけを厳しく除外して",
            key="stopword_guidance",
        )
        max_attempts = st.slider("最大試行回数", 1, 5, 3, key="stopword_attempts")
        if st.button(
            "AIで除外候補を作成／再評価",
            disabled=not (api_key and consent),
            type="primary",
        ):
            cache_key = vocabulary_fingerprint(result.word_frequencies, model, guidance)
            cache = st.session_state.setdefault("stopword_review_cache", {})
            review = cache.get(cache_key)
            if review is None:
                retry_status = st.empty()
                try:
                    with st.spinner("全語彙を評価しています…"):
                        review = review_stopwords(
                            result.word_frequencies,
                            api_key=api_key,
                            model=model,
                            user_guidance=guidance,
                            max_attempts=max_attempts,
                            on_retry=lambda attempt, error: retry_status.warning(
                                f"結果を検証できなかったため再試行します "
                                f"({attempt}/{max_attempts}): {error}"
                            ),
                        )
                    cache[cache_key] = review
                    retry_status.empty()
                except Exception as exc:
                    st.error(f"候補生成に失敗しました: {exc}")
                    review = None
            else:
                st.info("同じ語彙・モデル・追加指示のキャッシュを利用しました。")
            if review is not None:
                result.stopword_suggestions = [
                    {
                        "除外する": item.confidence >= 0.65,
                        "単語": item.word,
                        "分類": item.category,
                        "理由": item.reason,
                        "確信度": item.confidence,
                    }
                    for item in review.suggestions
                ]
                result.stopword_review_metadata = {
                    "summary": review.summary,
                    "model": review.model,
                    "attempts": review.attempts,
                    "request_id": review.request_id,
                }
                st.rerun()
    if result.stopword_suggestions:
        metadata = result.stopword_review_metadata
        st.caption(
            f"{metadata.get('summary', '')}　"
            f"モデル: {metadata.get('model', '—')} / 試行: {metadata.get('attempts', '—')}"
        )
        suggestion_frame = pd.DataFrame(result.stopword_suggestions)
        edited = st.data_editor(
            suggestion_frame,
            hide_index=True,
            use_container_width=True,
            disabled=["単語", "分類", "理由", "確信度"],
            column_config={
                "除外する": st.column_config.CheckboxColumn("除外する"),
                "確信度": st.column_config.ProgressColumn(
                    "確信度", min_value=0.0, max_value=1.0, format="%.0%%"
                ),
            },
            key="stopword_review_editor",
        )
        selected_words = set(edited.loc[edited["除外する"], "単語"].astype(str))
        st.write(f"選択中: {len(selected_words)}語")
        remember = st.checkbox(
            "この判断をセッション中の次回解析にも適用する",
            value=True,
            key="remember_stopword_feedback",
        )
        if st.button("選択した語を除外して分析結果を更新", disabled=not selected_words):
            analysis_settings = st.session_state.get("analysis_settings", AnalysisSettings())
            apply_stopword_feedback(
                result,
                selected_words,
                cooccurrence_window=analysis_settings.cooccurrence_window,
                cooccurrence_minimum=analysis_settings.cooccurrence_minimum,
            )
            result.stopword_suggestions = [
                row for row in result.stopword_suggestions if row["単語"] not in selected_words
            ]
            if remember:
                learned = st.session_state.setdefault("learned_stop_words", set())
                learned.update(selected_words)
            st.success(f"{len(selected_words)}語を除外し、関連する分析結果を更新しました。")
            st.rerun()
    if result.applied_stop_words:
        with st.expander(f"適用済み除外語（{len(result.applied_stop_words)}語）"):
            st.write("、".join(result.applied_stop_words))
            if st.button("適用済みのAI除外語をすべて元に戻す"):
                analysis_settings = st.session_state.get("analysis_settings", AnalysisSettings())
                restore_stopword_feedback(
                    result,
                    cooccurrence_window=analysis_settings.cooccurrence_window,
                    cooccurrence_minimum=analysis_settings.cooccurrence_minimum,
                )
                st.rerun()
    learned_words = st.session_state.get("learned_stop_words", set())
    if learned_words:
        with st.expander(f"次回解析にも適用する記憶（{len(learned_words)}語）"):
            st.write("、".join(sorted(learned_words)))
            if st.button("次回解析用の記憶を消去"):
                st.session_state.learned_stop_words = set()
                st.rerun()
    frequency = pd.DataFrame(list(result.word_frequencies.items()), columns=["単語", "回数"]).head(50)
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(px.bar(frequency.head(20), x="回数", y="単語", orientation="h",
                               title="頻出語", height=520), use_container_width=True)
    with right:
        try:
            from wordcloud import WordCloud
            font_candidates = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
            ]
            font_path = next((path for path in font_candidates if Path(path).exists()), None)
            cloud = WordCloud(
                width=1000, height=550, background_color="white",
                font_path=font_path, max_words=200, collocations=False,
            ).generate_from_frequencies(result.word_frequencies)
            st.image(cloud.to_array(), caption="ワードクラウド", use_container_width=True)
        except Exception as exc:
            st.info(f"ワードクラウドを生成できませんでした（頻出語表は利用できます）: {exc}")
    top_words = frequency["単語"].head(15).tolist()
    timeline = pd.DataFrame(result.keyword_rows)
    timeline = timeline[timeline["normalized_word"].isin(top_words)]
    st.plotly_chart(px.scatter(
        timeline, x="start", y="normalized_word", color="speaker_id",
        hover_data=["word", "segment_id"], title="キーワード出現位置",
        labels={"start": "時間（秒）", "normalized_word": "単語"},
    ), use_container_width=True)
    st.subheader("共起ネットワーク")
    if result.cooccurrence_edges:
        try:
            import networkx as nx
            graph = nx.Graph()
            for node in result.cooccurrence_nodes:
                graph.add_node(node["word"], size=node["frequency"])
            for edge in result.cooccurrence_edges:
                graph.add_edge(edge["source"], edge["target"], weight=edge["normalized_weight"])
            positions = nx.spring_layout(graph, seed=42)
            edge_x, edge_y = [], []
            for source, target in graph.edges():
                edge_x += [positions[source][0], positions[target][0], None]
                edge_y += [positions[source][1], positions[target][1], None]
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                        line=dict(width=0.7, color="#9AA5B1"), hoverinfo="skip"))
            figure.add_trace(go.Scatter(
                x=[positions[n][0] for n in graph], y=[positions[n][1] for n in graph],
                text=list(graph), mode="markers+text", textposition="top center",
                marker=dict(size=[8 + graph.nodes[n]["size"] * 2 for n in graph], color="#367BF5"),
            ))
            figure.update_layout(height=600, showlegend=False, xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(figure, use_container_width=True)
        except ImportError:
            st.dataframe(result.cooccurrence_edges, hide_index=True)
    else:
        st.caption("設定した最小出現数を満たす共起はありません。")


def _openai_api_key() -> str | None:
    """Read the API key without placing it in session state or logs."""
    try:
        secret = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret = None
    return str(secret) if secret else os.getenv("OPENAI_API_KEY")


def _openai_stopword_model() -> str:
    """Resolve the optional model override from Secrets or environment."""
    try:
        secret = st.secrets.get("OPENAI_STOPWORD_MODEL")
    except Exception:
        secret = None
    return str(secret) if secret else os.getenv("OPENAI_STOPWORD_MODEL", DEFAULT_MODEL)


def topics_tab(result) -> None:
    if not result.segments or result.points_2d is None:
        st.info("クラスタ結果がありません。")
        return
    topic_names = {}
    topic_ids = sorted(result.topic_keywords)
    if topic_ids:
        columns = st.columns(min(4, len(topic_ids)))
        for index, topic_id in enumerate(topic_ids):
            default = " / ".join(result.topic_keywords[topic_id][:3]) or f"Topic {topic_id}"
            topic_names[topic_id] = columns[index % len(columns)].text_input(
                f"Topic {topic_id} 表示名", default, key=f"topic_name_{topic_id}"
            )
    st.plotly_chart(cluster_chart(result), use_container_width=True)
    rows = [{
        "time": (s.start + s.end) / 2,
        "topic": topic_names.get(s.topic_id, f"Topic {s.topic_id}"), "text": s.text
    } for s in result.segments]
    st.plotly_chart(px.scatter(
        rows, x="time", y="topic", color="topic", hover_data=["text"],
        title="トピック推移", labels={"time": "時間（秒）", "topic": "トピック"},
    ), use_container_width=True)
    st.json({f"Topic {key}": value for key, value in result.topic_keywords.items()})


def highlights_tab(result) -> None:
    if not result.segments:
        st.info("ハイライト結果がありません。")
        return
    st.plotly_chart(highlight_chart(result), use_container_width=True)
    for item in result.highlights:
        with st.expander(f"#{item.id + 1}　{item.score:.1f}点　{format_timecode(item.start)}"):
            st.write(item.transcript)
            st.bar_chart(pd.DataFrame([item.score_components]))
            st.audio(result.audio_path, start_time=int(item.start), end_time=int(item.end))
    if not result.highlights:
        st.caption("閾値を超える候補はありません。詳細設定で閾値を下げて再解析できます。")
    else:
        export_rows = [asdict(item) for item in result.highlights]
        columns = st.columns(3)
        columns[0].download_button(
            "CSV", pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8-sig"),
            "highlights.csv", "text/csv",
        )
        columns[1].download_button(
            "JSON", json.dumps(export_rows, ensure_ascii=False, indent=2),
            "highlights.json", "application/json",
        )
        edl = "\n".join(
            f"{index + 1:03d}  AX       AA/V  C        "
            f"{format_timecode(item.start).replace(':', '')} "
            f"{format_timecode(item.end).replace(':', '')} "
            f"{format_timecode(item.start).replace(':', '')} "
            f"{format_timecode(item.end).replace(':', '')}"
            for index, item in enumerate(result.highlights)
        )
        columns[2].download_button("EDL", edl, "highlights.edl", "text/plain")


def search_tab(result) -> None:
    query = st.text_input("自然言語で検索", placeholder="例：料金について説明している箇所")
    top_k = st.slider("結果件数", 1, 20, 5)
    if query:
        rows = search(result, query, top_k)
        for row in rows:
            st.markdown(f"**類似度 {row['similarity']:.3f}　{format_timecode(row['start'])}**")
            st.write(row["text"])
            st.audio(result.audio_path, start_time=int(row["start"]), end_time=int(row["end"]))


def export_tab(result) -> None:
    st.write("解析結果をまとめてダウンロードできます。アップロード元ファイルはZIPに含めません。")
    bundle = build_export_zip(result)
    st.download_button(
        "解析結果ZIPをダウンロード", bundle,
        f"audio_intelligence_{result.job_id[:8]}.zip", "application/zip",
        type="primary",
    )
    st.download_button(
        "メタデータJSON", json.dumps(result.metadata(), ensure_ascii=False, indent=2),
        "metadata.json", "application/json",
    )


st.title("Audio Intelligence")
st.caption("音声・動画の音響、発話、キーワード、感情、トピック、ハイライトを一つの時間軸で分析")
input_mode, source, settings, hf_token = sidebar_settings()
if st.sidebar.button("解析を実行", type="primary", use_container_width=True):
    execute(input_mode, source, settings, hf_token)

result = st.session_state.get("result")
if result is None:
    st.info("左のサイドバーから音声・動画またはYouTube URLを指定し、「解析を実行」を押してください。")
    st.markdown("対応形式: WAV / MP3 / M4A / FLAC / OGG / MP4 / MOV / WEBM / MKV")
else:
    if result.features and result.features.duration > 7200:
        st.warning("120分を超える長時間音声です。処理時間とメモリ使用量にご注意ください。")
    page_names = [
        "概要", "音響", "文字起こし", "話者・感情", "キーワード",
        "トピック・クラスタ", "ハイライト", "意味検索", "エクスポート",
    ]
    renderers = [
        overview_tab, audio_tab, transcript_tab, speakers_emotions_tab,
        keywords_tab, topics_tab, highlights_tab, search_tab, export_tab,
    ]
    selected_page = st.radio(
        "表示ページ", page_names, horizontal=True,
        label_visibility="collapsed", key="analysis_page",
    )
    dict(zip(page_names, renderers))[selected_page](result)
