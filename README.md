<div align="center">

# 🎧 Audio Intelligence

**音声・動画を、見える情報へ。**

YouTubeやアップロードファイルを解析するStreamlitアプリ。

</div>

---

## Analyze

| Audio | Language | Insight |
|---|---|---|
| 波形・RMS・FFT・ピッチ・テンポ | Whisper・話者・感情 | 要約・トピック・ハイライト・意味検索 |

OpenAI APIで相づちや汎用語を提案し、確認後に分析対象から除外できます。

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11+とffmpegが必要です。

## Secrets

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..." # 単語フィードバック
HF_TOKEN = "hf_..."       # 話者分離（任意）
```

話者分離を使う場合は`pip install -r requirements-optional.txt`も実行してください。

## Deploy

Streamlit Community CloudはGitHubへのpushを自動反映します。YouTubeが403を返す場合はファイルアップロードをご利用ください。

```bash
docker build -t audio-intelligence .
docker run --rm -p 8501:8501 audio-intelligence
```

## Test

```bash
pip install -r requirements-dev.txt
pytest
```

詳細仕様は[仕様書](streamlit_audio_intelligence_spec.md)を参照してください。
