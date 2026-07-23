# Audio Intelligence

YouTube URLまたはアップロードした音声・動画を、共通の秒軸で分析するStreamlitアプリです。
音響特徴、Whisper文字起こし、頻出語、意味クラスタに加え、感情推定、話者分離、
要約・章立て、キーワード時間分析、共起、トピック、ハイライト、意味検索をまとめて表示します。

## 起動

Python 3.11または3.12と、OSの`ffmpeg`が必要です。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

初回のWhisperおよびSentenceTransformer実行時はモデルをダウンロードするため時間がかかります。
CPUでは`tiny`または`base`と「高速」モードを推奨します。CUDA対応PyTorchが利用可能な場合、
faster-whisperはGPUを自動利用します。

Dockerの場合:

```bash
docker build -t audio-intelligence .
docker run --rm -p 8501:8501 audio-intelligence
```

## 話者分離

話者分離だけは任意機能です。`pip install -r requirements-optional.txt`を実行し、
pyannoteのモデル利用条件へ同意したうえで、`.streamlit/secrets.toml`に以下を設定します。

```toml
HF_TOKEN = "hf_..."
```

Tokenまたは依存関係がない場合は話者分離だけをスキップし、ほかの解析は継続します。
既定では全発話を`SPEAKER_00`として扱います。

## OpenAIによる一般会話語フィードバック

「キーワード」タブでは、抽出された全語彙と出現回数をOpenAI APIへ送り、
フィラー・相づち・汎用語などの除外候補を理由と確信度付きで提案できます。
候補は自動削除されません。テーブルで確認・選択してから適用し、適用後もまとめて元に戻せます。
承認した語は任意でStreamlitセッション内に記憶し、同じブラウザセッション中の次回解析へ反映できます。

APIキーはコードへ書かず、環境変数または`.streamlit/secrets.toml`へ設定してください。

```bash
export OPENAI_API_KEY="sk-..."
# 任意。未設定時は gpt-5.6-sol
export OPENAI_STOPWORD_MODEL="gpt-5.6-sol"
```

```toml
OPENAI_API_KEY = "sk-..."
```

API呼び出しにはResponses APIの構造化出力を使用します。SDK側で接続エラー、408、409、
429、5xxを最大2回再試行し、さらにアプリ側で空結果・スキーマ不整合・入力にない単語などを
検出した場合も指数バックオフ付きで再試行します。401など設定起因のエラーは無駄に再試行しません。

外部へ送る内容は現在の抽出語、出現回数、画面で入力した追加指示です。音声ファイルや全文文字起こしは
この機能では送信しません。送信前に画面上の同意チェックが必要です。

## 実装上のフォールバック

- SentenceTransformerを取得できない場合、文字n-gram TF-IDFでクラスタリングします。
- BERTopicは初期版の必須依存にせず、共有埋め込みのKMeans結果をトピックとして表示します。
- 感情は語彙規則と区間RMSを統合した軽量推定です。画面上でも推定であることを明示します。
- 要約は外部APIへ送信しない抽出的要約です。
- 日本語形態素解析はJanome、利用不可時はUnicode正規表現へフォールバックします。
- Streamlit標準プレイヤーの区間再生機能を使用します。ブラウザによって終了位置の挙動が異なります。

## プライバシーと制限

アップロードファイルは推測不能なジョブ用一時ディレクトリに保存し、エクスポートZIPには含めません。
外部送信はYouTube取得とモデル初回ダウンロードのみです。アプリプロセス終了前の一時ファイルTTL削除は
未実装のため、長期稼働環境ではOSの一時領域清掃を設定してください。推奨最大時間は120分、
最大アップロードは500 MBです。非常に長い音声のWhisperチャンク処理はfaster-whisper内部に依存します。

## Notebook移行対応表

| Notebook機能 | 実装先 |
|---|---|
| YouTube取得 / WAV化 | `utils/files.py` |
| STFT・RMS・ピーク・FFT・pyin・entropy・HPSS・tempo | `services/acoustic_analyzer.py` |
| faster-whisper文字起こし | `services/transcriber.py` |
| Janome頻出語 | `services/text_processor.py` |
| SentenceTransformer / KMeans / PCA | `services/text_processor.py` |
| 追加インサイト | `services/insight_analyzers.py` |
| 意味検索 | `services/semantic_search.py` |
| UI | `app.py`, `ui/charts.py` |

元Notebookの`frame_length=2048`、`hop_length=512`、RMSピーク検出、C2〜C7のpyin、
正規化スペクトルエントロピー、HPSS、テンポ計算、Hanning窓FFTを維持しています。

## 使用モデルとライセンス

- faster-whisper / CTranslate2: MIT（Whisperモデルの利用条件も確認してください）
- `paraphrase-multilingual-MiniLM-L12-v2`: Apache-2.0
- pyannote.audio: MIT。各Hugging Faceモデルには個別の利用条件があります

依存パッケージとモデルのライセンスは配布・商用利用前に各プロジェクトの最新版を確認してください。

## テスト

```bash
pip install -r requirements-dev.txt
python3 -m pytest
python3 -m compileall app.py core services ui utils
```
