# 音声・動画インテリジェンス分析Webアプリ 仕様書
## Codexエージェント向け実装指示書

## 1. アプリ概要

本アプリは、YouTube URLまたはアップロードされた音声・動画ファイルを解析し、音響特徴、文字起こし、頻出語、意味クラスタ、話者、感情、トピック、盛り上がり、類似発話を統合的に可視化するStreamlit Webアプリである。

元のJupyter Notebookに実装済みの以下の機能を維持する。

- YouTube音声取得
- WAV変換
- スペクトログラム
- 波形表示
- RMS音量ピーク検出
- FFT周波数解析
- ピッチ抽出
- スペクトルエントロピー
- ハーモニック・パーカッシブ音源分離
- テンポ・ビート推定
- Whisperによる文字起こし
- 日本語形態素解析
- 頻出語集計
- ワードクラウド
- 文埋め込み
- KMeansクラスタリング
- PCAによるクラスタ可視化

これらに加えて、以下を実装する。

1. 感情・雰囲気の時系列分析
2. 話者分離
3. 要約と自動章立て
4. キーワードの時間マッピング
5. 共起ネットワーク
6. トピック推移分析
10. 盛り上がり箇所の自動検出
12. 類似発話検索

## 2. 最重要要件

- 元Notebookの解析機能を削除しない。
- Notebookの逐次実行コードを、再利用可能なPythonモジュールへ分割する。
- UI処理と解析処理を分離する。
- 長時間処理には進捗状況を表示する。
- 解析途中でエラーが起きても、完了済みの結果は表示できるようにする。
- 解析結果をセッション単位で保持する。
- 同一入力・同一設定ではキャッシュを利用する。
- グラフは原則としてPlotlyを使用し、ズーム、ホバー、凡例切替、範囲選択を可能にする。
- 音声タイムライン上の各分析結果は、秒単位の共通時間軸で同期させる。
- 日本語を標準対応言語とし、Whisperの自動言語判定も選択可能にする。
- CPU環境でも動作し、GPUがある場合は自動的に利用する。
- Streamlit Community CloudなどのCPU・メモリ制約を考慮する。
- YouTube取得が利用できない環境でも、ファイルアップロードによる解析は必ず利用可能にする。

## 3. 想定ユーザー

- YouTube動画やポッドキャストを分析したいユーザー
- 会議、インタビュー、講演を分析したいユーザー
- 動画のハイライト候補を抽出したい編集者
- 発話内容、話者、感情、トピックの変化を調査したい研究者
- 長時間音声から特定の話題を意味検索したいユーザー

## 4. 入力仕様

### 4.1 入力方式

サイドバーで次のいずれかを選択する。

- YouTube URL
- 音声ファイルアップロード
- 動画ファイルアップロード

### 4.2 対応形式

- 音声: WAV、MP3、M4A、FLAC、OGG
- 動画: MP4、MOV、WEBM、MKV
- YouTube URL: yt-dlpで取得可能な公開URL

### 4.3 入力制限

初期値として以下を採用し、設定ファイルで変更可能にする。

- 最大ファイルサイズ: 500 MB
- 推奨最大時間: 120分
- 120分を超える場合は警告を表示する
- 長時間音声はチャンク分割して処理する
- 一時ファイル名にユーザー入力文字列を直接使用しない
- URLは許可されたスキームのみ受け付ける
- 一時ディレクトリはセッションまたはジョブ単位で分離する

## 5. 解析設定UI

サイドバーに以下を配置する。

### 基本設定

- 入力方式
- 言語: 自動判定、日本語、英語など
- Whisperモデル: tiny、base、small、medium、large-v3
- 解析対象区間
- サンプルレート
- 高速モード
- 高精度モード

### 個別機能の有効化

- 音響分析
- 文字起こし
- 話者分離
- 感情分析
- 要約・章立て
- ワードクラウド
- 意味クラスタリング
- キーワード時間分析
- 共起ネットワーク
- トピック推移
- 盛り上がり検出
- 類似発話検索

### 詳細設定

- RMSピーク閾値
- 最小ピーク間隔
- クラスタ数: 自動または手動
- 最大クラスタ数
- 最小発話長
- ストップワード編集
- ワードクラウド最大語数
- 共起ウィンドウサイズ
- 共起ネットワーク最小出現数
- トピック最小サイズ
- 盛り上がり検出の重み
- 類似検索の取得件数

## 6. 画面構成

トップレベルは以下のタブ構成とする。

1. 概要
2. 音響
3. 文字起こし
4. 話者・感情
5. キーワード
6. トピック・クラスタ
7. ハイライト
8. 意味検索
9. エクスポート

### 6.1 概要タブ

上部にメトリクスカードを表示する。

- 音声時間
- 推定テンポ
- 話者数
- 発話区間数
- 総文字数
- 推定トピック数
- ハイライト数
- 言語と判定確率

続いて以下を表示する。

- 音声プレイヤー
- 解析ステータス
- 自動要約
- 自動生成チャプター
- 上位キーワード
- 感情構成比
- 代表的なハイライト

### 6.2 音響タブ

元Notebookの機能をインタラクティブに表示する。

- 波形
- RMS
- RMSピーク
- スペクトログラム
- FFT
- ピッチ
- スペクトルエントロピー
- ビート位置
- テンポ
- ハーモニック音源プレイヤー
- パーカッシブ音源プレイヤー

すべての時系列グラフは共通の秒軸を使用する。

Plotlyのrange sliderまたは範囲選択を利用し、選択区間を拡大できるようにする。

### 6.3 文字起こしタブ

発話を行単位で表示する。

表示項目:

- 開始時刻
- 終了時刻
- 話者
- 発話テキスト
- Whisper信頼度
- 感情
- トピック
- クラスタ
- ハイライトスコア

機能:

- キーワード検索
- 話者フィルタ
- 感情フィルタ
- トピックフィルタ
- クラスタフィルタ
- 時間範囲フィルタ
- CSVダウンロード
- JSONダウンロード
- TXTダウンロード
- 行選択時に対応位置を確認できるタイムスタンプリンクまたは再生開始時刻表示

Streamlit標準の音声プレイヤーに直接シーク制御できない場合は、区間ごとの短い音声クリップを生成し、選択した発話を個別再生できるようにする。

## 7. 機能仕様

## 7.1 感情・雰囲気の時系列分析

### 目的

発話内容と音声特徴から、動画内の感情・雰囲気の変化を時系列で表示する。

### 入力

- 発話テキスト
- 発話開始・終了時刻
- 区間RMS
- 区間ピッチ統計
- 話速
- 無音率
- スペクトル特徴

### 出力

各発話に以下を付与する。

- emotion_label
- emotion_score
- emotion_probabilities
- valence
- arousal
- confidence

標準ラベル:

- neutral
- joy
- sadness
- anger
- fear
- surprise

### 実装方針

第一段階はテキスト感情モデルを利用する。
第二段階として音声特徴を統合し、valence-arousalスコアを補正する。

モデルは抽象化し、以下のインターフェースを持たせる。

```python
class EmotionAnalyzer:
    def analyze(
        self,
        segments: list[TranscriptSegment],
        audio_features: AudioFeatureSet,
    ) -> list[EmotionResult]:
        ...
```

### 可視化

- 感情ラベルの時系列帯
- valenceの折れ線
- arousalの折れ線
- 感情構成比
- 話者別感情比較
- 特定感情だけを表示するフィルタ

感情分析結果は断定表現を避け、「モデルによる推定」と明記する。

## 7.2 話者分離

### 目的

複数人の音声から、誰がどの区間を話したかを推定する。

### 出力

- speaker_id
- start
- end
- duration
- confidence
- overlapフラグ

標準表示名は「話者1」「話者2」とする。
ユーザーが表示名を編集できるようにする。

### 実装方針

- pyannote.audioなどの話者ダイアライゼーションモデルを利用できる構成
- Hugging Face Tokenが必要な実装は、環境変数またはStreamlit Secretsから読み込む
- Token未設定時は機能を無効化し、ほかの解析を継続する
- 話者区間とWhisper発話区間を時間重複率で対応付ける
- 重複発話はoverlapとして保持する
- 話者数は自動推定を既定とし、任意で最小・最大話者数を指定可能にする

### 可視化

- 話者別タイムライン
- 話者別発話時間
- 話者別発話回数
- 話者別平均話速
- 話者別感情分布
- 話者フィルタ付き文字起こし

## 7.3 要約と自動章立て

### 目的

長時間の文字起こしを短く要約し、話題の転換点からチャプターを生成する。

### 出力

- short_summary
- detailed_summary
- key_points
- chapters
- chapter_title
- chapter_summary
- chapter_start
- chapter_end
- representative_segments

### 要約方式

長文を直接1回で処理せず、階層型要約を採用する。

1. 発話を時間またはトークン数でチャンク化
2. 各チャンクを要約
3. チャンク要約を統合
4. 全体要約と重要ポイントを生成

外部LLM APIを必須にしない。
ローカルモデルまたは抽出的要約へフォールバック可能にする。

### 章立て方式

以下を組み合わせて境界候補を算出する。

- 連続発話間の埋め込み類似度低下
- トピックラベル変化
- 長い無音
- 話者交代
- キーワード分布変化

隣接章が短すぎる場合は統合する。
初期値として最小チャプター長を60秒とする。

### 可視化

- チャプター一覧
- 横方向タイムライン
- 各章のタイトル、要約、開始時刻
- 章クリックで該当発話をフィルタ
- YouTube概要欄用のタイムスタンプ形式をコピーまたはダウンロード

## 7.4 キーワードの時間マッピング

### 目的

単語が動画のどこで、どの程度使われたかを可視化する。

### データ生成

Whisperのword_timestampsを利用する。
形態素解析で原形化し、ストップワードを除去する。

各単語について以下を保持する。

- word
- normalized_word
- start
- end
- segment_id
- speaker_id
- topic_id
- sentiment
- confidence

### 可視化

- 上位語の時間ヒートマップ
- 単語ごとの出現位置ラグプロット
- 時間窓ごとの頻度推移
- 話者別キーワード比較
- キーワード選択時の該当発話一覧

時間窓は30秒、60秒、120秒などから選択できるようにする。

## 7.5 共起ネットワーク

### 目的

同じ発話または近接する文脈で使用される語の関係をネットワーク化する。

### 共起定義

ユーザーが次から選択できるようにする。

- 同一発話内
- 前後N語
- 同一時間窓内
- 同一チャプター内

### ノード属性

- word
- frequency
- centrality
- community
- topic_distribution

### エッジ属性

- source
- target
- cooccurrence_count
- normalized_weight

### フィルタ

- 最小単語頻度
- 最小共起回数
- 最大ノード数
- 品詞
- 話者
- チャプター
- トピック

### 可視化

PyVisまたはPlotlyを用いてインタラクティブネットワークを表示する。

- ノードサイズ: 出現頻度
- エッジ太さ: 共起強度
- ノード色: コミュニティまたはトピック
- ホバー: 頻度、中心性、代表発話
- ノード選択: 関連発話一覧を更新

ネットワークが過密になる場合は上位ノードへ自動制限する。

## 7.6 トピック推移分析

### 目的

動画内の話題を抽出し、時間とともにどのトピックへ移行したかを表示する。

### 実装方針

第一候補はBERTopicとする。
依存関係またはリソース制約がある場合は、既存の文埋め込みとクラスタリングを利用するフォールバックを設ける。

- 埋め込みは既存のSentenceTransformer結果を再利用する
- UMAP、HDBSCAN、c-TF-IDFを利用可能にする
- 短い発話は一定時間窓でまとめてからトピック分析する
- 外れ値トピックは「その他」として扱う
- トピック名は代表語から自動生成する
- ユーザーがトピック名を編集できるようにする

### 出力

- topic_id
- topic_label
- topic_probability
- topic_keywords
- segment_ids
- start
- end

### 可視化

- トピック占有率
- トピック時系列
- トピック別キーワード
- トピック間距離
- トピック別代表発話
- 話者別トピック分布

## 7.7 盛り上がり箇所の自動検出

### 目的

音響、発話、感情、内容を統合し、動画のハイライト候補を検出する。

### 特徴量

時間窓ごとに以下を計算する。

- RMS平均・最大・変化量
- ピッチ平均・変化量
- onset strength
- beat density
- 発話密度
- 話速
- 感情arousal
- surprise確率
- キーワード重要度
- トピック転換度
- 話者交代頻度
- 無音後の発話開始
- スペクトルエントロピー
- 音響新規性

### スコア

各特徴を0〜1へ正規化して加重和を計算する。

```text
highlight_score =
    w_rms * rms_score
  + w_pitch * pitch_score
  + w_onset * onset_score
  + w_speech * speech_density_score
  + w_emotion * arousal_score
  + w_topic * topic_change_score
  + w_keyword * keyword_score
```

初期重みを設定し、UIから変更可能にする。
スコアは必ず0〜100に変換する。

### 区間生成

- スコアが閾値を超えた時間窓を候補とする
- 近接候補を結合する
- 最小長と最大長を適用する
- 前後に余白を追加する
- 重複候補は高スコア側へ統合する

初期値:

- 時間窓: 10秒
- ステップ: 2秒
- 最小ハイライト長: 15秒
- 最大ハイライト長: 90秒
- 前後余白: 3秒

### 可視化

- 0〜100の盛り上がり曲線
- ハイライト区間の帯表示
- ハイライトランキング
- 各候補のスコア内訳
- 各区間の音声クリップ再生
- CSV、JSON、EDL形式の出力

## 7.8 類似発話検索

### 目的

キーワード完全一致ではなく、意味が近い発話を検索する。

### データ

既存のSentenceTransformer埋め込みを再利用する。
発話単位とチャプター単位の両方を検索対象にする。

### 検索処理

1. ユーザーの自然言語クエリを埋め込みへ変換
2. セグメント埋め込みとのコサイン類似度を計算
3. 上位K件を返す
4. 必要に応じてMMRで重複を抑制する

### UI

- 検索入力欄
- 検索対象: 発話、チャプター、両方
- 結果件数
- 話者フィルタ
- 時間範囲
- トピックフィルタ
- 類似度閾値

### 結果表示

- 類似度
- 開始・終了時刻
- 話者
- 発話内容
- トピック
- 感情
- 対応する短い音声クリップ
- 前後の文脈

## 8. 共通データモデル

dataclassesまたはPydanticモデルを使用する。

```python
class TranscriptWord:
    start: float | None
    end: float | None
    text: str
    normalized: str | None
    probability: float | None

class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord]
    speaker_id: str | None
    emotion_label: str | None
    emotion_score: float | None
    valence: float | None
    arousal: float | None
    cluster_id: int | None
    topic_id: int | None
    highlight_score: float | None

class AudioFeatureSet:
    sample_rate: int
    duration: float
    tempo: float | None
    waveform: object
    rms: object
    rms_times: object
    pitch: object
    pitch_times: object
    entropy: object
    entropy_times: object
    beat_times: object
    peak_times: object
    spectrogram_db: object
    fft_frequencies: object
    fft_values: object

class Chapter:
    id: int
    title: str
    summary: str
    start: float
    end: float
    segment_ids: list[int]

class Highlight:
    id: int
    start: float
    end: float
    score: float
    score_components: dict[str, float]
    title: str
    transcript: str
```

NumPy配列はJSONへ直接保存せず、NPZ、Parquet、CSVなど用途に応じた形式へ保存する。

## 9. 推奨ディレクトリ構成

```text
project/
├── app.py
├── pages/
│   ├── overview.py
│   ├── audio_analysis.py
│   ├── transcript.py
│   ├── speakers_emotions.py
│   ├── keywords.py
│   ├── topics_clusters.py
│   ├── highlights.py
│   ├── semantic_search.py
│   └── export.py
├── core/
│   ├── config.py
│   ├── models.py
│   ├── pipeline.py
│   ├── job_manager.py
│   └── cache.py
├── services/
│   ├── media_loader.py
│   ├── audio_preprocessor.py
│   ├── acoustic_analyzer.py
│   ├── transcriber.py
│   ├── diarizer.py
│   ├── emotion_analyzer.py
│   ├── text_processor.py
│   ├── embedding_service.py
│   ├── cluster_analyzer.py
│   ├── topic_analyzer.py
│   ├── chapter_generator.py
│   ├── keyword_timeline.py
│   ├── cooccurrence.py
│   ├── highlight_detector.py
│   ├── semantic_search.py
│   └── exporter.py
├── ui/
│   ├── components.py
│   ├── charts.py
│   ├── timeline.py
│   └── theme.py
├── utils/
│   ├── audio.py
│   ├── timecode.py
│   ├── files.py
│   └── logging.py
├── tests/
│   ├── test_acoustic.py
│   ├── test_transcription.py
│   ├── test_diarization.py
│   ├── test_topics.py
│   ├── test_highlights.py
│   └── test_search.py
├── assets/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── requirements.txt
├── README.md
└── Dockerfile
```

## 10. パイプライン

```text
入力
  ↓
安全な一時保存
  ↓
ffmpegでWAV正規化
  ↓
音響解析
  ↓
Whisper文字起こし
  ↓
話者分離
  ↓
発話と話者の時間対応付け
  ↓
形態素解析・単語統計
  ↓
文埋め込み
  ↓
意味クラスタ
  ↓
トピック分析
  ↓
感情分析
  ↓
要約・章立て
  ↓
キーワード時間マッピング
  ↓
共起ネットワーク
  ↓
盛り上がり検出
  ↓
意味検索インデックス
  ↓
可視化・エクスポート
```

音響解析、文字起こし、話者分離など独立性の高い処理は、中間結果を保存して再利用する。

## 11. Streamlit状態管理

`st.session_state`には大型モデルや巨大配列を直接複製しない。

保持対象:

- job_id
- input metadata
- selected settings
- result paths
- summary metrics
- UI filters
- speaker display names
- topic display names

モデルは`st.cache_resource`でキャッシュする。

- WhisperModel
- SentenceTransformer
- 感情モデル
- 話者分離モデル
- トピックモデル

解析結果は`st.cache_data`またはジョブディレクトリに保存する。
キャッシュキーには以下を含める。

- 入力ファイルハッシュ
- モデル名
- 言語
- 解析設定
- アプリ側のパイプラインバージョン

## 12. 進捗・エラー処理

解析中は以下の段階を表示する。

1. 入力確認
2. 音声変換
3. 音響解析
4. 文字起こし
5. 話者分離
6. 感情分析
7. テキスト分析
8. トピック分析
9. 要約・章立て
10. ハイライト生成
11. 可視化準備
12. 完了

各段階で進捗率とメッセージを更新する。

エラーは機能単位で捕捉する。

例:

- 話者分離失敗 → 話者をUNKNOWNとして継続
- 感情モデル失敗 → 感情タブのみ無効化
- BERTopic失敗 → KMeansクラスタへフォールバック
- GPUメモリ不足 → CPUまたは小型モデルへ切替
- YouTube取得失敗 → ファイルアップロードを案内
- 日本語フォントなし → テーブル表示を継続し、ワードクラウドのみ停止

## 13. エクスポート仕様

ZIP一括ダウンロードを提供する。

最低限含める。

```text
analysis_result/
├── metadata.json
├── summary.json
├── transcript.txt
├── transcript_segments.json
├── transcript_segments.csv
├── word_frequencies.csv
├── clusters.csv
├── topics.csv
├── chapters.csv
├── chapters_youtube.txt
├── emotions.csv
├── speaker_stats.csv
├── keyword_timeline.csv
├── cooccurrence_nodes.csv
├── cooccurrence_edges.csv
├── highlights.csv
├── highlights.json
├── semantic_search_index.npz
├── harmonic.wav
├── percussive.wav
└── figures/
```

個別ダウンロードも可能にする。

## 14. 非機能要件

### パフォーマンス

- モデルは再ロードしない
- 埋め込みは1回生成してクラスタ、トピック、検索で共有する
- 波形表示用データはダウンサンプリングする
- スペクトログラムは表示解像度に合わせて制限する
- 長時間音声はチャンク処理する
- 大規模ネットワークはノード数を制限する
- UI操作だけで重い解析を再実行しない

### セキュリティ

- アップロードファイル名を信用しない
- shell=Trueを使用しない
- subprocessは引数配列で呼び出す
- API Tokenを画面やログへ出力しない
- 一時ファイルを一定時間後に削除する
- 外部URL取得にはタイムアウトを設定する
- MIMEタイプと実ファイル形式を確認する

### プライバシー

- アップロードデータを永続保存しないことを既定とする
- セッション終了または有効期限後に削除する
- 外部APIへ送信する機能は明示的な同意を必要とする
- ローカル処理と外部API処理をUI上で区別する

### アクセシビリティ

- 色だけで状態を区別しない
- グラフにタイトル、軸名、単位を付ける
- テーブルでも同じ情報へアクセス可能にする
- コントラストを確保する
- モバイル幅でも主要操作を可能にする

## 15. 推奨依存関係

既存Notebookの以下を維持する。

- yt-dlp
- librosa
- numpy
- scipy
- soundfile
- faster-whisper
- wordcloud
- janome
- sentence-transformers
- scikit-learn
- pandas
- matplotlib

追加候補:

- streamlit
- plotly
- pyannote.audio
- transformers
- torch
- bertopic
- umap-learn
- hdbscan
- networkx
- pyvis
- pydantic
- pyarrow

依存関係は固定しすぎず、互換性を確認した上でrequirements.txtへ記載する。
CPU専用環境とGPU環境の導入方法をREADMEで分ける。

## 16. テスト要件

### 単体テスト

- 秒からタイムコードへの変換
- 発話と話者区間の対応付け
- ストップワード除去
- 共起行列生成
- チャプター境界生成
- ハイライト候補の結合
- コサイン類似検索
- 空入力、短音声、無音音声の処理

### 統合テスト

- ローカル音声から全解析が完了する
- YouTube URLから音声取得できる
- GPUなしで処理できる
- 話者分離Tokenなしでも処理が継続する
- 解析途中の一機能失敗で全体が停止しない
- ZIP出力を展開できる

### UIテスト

- 入力方式切替
- 設定変更
- フィルタ
- タブ移動
- ダウンロード
- 再解析
- エラー表示
- スマートフォン幅

## 17. 受け入れ条件

以下をすべて満たした時点で初期版完成とする。

1. YouTube URLまたはアップロードファイルを解析できる。
2. 元Notebookの音響分析結果がWeb画面で確認できる。
3. 文字起こし、ワードクラウド、意味クラスタが表示される。
4. 話者別に発話を分類できる。
5. 感情の時間変化が表示される。
6. 要約とタイムスタンプ付きチャプターが生成される。
7. キーワードの出現位置を時間軸で確認できる。
8. 共起ネットワークを操作できる。
9. トピックの時間推移を確認できる。
10. 盛り上がり候補がランキング表示され、該当音声を再生できる。
11. 自然言語クエリで類似発話を検索できる。
12. CSV、JSON、TXT、画像、音声、ZIPをダウンロードできる。
13. 一部の追加モデルが利用できなくても、既存機能は動作する。
14. READMEの手順だけでローカル起動できる。
15. Dockerで起動できる。

## 18. Codexエージェントへの実装指示

以下の順番で実装すること。

### Phase 1: Notebookのリファクタリング

- 元Notebookを読み、既存の計算式、パラメータ、出力を確認する。
- 既存コードをそのままapp.pyへ貼り付けない。
- 音声取得、前処理、音響解析、文字起こし、テキスト解析を関数・クラスへ分割する。
- 元Notebookと同じ入力に対して主要な統計値が大きく変わらないことを確認する。

### Phase 2: Streamlit基本画面

- 入力
- 設定
- 解析実行
- 進捗
- 概要
- 音響
- 文字起こし
- エクスポート

を先に完成させる。

### Phase 3: 追加分析

以下の順で実装する。

1. 埋め込み共有基盤
2. 類似発話検索
3. キーワード時間マッピング
4. 感情分析
5. 話者分離
6. トピック推移
7. 自動章立て
8. 共起ネットワーク
9. 盛り上がり検出

### Phase 4: 品質改善

- Plotlyによるインタラクティブ化
- キャッシュ
- 長時間音声対応
- フォールバック
- テスト
- Docker
- README
- UI調整

## 19. Codexへの制約

- 元Notebookを唯一の既存仕様として扱う。
- 元機能を削除または簡略化する場合は、理由をコメントとREADMEへ記載する。
- 解析ロジックにUI依存コードを入れない。
- グローバル変数への解析状態保存を避ける。
- モデルロードをループ内で行わない。
- エラーを握りつぶさない。
- ダミーデータで完成扱いにしない。
- 未実装機能は明示的に「未実装」と表示する。
- APIキーやTokenをコードへ直書きしない。
- 大型生成AI APIを必須依存にしない。
- 各サービスは型ヒントとdocstringを持たせる。
- 主要関数にはテストを追加する。

## 20. 完成時の提出物

- Streamlitアプリ一式
- requirements.txt
- Dockerfile
- README.md
- .streamlit/config.toml
- .streamlit/secrets.toml.example
- tests
- サンプル出力
- Notebookからの移行対応表
- 既知の制限一覧
- 使用モデルとライセンス一覧
