# btc-edge-lab

BTC/USDT の過去データを複数の分析手法で解析し、トレーディングエッジを発見するプロジェクト。

## 概要

Binance API からローソク足データ、blockchain.info API からオンチェーンデータを取得し、以下の 3 つの観点から統計的に有意なエッジを探索する。

- **テクニカル分析**: RSI, MACD, ボリンジャーバンド, ATR, OBV 等の指標ベースのエッジ検証
- **統計的アノマリー検出**: 曜日効果、月別効果、時間帯効果、ボラティリティクラスタリング等
- **オンチェーン分析**: NVT Ratio、アクティブアドレス数、ハッシュレート等と価格の相関分析

## セットアップ

```bash
uv sync
```

Python 3.11+ が必要です。

## Agent Team での実行

本プロジェクトは Claude Code の Agent Team 機能を使い、以下の 4 エージェントが並行して分析を行います。

| エージェント | 担当ディレクトリ | 役割 |
|---|---|---|
| data-engineer | `src/fetcher/` | データ取得・クリーニング |
| ta-analyst | `src/technical/` | テクニカル指標によるエッジ検証 |
| anomaly-detective | `src/anomaly/` | 統計的アノマリーの検出 |
| onchain-analyst | `src/onchain/` | オンチェーンデータ分析 |

### 起動プロンプト

Claude Code で以下のプロンプトを貼り付けると、Agent Team が起動して分析を開始します。

<details>
<summary>クリックして起動プロンプトを表示</summary>

```
CLAUDE.mdを読んでプロジェクト規約を確認した上で、以下の構成でAgent Teamを作成し分析を開始してください。

## チーム構成

**Teammate "data-engineer"**（delegateモードで起動）
spawn prompt:
「あなたはdata-engineerです。CLAUDE.mdを読んでプロジェクト規約を理解してください。
あなたの担当は src/fetcher/ のみです。

タスク:
1. src/fetcher/binance_client.py にBinance Kline APIクライアントを実装
   - BTCUSDT の 1d, 4h, 1h の3つの時間足を取得
   - 過去3年分のデータをページネーションで取得（1回1500件、startTimeをずらす）
   - レート制限対策: リクエスト間に0.1秒のsleep
2. data/raw/ にタイムフレームごとのCSV保存（btcusdt_1d.csv, btcusdt_4h.csv, btcusdt_1h.csv）
3. data/processed/ にクリーニング済みCSV保存
   - 欠損行の検出と前方補完
   - float型への変換確認
   - timestamp列をUTC datetimeに変換
   - 重複行の除去
4. 全データの準備が完了したら、ta-analyst と anomaly-detective にメッセージを送信:
   「data/processed/ にクリーニング済みデータを配置しました。btcusdt_1d.csv, btcusdt_4h.csv, btcusdt_1h.csv が利用可能です。」」

**Teammate "ta-analyst"**（delegateモードで起動）
spawn prompt:
「あなたはta-analystです。CLAUDE.mdを読んでプロジェクト規約を理解してください。
あなたの担当は src/technical/ のみです。
data-engineerからデータ準備完了のメッセージが届くまで待機してください。

タスク:
1. src/technical/indicators.py にテクニカル分析モジュールを実装
2. data/processed/btcusdt_1d.csv を主に使用（4h, 1hは補助的に参照）
3. 以下の指標を計算し、各指標のエッジを検証:
   a. RSI(14): 20以下/80以上でのリバーサルエントリーの勝率・期待値
   b. MACD(12,26,9): ゴールデン/デッドクロス後5日・10日・20日のリターン分布
   c. ボリンジャーバンド(20,2): スクイーズ後のブレイクアウト方向と値幅
   d. ATR(14): ボラティリティレジーム別のリターン特性
   e. OBV: 価格とOBVのダイバージェンス検出、トレンド転換率
   f. 複合シグナル: RSI+MACD、ボリンジャー+出来高の組み合わせ
4. 各エッジのバックテスト:
   - 勝率、平均リターン、シャープレシオ、最大ドローダウン
   - 前半/後半分割でロバスト性確認
   - t検定（Bonferroni補正）
5. output/reports/technical/ に結果出力（edge_results.json + 可視化PNG）
6. 有望なエッジがあれば anomaly-detective と onchain-analyst にメッセージで共有」

**Teammate "anomaly-detective"**（delegateモードで起動）
spawn prompt:
「あなたはanomaly-detectiveです。CLAUDE.mdを読んでプロジェクト規約を理解してください。
あなたの担当は src/anomaly/ のみです。
data-engineerからデータ準備完了のメッセージが届くまで待機してください。

タスク:
1. src/anomaly/detector.py に統計的アノマリー検出モジュールを実装
2. data/processed/btcusdt_1d.csv および btcusdt_1h.csv を使用
3. 以下のアノマリーを検証:
   a. 曜日効果: 各曜日の日次リターン分布（Kruskal-Wallis検定 + 事後検定）
   b. 月別効果: 各月の月次リターン分布（特に1月、4月、10-12月に注目）
   c. 時間帯効果: Asian/European/US セッション別リターン
   d. 月末/月初効果: 月末5日間 vs 月初5日間 vs その他
   e. ボラティリティクラスタリング: 大きな日次変動（>2σ）後のリターン分布
   f. 連続陽線/陰線: N日連続上昇/下落後の翌日リターン（N=3,4,5,6,7）
   g. 出来高アノマリー: 出来量急増日（>2σ）の翌日・翌週リターン
4. 統計的検定: p値（多重比較補正）、Cohen's d、ブートストラップ95%CI、検定力の報告
5. output/reports/anomaly/ に結果出力（anomaly_results.json + 可視化PNG）
6. 統計的に有意なアノマリーがあれば他のチームメイトにメッセージで共有」

**Teammate "onchain-analyst"**（delegateモードで起動）
spawn prompt:
「あなたはonchain-analystです。CLAUDE.mdを読んでプロジェクト規約を理解してください。
あなたの担当は src/onchain/ のみです。
データ取得はdata-engineerと並行して開始してOKです。

タスク:
1. src/onchain/blockchain_client.py にblockchain.info APIクライアントを実装
2. 以下のオンチェーンデータを取得し data/onchain/ にCSV保存:
   - n-unique-addresses, hash-rate, miners-revenue, n-transactions
   - estimated-transaction-volume-usd, market-cap, trade-volume
3. data-engineerの価格データが準備できたら、価格との相関分析:
   a. 各指標と価格の相互相関（±30日のラグ相関）
   b. Granger因果性検定（最大ラグ14日）
   c. NVT Ratio = market-cap / estimated-transaction-volume-usd の計算
   d. NVTの極端値（>95パーセンタイル, <5パーセンタイル）での将来リターン
   e. ハッシュレート変化率と価格の関係
   f. アクティブアドレス数の急増/急減後の価格動向
4. オンチェーンシグナルのバックテスト（NVT、アクティブアドレス急増）
5. output/reports/onchain/ に結果出力（onchain_results.json + 可視化PNG）
6. ta-analystとanomaly-detectiveに有望な知見をメッセージで共有」

## リード（あなた）の行動指針
- delegateモード（Shift+Tab）を使い、自分では実装コードを書かない
- 各チームメイトの進捗を監視し、ブロッカーがあれば介入
- 全チームメイトの分析完了後、結果を統合して output/reports/final_report.md を作成
  - エグゼクティブサマリー（最も有望なエッジTop5）
  - 各分析カテゴリの詳細結果
  - エッジの複合活用提案（テクニカル×アノマリーの組み合わせなど）
  - 統計的注意事項（多重比較問題、過学習リスク、市場レジーム変化への耐性）
  - 次のステップ（ウォークフォワード検証、ペーパートレード計画）
- 最後に git add -A && git commit && git push
```

</details>

## ディレクトリ構造

```
btc-edge-lab/
├── CLAUDE.md              # プロジェクト規約（Agent Team 用）
├── requirements.txt       # Python 依存関係
├── data/
│   ├── raw/               # API から取得した生データ
│   ├── processed/         # クリーニング済みローソク足データ
│   └── onchain/           # オンチェーンデータ
├── src/
│   ├── common/            # 共通ユーティリティ
│   ├── fetcher/           # Binance API クライアント
│   ├── technical/         # テクニカル分析
│   ├── anomaly/           # アノマリー検出
│   ├── onchain/           # オンチェーン分析
│   └── report/            # レポート生成
└── output/
    └── reports/           # 分析結果（JSON + PNG）
        ├── technical/
        ├── anomaly/
        └── onchain/
```

## ライセンス

MIT
