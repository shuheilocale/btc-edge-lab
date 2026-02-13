# btc-edge-lab

BTC/USDT の過去データを複数の分析手法で解析し、トレーディングエッジを発見するプロジェクト。

## 概要

Binance API からローソク足データ、blockchain.info API からオンチェーンデータを取得し、以下の 3 つの観点から統計的に有意なエッジを探索する。

- **テクニカル分析**: RSI, MACD, ボリンジャーバンド, ATR, OBV 等の指標ベースのエッジ検証
- **統計的アノマリー検出**: 曜日効果、月別効果、時間帯効果、ボラティリティクラスタリング等
- **オンチェーン分析**: NVT Ratio、アクティブアドレス数、ハッシュレート等と価格の相関分析

## セットアップ

```bash
pip install -r requirements.txt
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
