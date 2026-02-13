# btc-edge-lab

## プロジェクト概要
Binance APIからBTC/USDTの過去データを取得し、複数の分析手法を組み合わせてトレーディングエッジを発見する。

## 技術スタック
- Python 3.11+
- 主要ライブラリ: pandas, numpy, scipy, scikit-learn, ta, matplotlib, plotly, statsmodels, arch
- データ: Binance REST API (認証不要), blockchain.info API (認証不要)

## Binance Kline API 仕様
- Endpoint: `GET https://api.binance.com/api/v3/klines`
- Params: symbol, interval (1m/5m/15m/1h/4h/1d/1w), startTime, endTime (unix ms), limit (max 1500)
- Response: 配列 [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
- Rate limit: 1200 requests/min (Weight: 2 per request)
- 認証不要

## blockchain.info API 仕様
- Endpoint: `GET https://api.blockchain.info/charts/{chart_name}?timespan=3years&format=json`
- 利用可能チャート: n-unique-addresses, hash-rate, miners-revenue, n-transactions, estimated-transaction-volume-usd, market-cap, trade-volume
- Rate limit: 明文化なし、1req/secを目安に
- 認証不要

## ディレクトリ規約（Agent Team用）
- 各チームメイトは **自分の担当 `src/` サブディレクトリのみ** 編集すること
  - data-engineer → `src/fetcher/`
  - ta-analyst → `src/technical/`
  - anomaly-detective → `src/anomaly/`
  - onchain-analyst → `src/onchain/`
- `src/common/` はリードが管理。チームメイトはimportのみ可
- データ共有は `data/` ディレクトリ経由（CSVファイル）
- 分析結果は `output/reports/{担当名}/` に JSON + PNG で出力

## コーディング規約
- 型ヒント必須
- docstring必須（Google style）
- 統計的主張には必ず p値・サンプルサイズ・効果量を併記
- ログ出力は `src/common/utils.py` の共通ロガーを使用
- CSVカラム名は英語snake_case統一

## 共通データフォーマット
### ローソク足 CSV（data/processed/）
timestamp (UTC datetime), open, high, low, close, volume

### オンチェーン CSV（data/onchain/）
timestamp (UTC datetime), metric_name (float)

### エッジレポート JSON
{
  "edge_name": "RSI極端値リバーサル",
  "category": "technical",
  "timeframe": "1d",
  "win_rate": 0.62,
  "expected_return": 0.015,
  "sharpe_ratio": 1.8,
  "max_drawdown": -0.12,
  "p_value": 0.003,
  "sample_size": 247,
  "test_period": "2022-01-01 to 2025-01-01",
  "description": "RSI(14)が20以下になった翌日のロングエントリー",
  "notes": "bear market期間に偏りあり、regime分析が必要"
}
