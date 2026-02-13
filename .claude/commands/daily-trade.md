# 日次ペーパートレード実行

毎日1回実行する BTC/USDT ペーパートレード PDCA サイクル。

## 実行手順

### Step 1: データ更新
最新のローソク足データとオンチェーンデータを取得する。

```python
from src.fetcher.binance_client import fetch_latest
from src.onchain.blockchain_client import fetch_metric

# 日次ローソク足の差分取得
df_1d = fetch_latest("1d")

# オンチェーンデータ更新（アクティブアドレス）
addr_df = fetch_metric("n-unique-addresses", timespan="3years")
```

`uv run python -c` で上記を実行し、data/processed/btcusdt_1d.csv と data/onchain/n_unique_addresses.csv を更新する。

### Step 2: シグナル生成 + トレード実行
state.json を読み込み、シグナル判定とトレード実行を行う。

```python
from src.trader.signal_generator import generate_all_signals
from src.trader.engine import PaperTradeEngine
from src.common.utils import read_csv
import pandas as pd

# データ読み込み
df = read_csv("processed", "btcusdt_1d.csv")
onchain_df = read_csv("onchain", "n_unique_addresses.csv")

# エンジン起動
engine = PaperTradeEngine()
loaded = engine.load_state()
if not loaded:
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    engine.initialize(today)

# シグナル生成
signals = generate_all_signals(df, onchain_df, engine.edge_configs)

# トレード実行
today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
current_price = float(df["close"].iloc[-1])
summary = engine.process_day(today, current_price, signals)
```

`uv run python -c` で上記を実行する。

### Step 3: パフォーマンス評価（30日サイクル）
engine.is_evaluation_due() が True の場合のみ実行する。

```python
from src.trader.evaluator import PerformanceEvaluator

evaluator = PerformanceEvaluator()
result = evaluator.evaluate(engine.edge_configs)

# 自動改善の適用
applied = evaluator.apply_auto_improvements(result["improvements"], engine.edge_configs)
if applied:
    engine.save_state()
```

### Step 4: 日次サマリー出力
以下の情報をユーザーに報告する：

1. **データ更新**: 取得した最新データの日時範囲
2. **シグナル**: 検出されたシグナル一覧（エッジ名、方向、確信度）
3. **トレード**: 新規エントリーと決済の詳細
4. **ポジション**: 現在のオープンポジション一覧
5. **エクイティ**: 現在の資産評価額と前日比
6. **監視**: BB Squeeze 等の監視モードシグナル
7. **PDCA**: 評価日の場合は改善提案と適用結果

## 注意事項
- `data/trade/state.json` が存在しない場合は初回実行として初期化する
- BB Squeeze Abs Return は監視モード（ポジションは取らない）
- 同一エッジの重複ポジションは禁止
- 最大同時ポジション数は 3
- 取引コストは片道 0.15%（手数料 + スリッページ）
