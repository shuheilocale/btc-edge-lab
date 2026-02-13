"""オンチェーン分析モジュール。

価格データとオンチェーンメトリクスの相関分析、Granger因果性検定、
NVT分析、シグナルバックテストを実行する。
"""

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

from src.common.utils import (
    PROCESSED_DIR,
    setup_logger,
    setup_plot_style,
    save_figure,
    save_report_json,
)
from src.onchain.blockchain_client import load_onchain_data

logger = setup_logger(__name__)

REPORT_CATEGORY = "onchain"


def load_price_data() -> pd.DataFrame:
    """日次価格データを読み込む。

    Returns:
        日次ローソク足 DataFrame。
    """
    path = PROCESSED_DIR / "btcusdt_1d.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("Loaded price data: %d rows", len(df))
    return df


def merge_with_price(
    price_df: pd.DataFrame, onchain_df: pd.DataFrame, metric_col: str
) -> pd.DataFrame:
    """価格データとオンチェーンデータを日付でマージする。

    Args:
        price_df: 日次価格データ。
        onchain_df: オンチェーンメトリクスデータ。
        metric_col: メトリクスのカラム名。

    Returns:
        マージ済み DataFrame。
    """
    price = price_df[["timestamp", "close"]].copy()
    price["date"] = price["timestamp"].dt.date

    onchain = onchain_df[["timestamp", metric_col]].copy()
    onchain["date"] = onchain["timestamp"].dt.date

    merged = pd.merge(price, onchain, on="date", how="inner", suffixes=("_price", "_onchain"))
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.dropna(subset=["close", metric_col])
    return merged


def compute_cross_correlation(
    price_df: pd.DataFrame, onchain_df: pd.DataFrame, metric_col: str, max_lag: int = 30
) -> dict[str, Any]:
    """価格とオンチェーンメトリクスの相互相関を計算する。

    Args:
        price_df: 日次価格データ。
        onchain_df: オンチェーンメトリクスデータ。
        metric_col: メトリクスのカラム名。
        max_lag: 最大ラグ日数。

    Returns:
        相互相関の結果辞書（ラグごとの相関係数とp値）。
    """
    merged = merge_with_price(price_df, onchain_df, metric_col)
    if len(merged) < max_lag * 2:
        logger.warning("Not enough data for cross-correlation of %s", metric_col)
        return {"metric": metric_col, "error": "insufficient_data"}

    price_returns = merged["close"].pct_change().dropna()
    metric_changes = merged[metric_col].pct_change().dropna()

    min_len = min(len(price_returns), len(metric_changes))
    price_returns = price_returns.iloc[:min_len].values
    metric_changes = metric_changes.iloc[:min_len].values

    lags = list(range(-max_lag, max_lag + 1))
    correlations: list[dict[str, Any]] = []

    for lag in lags:
        if lag > 0:
            x = metric_changes[:-lag] if lag < len(metric_changes) else np.array([])
            y = price_returns[lag:] if lag < len(price_returns) else np.array([])
        elif lag < 0:
            x = metric_changes[-lag:]
            y = price_returns[:lag]
        else:
            x = metric_changes
            y = price_returns

        if len(x) < 30:
            continue

        corr, p_value = stats.pearsonr(x, y)
        correlations.append({
            "lag": lag,
            "correlation": float(corr),
            "p_value": float(p_value),
            "sample_size": len(x),
        })

    best = max(correlations, key=lambda c: abs(c["correlation"])) if correlations else None

    return {
        "metric": metric_col,
        "correlations": correlations,
        "best_lag": best,
        "sample_size": min_len,
    }


def granger_causality_test(
    price_df: pd.DataFrame, onchain_df: pd.DataFrame, metric_col: str, max_lag: int = 14
) -> dict[str, Any]:
    """Granger因果性検定を実行する。

    Args:
        price_df: 日次価格データ。
        onchain_df: オンチェーンメトリクスデータ。
        metric_col: メトリクスのカラム名。
        max_lag: 最大ラグ日数。

    Returns:
        各ラグでのF統計量とp値。
    """
    merged = merge_with_price(price_df, onchain_df, metric_col)
    if len(merged) < max_lag * 3:
        return {"metric": metric_col, "error": "insufficient_data"}

    price_returns = merged["close"].pct_change().dropna()
    metric_changes = merged[metric_col].pct_change().dropna()

    min_len = min(len(price_returns), len(metric_changes))
    test_data = pd.DataFrame({
        "price_return": price_returns.iloc[:min_len].values,
        "metric_change": metric_changes.iloc[:min_len].values,
    }).dropna()

    if len(test_data) < max_lag * 3:
        return {"metric": metric_col, "error": "insufficient_data_after_cleaning"}

    results_by_lag: list[dict[str, Any]] = []
    try:
        gc_results = grangercausalitytests(
            test_data[["price_return", "metric_change"]], maxlag=max_lag, verbose=False
        )
        for lag_val in range(1, max_lag + 1):
            test_result = gc_results[lag_val]
            f_stat = test_result[0]["ssr_ftest"][0]
            p_val = test_result[0]["ssr_ftest"][1]
            results_by_lag.append({
                "lag": lag_val,
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "significant_5pct": p_val < 0.05,
            })
    except Exception as e:
        logger.warning("Granger test failed for %s: %s", metric_col, e)
        return {"metric": metric_col, "error": str(e)}

    significant_lags = [r for r in results_by_lag if r["significant_5pct"]]

    return {
        "metric": metric_col,
        "direction": "metric -> price_return",
        "results_by_lag": results_by_lag,
        "significant_lags_count": len(significant_lags),
        "most_significant": min(results_by_lag, key=lambda r: r["p_value"]) if results_by_lag else None,
        "sample_size": len(test_data),
    }


def compute_nvt_ratio(price_df: pd.DataFrame, onchain_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """NVT Ratio (Network Value to Transactions) を計算する。

    NVT = market_cap / estimated_transaction_volume_usd

    Args:
        price_df: 日次価格データ。
        onchain_data: オンチェーンデータ辞書。

    Returns:
        NVT Ratioを含む DataFrame。
    """
    if "market-cap" not in onchain_data or "estimated-transaction-volume-usd" not in onchain_data:
        logger.error("Required metrics for NVT not available")
        return pd.DataFrame()

    mcap_df = onchain_data["market-cap"].copy()
    mcap_df["date"] = mcap_df["timestamp"].dt.date

    txvol_df = onchain_data["estimated-transaction-volume-usd"].copy()
    txvol_df["date"] = txvol_df["timestamp"].dt.date

    nvt_df = pd.merge(mcap_df, txvol_df, on="date", how="inner", suffixes=("_mcap", "_txvol"))
    nvt_df["nvt_ratio"] = nvt_df["market_cap"] / nvt_df["estimated_transaction_volume_usd"].replace(0, np.nan)
    nvt_df = nvt_df.dropna(subset=["nvt_ratio"])

    price = price_df[["timestamp", "close"]].copy()
    price["date"] = price["timestamp"].dt.date

    nvt_df = pd.merge(nvt_df, price, on="date", how="inner")
    nvt_df = nvt_df.sort_values("date").reset_index(drop=True)

    logger.info("NVT Ratio computed: %d rows, mean=%.1f, median=%.1f",
                len(nvt_df), nvt_df["nvt_ratio"].mean(), nvt_df["nvt_ratio"].median())
    return nvt_df


def analyze_nvt_extremes(nvt_df: pd.DataFrame, forward_days: list[int] | None = None) -> dict[str, Any]:
    """NVT Ratioの極端値での将来リターンを分析する。

    Args:
        nvt_df: NVT Ratioを含む DataFrame。
        forward_days: 将来リターンを計算する日数リスト。

    Returns:
        NVT極端値分析の結果辞書。
    """
    if forward_days is None:
        forward_days = [1, 3, 7, 14, 30]

    if nvt_df.empty:
        return {"error": "empty_dataframe"}

    p95 = nvt_df["nvt_ratio"].quantile(0.95)
    p05 = nvt_df["nvt_ratio"].quantile(0.05)

    high_nvt = nvt_df[nvt_df["nvt_ratio"] > p95].index
    low_nvt = nvt_df[nvt_df["nvt_ratio"] < p05].index

    results: dict[str, Any] = {
        "nvt_p95_threshold": float(p95),
        "nvt_p05_threshold": float(p05),
        "nvt_mean": float(nvt_df["nvt_ratio"].mean()),
        "nvt_median": float(nvt_df["nvt_ratio"].median()),
        "high_nvt_signals": [],
        "low_nvt_signals": [],
    }

    for days in forward_days:
        nvt_df[f"fwd_return_{days}d"] = nvt_df["close"].shift(-days) / nvt_df["close"] - 1

    for label, signal_idx in [("high_nvt_signals", high_nvt), ("low_nvt_signals", low_nvt)]:
        for days in forward_days:
            col = f"fwd_return_{days}d"
            returns = nvt_df.loc[signal_idx, col].dropna()
            if len(returns) < 5:
                continue

            mean_ret = returns.mean()
            t_stat, p_val = stats.ttest_1samp(returns, 0)

            all_returns = nvt_df[col].dropna()
            all_mean = all_returns.mean()
            cohens_d = (mean_ret - all_mean) / returns.std() if returns.std() > 0 else 0.0

            results[label].append({
                "forward_days": days,
                "mean_return": float(mean_ret),
                "median_return": float(returns.median()),
                "std_return": float(returns.std()),
                "win_rate": float((returns > 0).mean()),
                "sample_size": int(len(returns)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(cohens_d),
                "baseline_mean_return": float(all_mean),
            })

    return results


def analyze_hashrate_price(
    price_df: pd.DataFrame, onchain_data: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """ハッシュレート変化率と価格の関係を分析する。

    Args:
        price_df: 日次価格データ。
        onchain_data: オンチェーンデータ辞書。

    Returns:
        分析結果辞書。
    """
    if "hash-rate" not in onchain_data:
        return {"error": "hash-rate data not available"}

    hr_df = onchain_data["hash-rate"]
    merged = merge_with_price(price_df, hr_df, "hash_rate")

    if len(merged) < 60:
        return {"error": "insufficient_data"}

    merged["hr_change_7d"] = merged["hash_rate"].pct_change(7)
    merged["hr_change_30d"] = merged["hash_rate"].pct_change(30)
    merged["price_return_7d"] = merged["close"].shift(-7) / merged["close"] - 1
    merged["price_return_14d"] = merged["close"].shift(-14) / merged["close"] - 1
    merged["price_return_30d"] = merged["close"].shift(-30) / merged["close"] - 1

    results: dict[str, Any] = {"hashrate_price_correlations": []}

    for hr_col in ["hr_change_7d", "hr_change_30d"]:
        for price_col in ["price_return_7d", "price_return_14d", "price_return_30d"]:
            valid = merged[[hr_col, price_col]].dropna()
            if len(valid) < 30:
                continue
            corr, p_val = stats.pearsonr(valid[hr_col], valid[price_col])
            results["hashrate_price_correlations"].append({
                "hashrate_window": hr_col,
                "price_window": price_col,
                "correlation": float(corr),
                "p_value": float(p_val),
                "sample_size": len(valid),
            })

    # Hashrate drop analysis
    merged_clean = merged.dropna(subset=["hr_change_30d"])
    hr_drop_threshold = merged_clean["hr_change_30d"].quantile(0.1)
    hr_surge_threshold = merged_clean["hr_change_30d"].quantile(0.9)

    results["hashrate_extremes"] = {
        "drop_threshold_10pct": float(hr_drop_threshold),
        "surge_threshold_90pct": float(hr_surge_threshold),
    }

    for label, mask in [
        ("after_hr_drop", merged_clean["hr_change_30d"] < hr_drop_threshold),
        ("after_hr_surge", merged_clean["hr_change_30d"] > hr_surge_threshold),
    ]:
        subset = merged_clean[mask]
        for ret_col in ["price_return_7d", "price_return_14d", "price_return_30d"]:
            returns = subset[ret_col].dropna()
            if len(returns) < 5:
                continue
            mean_ret = returns.mean()
            t_stat, p_val = stats.ttest_1samp(returns, 0)
            results["hashrate_extremes"][f"{label}_{ret_col}"] = {
                "mean_return": float(mean_ret),
                "win_rate": float((returns > 0).mean()),
                "sample_size": int(len(returns)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            }

    return results


def analyze_active_addresses(
    price_df: pd.DataFrame, onchain_data: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """アクティブアドレス数の急増/急減後の価格動向を分析する。

    Args:
        price_df: 日次価格データ。
        onchain_data: オンチェーンデータ辞書。

    Returns:
        分析結果辞書。
    """
    if "n-unique-addresses" not in onchain_data:
        return {"error": "n-unique-addresses data not available"}

    addr_df = onchain_data["n-unique-addresses"]
    merged = merge_with_price(price_df, addr_df, "n_unique_addresses")

    if len(merged) < 60:
        return {"error": "insufficient_data"}

    merged["addr_change_7d"] = merged["n_unique_addresses"].pct_change(7)
    merged["addr_zscore"] = (
        (merged["n_unique_addresses"] - merged["n_unique_addresses"].rolling(30).mean())
        / merged["n_unique_addresses"].rolling(30).std()
    )
    merged["price_return_7d"] = merged["close"].shift(-7) / merged["close"] - 1
    merged["price_return_14d"] = merged["close"].shift(-14) / merged["close"] - 1
    merged["price_return_30d"] = merged["close"].shift(-30) / merged["close"] - 1

    merged_clean = merged.dropna(subset=["addr_zscore"])

    results: dict[str, Any] = {"address_surge_analysis": {}, "address_drop_analysis": {}}

    surge_mask = merged_clean["addr_zscore"] > 2.0
    drop_mask = merged_clean["addr_zscore"] < -2.0

    for label, mask in [("address_surge_analysis", surge_mask), ("address_drop_analysis", drop_mask)]:
        subset = merged_clean[mask]
        results[label]["signal_count"] = int(len(subset))

        for ret_col in ["price_return_7d", "price_return_14d", "price_return_30d"]:
            returns = subset[ret_col].dropna()
            if len(returns) < 3:
                results[label][ret_col] = {"error": "insufficient_signals", "count": int(len(returns))}
                continue

            mean_ret = returns.mean()
            baseline = merged_clean[ret_col].dropna()
            baseline_mean = baseline.mean()

            if len(returns) >= 5:
                t_stat, p_val = stats.ttest_1samp(returns, 0)
            else:
                t_stat, p_val = float("nan"), float("nan")

            cohens_d = (mean_ret - baseline_mean) / returns.std() if returns.std() > 0 else 0.0

            results[label][ret_col] = {
                "mean_return": float(mean_ret),
                "median_return": float(returns.median()),
                "win_rate": float((returns > 0).mean()),
                "sample_size": int(len(returns)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(cohens_d),
                "baseline_mean_return": float(baseline_mean),
            }

    return results


def backtest_nvt_signal(nvt_df: pd.DataFrame) -> dict[str, Any]:
    """NVT Ratioに基づくシグナルのバックテストを実行する。

    戦略: NVT < 5th percentile → ロング（ネットワーク過小評価）
          NVT > 95th percentile → ショート/キャッシュ（ネットワーク過大評価）

    Args:
        nvt_df: NVT Ratioを含む DataFrame。

    Returns:
        バックテスト結果辞書。
    """
    if nvt_df.empty:
        return {"error": "empty_dataframe"}

    df = nvt_df.copy()
    p95 = df["nvt_ratio"].quantile(0.95)
    p05 = df["nvt_ratio"].quantile(0.05)

    df["signal"] = 0
    df.loc[df["nvt_ratio"] < p05, "signal"] = 1   # Long
    df.loc[df["nvt_ratio"] > p95, "signal"] = -1   # Short/Cash

    df["daily_return"] = df["close"].pct_change()

    # Hold position for 14 days after signal
    df["position"] = 0
    hold_period = 14
    for i in range(len(df)):
        if df.iloc[i]["signal"] != 0:
            end_idx = min(i + hold_period, len(df))
            df.iloc[i:end_idx, df.columns.get_loc("position")] = df.iloc[i]["signal"]

    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]
    df = df.dropna(subset=["strategy_return"])

    total_trades_long = int((df["signal"] == 1).sum())
    total_trades_short = int((df["signal"] == -1).sum())

    strategy_returns = df[df["position"] != 0]["strategy_return"]
    if len(strategy_returns) == 0:
        return {"error": "no_trades"}

    cumulative = (1 + df["strategy_return"]).cumprod()
    buy_hold = (1 + df["daily_return"]).cumprod()

    total_return = float(cumulative.iloc[-1] - 1)
    buy_hold_return = float(buy_hold.iloc[-1] - 1)

    ann_factor = np.sqrt(365)
    sharpe = float(strategy_returns.mean() / strategy_returns.std() * ann_factor) if strategy_returns.std() > 0 else 0.0

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    win_rate = float((strategy_returns > 0).sum() / len(strategy_returns)) if len(strategy_returns) > 0 else 0.0

    return {
        "edge_name": "NVT Ratio Extreme Signal",
        "category": "onchain",
        "timeframe": "1d",
        "strategy": "Long when NVT < 5th pctl, Short when NVT > 95th pctl, hold 14 days",
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_long_signals": total_trades_long,
        "total_short_signals": total_trades_short,
        "active_days": int((df["position"] != 0).sum()),
        "total_days": len(df),
        "sample_size": int(len(strategy_returns)),
        "nvt_p05": float(p05),
        "nvt_p95": float(p95),
        "test_period": f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}",
    }


def backtest_address_surge(
    price_df: pd.DataFrame, onchain_data: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """アクティブアドレス急増シグナルのバックテストを実行する。

    戦略: アドレス数 z-score > 2 → ロング（14日保持）

    Args:
        price_df: 日次価格データ。
        onchain_data: オンチェーンデータ辞書。

    Returns:
        バックテスト結果辞書。
    """
    if "n-unique-addresses" not in onchain_data:
        return {"error": "n-unique-addresses data not available"}

    addr_df = onchain_data["n-unique-addresses"]
    merged = merge_with_price(price_df, addr_df, "n_unique_addresses")

    if len(merged) < 60:
        return {"error": "insufficient_data"}

    df = merged.copy()
    df["addr_zscore"] = (
        (df["n_unique_addresses"] - df["n_unique_addresses"].rolling(30).mean())
        / df["n_unique_addresses"].rolling(30).std()
    )
    df["daily_return"] = df["close"].pct_change()

    df["signal"] = 0
    df.loc[df["addr_zscore"] > 2.0, "signal"] = 1

    hold_period = 14
    df["position"] = 0
    for i in range(len(df)):
        if df.iloc[i]["signal"] == 1:
            end_idx = min(i + hold_period, len(df))
            df.iloc[i:end_idx, df.columns.get_loc("position")] = 1

    df["strategy_return"] = df["position"].shift(1) * df["daily_return"]
    df = df.dropna(subset=["strategy_return"])

    strategy_returns = df[df["position"] != 0]["strategy_return"]
    if len(strategy_returns) == 0:
        return {"error": "no_trades", "signal_count": 0}

    cumulative = (1 + df["strategy_return"]).cumprod()
    buy_hold = (1 + df["daily_return"]).cumprod()

    total_return = float(cumulative.iloc[-1] - 1)
    buy_hold_return = float(buy_hold.iloc[-1] - 1)

    ann_factor = np.sqrt(365)
    sharpe = float(strategy_returns.mean() / strategy_returns.std() * ann_factor) if strategy_returns.std() > 0 else 0.0

    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min())

    win_rate = float((strategy_returns > 0).sum() / len(strategy_returns)) if len(strategy_returns) > 0 else 0.0

    signal_count = int((df["signal"] == 1).sum())

    return {
        "edge_name": "Active Address Surge Signal",
        "category": "onchain",
        "timeframe": "1d",
        "strategy": "Long when address z-score > 2 (30d rolling), hold 14 days",
        "total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_return": total_return - buy_hold_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "signal_count": signal_count,
        "active_days": int((df["position"] != 0).sum()),
        "total_days": len(df),
        "sample_size": int(len(strategy_returns)),
        "test_period": f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}",
    }


def analyze_bb_squeeze_with_onchain(
    price_df: pd.DataFrame, onchain_data: dict[str, pd.DataFrame]
) -> dict[str, Any]:
    """BBスクイーズ期間中のオンチェーン指標状態を分析する。

    ta-analystのBBスクイーズ定義と同じロジックで、スクイーズ期間中に
    アクティブアドレスz-score > 2が発生している場合の方向予測を検証する。

    Args:
        price_df: 日次価格データ。
        onchain_data: オンチェーンデータ辞書。

    Returns:
        複合分析結果辞書。
    """
    if "n-unique-addresses" not in onchain_data:
        return {"error": "n-unique-addresses data not available"}

    # 1. Compute BB squeeze (same definition as ta-analyst)
    close = price_df["close"].copy()
    bb_period = 20
    bb_std = 2.0
    middle = close.rolling(window=bb_period).mean()
    rolling_std = close.rolling(window=bb_period).std()
    upper = middle + bb_std * rolling_std
    lower = middle - bb_std * rolling_std
    bandwidth = (upper - lower) / middle

    bw_pctl = bandwidth.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    squeeze = bw_pctl <= 0.25

    # 2. Merge with address data
    addr_df = onchain_data["n-unique-addresses"]
    merged = merge_with_price(price_df, addr_df, "n_unique_addresses")
    if len(merged) < 60:
        return {"error": "insufficient_data"}

    # Align squeeze with merged df by date
    price_dates = price_df[["timestamp"]].copy()
    price_dates["date"] = price_dates["timestamp"].dt.date
    price_dates["squeeze"] = squeeze.values
    price_dates["upper"] = upper.values
    price_dates["lower"] = lower.values

    merged = pd.merge(merged, price_dates[["date", "squeeze", "upper", "lower"]], on="date", how="inner")

    merged["addr_zscore"] = (
        (merged["n_unique_addresses"] - merged["n_unique_addresses"].rolling(30).mean())
        / merged["n_unique_addresses"].rolling(30).std()
    )

    merged = merged.dropna(subset=["squeeze", "addr_zscore"])

    # 3. Define composite signals
    squeeze_mask = merged["squeeze"] == True  # noqa: E712
    addr_surge_mask = merged["addr_zscore"] > 2.0
    addr_drop_mask = merged["addr_zscore"] < -2.0

    # Squeeze release + direction
    squeeze_release = squeeze_mask.shift(1).fillna(False) & ~squeeze_mask
    breakout_up = squeeze_release & (merged["close"] > merged["upper"].shift(1))
    breakout_down = squeeze_release & (merged["close"] < merged["lower"].shift(1))

    # Composite: Squeeze + address surge (look at addr_zscore during squeeze window)
    # Check if address surge occurred in the 7 days before squeeze release
    addr_surge_recent = addr_surge_mask.rolling(7, min_periods=1).max().astype(bool)
    addr_drop_recent = addr_drop_mask.rolling(7, min_periods=1).max().astype(bool)

    combo_squeeze_addr_surge = squeeze_mask & addr_surge_recent
    combo_squeeze_addr_drop = squeeze_mask & addr_drop_recent
    combo_squeeze_no_addr_signal = squeeze_mask & ~addr_surge_recent & ~addr_drop_recent

    # 4. Compute forward returns
    results: dict[str, Any] = {"composite_signals": {}}

    for days in [5, 7, 10, 14]:
        merged[f"fwd_return_{days}d"] = merged["close"].shift(-days) / merged["close"] - 1

    signal_configs = [
        ("squeeze_only", squeeze_mask, "BBスクイーズ中（全期間）"),
        ("squeeze_with_addr_surge", combo_squeeze_addr_surge, "BBスクイーズ中 + アドレス急増(z>2)"),
        ("squeeze_with_addr_drop", combo_squeeze_addr_drop, "BBスクイーズ中 + アドレス急減(z<-2)"),
        ("squeeze_no_addr_signal", combo_squeeze_no_addr_signal, "BBスクイーズ中 + アドレス中立"),
        ("breakout_up_with_addr_surge", breakout_up & addr_surge_recent, "BB上方ブレイク + アドレス急増"),
        ("breakout_up_no_addr_surge", breakout_up & ~addr_surge_recent, "BB上方ブレイク + アドレス急増なし"),
    ]

    for label, mask, description in signal_configs:
        signal_data: dict[str, Any] = {
            "description": description,
            "signal_count": int(mask.sum()),
        }

        for days in [5, 7, 10, 14]:
            col = f"fwd_return_{days}d"
            returns = merged.loc[mask, col].dropna()
            if len(returns) < 3:
                signal_data[f"{days}d"] = {"error": "insufficient_signals", "count": int(len(returns))}
                continue

            mean_ret = returns.mean()
            baseline = merged[col].dropna()
            baseline_mean = baseline.mean()

            if len(returns) >= 5:
                t_stat, p_val = stats.ttest_1samp(returns, 0)
            else:
                t_stat, p_val = float("nan"), float("nan")

            cohens_d = (mean_ret - baseline_mean) / returns.std() if returns.std() > 0 else 0.0

            signal_data[f"{days}d"] = {
                "mean_return": float(mean_ret),
                "median_return": float(returns.median()),
                "win_rate": float((returns > 0).mean()),
                "sample_size": int(len(returns)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(cohens_d),
                "baseline_mean_return": float(baseline_mean),
            }

        results["composite_signals"][label] = signal_data

    # 5. Also check onchain metrics during squeeze periods for hashrate and miner revenue
    for metric_key, col_name in [("hash-rate", "hash_rate"), ("miners-revenue", "miners_revenue")]:
        if metric_key not in onchain_data:
            continue
        m_df = onchain_data[metric_key]
        m_merged = merge_with_price(price_df, m_df, col_name)
        m_merged = pd.merge(m_merged, price_dates[["date", "squeeze"]], on="date", how="inner")
        m_merged[f"{col_name}_change_7d"] = m_merged[col_name].pct_change(7)
        m_merged = m_merged.dropna(subset=["squeeze", f"{col_name}_change_7d"])

        sq_mask = m_merged["squeeze"] == True  # noqa: E712
        rising_mask = sq_mask & (m_merged[f"{col_name}_change_7d"] > 0)
        falling_mask = sq_mask & (m_merged[f"{col_name}_change_7d"] < 0)

        for days in [7, 14]:
            m_merged[f"fwd_return_{days}d"] = m_merged["close"].shift(-days) / m_merged["close"] - 1

        for sub_label, sub_mask in [
            (f"squeeze_{col_name}_rising", rising_mask),
            (f"squeeze_{col_name}_falling", falling_mask),
        ]:
            sub_data: dict[str, Any] = {"signal_count": int(sub_mask.sum())}
            for days in [7, 14]:
                col = f"fwd_return_{days}d"
                returns = m_merged.loc[sub_mask, col].dropna()
                if len(returns) < 5:
                    sub_data[f"{days}d"] = {"error": "insufficient", "count": int(len(returns))}
                    continue
                mean_ret = returns.mean()
                t_stat, p_val = stats.ttest_1samp(returns, 0)
                sub_data[f"{days}d"] = {
                    "mean_return": float(mean_ret),
                    "win_rate": float((returns > 0).mean()),
                    "sample_size": int(len(returns)),
                    "p_value": float(p_val),
                }
            results["composite_signals"][sub_label] = sub_data

    logger.info("BB Squeeze + Onchain composite analysis complete: %d signal configs",
                len(results["composite_signals"]))
    return results


def create_visualizations(
    nvt_df: pd.DataFrame,
    cross_corr_results: list[dict[str, Any]],
    price_df: pd.DataFrame,
    onchain_data: dict[str, pd.DataFrame],
) -> None:
    """分析結果の可視化PNGを生成する。

    Args:
        nvt_df: NVT Ratio DataFrame。
        cross_corr_results: 相互相関結果リスト。
        price_df: 日次価格データ。
        onchain_data: オンチェーンデータ辞書。
    """
    setup_plot_style()

    # 1. NVT Ratio vs Price
    if not nvt_df.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax1.plot(nvt_df["date"], nvt_df["close"], color="cyan", label="BTC Price")
        ax1.set_ylabel("Price (USD)")
        ax1.set_title("BTC Price vs NVT Ratio")
        ax1.legend()

        p95 = nvt_df["nvt_ratio"].quantile(0.95)
        p05 = nvt_df["nvt_ratio"].quantile(0.05)
        ax2.plot(nvt_df["date"], nvt_df["nvt_ratio"], color="orange", label="NVT Ratio", alpha=0.8)
        ax2.axhline(y=p95, color="red", linestyle="--", alpha=0.7, label=f"95th pctl ({p95:.0f})")
        ax2.axhline(y=p05, color="green", linestyle="--", alpha=0.7, label=f"5th pctl ({p05:.0f})")
        ax2.set_ylabel("NVT Ratio")
        ax2.set_xlabel("Date")
        ax2.legend()

        fig.tight_layout()
        save_figure(fig, REPORT_CATEGORY, "nvt_ratio_vs_price.png")

    # 2. Cross-correlation heatmap
    valid_results = [r for r in cross_corr_results if "correlations" in r]
    if valid_results:
        fig, ax = plt.subplots(figsize=(14, 8))
        for result in valid_results:
            lags = [c["lag"] for c in result["correlations"]]
            corrs = [c["correlation"] for c in result["correlations"]]
            ax.plot(lags, corrs, label=result["metric"], alpha=0.8)
        ax.axhline(y=0, color="white", linestyle="-", alpha=0.3)
        ax.axvline(x=0, color="white", linestyle="-", alpha=0.3)
        ax.set_xlabel("Lag (days, positive = metric leads price)")
        ax.set_ylabel("Correlation")
        ax.set_title("Cross-Correlation: On-chain Metrics vs BTC Price Returns")
        ax.legend(fontsize=8)
        save_figure(fig, REPORT_CATEGORY, "cross_correlations.png")

    # 3. Active Addresses vs Price
    if "n-unique-addresses" in onchain_data:
        addr_df = onchain_data["n-unique-addresses"]
        merged = merge_with_price(price_df, addr_df, "n_unique_addresses")
        if not merged.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            ax1.plot(merged["date"], merged["close"], color="cyan", label="BTC Price")
            ax1.set_ylabel("Price (USD)")
            ax1.set_title("BTC Price vs Active Addresses")
            ax1.legend()

            ax2.plot(merged["date"], merged["n_unique_addresses"], color="lime", label="Unique Addresses")
            ax2.set_ylabel("Unique Addresses")
            ax2.set_xlabel("Date")
            ax2.legend()

            fig.tight_layout()
            save_figure(fig, REPORT_CATEGORY, "active_addresses_vs_price.png")

    # 4. Hashrate vs Price
    if "hash-rate" in onchain_data:
        hr_df = onchain_data["hash-rate"]
        merged = merge_with_price(price_df, hr_df, "hash_rate")
        if not merged.empty:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            ax1.plot(merged["date"], merged["close"], color="cyan", label="BTC Price")
            ax1.set_ylabel("Price (USD)")
            ax1.set_title("BTC Price vs Hash Rate")
            ax1.legend()

            ax2.plot(merged["date"], merged["hash_rate"], color="magenta", label="Hash Rate")
            ax2.set_ylabel("Hash Rate (TH/s)")
            ax2.set_xlabel("Date")
            ax2.legend()

            fig.tight_layout()
            save_figure(fig, REPORT_CATEGORY, "hashrate_vs_price.png")


def run_full_analysis() -> dict[str, Any]:
    """全分析を実行してレポートを生成する。

    Returns:
        全分析結果を含む辞書。
    """
    logger.info("=== Starting full on-chain analysis ===")

    # Load data
    price_df = load_price_data()
    onchain_data = load_onchain_data()

    all_results: dict[str, Any] = {}

    # 1. Cross-correlation analysis
    logger.info("--- Cross-correlation analysis ---")
    cross_corr_results: list[dict[str, Any]] = []
    for metric_name, metric_df in onchain_data.items():
        col_name = metric_name.replace("-", "_")
        result = compute_cross_correlation(price_df, metric_df, col_name)
        cross_corr_results.append(result)
        if "best_lag" in result and result["best_lag"]:
            best = result["best_lag"]
            logger.info(
                "  %s: best lag=%d, corr=%.4f, p=%.4f",
                col_name, best["lag"], best["correlation"], best["p_value"],
            )
    all_results["cross_correlations"] = cross_corr_results

    # 2. Granger causality
    logger.info("--- Granger causality tests ---")
    granger_results: list[dict[str, Any]] = []
    for metric_name, metric_df in onchain_data.items():
        col_name = metric_name.replace("-", "_")
        result = granger_causality_test(price_df, metric_df, col_name)
        granger_results.append(result)
        if "most_significant" in result and result["most_significant"]:
            ms = result["most_significant"]
            logger.info(
                "  %s: best lag=%d, F=%.2f, p=%.4f, sig_lags=%d",
                col_name, ms["lag"], ms["f_statistic"], ms["p_value"],
                result["significant_lags_count"],
            )
    all_results["granger_causality"] = granger_results

    # 3. NVT analysis
    logger.info("--- NVT Ratio analysis ---")
    nvt_df = compute_nvt_ratio(price_df, onchain_data)
    nvt_extremes = analyze_nvt_extremes(nvt_df)
    all_results["nvt_analysis"] = nvt_extremes

    # 4. Hashrate analysis
    logger.info("--- Hashrate analysis ---")
    hr_results = analyze_hashrate_price(price_df, onchain_data)
    all_results["hashrate_analysis"] = hr_results

    # 5. Active address analysis
    logger.info("--- Active address analysis ---")
    addr_results = analyze_active_addresses(price_df, onchain_data)
    all_results["active_address_analysis"] = addr_results

    # 6. BB Squeeze + Onchain composite analysis
    logger.info("--- BB Squeeze + Onchain composite analysis ---")
    composite_results = analyze_bb_squeeze_with_onchain(price_df, onchain_data)
    all_results["bb_squeeze_onchain_composite"] = composite_results
    if "composite_signals" in composite_results:
        for sig_name, sig_data in composite_results["composite_signals"].items():
            count = sig_data.get("signal_count", 0)
            ret_14d = sig_data.get("14d", {})
            if isinstance(ret_14d, dict) and "mean_return" in ret_14d:
                logger.info("  %s: n=%d, 14d_ret=%.2f%%, wr=%.1f%%, p=%.4f",
                            sig_name, count,
                            ret_14d["mean_return"] * 100,
                            ret_14d["win_rate"] * 100,
                            ret_14d["p_value"])

    # 7. Backtests
    logger.info("--- Backtesting NVT signal ---")
    nvt_backtest = backtest_nvt_signal(nvt_df)
    all_results["nvt_backtest"] = nvt_backtest
    logger.info("  NVT backtest: return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%",
                nvt_backtest.get("total_return", 0) * 100,
                nvt_backtest.get("sharpe_ratio", 0),
                nvt_backtest.get("max_drawdown", 0) * 100)

    logger.info("--- Backtesting address surge signal ---")
    addr_backtest = backtest_address_surge(price_df, onchain_data)
    all_results["address_surge_backtest"] = addr_backtest
    logger.info("  Address surge backtest: return=%.2f%%, sharpe=%.2f, max_dd=%.2f%%",
                addr_backtest.get("total_return", 0) * 100,
                addr_backtest.get("sharpe_ratio", 0),
                addr_backtest.get("max_drawdown", 0) * 100)

    # 8. Generate edge reports
    edge_reports = []
    if "error" not in nvt_backtest:
        edge_reports.append({
            "edge_name": nvt_backtest["edge_name"],
            "category": "onchain",
            "timeframe": "1d",
            "win_rate": nvt_backtest["win_rate"],
            "expected_return": nvt_backtest["total_return"] / max(nvt_backtest["total_days"], 1) * 365,
            "sharpe_ratio": nvt_backtest["sharpe_ratio"],
            "max_drawdown": nvt_backtest["max_drawdown"],
            "p_value": nvt_extremes.get("low_nvt_signals", [{}])[0].get("p_value", None) if nvt_extremes.get("low_nvt_signals") else None,
            "sample_size": nvt_backtest["sample_size"],
            "test_period": nvt_backtest["test_period"],
            "description": nvt_backtest["strategy"],
            "notes": f"Long signals: {nvt_backtest['total_long_signals']}, Short signals: {nvt_backtest['total_short_signals']}",
        })

    if "error" not in addr_backtest:
        edge_reports.append({
            "edge_name": addr_backtest["edge_name"],
            "category": "onchain",
            "timeframe": "1d",
            "win_rate": addr_backtest["win_rate"],
            "expected_return": addr_backtest["total_return"] / max(addr_backtest["total_days"], 1) * 365,
            "sharpe_ratio": addr_backtest["sharpe_ratio"],
            "max_drawdown": addr_backtest["max_drawdown"],
            "p_value": addr_results.get("address_surge_analysis", {}).get("price_return_14d", {}).get("p_value", None),
            "sample_size": addr_backtest["sample_size"],
            "test_period": addr_backtest["test_period"],
            "description": addr_backtest["strategy"],
            "notes": f"Signal count: {addr_backtest['signal_count']}",
        })

    all_results["edge_reports"] = edge_reports

    # 8. Visualizations
    logger.info("--- Creating visualizations ---")
    create_visualizations(nvt_df, cross_corr_results, price_df, onchain_data)

    # 9. Save report
    save_report_json(all_results, REPORT_CATEGORY, "onchain_results.json")
    logger.info("=== On-chain analysis complete ===")

    return all_results


if __name__ == "__main__":
    results = run_full_analysis()
