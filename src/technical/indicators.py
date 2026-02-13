"""テクニカル分析モジュール。

RSI, MACD, ボリンジャーバンド, ATR, OBV および複合シグナルを計算し、
各指標のトレーディングエッジを統計的に検証する。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.common.utils import (
    setup_logger,
    read_csv,
    save_report_json,
    setup_plot_style,
    save_figure,
    REPORTS_DIR,
)

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class EdgeResult:
    """個別エッジ検証結果を保持するデータクラス。"""

    edge_name: str
    category: str = "technical"
    timeframe: str = "1d"
    win_rate: float = 0.0
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    p_value: float = 1.0
    sample_size: int = 0
    test_period: str = ""
    description: str = ""
    notes: str = ""
    cohens_d: float = 0.0
    first_half_win_rate: float = 0.0
    second_half_win_rate: float = 0.0
    first_half_expected_return: float = 0.0
    second_half_expected_return: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return asdict(self)


# ---------------------------------------------------------------------------
# 指標計算
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index) を計算する。

    Args:
        series: 終値の Series。
        period: 計算期間。

    Returns:
        RSI 値の Series。
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD を計算する。

    Args:
        series: 終値の Series。
        fast: 短期EMA期間。
        slow: 長期EMA期間。
        signal: シグナルライン期間。

    Returns:
        (MACD line, Signal line, Histogram) のタプル。
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """ボリンジャーバンドを計算する。

    Args:
        series: 終値の Series。
        period: 移動平均期間。
        num_std: 標準偏差の倍数。

    Returns:
        (upper, middle, lower, bandwidth) のタプル。
    """
    middle = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR (Average True Range) を計算する。

    Args:
        high: 高値の Series。
        low: 安値の Series。
        close: 終値の Series。
        period: 計算期間。

    Returns:
        ATR 値の Series。
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    return atr


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV (On-Balance Volume) を計算する。

    Args:
        close: 終値の Series。
        volume: 出来高の Series。

    Returns:
        OBV 値の Series。
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    return obv


# ---------------------------------------------------------------------------
# エッジ検証ヘルパー
# ---------------------------------------------------------------------------

def _future_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """N日後のリターンを計算する。

    Args:
        close: 終値の Series。
        periods: 先読み日数。

    Returns:
        将来リターンの Series。
    """
    return close.shift(-periods) / close - 1


def _compute_edge_stats(
    returns: pd.Series,
    edge_name: str,
    description: str,
    timeframe: str = "1d",
    test_period: str = "",
    num_tests: int = 1,
) -> EdgeResult:
    """リターン系列からエッジ統計量を算出する。

    Args:
        returns: シグナル発生時のリターン Series。
        edge_name: エッジ名。
        description: エッジの説明。
        timeframe: タイムフレーム。
        test_period: テスト期間文字列。
        num_tests: Bonferroni補正用のテスト数。

    Returns:
        EdgeResult オブジェクト。
    """
    returns = returns.dropna()
    n = len(returns)
    if n < 5:
        logger.warning("サンプルサイズ不足: %s (n=%d)", edge_name, n)
        return EdgeResult(
            edge_name=edge_name,
            description=description,
            timeframe=timeframe,
            sample_size=n,
            test_period=test_period,
            notes="サンプルサイズ不足",
        )

    win_rate = (returns > 0).mean()
    expected_return = returns.mean()
    std = returns.std()
    sharpe = expected_return / std * np.sqrt(252) if std > 0 else 0.0

    # 最大ドローダウン
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    max_dd = drawdowns.min()

    # t検定 (H0: mean return = 0) with Bonferroni correction
    t_stat, p_raw = stats.ttest_1samp(returns, 0)
    p_value = min(p_raw * num_tests, 1.0)

    # Cohen's d
    cohens_d = expected_return / std if std > 0 else 0.0

    # ロバスト性: 前半/後半分割
    mid = n // 2
    first_half = returns.iloc[:mid]
    second_half = returns.iloc[mid:]

    result = EdgeResult(
        edge_name=edge_name,
        timeframe=timeframe,
        win_rate=round(float(win_rate), 4),
        expected_return=round(float(expected_return), 6),
        sharpe_ratio=round(float(sharpe), 4),
        max_drawdown=round(float(max_dd), 4),
        p_value=round(float(p_value), 6),
        sample_size=n,
        test_period=test_period,
        description=description,
        cohens_d=round(float(cohens_d), 4),
        first_half_win_rate=round(float((first_half > 0).mean()), 4),
        second_half_win_rate=round(float((second_half > 0).mean()), 4),
        first_half_expected_return=round(float(first_half.mean()), 6),
        second_half_expected_return=round(float(second_half.mean()), 6),
    )
    return result


# ---------------------------------------------------------------------------
# 個別エッジ検証
# ---------------------------------------------------------------------------

def edge_rsi_reversal(df: pd.DataFrame, num_tests: int = 1) -> list[EdgeResult]:
    """RSI(14) 極端値でのリバーサルエッジを検証する。

    Args:
        df: ローソク足 DataFrame。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("RSI リバーサルエッジ検証開始")
    rsi = compute_rsi(df["close"])
    ret_1d = _future_returns(df["close"], 1)
    ret_5d = _future_returns(df["close"], 5)
    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"

    results: list[EdgeResult] = []

    # RSI <= 20 → ロング (1日後)
    mask_oversold = rsi <= 20
    results.append(_compute_edge_stats(
        ret_1d[mask_oversold], "RSI<=20 Long 1d",
        "RSI(14)が20以下のとき翌日ロング", "1d", period_str, num_tests,
    ))

    # RSI <= 20 → ロング (5日後)
    results.append(_compute_edge_stats(
        ret_5d[mask_oversold], "RSI<=20 Long 5d",
        "RSI(14)が20以下のとき5日後ロング", "1d", period_str, num_tests,
    ))

    # RSI >= 80 → ショート (1日後)
    mask_overbought = rsi >= 80
    results.append(_compute_edge_stats(
        -ret_1d[mask_overbought], "RSI>=80 Short 1d",
        "RSI(14)が80以上のとき翌日ショート（リターン反転）", "1d", period_str, num_tests,
    ))

    # RSI >= 80 → ショート (5日後)
    results.append(_compute_edge_stats(
        -ret_5d[mask_overbought], "RSI>=80 Short 5d",
        "RSI(14)が80以上のとき5日後ショート（リターン反転）", "1d", period_str, num_tests,
    ))

    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size)
    return results


def edge_macd_cross(df: pd.DataFrame, num_tests: int = 1) -> list[EdgeResult]:
    """MACD ゴールデン/デッドクロス後のリターンを検証する。

    Args:
        df: ローソク足 DataFrame。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("MACD クロスエッジ検証開始")
    macd_line, signal_line, _ = compute_macd(df["close"])
    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"

    # クロス検出
    prev_diff = (macd_line - signal_line).shift(1)
    curr_diff = macd_line - signal_line
    golden_cross = (prev_diff < 0) & (curr_diff >= 0)
    dead_cross = (prev_diff > 0) & (curr_diff <= 0)

    results: list[EdgeResult] = []
    for days in [5, 10, 20]:
        ret = _future_returns(df["close"], days)

        results.append(_compute_edge_stats(
            ret[golden_cross], f"MACD Golden Cross {days}d",
            f"MACDゴールデンクロス後{days}日のロングリターン", "1d", period_str, num_tests,
        ))
        results.append(_compute_edge_stats(
            -ret[dead_cross], f"MACD Dead Cross {days}d",
            f"MACDデッドクロス後{days}日のショートリターン", "1d", period_str, num_tests,
        ))

    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size)
    return results


def edge_bollinger_squeeze(df: pd.DataFrame, num_tests: int = 1) -> list[EdgeResult]:
    """ボリンジャーバンド スクイーズ後ブレイクアウトを検証する。

    Args:
        df: ローソク足 DataFrame。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("ボリンジャーバンド スクイーズエッジ検証開始")
    upper, middle, lower, bandwidth = compute_bollinger_bands(df["close"])
    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"

    # スクイーズ: bandwidth が過去50日の最小値付近（25パーセンタイル以下）
    bw_pctl = bandwidth.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    squeeze = bw_pctl <= 0.25

    # スクイーズ解除後のブレイクアウト方向
    squeeze_release = squeeze.shift(1) & ~squeeze
    breakout_up = squeeze_release & (df["close"] > upper.shift(1))
    breakout_down = squeeze_release & (df["close"] < lower.shift(1))

    results: list[EdgeResult] = []
    for days in [5, 10]:
        ret = _future_returns(df["close"], days)

        results.append(_compute_edge_stats(
            ret[breakout_up], f"BB Squeeze Breakout Up {days}d",
            f"BBスクイーズ後上方ブレイクアウト{days}日リターン", "1d", period_str, num_tests,
        ))
        results.append(_compute_edge_stats(
            -ret[breakout_down], f"BB Squeeze Breakout Down {days}d",
            f"BBスクイーズ後下方ブレイクアウト{days}日ショートリターン", "1d", period_str, num_tests,
        ))

    # スクイーズ中の絶対リターン（方向不問、ボラ拡大狙い）
    ret_10d = _future_returns(df["close"], 10)
    results.append(_compute_edge_stats(
        ret_10d[squeeze].abs(), "BB Squeeze Abs Return 10d",
        "BBスクイーズ中の10日後絶対リターン（ボラ拡大期待）", "1d", period_str, num_tests,
    ))

    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size)
    return results


def edge_atr_regime(df: pd.DataFrame, num_tests: int = 1) -> list[EdgeResult]:
    """ATR(14) ボラティリティレジーム別リターン特性を検証する。

    Args:
        df: ローソク足 DataFrame。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("ATR レジームエッジ検証開始")
    atr = compute_atr(df["high"], df["low"], df["close"])
    atr_pct = atr / df["close"]  # ATR%
    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"

    # レジーム分類: ATR%の3分位
    q33 = atr_pct.quantile(0.33)
    q66 = atr_pct.quantile(0.66)
    low_vol = atr_pct <= q33
    mid_vol = (atr_pct > q33) & (atr_pct <= q66)
    high_vol = atr_pct > q66

    ret_1d = _future_returns(df["close"], 1)
    ret_5d = _future_returns(df["close"], 5)

    results: list[EdgeResult] = []

    for label, mask in [("LowVol", low_vol), ("MidVol", mid_vol), ("HighVol", high_vol)]:
        results.append(_compute_edge_stats(
            ret_1d[mask], f"ATR {label} 1d Return",
            f"ATRレジーム({label})での翌日リターン", "1d", period_str, num_tests,
        ))
        results.append(_compute_edge_stats(
            ret_5d[mask], f"ATR {label} 5d Return",
            f"ATRレジーム({label})での5日後リターン", "1d", period_str, num_tests,
        ))

    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size)
    return results


def edge_obv_divergence(df: pd.DataFrame, num_tests: int = 1) -> list[EdgeResult]:
    """OBV ダイバージェンスによるトレンド転換を検証する。

    Args:
        df: ローソク足 DataFrame。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("OBV ダイバージェンスエッジ検証開始")
    obv = compute_obv(df["close"], df["volume"])
    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"

    # 20日間の変化方向で比較
    lookback = 20
    price_change = df["close"].diff(lookback)
    obv_change = obv.diff(lookback)

    # 弱気ダイバージェンス: 価格上昇 + OBV下降
    bearish_div = (price_change > 0) & (obv_change < 0)
    # 強気ダイバージェンス: 価格下降 + OBV上昇
    bullish_div = (price_change < 0) & (obv_change > 0)

    ret_5d = _future_returns(df["close"], 5)
    ret_10d = _future_returns(df["close"], 10)

    results: list[EdgeResult] = []

    results.append(_compute_edge_stats(
        ret_5d[bullish_div], "OBV Bullish Div 5d",
        "強気ダイバージェンス（価格↓OBV↑）後5日ロング", "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        ret_10d[bullish_div], "OBV Bullish Div 10d",
        "強気ダイバージェンス（価格↓OBV↑）後10日ロング", "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        -ret_5d[bearish_div], "OBV Bearish Div 5d",
        "弱気ダイバージェンス（価格↑OBV↓）後5日ショート", "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        -ret_10d[bearish_div], "OBV Bearish Div 10d",
        "弱気ダイバージェンス（価格↑OBV↓）後10日ショート", "1d", period_str, num_tests,
    ))

    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size)
    return results


def edge_composite_signals(df: pd.DataFrame, num_tests: int = 1) -> list[EdgeResult]:
    """複合シグナル（RSI+MACD、ボリンジャー+出来高）を検証する。

    Args:
        df: ローソク足 DataFrame。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("複合シグナルエッジ検証開始")
    rsi = compute_rsi(df["close"])
    macd_line, signal_line, _ = compute_macd(df["close"])
    upper, middle, lower, bandwidth = compute_bollinger_bands(df["close"])
    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"

    prev_diff = (macd_line - signal_line).shift(1)
    curr_diff = macd_line - signal_line
    golden_cross = (prev_diff < 0) & (curr_diff >= 0)
    dead_cross = (prev_diff > 0) & (curr_diff <= 0)

    ret_5d = _future_returns(df["close"], 5)
    ret_10d = _future_returns(df["close"], 10)

    results: list[EdgeResult] = []

    # RSI + MACD: RSI<=30 かつ MACDゴールデンクロス
    rsi_macd_long = (rsi <= 30) & golden_cross
    results.append(_compute_edge_stats(
        ret_5d[rsi_macd_long], "RSI+MACD Long 5d",
        "RSI<=30 かつ MACDゴールデンクロスの5日後ロング", "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        ret_10d[rsi_macd_long], "RSI+MACD Long 10d",
        "RSI<=30 かつ MACDゴールデンクロスの10日後ロング", "1d", period_str, num_tests,
    ))

    # RSI + MACD: RSI>=70 かつ MACDデッドクロス
    rsi_macd_short = (rsi >= 70) & dead_cross
    results.append(_compute_edge_stats(
        -ret_5d[rsi_macd_short], "RSI+MACD Short 5d",
        "RSI>=70 かつ MACDデッドクロスの5日後ショート", "1d", period_str, num_tests,
    ))

    # ボリンジャー + 出来高: BB上限ブレイク かつ 出来高が20日平均の1.5倍超
    vol_ma20 = df["volume"].rolling(20).mean()
    high_vol = df["volume"] > vol_ma20 * 1.5
    bb_break_up_vol = (df["close"] > upper) & high_vol
    bb_break_down_vol = (df["close"] < lower) & high_vol

    results.append(_compute_edge_stats(
        ret_5d[bb_break_up_vol], "BB+Vol Breakout Up 5d",
        "BB上限突破+高出来高の5日後リターン（トレンド継続）", "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        -ret_5d[bb_break_down_vol], "BB+Vol Breakout Down 5d",
        "BB下限突破+高出来高の5日後ショートリターン", "1d", period_str, num_tests,
    ))

    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size)
    return results


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------

def _plot_edge_summary(all_results: list[EdgeResult]) -> plt.Figure:
    """エッジ結果のサマリーチャートを作成する。

    Args:
        all_results: 全エッジ結果のリスト。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()

    valid = [r for r in all_results if r.sample_size >= 10]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid edges found", ha="center", va="center")
        return fig

    names = [r.edge_name for r in valid]
    win_rates = [r.win_rate for r in valid]
    expected_rets = [r.expected_return * 100 for r in valid]
    p_values = [r.p_value for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(8, len(valid) * 0.4)))

    # Win Rate
    colors_wr = ["#4CAF50" if wr > 0.5 else "#F44336" for wr in win_rates]
    axes[0].barh(names, win_rates, color=colors_wr, alpha=0.8)
    axes[0].axvline(x=0.5, color="white", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Win Rate")
    axes[0].set_title("Win Rate by Edge")

    # Expected Return
    colors_er = ["#4CAF50" if er > 0 else "#F44336" for er in expected_rets]
    axes[1].barh(names, expected_rets, color=colors_er, alpha=0.8)
    axes[1].axvline(x=0, color="white", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Expected Return (%)")
    axes[1].set_title("Expected Return by Edge")

    # p-value (log scale)
    p_colors = ["#4CAF50" if p < 0.05 else "#FFC107" if p < 0.1 else "#F44336" for p in p_values]
    axes[2].barh(names, p_values, color=p_colors, alpha=0.8)
    axes[2].axvline(x=0.05, color="yellow", linestyle="--", alpha=0.7, label="p=0.05")
    axes[2].set_xlabel("p-value (Bonferroni corrected)")
    axes[2].set_title("Statistical Significance")
    axes[2].set_xscale("log")
    axes[2].legend()

    fig.suptitle("BTC/USDT Technical Edge Analysis", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


def _plot_robustness(all_results: list[EdgeResult]) -> plt.Figure:
    """前半/後半のロバスト性比較チャートを作成する。

    Args:
        all_results: 全エッジ結果のリスト。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()

    valid = [r for r in all_results if r.sample_size >= 20]
    if not valid:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No valid edges for robustness check", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(valid) * 0.4)))

    names = [r.edge_name for r in valid]
    first_wr = [r.first_half_win_rate for r in valid]
    second_wr = [r.second_half_win_rate for r in valid]
    first_er = [r.first_half_expected_return * 100 for r in valid]
    second_er = [r.second_half_expected_return * 100 for r in valid]

    y = np.arange(len(names))
    h = 0.35

    axes[0].barh(y - h / 2, first_wr, h, label="First Half", color="#2196F3", alpha=0.8)
    axes[0].barh(y + h / 2, second_wr, h, label="Second Half", color="#FF9800", alpha=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].axvline(x=0.5, color="white", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Win Rate")
    axes[0].set_title("Win Rate: First Half vs Second Half")
    axes[0].legend()

    axes[1].barh(y - h / 2, first_er, h, label="First Half", color="#2196F3", alpha=0.8)
    axes[1].barh(y + h / 2, second_er, h, label="Second Half", color="#FF9800", alpha=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(names)
    axes[1].axvline(x=0, color="white", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Expected Return (%)")
    axes[1].set_title("Expected Return: First Half vs Second Half")
    axes[1].legend()

    fig.suptitle("Robustness Check (Time-Split)", fontsize=16, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# クロスドメイン複合エッジ検証
# ---------------------------------------------------------------------------

def edge_bb_squeeze_onchain(
    df: pd.DataFrame,
    onchain_df: pd.DataFrame,
    num_tests: int = 1,
) -> list[EdgeResult]:
    """BBスクイーズ + オンチェーン指標の複合エッジを検証する。

    Args:
        df: ローソク足 DataFrame。
        onchain_df: オンチェーン DataFrame (timestamp, n_unique_addresses)。
        num_tests: Bonferroni補正のテスト総数。

    Returns:
        EdgeResult のリスト。
    """
    logger.info("BBスクイーズ + オンチェーン 複合エッジ検証開始")

    # BB指標の計算
    upper, middle, lower, bandwidth = compute_bollinger_bands(df["close"])
    bw_pctl = bandwidth.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    squeeze = bw_pctl <= 0.25

    # オンチェーンデータをマージ（日付ベース）
    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["timestamp"]).dt.date
    onchain_work = onchain_df.copy()
    onchain_work["date"] = pd.to_datetime(onchain_work["timestamp"]).dt.date
    merged = df_work.merge(
        onchain_work[["date", "n_unique_addresses"]],
        on="date",
        how="left",
    )
    merged["n_unique_addresses"] = merged["n_unique_addresses"].ffill()

    # アクティブアドレスの z-score (30日ローリング)
    addr = merged["n_unique_addresses"]
    addr_mean = addr.rolling(30, min_periods=20).mean()
    addr_std = addr.rolling(30, min_periods=20).std()
    addr_zscore = (addr - addr_mean) / addr_std

    period_str = f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}"
    results: list[EdgeResult] = []

    # --- シグナル定義 ---
    addr_surge = addr_zscore > 2  # アドレス急増
    addr_drop = addr_zscore < -2  # アドレス急減

    # スクイーズ解除検出
    squeeze_release = squeeze.shift(1).fillna(False) & ~squeeze

    # ブレイクアウト方向
    breakout_up = squeeze_release & (df["close"].values > upper.shift(1).values)
    breakout_down = squeeze_release & (df["close"].values < lower.shift(1).values)

    ret_5d = _future_returns(df["close"], 5)
    ret_10d = _future_returns(df["close"], 10)
    ret_14d = _future_returns(df["close"], 14)

    # --- 1. BBスクイーズ中 + アドレス急増 → ロング ---
    squeeze_addr_surge = squeeze & addr_surge
    results.append(_compute_edge_stats(
        ret_5d[squeeze_addr_surge], "BB Squeeze + Addr Surge 5d",
        "BBスクイーズ中にアドレス急増(z>2)が発生、5日後ロング",
        "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        ret_10d[squeeze_addr_surge], "BB Squeeze + Addr Surge 10d",
        "BBスクイーズ中にアドレス急増(z>2)が発生、10日後ロング",
        "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        ret_14d[squeeze_addr_surge], "BB Squeeze + Addr Surge 14d",
        "BBスクイーズ中にアドレス急増(z>2)が発生、14日後ロング",
        "1d", period_str, num_tests,
    ))

    # --- 2. BBスクイーズ中 + アドレス急減 → ショート ---
    squeeze_addr_drop = squeeze & addr_drop
    results.append(_compute_edge_stats(
        -ret_5d[squeeze_addr_drop], "BB Squeeze + Addr Drop 5d",
        "BBスクイーズ中にアドレス急減(z<-2)が発生、5日後ショート",
        "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        -ret_10d[squeeze_addr_drop], "BB Squeeze + Addr Drop 10d",
        "BBスクイーズ中にアドレス急減(z<-2)が発生、10日後ショート",
        "1d", period_str, num_tests,
    ))

    # --- 3. スクイーズ解除時のアドレス状態別ブレイクアウト方向 ---
    # 直近5日間にアドレス急増があったか
    addr_surge_recent = addr_surge.rolling(5, min_periods=1).max().astype(bool)
    release_with_addr_surge = squeeze_release & addr_surge_recent

    results.append(_compute_edge_stats(
        ret_5d[release_with_addr_surge], "Squeeze Release + Recent Addr Surge 5d",
        "スクイーズ解除時に直近5日間アドレス急増あり、5日後ロング",
        "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        ret_10d[release_with_addr_surge], "Squeeze Release + Recent Addr Surge 10d",
        "スクイーズ解除時に直近5日間アドレス急増あり、10日後ロング",
        "1d", period_str, num_tests,
    ))

    # --- 4. BB上限ブレイク + 高出来高 + アドレス急増(直近5日) ---
    vol_ma20 = df["volume"].rolling(20).mean()
    high_vol = df["volume"] > vol_ma20 * 1.5
    bb_up_vol_addr = (df["close"] > upper) & high_vol & addr_surge_recent

    results.append(_compute_edge_stats(
        ret_5d[bb_up_vol_addr], "BB+Vol+Addr Breakout Up 5d",
        "BB上限突破+高出来高+直近アドレス急増、5日後ロング",
        "1d", period_str, num_tests,
    ))
    results.append(_compute_edge_stats(
        ret_10d[bb_up_vol_addr], "BB+Vol+Addr Breakout Up 10d",
        "BB上限突破+高出来高+直近アドレス急増、10日後ロング",
        "1d", period_str, num_tests,
    ))

    # --- ログ出力 ---
    for r in results:
        logger.info("  %s: win=%.2f%%, E[r]=%.4f%%, p=%.4f, n=%d, d=%.4f",
                     r.edge_name, r.win_rate * 100, r.expected_return * 100,
                     r.p_value, r.sample_size, r.cohens_d)

    # シグナル発生状況の統計
    logger.info("シグナル発生統計:")
    logger.info("  BBスクイーズ日数: %d / %d", squeeze.sum(), len(df))
    logger.info("  アドレス急増(z>2)日数: %d", addr_surge.sum())
    logger.info("  スクイーズ+アドレス急増: %d", squeeze_addr_surge.sum())
    logger.info("  スクイーズ解除+直近アドレス急増: %d", release_with_addr_surge.sum())
    logger.info("  BB上限突破+高出来高+アドレス急増: %d", bb_up_vol_addr.sum())

    return results


def _plot_composite_edge(
    df: pd.DataFrame,
    onchain_df: pd.DataFrame,
) -> plt.Figure:
    """BBスクイーズ + アドレス急増の複合シグナル可視化。

    Args:
        df: ローソク足 DataFrame。
        onchain_df: オンチェーン DataFrame。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()

    upper, middle, lower, bandwidth = compute_bollinger_bands(df["close"])
    bw_pctl = bandwidth.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    squeeze = bw_pctl <= 0.25

    # アドレス z-score
    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["timestamp"]).dt.date
    onchain_work = onchain_df.copy()
    onchain_work["date"] = pd.to_datetime(onchain_work["timestamp"]).dt.date
    merged = df_work.merge(
        onchain_work[["date", "n_unique_addresses"]],
        on="date", how="left",
    )
    merged["n_unique_addresses"] = merged["n_unique_addresses"].ffill()
    addr = merged["n_unique_addresses"]
    addr_mean = addr.rolling(30, min_periods=20).mean()
    addr_std = addr.rolling(30, min_periods=20).std()
    addr_zscore = (addr - addr_mean) / addr_std

    combo_signal = squeeze & (addr_zscore > 2)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Price + BB
    ts = df["timestamp"]
    axes[0].plot(ts, df["close"], color="white", linewidth=1, label="Close")
    axes[0].plot(ts, upper, color="#FF5722", linewidth=0.8, alpha=0.5, label="BB Upper")
    axes[0].plot(ts, lower, color="#2196F3", linewidth=0.8, alpha=0.5, label="BB Lower")
    axes[0].fill_between(ts, lower, upper, alpha=0.05, color="white")
    # Highlight squeeze periods
    for i in range(len(ts)):
        if squeeze.iloc[i]:
            axes[0].axvspan(ts.iloc[i], ts.iloc[min(i + 1, len(ts) - 1)],
                           alpha=0.15, color="yellow")
    # Mark combo signals
    combo_idx = combo_signal[combo_signal].index
    if len(combo_idx) > 0:
        axes[0].scatter(ts.iloc[combo_idx], df["close"].iloc[combo_idx],
                       color="#00FF00", marker="^", s=100, zorder=5,
                       label="Squeeze + Addr Surge")
    axes[0].set_ylabel("Price (USDT)")
    axes[0].set_title("BTC/USDT Price with BB Squeeze + Address Surge Signals")
    axes[0].legend(loc="upper left", fontsize=9)

    # BB Bandwidth percentile
    axes[1].plot(ts, bw_pctl, color="#FFC107", linewidth=1)
    axes[1].axhline(y=0.25, color="red", linestyle="--", alpha=0.7, label="Squeeze threshold (25th pctl)")
    axes[1].fill_between(ts, 0, bw_pctl.clip(upper=0.25), alpha=0.3, color="yellow")
    axes[1].set_ylabel("BB Width Percentile")
    axes[1].set_title("Bollinger Bandwidth Percentile (50d rolling)")
    axes[1].legend(loc="upper right", fontsize=9)

    # Address z-score
    axes[2].plot(ts, addr_zscore, color="#9C27B0", linewidth=1)
    axes[2].axhline(y=2, color="green", linestyle="--", alpha=0.7, label="z = +2")
    axes[2].axhline(y=-2, color="red", linestyle="--", alpha=0.7, label="z = -2")
    axes[2].fill_between(ts, 2, addr_zscore.clip(lower=2), alpha=0.3, color="green")
    axes[2].fill_between(ts, -2, addr_zscore.clip(upper=-2), alpha=0.3, color="red")
    axes[2].set_ylabel("Address z-score")
    axes[2].set_title("Active Address z-score (30d rolling)")
    axes[2].legend(loc="upper right", fontsize=9)

    fig.suptitle("Composite Edge: BB Squeeze + Active Address Surge", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------

def run_analysis() -> list[EdgeResult]:
    """全テクニカル分析を実行し、結果を保存する。

    Returns:
        全エッジ結果のリスト。
    """
    logger.info("=== テクニカル分析開始 ===")

    # データ読み込み
    df = read_csv("processed", "btcusdt_1d.csv")
    logger.info("データ読み込み完了: %d行 (%s ~ %s)",
                len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # 総テスト数（Bonferroni補正用）
    # RSI: 4, MACD: 6, BB: 5, ATR: 6, OBV: 4, Composite: 6 = 31
    total_tests = 31

    # 各エッジ検証
    all_results: list[EdgeResult] = []
    all_results.extend(edge_rsi_reversal(df, total_tests))
    all_results.extend(edge_macd_cross(df, total_tests))
    all_results.extend(edge_bollinger_squeeze(df, total_tests))
    all_results.extend(edge_atr_regime(df, total_tests))
    all_results.extend(edge_obv_divergence(df, total_tests))
    all_results.extend(edge_composite_signals(df, total_tests))

    # 結果保存 (JSON)
    results_dicts = [r.to_dict() for r in all_results]
    save_report_json(results_dicts, "technical", "edge_results.json")

    # 有望なエッジのサマリー
    promising = [r for r in all_results if r.p_value < 0.05 and r.sample_size >= 20]
    if promising:
        logger.info("=== 有望なエッジ (p < 0.05, n >= 20) ===")
        for r in promising:
            logger.info("  %s: win=%.1f%%, E[r]=%.3f%%, Sharpe=%.2f, p=%.4f, n=%d, d=%.3f",
                         r.edge_name, r.win_rate * 100, r.expected_return * 100,
                         r.sharpe_ratio, r.p_value, r.sample_size, r.cohens_d)
    else:
        logger.info("有望なエッジは見つかりませんでした（p < 0.05 かつ n >= 20 の条件）")

    # 可視化
    fig_summary = _plot_edge_summary(all_results)
    save_figure(fig_summary, "technical", "edge_summary.png")

    fig_robust = _plot_robustness(all_results)
    save_figure(fig_robust, "technical", "robustness_check.png")

    logger.info("=== テクニカル分析完了 ===")
    return all_results


def run_cross_validation() -> list[EdgeResult]:
    """BBスクイーズ + オンチェーンの複合エッジ検証を実行する。

    Returns:
        複合エッジ結果のリスト。
    """
    logger.info("=== クロスドメイン複合エッジ検証開始 ===")

    df = read_csv("processed", "btcusdt_1d.csv")
    onchain_df = read_csv("onchain", "n_unique_addresses.csv")
    logger.info("価格データ: %d行, オンチェーンデータ: %d行", len(df), len(onchain_df))

    # テスト数 (Bonferroni補正: この検証内の10テスト)
    num_tests = 10

    results = edge_bb_squeeze_onchain(df, onchain_df, num_tests)

    # JSON保存
    results_dicts = [r.to_dict() for r in results]
    save_report_json(results_dicts, "technical", "composite_edge_results.json")

    # 可視化
    fig = _plot_composite_edge(df, onchain_df)
    save_figure(fig, "technical", "composite_bb_address.png")

    # 有望なエッジ
    promising = [r for r in results if r.p_value < 0.10 and r.sample_size >= 5]
    if promising:
        logger.info("=== 注目エッジ (p < 0.10, n >= 5) ===")
        for r in promising:
            logger.info("  %s: win=%.1f%%, E[r]=%.3f%%, Sharpe=%.2f, p=%.4f, n=%d, d=%.3f",
                         r.edge_name, r.win_rate * 100, r.expected_return * 100,
                         r.sharpe_ratio, r.p_value, r.sample_size, r.cohens_d)
    else:
        logger.info("注目エッジなし（p < 0.10 かつ n >= 5）")

    logger.info("=== クロスドメイン複合エッジ検証完了 ===")
    return results


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) > 1 and _sys.argv[1] == "--cross":
        results = run_cross_validation()
        promising = [r for r in results if r.p_value < 0.10 and r.sample_size >= 5]
        print(f"\n複合エッジ: {len(results)} テスト、うち {len(promising)} 個が注目 (p < 0.10)")
        for r in promising:
            print(f"  {r.edge_name}: WR={r.win_rate:.1%}, E[r]={r.expected_return:.3%}, "
                  f"Sharpe={r.sharpe_ratio:.2f}, p={r.p_value:.4f}, n={r.sample_size}")
    else:
        results = run_analysis()
        promising = [r for r in results if r.p_value < 0.05 and r.sample_size >= 20]
        print(f"\n合計 {len(results)} エッジを検証、うち {len(promising)} 個が統計的に有意 (p < 0.05)")
        for r in promising:
            print(f"  {r.edge_name}: WR={r.win_rate:.1%}, E[r]={r.expected_return:.3%}, "
                  f"Sharpe={r.sharpe_ratio:.2f}, p={r.p_value:.4f}, n={r.sample_size}")
