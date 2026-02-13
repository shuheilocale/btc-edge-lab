"""統計的アノマリー検出モジュール。

BTC/USDTの価格データに対して複数の統計的アノマリー検定を実施し、
トレーディングエッジの存在を検証する。

検出対象:
- 曜日効果（Day-of-Week Effect）
- 月別効果（Month-of-Year Effect）
- 時間帯効果（Trading Session Effect）
- 月末/月初効果（Turn-of-Month Effect）
- ボラティリティクラスタリング後リターン
- 連続陽線/陰線後リターン
- 出来高アノマリー
"""

import json
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import sys
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
# データ型定義
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult:
    """個別アノマリー検定の結果。"""

    edge_name: str
    category: str = "anomaly"
    timeframe: str = "1d"
    description: str = ""
    test_statistic: float = 0.0
    p_value: float = 1.0
    p_value_corrected: float = 1.0
    effect_size: float = 0.0
    effect_size_label: str = ""
    sample_size: int = 0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    power: float = 0.0
    significant: bool = False
    win_rate: float = 0.0
    expected_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    test_period: str = ""
    notes: str = ""
    sub_results: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d 効果量を計算する。

    Args:
        group1: グループ1のデータ。
        group2: グループ2のデータ。

    Returns:
        Cohen's d の値。
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def _effect_size_label(d: float) -> str:
    """Cohen's d の大きさラベルを返す。

    Args:
        d: Cohen's d の絶対値。

    Returns:
        効果量のラベル文字列。
    """
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def _bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95
) -> tuple[float, float]:
    """ブートストラップ法で平均の信頼区間を計算する。

    Args:
        data: 元データ。
        n_bootstrap: ブートストラップ繰り返し回数。
        ci: 信頼水準。

    Returns:
        (下限, 上限) のタプル。
    """
    if len(data) < 2:
        m = float(np.mean(data)) if len(data) > 0 else 0.0
        return (m, m)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - ci
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lower, upper)


def _compute_power_ttest(
    effect_size: float, n: int, alpha: float = 0.05
) -> float:
    """2群の t 検定に対する近似検定力を計算する。

    Args:
        effect_size: Cohen's d。
        n: 各群のサンプルサイズ（最小の群）。
        alpha: 有意水準。

    Returns:
        検定力（0〜1）。
    """
    if n < 2 or effect_size == 0:
        return 0.0
    # noncentrality parameter
    ncp = abs(effect_size) * np.sqrt(n / 2)
    df = 2 * n - 2
    critical_t = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - stats.nct.cdf(critical_t, df, ncp) + stats.nct.cdf(-critical_t, df, ncp)
    return float(np.clip(power, 0, 1))


def _bonferroni(p_values: list[float]) -> list[float]:
    """Bonferroni 多重比較補正を適用する。

    Args:
        p_values: 元の p 値リスト。

    Returns:
        補正後の p 値リスト。
    """
    m = len(p_values)
    return [min(p * m, 1.0) for p in p_values]


def _compute_edge_metrics(
    returns: np.ndarray,
) -> tuple[float, float, float, float]:
    """リターン系列からエッジ指標を計算する。

    Args:
        returns: リターン配列。

    Returns:
        (win_rate, expected_return, sharpe_ratio, max_drawdown) のタプル。
    """
    if len(returns) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    win_rate = float(np.mean(returns > 0))
    expected_return = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 1.0
    sharpe = (expected_return / std * np.sqrt(252)) if std > 0 else 0.0
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    return (win_rate, expected_return, float(sharpe), max_dd)


def _prepare_daily(df: pd.DataFrame) -> pd.DataFrame:
    """日次データを準備し、リターン列を追加する。

    Args:
        df: ローソク足 DataFrame（timestamp, open, high, low, close, volume）。

    Returns:
        リターン列が追加された DataFrame。
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna(subset=["return"]).reset_index(drop=True)
    return df


def _prepare_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """時間足データを準備し、リターン列を追加する。

    Args:
        df: ローソク足 DataFrame。

    Returns:
        リターン列が追加された DataFrame。
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["return"] = df["close"].pct_change()
    df = df.dropna(subset=["return"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# アノマリー検定: 曜日効果
# ---------------------------------------------------------------------------

def test_day_of_week(df: pd.DataFrame) -> AnomalyResult:
    """曜日効果を検定する（Kruskal-Wallis + 事後検定）。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing day-of-week effect...")
    df = df.copy()
    df["dow"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    groups = [df.loc[df["dow"] == i, "return"].values for i in range(7)]
    stat, p_val = stats.kruskal(*[g for g in groups if len(g) > 0])

    # 事後: 各曜日ペアの Mann-Whitney
    sub_results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    pair_labels: list[str] = []

    for i in range(7):
        mean_ret = float(np.mean(groups[i])) if len(groups[i]) > 0 else 0.0
        ci_lo, ci_hi = _bootstrap_ci(groups[i]) if len(groups[i]) > 1 else (0.0, 0.0)
        sub_results.append({
            "day": day_names[i],
            "mean_return": round(mean_ret, 6),
            "median_return": round(float(np.median(groups[i])), 6) if len(groups[i]) > 0 else 0.0,
            "std": round(float(np.std(groups[i], ddof=1)), 6) if len(groups[i]) > 1 else 0.0,
            "n": len(groups[i]),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
        })

    for i in range(7):
        for j in range(i + 1, 7):
            if len(groups[i]) > 0 and len(groups[j]) > 0:
                _, p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
                raw_p_values.append(p)
                pair_labels.append(f"{day_names[i]} vs {day_names[j]}")

    corrected = _bonferroni(raw_p_values)
    for k, label in enumerate(pair_labels):
        sub_results.append({
            "pair": label,
            "p_value_raw": round(raw_p_values[k], 6),
            "p_value_corrected": round(corrected[k], 6),
        })

    # 最大差の効果量: best day vs worst day
    means = [float(np.mean(g)) for g in groups if len(g) > 0]
    best_idx = int(np.argmax(means))
    worst_idx = int(np.argmin(means))
    d = _cohens_d(groups[best_idx], groups[worst_idx])
    min_n = min(len(groups[best_idx]), len(groups[worst_idx]))
    power = _compute_power_ttest(d, min_n)

    test_period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"

    # 有意判定 (corrected p < 0.05)
    p_corrected = min(p_val * 7, 1.0)  # overall のBonferroni近似
    significant = p_corrected < 0.05

    wr, er, sr, mdd = _compute_edge_metrics(groups[best_idx])

    return AnomalyResult(
        edge_name="Day-of-Week Effect",
        timeframe="1d",
        description=f"Kruskal-Wallis test for return differences across weekdays. Best: {day_names[best_idx]}, Worst: {day_names[worst_idx]}",
        test_statistic=float(stat),
        p_value=float(p_val),
        p_value_corrected=p_corrected,
        effect_size=d,
        effect_size_label=_effect_size_label(d),
        sample_size=len(df),
        ci_lower=sub_results[best_idx]["ci_95_lower"],
        ci_upper=sub_results[best_idx]["ci_95_upper"],
        power=power,
        significant=significant,
        win_rate=wr,
        expected_return=er,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        test_period=test_period,
        notes=f"Best day: {day_names[best_idx]} (mean={means[best_idx]:.6f}), Worst: {day_names[worst_idx]} (mean={means[worst_idx]:.6f})",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# アノマリー検定: 月別効果
# ---------------------------------------------------------------------------

def test_month_of_year(df: pd.DataFrame) -> AnomalyResult:
    """月別効果を検定する。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing month-of-year effect...")
    df = df.copy()
    df["month"] = df["timestamp"].dt.month
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    # 月次集計: 各月のリターンを合算
    df["year"] = df["timestamp"].dt.year
    monthly = df.groupby(["year", "month"])["return"].sum().reset_index()

    groups = [monthly.loc[monthly["month"] == m, "return"].values for m in range(1, 13)]
    non_empty = [g for g in groups if len(g) > 0]

    if len(non_empty) < 2:
        return AnomalyResult(edge_name="Month-of-Year Effect", notes="Insufficient data")

    stat, p_val = stats.kruskal(*non_empty)

    sub_results: list[dict[str, Any]] = []
    for i, m in enumerate(range(1, 13)):
        g = groups[i]
        if len(g) == 0:
            continue
        ci_lo, ci_hi = _bootstrap_ci(g) if len(g) > 1 else (0.0, 0.0)
        sub_results.append({
            "month": month_names[i],
            "mean_monthly_return": round(float(np.mean(g)), 6),
            "median_monthly_return": round(float(np.median(g)), 6),
            "std": round(float(np.std(g, ddof=1)), 6) if len(g) > 1 else 0.0,
            "n_months": len(g),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
        })

    # 注目月（1,4,10,11,12）vs その他
    focus_months = {1, 4, 10, 11, 12}
    focus_returns = monthly.loc[monthly["month"].isin(focus_months), "return"].values
    other_returns = monthly.loc[~monthly["month"].isin(focus_months), "return"].values

    if len(focus_returns) > 0 and len(other_returns) > 0:
        _, p_focus = stats.mannwhitneyu(focus_returns, other_returns, alternative="two-sided")
        d_focus = _cohens_d(focus_returns, other_returns)
    else:
        p_focus = 1.0
        d_focus = 0.0

    sub_results.append({
        "comparison": "Focus months (Jan,Apr,Oct-Dec) vs Others",
        "p_value": round(p_focus, 6),
        "cohens_d": round(d_focus, 4),
    })

    means = [float(np.mean(g)) for g in groups if len(g) > 0]
    valid_indices = [i for i in range(12) if len(groups[i]) > 0]
    best_valid = valid_indices[int(np.argmax(means))]
    worst_valid = valid_indices[int(np.argmin(means))]

    d = _cohens_d(groups[best_valid], groups[worst_valid])
    min_n = min(len(groups[best_valid]), len(groups[worst_valid]))
    power = _compute_power_ttest(d, min_n)

    p_corrected = min(p_val * 12, 1.0)
    significant = p_corrected < 0.05

    test_period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"

    wr, er, sr, mdd = _compute_edge_metrics(groups[best_valid])

    return AnomalyResult(
        edge_name="Month-of-Year Effect",
        timeframe="1M",
        description=f"Monthly return seasonality. Best: {month_names[best_valid]}, Worst: {month_names[worst_valid]}",
        test_statistic=float(stat),
        p_value=float(p_val),
        p_value_corrected=p_corrected,
        effect_size=d,
        effect_size_label=_effect_size_label(d),
        sample_size=len(monthly),
        ci_lower=0.0,
        ci_upper=0.0,
        power=power,
        significant=significant,
        win_rate=wr,
        expected_return=er,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        test_period=test_period,
        notes=f"Focus months p={p_focus:.4f}, Cohen's d={d_focus:.4f}",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# アノマリー検定: 時間帯効果
# ---------------------------------------------------------------------------

def test_session_effect(df_hourly: pd.DataFrame) -> AnomalyResult:
    """Asian/European/US セッション別リターンを検定する。

    セッション定義（UTC）:
    - Asian: 00:00-08:00
    - European: 08:00-16:00
    - US: 16:00-24:00

    Args:
        df_hourly: _prepare_hourly 済みの時間足 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing trading session effect...")
    df = df_hourly.copy()
    df["hour"] = df["timestamp"].dt.hour

    def assign_session(hour: int) -> str:
        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 16:
            return "European"
        else:
            return "US"

    df["session"] = df["hour"].apply(assign_session)

    # 日次セッションリターンに集約
    df["date"] = df["timestamp"].dt.date
    session_daily = df.groupby(["date", "session"])["return"].sum().reset_index()

    groups_dict: dict[str, np.ndarray] = {}
    for sess in ["Asian", "European", "US"]:
        groups_dict[sess] = session_daily.loc[
            session_daily["session"] == sess, "return"
        ].values

    non_empty = [v for v in groups_dict.values() if len(v) > 0]
    if len(non_empty) < 2:
        return AnomalyResult(edge_name="Trading Session Effect", notes="Insufficient data")

    stat, p_val = stats.kruskal(*non_empty)

    sub_results: list[dict[str, Any]] = []
    raw_p: list[float] = []
    pair_labels: list[str] = []
    sessions = ["Asian", "European", "US"]

    for sess in sessions:
        g = groups_dict[sess]
        ci_lo, ci_hi = _bootstrap_ci(g) if len(g) > 1 else (0.0, 0.0)
        sub_results.append({
            "session": sess,
            "mean_return": round(float(np.mean(g)), 6) if len(g) > 0 else 0.0,
            "median_return": round(float(np.median(g)), 6) if len(g) > 0 else 0.0,
            "std": round(float(np.std(g, ddof=1)), 6) if len(g) > 1 else 0.0,
            "n": len(g),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
        })

    for i in range(3):
        for j in range(i + 1, 3):
            g1, g2 = groups_dict[sessions[i]], groups_dict[sessions[j]]
            if len(g1) > 0 and len(g2) > 0:
                _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                raw_p.append(p)
                pair_labels.append(f"{sessions[i]} vs {sessions[j]}")

    corrected = _bonferroni(raw_p)
    for k, label in enumerate(pair_labels):
        sub_results.append({
            "pair": label,
            "p_value_raw": round(raw_p[k], 6),
            "p_value_corrected": round(corrected[k], 6),
        })

    means = {s: float(np.mean(groups_dict[s])) for s in sessions if len(groups_dict[s]) > 0}
    best_sess = max(means, key=means.get)  # type: ignore[arg-type]
    worst_sess = min(means, key=means.get)  # type: ignore[arg-type]
    d = _cohens_d(groups_dict[best_sess], groups_dict[worst_sess])
    min_n = min(len(groups_dict[best_sess]), len(groups_dict[worst_sess]))
    power = _compute_power_ttest(d, min_n)

    p_corrected = min(p_val * 3, 1.0)
    significant = p_corrected < 0.05

    test_period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"

    wr, er, sr, mdd = _compute_edge_metrics(groups_dict[best_sess])

    return AnomalyResult(
        edge_name="Trading Session Effect",
        timeframe="1h",
        description=f"Return differences across Asian/European/US sessions. Best: {best_sess}",
        test_statistic=float(stat),
        p_value=float(p_val),
        p_value_corrected=p_corrected,
        effect_size=d,
        effect_size_label=_effect_size_label(d),
        sample_size=len(session_daily),
        ci_lower=0.0,
        ci_upper=0.0,
        power=power,
        significant=significant,
        win_rate=wr,
        expected_return=er,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        test_period=test_period,
        notes=f"Best session: {best_sess} (mean={means[best_sess]:.6f})",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# アノマリー検定: 月末/月初効果
# ---------------------------------------------------------------------------

def test_turn_of_month(df: pd.DataFrame) -> AnomalyResult:
    """月末5日間 vs 月初5日間 vs その他のリターンを検定する。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing turn-of-month effect...")
    df = df.copy()
    df["day"] = df["timestamp"].dt.day
    df["days_in_month"] = df["timestamp"].dt.days_in_month

    def classify(row: pd.Series) -> str:
        if row["day"] <= 5:
            return "month_start"
        elif row["day"] > row["days_in_month"] - 5:
            return "month_end"
        else:
            return "mid_month"

    df["period"] = df.apply(classify, axis=1)

    groups_dict: dict[str, np.ndarray] = {}
    for period in ["month_start", "month_end", "mid_month"]:
        groups_dict[period] = df.loc[df["period"] == period, "return"].values

    non_empty = [v for v in groups_dict.values() if len(v) > 0]
    if len(non_empty) < 2:
        return AnomalyResult(edge_name="Turn-of-Month Effect", notes="Insufficient data")

    stat, p_val = stats.kruskal(*non_empty)

    sub_results: list[dict[str, Any]] = []
    periods = ["month_start", "month_end", "mid_month"]
    raw_p: list[float] = []
    pair_labels: list[str] = []

    for period in periods:
        g = groups_dict[period]
        ci_lo, ci_hi = _bootstrap_ci(g) if len(g) > 1 else (0.0, 0.0)
        sub_results.append({
            "period": period,
            "mean_return": round(float(np.mean(g)), 6) if len(g) > 0 else 0.0,
            "median_return": round(float(np.median(g)), 6) if len(g) > 0 else 0.0,
            "std": round(float(np.std(g, ddof=1)), 6) if len(g) > 1 else 0.0,
            "n": len(g),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
        })

    for i in range(3):
        for j in range(i + 1, 3):
            g1, g2 = groups_dict[periods[i]], groups_dict[periods[j]]
            if len(g1) > 0 and len(g2) > 0:
                _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                raw_p.append(p)
                pair_labels.append(f"{periods[i]} vs {periods[j]}")

    corrected = _bonferroni(raw_p)
    for k, label in enumerate(pair_labels):
        sub_results.append({
            "pair": label,
            "p_value_raw": round(raw_p[k], 6),
            "p_value_corrected": round(corrected[k], 6),
        })

    # month_start + month_end (turn) vs mid_month
    turn = np.concatenate([groups_dict["month_start"], groups_dict["month_end"]])
    mid = groups_dict["mid_month"]
    if len(turn) > 0 and len(mid) > 0:
        _, p_turn = stats.mannwhitneyu(turn, mid, alternative="two-sided")
        d_turn = _cohens_d(turn, mid)
    else:
        p_turn = 1.0
        d_turn = 0.0

    sub_results.append({
        "comparison": "Turn-of-month (start+end) vs Mid-month",
        "p_value": round(p_turn, 6),
        "cohens_d": round(d_turn, 4),
    })

    means = {p: float(np.mean(groups_dict[p])) for p in periods if len(groups_dict[p]) > 0}
    best_period = max(means, key=means.get)  # type: ignore[arg-type]
    worst_period = min(means, key=means.get)  # type: ignore[arg-type]
    d = _cohens_d(groups_dict[best_period], groups_dict[worst_period])
    min_n = min(len(groups_dict[best_period]), len(groups_dict[worst_period]))
    power = _compute_power_ttest(d, min_n)

    p_corrected = min(p_val * 3, 1.0)
    significant = p_corrected < 0.05

    test_period_str = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"

    wr, er, sr, mdd = _compute_edge_metrics(groups_dict[best_period])

    return AnomalyResult(
        edge_name="Turn-of-Month Effect",
        timeframe="1d",
        description=f"Returns in first/last 5 days vs mid-month. Best: {best_period}",
        test_statistic=float(stat),
        p_value=float(p_val),
        p_value_corrected=p_corrected,
        effect_size=d,
        effect_size_label=_effect_size_label(d),
        sample_size=len(df),
        ci_lower=0.0,
        ci_upper=0.0,
        power=power,
        significant=significant,
        win_rate=wr,
        expected_return=er,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        test_period=test_period_str,
        notes=f"Turn vs Mid p={p_turn:.4f}, Cohen's d={d_turn:.4f}",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# アノマリー検定: ボラティリティクラスタリング
# ---------------------------------------------------------------------------

def test_volatility_clustering(df: pd.DataFrame) -> AnomalyResult:
    """大きな日次変動（>2σ）後の翌日リターンを検定する。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing volatility clustering effect...")
    df = df.copy()
    abs_ret = df["return"].abs()
    mean_abs = abs_ret.mean()
    std_abs = abs_ret.std()
    threshold = mean_abs + 2 * std_abs

    df["big_move"] = abs_ret > threshold
    df["next_return"] = df["return"].shift(-1)
    df = df.dropna(subset=["next_return"])

    after_big = df.loc[df["big_move"], "next_return"].values
    after_normal = df.loc[~df["big_move"], "next_return"].values

    if len(after_big) < 5:
        return AnomalyResult(
            edge_name="Volatility Clustering Effect",
            notes="Too few large moves detected",
        )

    stat, p_val = stats.mannwhitneyu(after_big, after_normal, alternative="two-sided")
    d = _cohens_d(after_big, after_normal)
    ci_lo, ci_hi = _bootstrap_ci(after_big)
    min_n = min(len(after_big), len(after_normal))
    power = _compute_power_ttest(d, min_n)

    # Positive big moves vs Negative big moves (direction matters)
    df["big_up"] = (df["return"] > threshold)
    df["big_down"] = (df["return"] < -threshold)

    after_big_up = df.loc[df["big_up"], "next_return"].values
    after_big_down = df.loc[df["big_down"], "next_return"].values

    sub_results: list[dict[str, Any]] = [
        {
            "group": "After big move (>2σ)",
            "mean_next_return": round(float(np.mean(after_big)), 6),
            "median_next_return": round(float(np.median(after_big)), 6),
            "std": round(float(np.std(after_big, ddof=1)), 6),
            "n": len(after_big),
        },
        {
            "group": "After normal move",
            "mean_next_return": round(float(np.mean(after_normal)), 6),
            "median_next_return": round(float(np.median(after_normal)), 6),
            "std": round(float(np.std(after_normal, ddof=1)), 6),
            "n": len(after_normal),
        },
    ]

    if len(after_big_up) > 2:
        ci_up_lo, ci_up_hi = _bootstrap_ci(after_big_up)
        sub_results.append({
            "group": "After big UP move (>+2σ)",
            "mean_next_return": round(float(np.mean(after_big_up)), 6),
            "n": len(after_big_up),
            "ci_95_lower": round(ci_up_lo, 6),
            "ci_95_upper": round(ci_up_hi, 6),
        })

    if len(after_big_down) > 2:
        ci_dn_lo, ci_dn_hi = _bootstrap_ci(after_big_down)
        sub_results.append({
            "group": "After big DOWN move (<-2σ)",
            "mean_next_return": round(float(np.mean(after_big_down)), 6),
            "n": len(after_big_down),
            "ci_95_lower": round(ci_dn_lo, 6),
            "ci_95_upper": round(ci_dn_hi, 6),
        })

    significant = p_val < 0.05
    test_period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"

    wr, er, sr, mdd = _compute_edge_metrics(after_big)

    return AnomalyResult(
        edge_name="Volatility Clustering Effect",
        timeframe="1d",
        description=f"Next-day return after daily moves >2σ (threshold={threshold:.4f})",
        test_statistic=float(stat),
        p_value=float(p_val),
        p_value_corrected=float(p_val),  # single test
        effect_size=d,
        effect_size_label=_effect_size_label(d),
        sample_size=len(after_big),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        power=power,
        significant=significant,
        win_rate=wr,
        expected_return=er,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        test_period=test_period,
        notes=f"Big moves: {len(after_big)} days ({len(after_big)/len(df)*100:.1f}%)",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# アノマリー検定: 連続陽線/陰線
# ---------------------------------------------------------------------------

def test_consecutive_candles(df: pd.DataFrame) -> AnomalyResult:
    """N日連続上昇/下落後の翌日リターンを検定する（N=3,4,5,6,7）。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing consecutive candles effect...")
    df = df.copy()
    df["up"] = (df["return"] > 0).astype(int)
    df["down"] = (df["return"] < 0).astype(int)

    # 連続上昇/下落のカウント
    df["consec_up"] = 0
    df["consec_down"] = 0
    consec_up = 0
    consec_down = 0
    for idx in df.index:
        if df.loc[idx, "up"] == 1:
            consec_up += 1
            consec_down = 0
        elif df.loc[idx, "down"] == 1:
            consec_down += 1
            consec_up = 0
        else:
            consec_up = 0
            consec_down = 0
        df.loc[idx, "consec_up"] = consec_up
        df.loc[idx, "consec_down"] = consec_down

    df["next_return"] = df["return"].shift(-1)
    df = df.dropna(subset=["next_return"])

    sub_results: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    baseline = df["next_return"].values

    for n in [3, 4, 5, 6, 7]:
        # After N consecutive up days
        mask_up = df["consec_up"] >= n
        ret_after_up = df.loc[mask_up, "next_return"].values
        if len(ret_after_up) >= 3:
            _, p = stats.mannwhitneyu(
                ret_after_up,
                df.loc[~mask_up, "next_return"].values,
                alternative="two-sided",
            )
            d = _cohens_d(ret_after_up, df.loc[~mask_up, "next_return"].values)
            ci_lo, ci_hi = _bootstrap_ci(ret_after_up)
        else:
            p = 1.0
            d = 0.0
            ci_lo = ci_hi = 0.0

        sub_results.append({
            "pattern": f"After {n}+ consecutive up days",
            "mean_next_return": round(float(np.mean(ret_after_up)), 6) if len(ret_after_up) > 0 else 0.0,
            "n": len(ret_after_up),
            "p_value": round(p, 6),
            "cohens_d": round(d, 4),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
        })
        raw_p_values.append(p)

        # After N consecutive down days
        mask_down = df["consec_down"] >= n
        ret_after_down = df.loc[mask_down, "next_return"].values
        if len(ret_after_down) >= 3:
            _, p = stats.mannwhitneyu(
                ret_after_down,
                df.loc[~mask_down, "next_return"].values,
                alternative="two-sided",
            )
            d = _cohens_d(ret_after_down, df.loc[~mask_down, "next_return"].values)
            ci_lo, ci_hi = _bootstrap_ci(ret_after_down)
        else:
            p = 1.0
            d = 0.0
            ci_lo = ci_hi = 0.0

        sub_results.append({
            "pattern": f"After {n}+ consecutive down days",
            "mean_next_return": round(float(np.mean(ret_after_down)), 6) if len(ret_after_down) > 0 else 0.0,
            "n": len(ret_after_down),
            "p_value": round(p, 6),
            "cohens_d": round(d, 4),
            "ci_95_lower": round(ci_lo, 6),
            "ci_95_upper": round(ci_hi, 6),
        })
        raw_p_values.append(p)

    # Bonferroni correction for all 10 tests
    corrected = _bonferroni(raw_p_values)
    for i, sr_item in enumerate(sub_results):
        sr_item["p_value_corrected"] = round(corrected[i], 6)

    best_p = min(raw_p_values)
    best_idx = raw_p_values.index(best_p)
    best_corrected = corrected[best_idx]
    significant = best_corrected < 0.05

    test_period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"

    # 最も有意なパターンのeffect_size
    best_d = sub_results[best_idx].get("cohens_d", 0.0)
    best_n = sub_results[best_idx].get("n", 0)
    power = _compute_power_ttest(best_d, best_n)

    return AnomalyResult(
        edge_name="Consecutive Candles Effect",
        timeframe="1d",
        description="Next-day return after N consecutive up/down days (N=3..7)",
        test_statistic=0.0,
        p_value=best_p,
        p_value_corrected=best_corrected,
        effect_size=best_d,
        effect_size_label=_effect_size_label(best_d),
        sample_size=len(df),
        ci_lower=sub_results[best_idx].get("ci_95_lower", 0.0),
        ci_upper=sub_results[best_idx].get("ci_95_upper", 0.0),
        power=power,
        significant=significant,
        win_rate=0.0,
        expected_return=sub_results[best_idx].get("mean_next_return", 0.0),
        sharpe_ratio=0.0,
        max_drawdown=0.0,
        test_period=test_period,
        notes=f"Most significant: {sub_results[best_idx]['pattern']}",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# アノマリー検定: 出来高アノマリー
# ---------------------------------------------------------------------------

def test_volume_anomaly(df: pd.DataFrame) -> AnomalyResult:
    """出来高急増日（>2σ）の翌日・翌週リターンを検定する。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。

    Returns:
        AnomalyResult。
    """
    logger.info("Testing volume anomaly...")
    df = df.copy()
    vol_mean = df["volume"].mean()
    vol_std = df["volume"].std()
    threshold = vol_mean + 2 * vol_std

    df["high_volume"] = df["volume"] > threshold
    df["next_return"] = df["return"].shift(-1)
    df["next_week_return"] = df["close"].pct_change(5).shift(-5)
    df = df.dropna(subset=["next_return"])

    hv = df.loc[df["high_volume"]]
    nv = df.loc[~df["high_volume"]]

    if len(hv) < 5:
        return AnomalyResult(edge_name="Volume Anomaly", notes="Too few high-volume days")

    # Next-day return
    hv_next = hv["next_return"].values
    nv_next = nv["next_return"].values
    stat_1d, p_1d = stats.mannwhitneyu(hv_next, nv_next, alternative="two-sided")
    d_1d = _cohens_d(hv_next, nv_next)
    ci_1d_lo, ci_1d_hi = _bootstrap_ci(hv_next)

    # Next-week return
    hv_week = hv["next_week_return"].dropna().values
    nv_week = nv["next_week_return"].dropna().values
    if len(hv_week) >= 3 and len(nv_week) >= 3:
        stat_1w, p_1w = stats.mannwhitneyu(hv_week, nv_week, alternative="two-sided")
        d_1w = _cohens_d(hv_week, nv_week)
        ci_1w_lo, ci_1w_hi = _bootstrap_ci(hv_week)
    else:
        stat_1w, p_1w = 0.0, 1.0
        d_1w = 0.0
        ci_1w_lo = ci_1w_hi = 0.0

    corrected = _bonferroni([p_1d, p_1w])

    sub_results: list[dict[str, Any]] = [
        {
            "horizon": "Next day",
            "high_vol_mean_return": round(float(np.mean(hv_next)), 6),
            "normal_vol_mean_return": round(float(np.mean(nv_next)), 6),
            "n_high_vol": len(hv_next),
            "p_value": round(p_1d, 6),
            "p_value_corrected": round(corrected[0], 6),
            "cohens_d": round(d_1d, 4),
            "ci_95_lower": round(ci_1d_lo, 6),
            "ci_95_upper": round(ci_1d_hi, 6),
        },
        {
            "horizon": "Next week (5 days)",
            "high_vol_mean_return": round(float(np.mean(hv_week)), 6) if len(hv_week) > 0 else 0.0,
            "normal_vol_mean_return": round(float(np.mean(nv_week)), 6) if len(nv_week) > 0 else 0.0,
            "n_high_vol": len(hv_week),
            "p_value": round(p_1w, 6),
            "p_value_corrected": round(corrected[1], 6),
            "cohens_d": round(d_1w, 4),
            "ci_95_lower": round(ci_1w_lo, 6),
            "ci_95_upper": round(ci_1w_hi, 6),
        },
    ]

    # Also check direction: high volume + up day vs high volume + down day
    hv_up = df.loc[df["high_volume"] & (df["return"] > 0), "next_return"].values
    hv_down = df.loc[df["high_volume"] & (df["return"] < 0), "next_return"].values
    if len(hv_up) >= 3 and len(hv_down) >= 3:
        _, p_dir = stats.mannwhitneyu(hv_up, hv_down, alternative="two-sided")
        d_dir = _cohens_d(hv_up, hv_down)
        sub_results.append({
            "comparison": "High-vol up-day vs High-vol down-day (next return)",
            "hv_up_mean": round(float(np.mean(hv_up)), 6),
            "hv_down_mean": round(float(np.mean(hv_down)), 6),
            "n_up": len(hv_up),
            "n_down": len(hv_down),
            "p_value": round(p_dir, 6),
            "cohens_d": round(d_dir, 4),
        })

    best_p = min(p_1d, p_1w)
    best_corrected = min(corrected)
    significant = best_corrected < 0.05

    test_period = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    min_n = min(len(hv_next), len(nv_next))
    power = _compute_power_ttest(d_1d, min_n)

    wr, er, sr, mdd = _compute_edge_metrics(hv_next)

    return AnomalyResult(
        edge_name="Volume Anomaly",
        timeframe="1d",
        description=f"Next-day and next-week returns after volume spikes (>2σ, threshold={threshold:.0f})",
        test_statistic=float(stat_1d),
        p_value=best_p,
        p_value_corrected=best_corrected,
        effect_size=d_1d,
        effect_size_label=_effect_size_label(d_1d),
        sample_size=len(hv),
        ci_lower=ci_1d_lo,
        ci_upper=ci_1d_hi,
        power=power,
        significant=significant,
        win_rate=wr,
        expected_return=er,
        sharpe_ratio=sr,
        max_drawdown=mdd,
        test_period=test_period,
        notes=f"High-volume days: {len(hv)} ({len(hv)/len(df)*100:.1f}%)",
        sub_results=sub_results,
    )


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------

def plot_day_of_week(df: pd.DataFrame, result: AnomalyResult) -> plt.Figure:
    """曜日別リターン分布の箱ひげ図を作成する。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。
        result: 曜日効果の検定結果。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df = df.copy()
    df["dow"] = df["timestamp"].dt.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Box plot
    box_data = [df.loc[df["dow"] == i, "return"].values for i in range(7)]
    bp = axes[0].boxplot(box_data, tick_labels=day_names, patch_artist=True, showmeans=True)
    colors = plt.cm.Set2(np.linspace(0, 1, 7))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title("Daily Return Distribution by Weekday")
    axes[0].set_ylabel("Return")
    axes[0].axhline(y=0, color="white", linestyle="--", alpha=0.5)

    # Mean return bar chart
    means = [float(np.mean(g)) for g in box_data]
    bar_colors = ["#4CAF50" if m > 0 else "#F44336" for m in means]
    axes[1].bar(day_names, means, color=bar_colors, alpha=0.8)
    axes[1].set_title(f"Mean Daily Return by Weekday (p={result.p_value:.4f})")
    axes[1].set_ylabel("Mean Return")
    axes[1].axhline(y=0, color="white", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Day-of-Week Effect | KW stat={result.test_statistic:.2f}, "
        f"p={result.p_value:.4f}, d={result.effect_size:.3f} ({result.effect_size_label})",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_month_of_year(df: pd.DataFrame, result: AnomalyResult) -> plt.Figure:
    """月別リターンの棒グラフを作成する。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。
        result: 月別効果の検定結果。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df = df.copy()
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly = df.groupby(["year", "month"])["return"].sum().reset_index()
    month_stats = monthly.groupby("month")["return"].agg(["mean", "std", "count"]).reset_index()

    focus = {1, 4, 10, 11, 12}
    colors = ["#FFD700" if m in focus else "#4FC3F7" for m in range(1, 13)]

    axes[0].bar(
        [month_names[m - 1] for m in month_stats["month"]],
        month_stats["mean"],
        yerr=month_stats["std"] / np.sqrt(month_stats["count"]),
        color=colors,
        alpha=0.8,
        capsize=3,
    )
    axes[0].set_title("Mean Monthly Return (±SE)")
    axes[0].set_ylabel("Mean Monthly Return")
    axes[0].axhline(y=0, color="white", linestyle="--", alpha=0.5)
    axes[0].tick_params(axis="x", rotation=45)

    # Box plot
    box_data = [monthly.loc[monthly["month"] == m, "return"].values for m in range(1, 13)]
    bp = axes[1].boxplot(
        [b for b in box_data if len(b) > 0],
        tick_labels=[month_names[i] for i in range(12) if len(box_data[i]) > 0],
        patch_artist=True,
        showmeans=True,
    )
    valid_colors = [colors[i] for i in range(12) if len(box_data[i]) > 0]
    for patch, color in zip(bp["boxes"], valid_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Monthly Return Distribution")
    axes[1].set_ylabel("Monthly Return")
    axes[1].axhline(y=0, color="white", linestyle="--", alpha=0.5)
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(
        f"Month-of-Year Effect | KW stat={result.test_statistic:.2f}, "
        f"p={result.p_value:.4f}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_session_effect(df_hourly: pd.DataFrame, result: AnomalyResult) -> plt.Figure:
    """セッション別リターンの可視化。

    Args:
        df_hourly: _prepare_hourly 済みの時間足 DataFrame。
        result: セッション効果の検定結果。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df = df_hourly.copy()
    df["hour"] = df["timestamp"].dt.hour

    def assign_session(hour: int) -> str:
        if 0 <= hour < 8:
            return "Asian"
        elif 8 <= hour < 16:
            return "European"
        else:
            return "US"

    df["session"] = df["hour"].apply(assign_session)
    df["date"] = df["timestamp"].dt.date
    session_daily = df.groupby(["date", "session"])["return"].sum().reset_index()

    session_colors = {"Asian": "#FF6B6B", "European": "#4ECDC4", "US": "#45B7D1"}
    sessions = ["Asian", "European", "US"]

    # Bar chart
    means = [session_daily.loc[session_daily["session"] == s, "return"].mean() for s in sessions]
    stes = [
        session_daily.loc[session_daily["session"] == s, "return"].std()
        / np.sqrt(len(session_daily.loc[session_daily["session"] == s]))
        for s in sessions
    ]
    axes[0].bar(
        sessions,
        means,
        yerr=stes,
        color=[session_colors[s] for s in sessions],
        alpha=0.8,
        capsize=5,
    )
    axes[0].set_title(f"Mean Session Return (±SE) | p={result.p_value:.4f}")
    axes[0].set_ylabel("Mean Return")
    axes[0].axhline(y=0, color="white", linestyle="--", alpha=0.5)

    # Hourly return heatmap-like bar
    hourly_means = df.groupby("hour")["return"].mean()
    bar_colors = ["#FF6B6B"] * 8 + ["#4ECDC4"] * 8 + ["#45B7D1"] * 8
    axes[1].bar(hourly_means.index, hourly_means.values, color=bar_colors, alpha=0.8)
    axes[1].set_title("Mean Hourly Return by Hour (UTC)")
    axes[1].set_xlabel("Hour (UTC)")
    axes[1].set_ylabel("Mean Return")
    axes[1].axhline(y=0, color="white", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"Trading Session Effect | d={result.effect_size:.3f} ({result.effect_size_label})",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_volatility_clustering(df: pd.DataFrame, result: AnomalyResult) -> plt.Figure:
    """ボラティリティクラスタリング効果の可視化。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。
        result: ボラティリティクラスタリングの検定結果。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df = df.copy()
    abs_ret = df["return"].abs()
    threshold = abs_ret.mean() + 2 * abs_ret.std()

    df["big_move"] = abs_ret > threshold
    df["next_return"] = df["return"].shift(-1)
    df = df.dropna(subset=["next_return"])

    after_big = df.loc[df["big_move"], "next_return"].values
    after_normal = df.loc[~df["big_move"], "next_return"].values

    # Histogram comparison
    axes[0].hist(after_normal, bins=50, alpha=0.6, label=f"After normal (n={len(after_normal)})",
                 density=True, color="#4FC3F7")
    axes[0].hist(after_big, bins=30, alpha=0.7, label=f"After >2σ (n={len(after_big)})",
                 density=True, color="#FF6B6B")
    axes[0].set_title("Next-Day Return Distribution")
    axes[0].set_xlabel("Return")
    axes[0].legend()

    # Scatter: big move size vs next return
    big_move_df = df.loc[df["big_move"]]
    axes[1].scatter(
        big_move_df["return"], big_move_df["next_return"],
        alpha=0.5, c=np.where(big_move_df["return"] > 0, "#4CAF50", "#F44336"),
        s=30,
    )
    axes[1].axhline(y=0, color="white", linestyle="--", alpha=0.5)
    axes[1].axvline(x=0, color="white", linestyle="--", alpha=0.5)
    axes[1].set_title("Big Move vs Next-Day Return")
    axes[1].set_xlabel("Big Move Return")
    axes[1].set_ylabel("Next-Day Return")

    fig.suptitle(
        f"Volatility Clustering | p={result.p_value:.4f}, d={result.effect_size:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_consecutive_candles(result: AnomalyResult) -> plt.Figure:
    """連続陽線/陰線効果の可視化。

    Args:
        result: 連続陽線/陰線の検定結果。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))

    patterns = []
    means = []
    ns = []
    p_vals = []
    colors = []

    for sr_item in result.sub_results:
        if "pattern" in sr_item:
            patterns.append(sr_item["pattern"].replace("After ", "").replace(" consecutive ", "\nconsec "))
            means.append(sr_item["mean_next_return"])
            ns.append(sr_item["n"])
            p_vals.append(sr_item.get("p_value_corrected", 1.0))
            colors.append("#4CAF50" if "up" in sr_item["pattern"] else "#F44336")

    y_pos = range(len(patterns))
    bars = ax.barh(y_pos, means, color=colors, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(patterns, fontsize=9)
    ax.set_xlabel("Mean Next-Day Return")
    ax.set_title("Consecutive Candle Patterns: Next-Day Return")
    ax.axvline(x=0, color="white", linestyle="--", alpha=0.5)

    for i, (bar, n, p) in enumerate(zip(bars, ns, p_vals)):
        sig = "*" if p < 0.05 else ""
        ax.text(
            bar.get_width() + 0.0001 if bar.get_width() >= 0 else bar.get_width() - 0.0001,
            bar.get_y() + bar.get_height() / 2,
            f"n={n} p={p:.3f}{sig}",
            ha="left" if bar.get_width() >= 0 else "right",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def plot_volume_anomaly(df: pd.DataFrame, result: AnomalyResult) -> plt.Figure:
    """出来高アノマリーの可視化。

    Args:
        df: _prepare_daily 済みの日次 DataFrame。
        result: 出来高アノマリーの検定結果。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    df = df.copy()
    vol_mean = df["volume"].mean()
    vol_std = df["volume"].std()
    threshold = vol_mean + 2 * vol_std

    df["high_volume"] = df["volume"] > threshold
    df["next_return"] = df["return"].shift(-1)
    df = df.dropna(subset=["next_return"])

    hv = df.loc[df["high_volume"], "next_return"].values
    nv = df.loc[~df["high_volume"], "next_return"].values

    # Histogram
    axes[0].hist(nv, bins=50, alpha=0.6, density=True, label=f"Normal vol (n={len(nv)})",
                 color="#4FC3F7")
    axes[0].hist(hv, bins=20, alpha=0.7, density=True, label=f"High vol (n={len(hv)})",
                 color="#FFD700")
    axes[0].set_title("Next-Day Return: High vs Normal Volume")
    axes[0].set_xlabel("Return")
    axes[0].legend()

    # Volume time series with high vol markers
    axes[1].plot(df["timestamp"], df["volume"], alpha=0.3, color="#4FC3F7", linewidth=0.5)
    hv_mask = df["high_volume"]
    axes[1].scatter(
        df.loc[hv_mask, "timestamp"],
        df.loc[hv_mask, "volume"],
        color="#FFD700",
        s=15,
        alpha=0.8,
        label="High volume days",
    )
    axes[1].axhline(y=threshold, color="#FF6B6B", linestyle="--", alpha=0.5, label=f"2σ threshold")
    axes[1].set_title("Volume Timeline with Spike Detection")
    axes[1].set_ylabel("Volume")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Volume Anomaly | p={result.p_value:.4f}, d={result.effect_size:.3f}",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


def plot_summary_heatmap(results: list[AnomalyResult]) -> plt.Figure:
    """全アノマリーのサマリーヒートマップを作成する。

    Args:
        results: 全検定結果のリスト。

    Returns:
        matplotlib Figure。
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, max(4, len(results) * 0.8 + 1)))

    names = [r.edge_name for r in results]
    p_values = [r.p_value for r in results]
    effect_sizes = [abs(r.effect_size) for r in results]
    significants = [r.significant for r in results]

    y_pos = range(len(names))
    bar_colors = ["#4CAF50" if s else "#757575" for s in significants]

    bars = ax.barh(y_pos, [-np.log10(max(p, 1e-10)) for p in p_values], color=bar_colors, alpha=0.8)
    ax.axvline(x=-np.log10(0.05), color="#FF6B6B", linestyle="--", alpha=0.8, label="p=0.05")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names)
    ax.set_xlabel("-log10(p-value)")
    ax.set_title("Anomaly Detection Summary: Statistical Significance")
    ax.legend()

    for i, (bar, es, sig) in enumerate(zip(bars, effect_sizes, significants)):
        label = f"|d|={es:.3f}" + (" *" if sig else "")
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# メイン実行
# ---------------------------------------------------------------------------

def run_all_tests() -> list[AnomalyResult]:
    """全アノマリー検定を実行し、結果を保存する。

    Returns:
        全検定結果のリスト。
    """
    logger.info("=" * 60)
    logger.info("Starting anomaly detection analysis")
    logger.info("=" * 60)

    # データ読み込み
    logger.info("Loading daily data...")
    df_daily_raw = read_csv("processed", "btcusdt_1d.csv")
    df_daily = _prepare_daily(df_daily_raw)
    logger.info("Daily data: %d rows (%s to %s)",
                len(df_daily),
                df_daily["timestamp"].min().strftime("%Y-%m-%d"),
                df_daily["timestamp"].max().strftime("%Y-%m-%d"))

    logger.info("Loading hourly data...")
    df_hourly_raw = read_csv("processed", "btcusdt_1h.csv")
    df_hourly = _prepare_hourly(df_hourly_raw)
    logger.info("Hourly data: %d rows (%s to %s)",
                len(df_hourly),
                df_hourly["timestamp"].min().strftime("%Y-%m-%d"),
                df_hourly["timestamp"].max().strftime("%Y-%m-%d"))

    # 検定実行
    results: list[AnomalyResult] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        results.append(test_day_of_week(df_daily))
        results.append(test_month_of_year(df_daily))
        results.append(test_session_effect(df_hourly))
        results.append(test_turn_of_month(df_daily))
        results.append(test_volatility_clustering(df_daily))
        results.append(test_consecutive_candles(df_daily))
        results.append(test_volume_anomaly(df_daily))

    # 結果サマリー
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    significant_edges: list[str] = []
    for r in results:
        sig_marker = "*** SIGNIFICANT ***" if r.significant else ""
        logger.info("  %s: p=%.4f (corrected=%.4f), d=%.3f (%s) %s",
                     r.edge_name, r.p_value, r.p_value_corrected,
                     r.effect_size, r.effect_size_label, sig_marker)
        if r.significant:
            significant_edges.append(r.edge_name)

    if significant_edges:
        logger.info("Significant anomalies found: %s", ", ".join(significant_edges))
    else:
        logger.info("No statistically significant anomalies found after correction.")

    # JSON保存
    report_data = [asdict(r) for r in results]
    save_report_json(report_data, "anomaly", "anomaly_results.json")

    # 可視化
    logger.info("Generating visualizations...")
    save_figure(plot_day_of_week(df_daily, results[0]), "anomaly", "day_of_week.png")
    save_figure(plot_month_of_year(df_daily, results[1]), "anomaly", "month_of_year.png")
    save_figure(plot_session_effect(df_hourly, results[2]), "anomaly", "session_effect.png")
    save_figure(plot_volatility_clustering(df_daily, results[4]), "anomaly", "volatility_clustering.png")
    save_figure(plot_consecutive_candles(results[5]), "anomaly", "consecutive_candles.png")
    save_figure(plot_volume_anomaly(df_daily, results[6]), "anomaly", "volume_anomaly.png")
    save_figure(plot_summary_heatmap(results), "anomaly", "summary_heatmap.png")

    logger.info("Anomaly detection complete. Reports saved to output/reports/anomaly/")
    return results


if __name__ == "__main__":
    run_all_tests()
