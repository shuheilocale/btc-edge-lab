"""シグナル生成モジュール。

バックテストで特定されたエッジに基づき、日次トレードシグナルを生成する。
各 check_* 関数は最新のローソク足データとオンチェーンデータを受け取り、
シグナルの有無と方向を返す。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.common.utils import setup_logger
from src.technical.indicators import compute_bollinger_bands

logger = setup_logger(__name__)


@dataclass
class Signal:
    """トレードシグナルを表すデータクラス。"""

    edge_name: str
    direction: str  # "long", "short", "monitor"
    confidence: float  # 0.0 - 1.0
    entry_price: float
    stop_loss: float
    hold_days: int
    metadata: dict[str, Any]


# ---------------------------------------------------------------------------
# エッジ設定デフォルト値
# ---------------------------------------------------------------------------

EDGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "bb_vol_addr": {
        "enabled": True,
        "direction": "long",
        "hold_days": 10,
        "stop_loss_pct": 0.05,
        "position_size_pct": 0.10,
        "expected_win_rate": 0.8182,
        "expected_return": 0.0787,
    },
    "address_surge": {
        "enabled": True,
        "direction": "long",
        "hold_days": 14,
        "stop_loss_pct": 0.07,
        "position_size_pct": 0.10,
        "expected_win_rate": 0.55,
        "expected_return": 0.03,
    },
    "bb_squeeze_abs": {
        "enabled": True,
        "direction": "monitor",
        "hold_days": 10,
        "stop_loss_pct": 0.0,
        "position_size_pct": 0.0,
        "expected_win_rate": 1.0,
        "expected_return": 0.052,
    },
    "bb_vol": {
        "enabled": True,
        "direction": "long",
        "hold_days": 5,
        "stop_loss_pct": 0.04,
        "position_size_pct": 0.10,
        "expected_win_rate": 0.60,
        "expected_return": 0.0175,
    },
}


# ---------------------------------------------------------------------------
# 共通指標計算
# ---------------------------------------------------------------------------

def _compute_bb_indicators(df: pd.DataFrame) -> dict[str, pd.Series]:
    """ボリンジャーバンド関連指標を計算する。

    Args:
        df: ローソク足 DataFrame (close, volume カラム必須)。

    Returns:
        upper, lower, bandwidth_pctl, squeeze, vol_ma20, high_vol の辞書。
    """
    upper, middle, lower, bandwidth = compute_bollinger_bands(df["close"])

    bw_pctl = bandwidth.rolling(50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False,
    )
    squeeze = bw_pctl <= 0.25

    vol_ma20 = df["volume"].rolling(20).mean()
    high_vol = df["volume"] > vol_ma20 * 1.5

    return {
        "upper": upper,
        "lower": lower,
        "bandwidth_pctl": bw_pctl,
        "squeeze": squeeze,
        "vol_ma20": vol_ma20,
        "high_vol": high_vol,
    }


def _compute_addr_zscore(
    df: pd.DataFrame, onchain_df: pd.DataFrame | None,
) -> pd.Series:
    """アクティブアドレスの z-score を計算する。

    Args:
        df: ローソク足 DataFrame。
        onchain_df: オンチェーン DataFrame (timestamp, n_unique_addresses)。

    Returns:
        z-score の Series（onchain_df が None の場合は NaN 埋め）。
    """
    if onchain_df is None or onchain_df.empty:
        return pd.Series(np.nan, index=df.index)

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

    addr = merged["n_unique_addresses"]
    addr_mean = addr.rolling(30, min_periods=20).mean()
    addr_std = addr.rolling(30, min_periods=20).std()
    zscore = (addr - addr_mean) / addr_std
    return zscore


# ---------------------------------------------------------------------------
# 個別エッジシグナルチェック
# ---------------------------------------------------------------------------

def check_bb_vol_addr(
    df: pd.DataFrame,
    bb: dict[str, pd.Series],
    addr_zscore: pd.Series,
    edge_config: dict[str, Any],
) -> Signal | None:
    """BB+Vol+Addr Breakout Up シグナルを判定する。

    条件: BB上限突破 + 出来高 > 20日平均 * 1.5 + 直近5日にアドレス急増(z>2)

    Args:
        df: ローソク足 DataFrame。
        bb: ボリンジャーバンド指標辞書。
        addr_zscore: アドレス z-score Series。
        edge_config: エッジ設定辞書。

    Returns:
        シグナル、または条件未達の場合 None。
    """
    if not edge_config.get("enabled", True):
        return None

    idx = len(df) - 1
    close = df["close"].iloc[idx]

    bb_break_up = close > bb["upper"].iloc[idx]
    high_vol = bb["high_vol"].iloc[idx]
    addr_surge_recent = addr_zscore.iloc[max(0, idx - 4):idx + 1].max() > 2

    if bb_break_up and high_vol and addr_surge_recent:
        sl_pct = edge_config.get("stop_loss_pct", 0.05)
        return Signal(
            edge_name="bb_vol_addr",
            direction="long",
            confidence=0.8,
            entry_price=close,
            stop_loss=close * (1 - sl_pct),
            hold_days=edge_config.get("hold_days", 10),
            metadata={
                "bb_upper": float(bb["upper"].iloc[idx]),
                "volume_ratio": float(df["volume"].iloc[idx] / bb["vol_ma20"].iloc[idx]),
                "addr_zscore_max_5d": float(addr_zscore.iloc[max(0, idx - 4):idx + 1].max()),
            },
        )
    return None


def check_address_surge(
    df: pd.DataFrame,
    addr_zscore: pd.Series,
    edge_config: dict[str, Any],
) -> Signal | None:
    """Address Surge シグナルを判定する。

    条件: アクティブアドレス z-score > 2

    Args:
        df: ローソク足 DataFrame。
        addr_zscore: アドレス z-score Series。
        edge_config: エッジ設定辞書。

    Returns:
        シグナル、または条件未達の場合 None。
    """
    if not edge_config.get("enabled", True):
        return None

    idx = len(df) - 1
    close = df["close"].iloc[idx]
    current_zscore = addr_zscore.iloc[idx]

    if pd.notna(current_zscore) and current_zscore > 2.0:
        sl_pct = edge_config.get("stop_loss_pct", 0.07)
        return Signal(
            edge_name="address_surge",
            direction="long",
            confidence=0.6,
            entry_price=close,
            stop_loss=close * (1 - sl_pct),
            hold_days=edge_config.get("hold_days", 14),
            metadata={
                "addr_zscore": float(current_zscore),
            },
        )
    return None


def check_bb_squeeze(
    df: pd.DataFrame,
    bb: dict[str, pd.Series],
    edge_config: dict[str, Any],
) -> Signal | None:
    """BB Squeeze Abs Return シグナルを判定する（監視モード）。

    条件: BB bandwidth パーセンタイル <= 25%
    方向不問のためポジションは取らず、ボラ拡大の予兆として記録する。

    Args:
        df: ローソク足 DataFrame。
        bb: ボリンジャーバンド指標辞書。
        edge_config: エッジ設定辞書。

    Returns:
        監視用シグナル、または条件未達の場合 None。
    """
    if not edge_config.get("enabled", True):
        return None

    idx = len(df) - 1
    close = df["close"].iloc[idx]
    is_squeeze = bb["squeeze"].iloc[idx]

    if is_squeeze:
        return Signal(
            edge_name="bb_squeeze_abs",
            direction="monitor",
            confidence=0.9,
            entry_price=close,
            stop_loss=0.0,
            hold_days=edge_config.get("hold_days", 10),
            metadata={
                "bandwidth_pctl": float(bb["bandwidth_pctl"].iloc[idx]),
                "bb_upper": float(bb["upper"].iloc[idx]),
                "bb_lower": float(bb["lower"].iloc[idx]),
            },
        )
    return None


def check_bb_vol(
    df: pd.DataFrame,
    bb: dict[str, pd.Series],
    edge_config: dict[str, Any],
) -> Signal | None:
    """BB+Vol Breakout Up シグナルを判定する。

    条件: BB上限突破 + 出来高 > 20日平均 * 1.5

    Args:
        df: ローソク足 DataFrame。
        bb: ボリンジャーバンド指標辞書。
        edge_config: エッジ設定辞書。

    Returns:
        シグナル、または条件未達の場合 None。
    """
    if not edge_config.get("enabled", True):
        return None

    idx = len(df) - 1
    close = df["close"].iloc[idx]

    bb_break_up = close > bb["upper"].iloc[idx]
    high_vol = bb["high_vol"].iloc[idx]

    if bb_break_up and high_vol:
        sl_pct = edge_config.get("stop_loss_pct", 0.04)
        return Signal(
            edge_name="bb_vol",
            direction="long",
            confidence=0.6,
            entry_price=close,
            stop_loss=close * (1 - sl_pct),
            hold_days=edge_config.get("hold_days", 5),
            metadata={
                "bb_upper": float(bb["upper"].iloc[idx]),
                "volume_ratio": float(df["volume"].iloc[idx] / bb["vol_ma20"].iloc[idx]),
            },
        )
    return None


# ---------------------------------------------------------------------------
# メインエントリポイント
# ---------------------------------------------------------------------------

def generate_all_signals(
    df: pd.DataFrame,
    onchain_df: pd.DataFrame | None = None,
    edge_configs: dict[str, dict[str, Any]] | None = None,
) -> list[Signal]:
    """全エッジに対してシグナル判定を実行する。

    Args:
        df: ローソク足 DataFrame（十分な履歴が必要: 最低50日）。
        onchain_df: オンチェーン DataFrame (timestamp, n_unique_addresses)。
        edge_configs: エッジ別の設定辞書。None の場合はデフォルト値を使用。

    Returns:
        検出されたシグナルのリスト。
    """
    if len(df) < 50:
        logger.warning("データ不足: %d行（最低50行必要）", len(df))
        return []

    configs = {}
    for key, defaults in EDGE_DEFAULTS.items():
        cfg = defaults.copy()
        if edge_configs and key in edge_configs:
            cfg.update(edge_configs[key])
        configs[key] = cfg

    bb = _compute_bb_indicators(df)
    addr_zscore = _compute_addr_zscore(df, onchain_df)

    signals: list[Signal] = []

    checkers = [
        lambda: check_bb_vol_addr(df, bb, addr_zscore, configs["bb_vol_addr"]),
        lambda: check_address_surge(df, addr_zscore, configs["address_surge"]),
        lambda: check_bb_squeeze(df, bb, configs["bb_squeeze_abs"]),
        lambda: check_bb_vol(df, bb, configs["bb_vol"]),
    ]

    for checker in checkers:
        sig = checker()
        if sig is not None:
            signals.append(sig)
            logger.info(
                "Signal: %s [%s] price=%.2f sl=%.2f hold=%dd",
                sig.edge_name, sig.direction, sig.entry_price,
                sig.stop_loss, sig.hold_days,
            )

    if not signals:
        logger.info("No signals detected today")

    return signals
