"""パフォーマンス評価 + 改善提案モジュール。

30日サイクルのPDCA評価を実行し、エッジの劣化検出と改善提案を生成する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.common.utils import TRADE_DIR, setup_logger
from src.trader.signal_generator import EDGE_DEFAULTS

logger = setup_logger(__name__)

PDCA_LOG_PATH = TRADE_DIR / "pdca_log.json"


@dataclass
class Improvement:
    """改善提案を表すデータクラス。"""

    edge_name: str
    condition: str
    proposal: str
    auto_apply: bool
    confidence: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return asdict(self)


class PerformanceEvaluator:
    """30日サイクルのパフォーマンス評価を実行する。

    Attributes:
        trades_df: トレード履歴 DataFrame。
        equity_df: エクイティカーブ DataFrame。
        edge_configs: エッジ別設定辞書。
    """

    def __init__(
        self,
        trades_path: str | None = None,
        equity_path: str | None = None,
    ) -> None:
        """評価器を初期化する。

        Args:
            trades_path: trades.csv のパス。
            equity_path: equity_curve.csv のパス。
        """
        tp = trades_path or str(TRADE_DIR / "trades.csv")
        ep = equity_path or str(TRADE_DIR / "equity_curve.csv")

        if pd.io.common.file_exists(tp):
            self.trades_df = pd.read_csv(tp)
        else:
            self.trades_df = pd.DataFrame()

        if pd.io.common.file_exists(ep):
            self.equity_df = pd.read_csv(ep)
        else:
            self.equity_df = pd.DataFrame()

    def evaluate(
        self,
        edge_configs: dict[str, dict[str, Any]],
        cycle_days: int = 30,
    ) -> dict[str, Any]:
        """30日サイクルのパフォーマンス評価を実行する。

        Args:
            edge_configs: エッジ別設定辞書。
            cycle_days: 評価対象の直近日数。

        Returns:
            評価結果辞書。
        """
        result: dict[str, Any] = {
            "evaluation_date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "cycle_days": cycle_days,
        }

        # --- 全体パフォーマンス ---
        result["overall"] = self._compute_overall_stats(cycle_days)

        # --- エッジ別パフォーマンス ---
        result["by_edge"] = self._compute_edge_stats(cycle_days)

        # --- Buy & Hold 比較 ---
        result["buy_hold_comparison"] = self._compute_buy_hold(cycle_days)

        # --- 改善提案 ---
        improvements = self._generate_improvements(
            result["overall"], result["by_edge"], edge_configs,
        )
        result["improvements"] = [imp.to_dict() for imp in improvements]

        # --- PDCA ログ追記 ---
        self._append_pdca_log(result)

        return result

    def _compute_overall_stats(self, cycle_days: int) -> dict[str, Any]:
        """全体のパフォーマンス指標を算出する。

        Args:
            cycle_days: 評価対象日数。

        Returns:
            全体統計辞書。
        """
        if self.trades_df.empty:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        recent = self._filter_recent_trades(cycle_days)
        if recent.empty:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

        wins = (recent["pnl"] > 0).sum()
        total = len(recent)
        win_rate = wins / total if total > 0 else 0.0

        total_pnl = recent["pnl"].sum()
        avg_pnl_pct = recent["pnl_pct"].mean()
        std_pnl_pct = recent["pnl_pct"].std()
        sharpe = avg_pnl_pct / std_pnl_pct * np.sqrt(252) if std_pnl_pct > 0 else 0.0

        # 最大連敗数
        streak = 0
        max_streak = 0
        for _, row in recent.iterrows():
            if row["pnl"] <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        # エクイティベースのドローダウン
        max_dd = self._compute_max_drawdown(cycle_days)

        return {
            "total_trades": total,
            "wins": int(wins),
            "losses": total - int(wins),
            "win_rate": round(float(win_rate), 4),
            "total_pnl": round(float(total_pnl), 2),
            "avg_pnl_pct": round(float(avg_pnl_pct), 6),
            "sharpe_ratio": round(float(sharpe), 4),
            "max_drawdown": round(float(max_dd), 4),
            "max_losing_streak": max_streak,
        }

    def _compute_edge_stats(self, cycle_days: int) -> dict[str, dict[str, Any]]:
        """エッジ別のパフォーマンス指標を算出する。

        Args:
            cycle_days: 評価対象日数。

        Returns:
            エッジ名 → 統計辞書。
        """
        if self.trades_df.empty:
            return {}

        recent = self._filter_recent_trades(cycle_days)
        if recent.empty:
            return {}

        result: dict[str, dict[str, Any]] = {}
        for edge_name, group in recent.groupby("edge_name"):
            wins = (group["pnl"] > 0).sum()
            total = len(group)
            win_rate = wins / total if total > 0 else 0.0

            # 連敗数
            streak = 0
            max_streak = 0
            for _, row in group.iterrows():
                if row["pnl"] <= 0:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0

            result[str(edge_name)] = {
                "total_trades": total,
                "wins": int(wins),
                "win_rate": round(float(win_rate), 4),
                "total_pnl": round(float(group["pnl"].sum()), 2),
                "avg_pnl_pct": round(float(group["pnl_pct"].mean()), 6),
                "max_losing_streak": max_streak,
            }

        return result

    def _compute_buy_hold(self, cycle_days: int) -> dict[str, Any]:
        """Buy & Hold との比較を算出する。

        Args:
            cycle_days: 評価対象日数。

        Returns:
            比較結果辞書。
        """
        if self.equity_df.empty or len(self.equity_df) < 2:
            return {"error": "insufficient_equity_data"}

        recent_eq = self.equity_df.tail(cycle_days)
        if len(recent_eq) < 2:
            return {"error": "insufficient_equity_data"}

        strategy_return = recent_eq["equity"].iloc[-1] / recent_eq["equity"].iloc[0] - 1
        buy_hold_return = recent_eq["btc_price"].iloc[-1] / recent_eq["btc_price"].iloc[0] - 1

        return {
            "strategy_return": round(float(strategy_return), 6),
            "buy_hold_return": round(float(buy_hold_return), 6),
            "excess_return": round(float(strategy_return - buy_hold_return), 6),
            "period_days": len(recent_eq),
        }

    def _compute_max_drawdown(self, cycle_days: int) -> float:
        """エクイティカーブから最大ドローダウンを算出する。

        Args:
            cycle_days: 評価対象日数。

        Returns:
            最大ドローダウン（負の値）。
        """
        if self.equity_df.empty:
            return 0.0

        recent = self.equity_df.tail(cycle_days)
        if len(recent) < 2:
            return 0.0

        equity = recent["equity"].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        return float(drawdowns.min())

    def _filter_recent_trades(self, cycle_days: int) -> pd.DataFrame:
        """直近N日のトレードをフィルタする。

        Args:
            cycle_days: 日数。

        Returns:
            フィルタ済み DataFrame。
        """
        if self.trades_df.empty:
            return pd.DataFrame()

        df = self.trades_df.copy()
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=cycle_days)
        # exit_date がタイムゾーン情報を持たない場合に対応
        if df["exit_date"].dt.tz is None:
            df["exit_date"] = df["exit_date"].dt.tz_localize("UTC")
        return df[df["exit_date"] >= cutoff]

    def _generate_improvements(
        self,
        overall: dict[str, Any],
        by_edge: dict[str, dict[str, Any]],
        edge_configs: dict[str, dict[str, Any]],
    ) -> list[Improvement]:
        """パフォーマンスに基づく改善提案を生成する。

        Args:
            overall: 全体統計。
            by_edge: エッジ別統計。
            edge_configs: エッジ別設定。

        Returns:
            改善提案のリスト。
        """
        improvements: list[Improvement] = []

        # --- 1. 勝率がバックテスト期待値から2σ以上低下 ---
        for edge_name, stats in by_edge.items():
            expected_wr = edge_configs.get(edge_name, {}).get(
                "expected_win_rate",
                EDGE_DEFAULTS.get(edge_name, {}).get("expected_win_rate", 0.5),
            )
            actual_wr = stats["win_rate"]
            n = stats["total_trades"]

            if n >= 5:
                std_wr = np.sqrt(expected_wr * (1 - expected_wr) / n)
                if std_wr > 0 and actual_wr < expected_wr - 2 * std_wr:
                    confidence = min(1.0, (expected_wr - actual_wr) / (2 * std_wr))
                    improvements.append(Improvement(
                        edge_name=edge_name,
                        condition=f"win_rate_degraded (expected={expected_wr:.2f}, actual={actual_wr:.2f}, n={n})",
                        proposal="パラメータ調整を推奨",
                        auto_apply=confidence >= 0.8,
                        confidence=round(confidence, 2),
                        details={
                            "expected_win_rate": expected_wr,
                            "actual_win_rate": actual_wr,
                            "std_win_rate": round(std_wr, 4),
                            "z_score": round((expected_wr - actual_wr) / std_wr, 2) if std_wr > 0 else 0,
                        },
                    ))

        # --- 2. 5連敗以上 → エッジ無効化 ---
        for edge_name, stats in by_edge.items():
            if stats["max_losing_streak"] >= 5:
                improvements.append(Improvement(
                    edge_name=edge_name,
                    condition=f"losing_streak={stats['max_losing_streak']}",
                    proposal="エッジ無効化（5連敗到達）",
                    auto_apply=True,
                    confidence=0.9,
                    details={"max_losing_streak": stats["max_losing_streak"]},
                ))

        # --- 3. DD > 10% → 全体ポジション縮小 ---
        max_dd = overall.get("max_drawdown", 0.0)
        if max_dd < -0.10:
            improvements.append(Improvement(
                edge_name="__global__",
                condition=f"drawdown={max_dd:.2%}",
                proposal="全エッジのposition_size_pctを50%に縮小",
                auto_apply=False,
                confidence=0.7,
                details={"max_drawdown": max_dd},
            ))

        # --- 4. 市場レジーム変化の簡易検出 ---
        if self.equity_df is not None and len(self.equity_df) >= 30:
            recent_prices = self.equity_df["btc_price"].tail(30).values
            if len(recent_prices) >= 20:
                vol_recent = np.std(np.diff(np.log(recent_prices[-10:])))
                vol_prev = np.std(np.diff(np.log(recent_prices[:10])))
                if vol_prev > 0 and vol_recent / vol_prev > 2.0:
                    improvements.append(Improvement(
                        edge_name="__global__",
                        condition=f"regime_change (vol_ratio={vol_recent / vol_prev:.2f})",
                        proposal="ボラティリティ急変 - レジーム適合エッジの探索を推奨",
                        auto_apply=False,
                        confidence=0.5,
                        details={
                            "recent_vol": round(float(vol_recent), 6),
                            "prev_vol": round(float(vol_prev), 6),
                            "vol_ratio": round(float(vol_recent / vol_prev), 2),
                        },
                    ))

        return improvements

    def apply_auto_improvements(
        self,
        improvements: list[dict[str, Any]],
        edge_configs: dict[str, dict[str, Any]],
    ) -> list[str]:
        """auto_apply=True の改善を適用する。

        Args:
            improvements: 改善提案リスト。
            edge_configs: エッジ別設定（変更が in-place で適用される）。

        Returns:
            適用された改善の説明リスト。
        """
        applied: list[str] = []

        for imp in improvements:
            if not imp.get("auto_apply", False):
                continue

            edge = imp["edge_name"]
            condition = imp["condition"]

            if "losing_streak" in condition and edge in edge_configs:
                edge_configs[edge]["enabled"] = False
                applied.append(f"{edge}: disabled (5+ losing streak)")

            if "win_rate_degraded" in condition and imp.get("confidence", 0) >= 0.8:
                if edge in edge_configs:
                    current_size = edge_configs[edge].get("position_size_pct", 0.10)
                    edge_configs[edge]["position_size_pct"] = current_size * 0.5
                    applied.append(
                        f"{edge}: position_size reduced to {current_size * 0.5:.2%} "
                        f"(win rate degraded)"
                    )

        return applied

    def _append_pdca_log(self, evaluation: dict[str, Any]) -> None:
        """PDCA ログファイルに評価結果を追記する。

        Args:
            evaluation: 評価結果辞書。
        """
        TRADE_DIR.mkdir(parents=True, exist_ok=True)
        log: list[dict[str, Any]] = []

        if PDCA_LOG_PATH.exists():
            with open(PDCA_LOG_PATH, encoding="utf-8") as f:
                log = json.load(f)

        log.append(evaluation)

        with open(PDCA_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2, default=str)

        logger.info("PDCA log updated (%d entries)", len(log))
