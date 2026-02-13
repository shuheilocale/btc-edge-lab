"""ペーパートレードエンジン。

ポジション管理、P&L計算、状態永続化を担当する。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.utils import TRADE_DIR, setup_logger
from src.trader.signal_generator import Signal, EDGE_DEFAULTS

logger = setup_logger(__name__)

INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 3
COMMISSION_RATE = 0.0015  # 片道 0.15%（手数料0.1% + スリッページ0.05%）


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """オープンポジションを表すデータクラス。"""

    edge_name: str
    direction: str
    entry_price: float
    entry_date: str
    size_usd: float
    stop_loss: float
    hold_days: int
    days_held: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書に変換する。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Position:
        """辞書からインスタンスを生成する。"""
        return cls(**d)


@dataclass
class TradeRecord:
    """決済済みトレードの記録。"""

    edge_name: str
    direction: str
    entry_price: float
    entry_date: str
    exit_price: float
    exit_date: str
    size_usd: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_days: int


# ---------------------------------------------------------------------------
# PaperTradeEngine
# ---------------------------------------------------------------------------

class PaperTradeEngine:
    """ペーパートレードエンジン。

    ポジションのオープン/クローズ、エクイティ計算、状態永続化を管理する。

    Attributes:
        capital: 現在の現金残高。
        positions: オープンポジションのリスト。
        edge_configs: エッジ別設定。
        pdca_cycle: PDCAサイクル状態。
    """

    def __init__(self, state_path: Path | None = None) -> None:
        """エンジンを初期化する。

        Args:
            state_path: state.json のパス。None の場合はデフォルトパスを使用。
        """
        self.state_path = state_path or (TRADE_DIR / "state.json")
        self.trades_path = TRADE_DIR / "trades.csv"
        self.equity_path = TRADE_DIR / "equity_curve.csv"

        self.capital: float = INITIAL_CAPITAL
        self.positions: list[Position] = []
        self.edge_configs: dict[str, dict[str, Any]] = {}
        self.pdca_cycle: dict[str, Any] = {
            "cycle_number": 0,
            "cycle_start_date": "",
            "last_evaluation_date": "",
            "days_in_cycle": 0,
        }
        self.start_date: str = ""

        TRADE_DIR.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> bool:
        """state.json から状態を復元する。

        Returns:
            状態ファイルが存在して正常に読み込めた場合 True。
        """
        if not self.state_path.exists():
            logger.info("No existing state file, starting fresh")
            return False

        with open(self.state_path, encoding="utf-8") as f:
            state = json.load(f)

        self.capital = state.get("capital", INITIAL_CAPITAL)
        self.positions = [
            Position.from_dict(p) for p in state.get("positions", [])
        ]
        self.edge_configs = state.get("edge_configs", {})
        self.pdca_cycle = state.get("pdca_cycle", self.pdca_cycle)
        self.start_date = state.get("start_date", "")

        logger.info(
            "State loaded: capital=%.2f, positions=%d, cycle=%d",
            self.capital, len(self.positions),
            self.pdca_cycle.get("cycle_number", 0),
        )
        return True

    def save_state(self) -> None:
        """現在の状態を state.json に保存する。"""
        state = {
            "capital": self.capital,
            "positions": [p.to_dict() for p in self.positions],
            "edge_configs": self.edge_configs,
            "pdca_cycle": self.pdca_cycle,
            "start_date": self.start_date,
            "last_updated": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        logger.info("State saved to %s", self.state_path)

    def initialize(self, today: str) -> None:
        """初回の状態を初期化する。

        Args:
            today: 今日の日付文字列 (YYYY-MM-DD)。
        """
        self.capital = INITIAL_CAPITAL
        self.positions = []
        self.start_date = today

        for key, defaults in EDGE_DEFAULTS.items():
            self.edge_configs[key] = defaults.copy()

        self.pdca_cycle = {
            "cycle_number": 1,
            "cycle_start_date": today,
            "last_evaluation_date": "",
            "days_in_cycle": 0,
        }
        self.save_state()
        logger.info("Engine initialized: capital=%.2f, date=%s", self.capital, today)

    def get_equity(self, current_price: float) -> float:
        """現在のエクイティ（現金 + 未実現損益）を計算する。

        Args:
            current_price: 現在の BTC 価格。

        Returns:
            エクイティの合計額。
        """
        unrealized = 0.0
        for pos in self.positions:
            if pos.direction == "long":
                unrealized += pos.size_usd * (current_price / pos.entry_price - 1)
            elif pos.direction == "short":
                unrealized += pos.size_usd * (1 - current_price / pos.entry_price)
        return self.capital + unrealized

    def _has_edge_position(self, edge_name: str) -> bool:
        """同一エッジのポジションが既にオープンしているかを確認する。"""
        return any(p.edge_name == edge_name for p in self.positions)

    def process_day(
        self,
        today: str,
        current_price: float,
        signals: list[Signal],
    ) -> dict[str, Any]:
        """日次処理を実行する。

        1. オープンポジションの決済判定
        2. 新規エントリー
        3. エクイティ記録

        Args:
            today: 今日の日付文字列。
            current_price: 現在の BTC 終値。
            signals: 本日のシグナルリスト。

        Returns:
            日次サマリー辞書。
        """
        closed_trades: list[TradeRecord] = []
        new_entries: list[str] = []
        monitor_signals: list[str] = []

        # --- Step 1: 決済判定 ---
        positions_to_keep: list[Position] = []
        for pos in self.positions:
            pos.days_held += 1
            exit_reason = self._check_exit(pos, current_price)

            if exit_reason:
                trade = self._close_position(pos, current_price, today, exit_reason)
                closed_trades.append(trade)
                logger.info(
                    "CLOSE %s [%s] pnl=%.2f (%.2f%%) reason=%s",
                    trade.edge_name, trade.direction,
                    trade.pnl, trade.pnl_pct * 100, trade.exit_reason,
                )
            else:
                positions_to_keep.append(pos)

        self.positions = positions_to_keep

        # --- Step 2: 新規エントリー ---
        for sig in signals:
            if sig.direction == "monitor":
                monitor_signals.append(sig.edge_name)
                continue

            if self._has_edge_position(sig.edge_name):
                logger.info("Skip %s: duplicate edge position", sig.edge_name)
                continue

            if len(self.positions) >= MAX_POSITIONS:
                logger.info("Skip %s: max positions reached (%d)", sig.edge_name, MAX_POSITIONS)
                break

            edge_cfg = self.edge_configs.get(sig.edge_name, EDGE_DEFAULTS.get(sig.edge_name, {}))
            size_pct = edge_cfg.get("position_size_pct", 0.10)
            equity = self.get_equity(current_price)
            size_usd = equity * size_pct

            commission = size_usd * COMMISSION_RATE
            self.capital -= commission

            pos = Position(
                edge_name=sig.edge_name,
                direction=sig.direction,
                entry_price=sig.entry_price,
                entry_date=today,
                size_usd=size_usd,
                stop_loss=sig.stop_loss,
                hold_days=sig.hold_days,
                days_held=0,
                metadata=sig.metadata,
            )
            self.positions.append(pos)
            new_entries.append(sig.edge_name)
            logger.info(
                "OPEN %s [%s] price=%.2f size=$%.2f sl=%.2f hold=%dd",
                pos.edge_name, pos.direction, pos.entry_price,
                pos.size_usd, pos.stop_loss, pos.hold_days,
            )

        # --- Step 3: エクイティ記録 ---
        equity = self.get_equity(current_price)
        self._append_equity(today, equity, current_price)

        # --- Step 4: トレード記録 ---
        for trade in closed_trades:
            self._append_trade(trade)

        # --- Step 5: PDCA サイクル更新 ---
        self.pdca_cycle["days_in_cycle"] = self.pdca_cycle.get("days_in_cycle", 0) + 1

        # --- 状態保存 ---
        self.save_state()

        summary = {
            "date": today,
            "current_price": current_price,
            "equity": equity,
            "capital": self.capital,
            "open_positions": len(self.positions),
            "closed_trades": len(closed_trades),
            "new_entries": new_entries,
            "monitor_signals": monitor_signals,
            "closed_details": [
                {
                    "edge": t.edge_name,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "reason": t.exit_reason,
                }
                for t in closed_trades
            ],
            "positions": [
                {
                    "edge": p.edge_name,
                    "direction": p.direction,
                    "entry_price": p.entry_price,
                    "days_held": p.days_held,
                    "unrealized_pnl_pct": (
                        (current_price / p.entry_price - 1)
                        if p.direction == "long"
                        else (1 - current_price / p.entry_price)
                    ),
                }
                for p in self.positions
            ],
            "pdca_days_in_cycle": self.pdca_cycle["days_in_cycle"],
        }
        return summary

    def _check_exit(self, pos: Position, current_price: float) -> str | None:
        """ポジションの決済条件を判定する。

        Args:
            pos: オープンポジション。
            current_price: 現在価格。

        Returns:
            決済理由の文字列。決済不要の場合 None。
        """
        # 保有期間満了
        if pos.days_held >= pos.hold_days:
            return "hold_expired"

        # ストップロス
        if pos.direction == "long" and current_price <= pos.stop_loss:
            return "stop_loss"
        if pos.direction == "short" and current_price >= pos.stop_loss:
            return "stop_loss"

        return None

    def _close_position(
        self, pos: Position, exit_price: float, exit_date: str, reason: str,
    ) -> TradeRecord:
        """ポジションを決済し、P&Lを計算する。

        Args:
            pos: クローズするポジション。
            exit_price: 決済価格。
            exit_date: 決済日。
            reason: 決済理由。

        Returns:
            TradeRecord。
        """
        if pos.direction == "long":
            pnl_pct = exit_price / pos.entry_price - 1
        else:
            pnl_pct = 1 - exit_price / pos.entry_price

        # 決済手数料
        commission = pos.size_usd * COMMISSION_RATE
        pnl_gross = pos.size_usd * pnl_pct
        pnl_net = pnl_gross - commission

        self.capital += pos.size_usd + pnl_net

        return TradeRecord(
            edge_name=pos.edge_name,
            direction=pos.direction,
            entry_price=pos.entry_price,
            entry_date=pos.entry_date,
            exit_price=exit_price,
            exit_date=exit_date,
            size_usd=pos.size_usd,
            pnl=pnl_net,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            hold_days=pos.days_held,
        )

    def _append_trade(self, trade: TradeRecord) -> None:
        """トレード履歴CSVに追記する。"""
        row = {
            "edge_name": trade.edge_name,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "entry_date": trade.entry_date,
            "exit_price": trade.exit_price,
            "exit_date": trade.exit_date,
            "size_usd": round(trade.size_usd, 2),
            "pnl": round(trade.pnl, 2),
            "pnl_pct": round(trade.pnl_pct, 6),
            "exit_reason": trade.exit_reason,
            "hold_days": trade.hold_days,
        }
        df = pd.DataFrame([row])

        if self.trades_path.exists():
            df.to_csv(self.trades_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.trades_path, index=False)

    def _append_equity(self, date: str, equity: float, price: float) -> None:
        """エクイティカーブCSVに追記する。"""
        row = {
            "date": date,
            "equity": round(equity, 2),
            "btc_price": round(price, 2),
            "open_positions": len(self.positions),
        }
        df = pd.DataFrame([row])

        if self.equity_path.exists():
            df.to_csv(self.equity_path, mode="a", header=False, index=False)
        else:
            df.to_csv(self.equity_path, index=False)

    def is_evaluation_due(self, cycle_days: int = 30) -> bool:
        """PDCAサイクルの評価日かどうかを判定する。

        Args:
            cycle_days: サイクル日数（デフォルト30日）。

        Returns:
            評価日であれば True。
        """
        return self.pdca_cycle.get("days_in_cycle", 0) >= cycle_days
