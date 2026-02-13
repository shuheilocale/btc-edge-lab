"""共通ユーティリティモジュール。

ロガー、timestamp変換、CSV/JSONヘルパー、matplotlibスタイル設定を提供する。
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# プロジェクトルートパス
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ONCHAIN_DIR = DATA_DIR / "onchain"
TRADE_DIR = DATA_DIR / "trade"
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """統一フォーマットのロガーをセットアップする。

    Args:
        name: ロガー名（通常はモジュールの __name__）。
        level: ログレベル。

    Returns:
        設定済みのロガーインスタンス。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ms_to_datetime(ms: int) -> datetime:
    """Unix ミリ秒を UTC datetime に変換する。

    Args:
        ms: Unixタイムスタンプ（ミリ秒）。

    Returns:
        UTC datetime オブジェクト。
    """
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def datetime_to_ms(dt: datetime) -> int:
    """datetime を Unix ミリ秒に変換する。

    Args:
        dt: datetime オブジェクト。

    Returns:
        Unixタイムスタンプ（ミリ秒）。
    """
    return int(dt.timestamp() * 1000)


def read_csv(subdir: str, filename: str) -> pd.DataFrame:
    """data/ 配下のCSVファイルを読み込む。

    Args:
        subdir: data/ 配下のサブディレクトリ名（"raw", "processed", "onchain"）。
        filename: CSVファイル名。

    Returns:
        読み込んだ DataFrame。
    """
    path = DATA_DIR / subdir / filename
    logger = setup_logger(__name__)
    logger.info("Reading CSV: %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def write_csv(df: pd.DataFrame, subdir: str, filename: str) -> Path:
    """data/ 配下にCSVファイルを書き出す。

    Args:
        df: 書き出す DataFrame。
        subdir: data/ 配下のサブディレクトリ名。
        filename: CSVファイル名。

    Returns:
        書き出したファイルのパス。
    """
    path = DATA_DIR / subdir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(__name__)
    logger.info("Writing CSV: %s (%d rows)", path, len(df))
    df.to_csv(path, index=False)
    return path


def save_report_json(data: dict[str, Any] | list[dict[str, Any]], category: str, filename: str) -> Path:
    """output/reports/ 配下にJSONレポートを保存する。

    Args:
        data: 保存するデータ（辞書またはリスト）。
        category: レポートカテゴリ（"technical", "anomaly", "onchain"）。
        filename: JSONファイル名。

    Returns:
        保存したファイルのパス。
    """
    path = REPORTS_DIR / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(__name__)
    logger.info("Saving report JSON: %s", path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path


def setup_plot_style() -> None:
    """matplotlib のプロットスタイルをダークテーマに統一設定する。"""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.figsize": (14, 7),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "lines.linewidth": 1.5,
            "font.size": 11,
        }
    )


def save_figure(fig: plt.Figure, category: str, filename: str) -> Path:
    """output/reports/ 配下にPNG画像を保存する。

    Args:
        fig: matplotlib Figure オブジェクト。
        category: レポートカテゴリ（"technical", "anomaly", "onchain"）。
        filename: PNGファイル名。

    Returns:
        保存したファイルのパス。
    """
    path = REPORTS_DIR / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(__name__)
    logger.info("Saving figure: %s", path)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
