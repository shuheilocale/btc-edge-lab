"""blockchain.info API クライアント。

オンチェーンメトリクスを取得し、CSV形式で保存する。
"""

import time
from typing import Any

import pandas as pd
import requests

from src.common.utils import ONCHAIN_DIR, setup_logger, write_csv

logger = setup_logger(__name__)

BASE_URL = "https://api.blockchain.info/charts"

METRICS: list[str] = [
    "n-unique-addresses",
    "hash-rate",
    "miners-revenue",
    "n-transactions",
    "estimated-transaction-volume-usd",
    "market-cap",
    "trade-volume",
]


def fetch_metric(chart_name: str, timespan: str = "3years") -> pd.DataFrame:
    """blockchain.info APIから単一メトリクスを取得する。

    Args:
        chart_name: チャート名（例: "n-unique-addresses"）。
        timespan: 取得期間（例: "3years"）。

    Returns:
        timestamp, {metric_column} カラムを持つ DataFrame。

    Raises:
        requests.HTTPError: APIリクエストが失敗した場合。
    """
    url = f"{BASE_URL}/{chart_name}"
    params = {"timespan": timespan, "format": "json"}
    logger.info("Fetching %s (timespan=%s)", chart_name, timespan)

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data: dict[str, Any] = resp.json()
    values: list[dict[str, Any]] = data.get("values", [])

    if not values:
        logger.warning("No data returned for %s", chart_name)
        return pd.DataFrame(columns=["timestamp", chart_name])

    df = pd.DataFrame(values)
    df["timestamp"] = pd.to_datetime(df["x"], unit="s", utc=True)
    col_name = chart_name.replace("-", "_")
    df = df.rename(columns={"y": col_name})
    df = df[["timestamp", col_name]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Fetched %d rows for %s", len(df), chart_name)
    return df


def fetch_all_metrics(timespan: str = "3years") -> dict[str, pd.DataFrame]:
    """全オンチェーンメトリクスを取得する。

    Args:
        timespan: 取得期間。

    Returns:
        メトリクス名をキー、DataFrameを値とする辞書。
    """
    results: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        try:
            df = fetch_metric(metric, timespan)
            results[metric] = df
            col_name = metric.replace("-", "_")
            write_csv(df, "onchain", f"{col_name}.csv")
            time.sleep(1.0)  # Rate limit: ~1 req/sec
        except Exception:
            logger.exception("Failed to fetch %s", metric)
    return results


def load_onchain_data() -> dict[str, pd.DataFrame]:
    """保存済みのオンチェーンCSVをすべて読み込む。

    Returns:
        メトリクス名をキー、DataFrameを値とする辞書。
    """
    results: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        col_name = metric.replace("-", "_")
        path = ONCHAIN_DIR / f"{col_name}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            results[metric] = df
            logger.info("Loaded %s (%d rows)", col_name, len(df))
        else:
            logger.warning("File not found: %s", path)
    return results


if __name__ == "__main__":
    logger.info("Starting onchain data fetch")
    fetch_all_metrics(timespan="3years")
    logger.info("Onchain data fetch complete")
