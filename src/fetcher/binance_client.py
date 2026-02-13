"""Binance Kline API クライアント。

BTCUSDT のローソク足データを取得し、raw / processed CSV に保存する。
"""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from src.common.utils import (
    PROCESSED_DIR,
    RAW_DIR,
    ms_to_datetime,
    setup_logger,
    write_csv,
)

logger = setup_logger(__name__)

KLINE_ENDPOINT = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVALS = ["1d", "4h", "1h"]
LIMIT = 1000
REQUEST_SLEEP = 0.1  # レート制限対策（秒）
YEARS_BACK = 3

# Binance Kline レスポンスのカラムインデックス
RAW_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """指定期間のローソク足データをページネーションで全件取得する。

    Args:
        symbol: 取引ペア（例: "BTCUSDT"）。
        interval: 時間足（例: "1d", "4h", "1h"）。
        start_ms: 取得開始時刻（Unix ミリ秒）。
        end_ms: 取得終了時刻（Unix ミリ秒）。

    Returns:
        Binance Kline レスポンスのリスト。
    """
    all_klines: list[list] = []
    current_start = start_ms

    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        resp = requests.get(KLINE_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        all_klines.extend(data)
        logger.info(
            "Fetched %d klines for %s %s (total: %d)",
            len(data),
            symbol,
            interval,
            len(all_klines),
        )

        # 次のページの開始は最後のレコードの close_time + 1ms
        current_start = data[-1][6] + 1

        if len(data) < LIMIT:
            break

        time.sleep(REQUEST_SLEEP)

    return all_klines


def klines_to_raw_df(klines: list[list]) -> pd.DataFrame:
    """Kline レスポンスを raw DataFrame に変換する。

    Args:
        klines: Binance Kline レスポンスのリスト。

    Returns:
        raw DataFrame（全カラム保持）。
    """
    df = pd.DataFrame(klines, columns=RAW_COLUMNS)
    return df


def clean_klines(raw_df: pd.DataFrame) -> pd.DataFrame:
    """raw DataFrame をクリーニングして processed フォーマットに変換する。

    処理内容:
    - open_time を UTC datetime の timestamp 列に変換
    - OHLCV カラムを float に変換
    - 重複行の除去
    - 欠損行の前方補完
    - processed フォーマットのカラムのみ保持

    Args:
        raw_df: raw DataFrame。

    Returns:
        クリーニング済み DataFrame。
    """
    df = raw_df.copy()

    # timestamp 変換
    df["timestamp"] = df["open_time"].apply(
        lambda x: ms_to_datetime(int(x)).strftime("%Y-%m-%d %H:%M:%S")
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # float 変換
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # 重複除去
    n_before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        logger.warning("Removed %d duplicate rows", n_dupes)

    # 時系列ソート
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 欠損検出と前方補完
    n_missing = df[["open", "high", "low", "close", "volume"]].isna().sum().sum()
    if n_missing > 0:
        logger.warning("Found %d missing values, applying forward fill", n_missing)
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].ffill()

    # processed フォーマットのカラムのみ保持
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    return df


def fetch_and_save(interval: str) -> pd.DataFrame:
    """指定時間足のデータを取得し、raw / processed CSV に保存する。

    Args:
        interval: 時間足（例: "1d", "4h", "1h"）。

    Returns:
        クリーニング済み DataFrame。
    """
    now = datetime.now(tz=timezone.utc)
    start = now - timedelta(days=365 * YEARS_BACK)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(now.timestamp() * 1000)

    filename = f"btcusdt_{interval}.csv"
    logger.info("=== Fetching %s %s (%s to %s) ===", SYMBOL, interval, start.date(), now.date())

    # API からデータ取得
    klines = fetch_klines(SYMBOL, interval, start_ms, end_ms)
    logger.info("Total klines fetched: %d", len(klines))

    # raw 保存
    raw_df = klines_to_raw_df(klines)
    write_csv(raw_df, "raw", filename)

    # クリーニングして processed 保存
    processed_df = clean_klines(raw_df)
    write_csv(processed_df, "processed", filename)
    logger.info("Processed %s: %d rows, %s to %s",
                filename, len(processed_df),
                processed_df["timestamp"].min(),
                processed_df["timestamp"].max())

    return processed_df


def fetch_all() -> dict[str, pd.DataFrame]:
    """全時間足のデータを取得して保存する。

    Returns:
        時間足 → DataFrame の辞書。
    """
    results: dict[str, pd.DataFrame] = {}
    for interval in INTERVALS:
        results[interval] = fetch_and_save(interval)
    return results


if __name__ == "__main__":
    fetch_all()
