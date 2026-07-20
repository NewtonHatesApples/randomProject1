import os
import shutil
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import logging

aggbook_url = "https://data.binance.vision/data/spot/daily/aggTrades"
pv_data_url = "https://data.binance.vision/data/spot/daily/klines"
valid_intervals = ["1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]


def download_crypto_aggbook(symbol: str, start: date, end: date, save_dir: str, tmp_dir: str = "tmp_data", suppress_info=True) -> None:
    os.makedirs(tmp_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    if suppress_info:
        logger.setLevel(logging.WARNING)

    # 1) Download and unzip daily files
    print("Started downloading data...")
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        zip_name = f"{symbol}-aggTrades-{ds}.zip"
        url = f"{aggbook_url}/{symbol}/{zip_name}"
        zip_path = os.path.join(tmp_dir, zip_name)

        logging.info("Fetching %s", url)
        r = requests.get(url)
        if r.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            os.remove(zip_path)  # keep only CSV
            logging.info("OK %s", ds)
        else:
            logging.critical("Missing/Error for %s on %s: %s", symbol, ds, r.status_code)
        d += timedelta(days=1)

    # 2) Concatenate all CSVs in time order
    #    Binance kline schema:
    #    0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume,
    #    6 close time, 7 quote asset volume, 8 number of trades,
    #    9 taker buy base volume, 10 taker buy quote volume, 11 ignore
    csv_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".csv") and f.startswith(f"{symbol}"))

    dfs = []
    for fname in csv_files:
        path = os.path.join(tmp_dir, fname)
        df = pd.read_csv(path, header=None)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSVs downloaded; check dates and URLs.")

    full = pd.concat(dfs, ignore_index=True)

    full = full.sort_values(0).drop_duplicates(subset=0)
    full = full.iloc[:, :-2]  # Drop last two column (useless)
    threshold = 1e14
    # Convert all time data in time column to ms
    full.iloc[:, 5] = np.where(full.iloc[:, 5] > threshold, full.iloc[:, 5] // 1000, full.iloc[:, 0])

    # 3) Save a single merged CSV
    os.makedirs(save_dir, exist_ok=True)
    out_file = f"{save_dir}/{symbol}-aggTrades-{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.csv"
    full.to_csv(out_file, index=False,
                header=["AggregateTradeID", "Price", "Quantity", "FirstTradeID", "LastTradeID", "TradeTimestamp[ms]"])
    print("Merged CSV written to:", out_file)

    # Clear all the temp .zip files
    root = Path(tmp_dir)
    for item in root.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    os.rmdir(tmp_dir)
    return None


def download_crypto_data(symbol: str, start: date, end: date, save_dir: str = "data", interval: str = "1m",
                         tmp_dir: str = "tmp_data", suppress_info=False) -> None:
    """
    Download crypto data from https://data.binance.vision/data/spot/daily/klines
    :param symbol: Crypto-pair symbol, e.g. ``"BTCUSDT"``. For symbols available, visit https://data.binance.vision/?prefix=data/spot/daily/klines/.
    :param start: Historical data start date, must be ``datetime.date`` objects, e.g. ``datetime.date(2019, 1, 1)``. Support up to ``max(date(2017, 8, 17), coin_release_date)``
    :param end: Historical data end date, must be ``datetime.date`` objects, e.g. ``datetime.date(2024, 1, 1)``
    :param save_dir: Relative path from current script location, where the csv will be saved, e.g. ``"data"``
    :param interval: Data granularity. Must be one of ``"1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"``
    :param tmp_dir: Relative path from current script location, where temporary zip files will be saved. You won't need this probably, e.g. ``"tmp_data"``
    :param suppress_info: If true, only error messages will be shown.
    :return: ``None``
    """
    os.makedirs(tmp_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger()
    if suppress_info:
        logger.setLevel(logging.WARNING)

    # Validate interval
    if interval not in valid_intervals:
        raise ValueError(f"Intervals values must be one of {valid_intervals}, received {interval}")

    # 1) Download and unzip daily files
    print("Started downloading data...")
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        zip_name = f"{symbol}-{interval}-{ds}.zip"
        url = f"{pv_data_url}/{symbol}/{interval}/{zip_name}"
        zip_path = os.path.join(tmp_dir, zip_name)

        logger.info("Fetching %s", url)
        r = requests.get(url)
        if r.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(r.content)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_dir)
            os.remove(zip_path)  # keep only CSV
            logger.info("OK %s", ds)
        else:
            logger.critical("Missing/Error for %s on %s: %s", symbol, ds, r.status_code)
        d += timedelta(days=1)

    # 2) Concatenate all CSVs in time order
    #    Binance kline schema:
    #    0 open time, 1 open, 2 high, 3 low, 4 close, 5 volume,
    #    6 close time, 7 quote asset volume, 8 number of trades,
    #    9 taker buy base volume, 10 taker buy quote volume, 11 ignore
    csv_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith(".csv") and f.startswith(f"{symbol}-{interval}-"))

    dfs = []
    for fname in csv_files:
        path = os.path.join(tmp_dir, fname)
        df = pd.read_csv(path, header=None)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No CSVs downloaded; check dates and URLs.")

    full = pd.concat(dfs, ignore_index=True)

    full = full.sort_values(0).drop_duplicates(subset=0)
    full = full.iloc[:, :-1]  # Drop last column (useless)
    threshold = 1e14
    # Convert all time data in time column to ms
    full.iloc[:, 0] = np.where(full.iloc[:, 0] > threshold, full.iloc[:, 0] // 1000, full.iloc[:, 0])
    full.iloc[:, 6] = np.where(full.iloc[:, 6] > threshold, full.iloc[:, 6] // 1000, full.iloc[:, 6])

    # 3) Save a single merged CSV
    os.makedirs(save_dir, exist_ok=True)
    out_file = f"{save_dir}/{symbol}-{interval}-{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.csv"
    full.to_csv(out_file, index=False,
                header=["OpenTimestamp[ms]", "Open", "High", "Low", "Close", "Volume", "CloseTimestamp[ms]",
                        "QuoteAssetVolume", "NumberOfTrades", "TakerBuyBaseVolume", "TakerBuyQuoteVolume"])
    print(f"Merged CSV written to: {out_file}")

    # Clear all the temp .zip files
    root = Path(tmp_dir)
    for item in root.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    os.rmdir(tmp_dir)

    return None


if __name__ == "__main__":
    download_crypto_data(symbol="ETHUSDT", start=date(2025, 1, 1), end=date(2026, 6, 30), interval="5m", suppress_info=False, tmp_dir="tmp_data2")
