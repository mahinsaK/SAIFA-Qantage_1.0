"""Public-data acquisition and return construction."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import AnalysisConfig

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _download_series(series: str, raw_dir: Path, force: bool = False) -> pd.Series:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{series}.csv"
    request = Request(FRED_URL.format(series=series), headers={"User-Agent": "Mozilla/5.0 QuantEdge/1.0"})
    last_error: Exception | None = None
    text = raw_path.read_text(encoding="utf-8") if raw_path.exists() and not force else ""
    if text:
        frame = pd.read_csv(StringIO(text))
        date_col = frame.columns[0]
        frame[date_col] = pd.to_datetime(frame[date_col])
        values = pd.to_numeric(frame[series], errors="coerce")
        return pd.Series(values.to_numpy(), index=frame[date_col], name=series).dropna()
    for attempt in range(3):
        try:
            with urlopen(request, timeout=180) as response:
                text = response.read().decode("utf-8")
            raw_path.write_text(text, encoding="utf-8")
            break
        except Exception as error:  # network failures are retried, then surfaced clearly
            last_error = error
            if raw_path.exists():
                text = raw_path.read_text(encoding="utf-8")
                break
            if attempt < 2:
                time.sleep(2**attempt)
    if not text:
        raise RuntimeError(f"Could not download FRED series {series}") from last_error
    frame = pd.read_csv(StringIO(text))
    date_col = frame.columns[0]
    frame[date_col] = pd.to_datetime(frame[date_col])
    values = pd.to_numeric(frame[series], errors="coerce")
    return pd.Series(values.to_numpy(), index=frame[date_col], name=series).dropna()


def load_prices(root: Path, config: AnalysisConfig, refresh: bool = False) -> pd.DataFrame:
    """Load the included snapshot or download the configured FRED series."""
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "market_prices.csv"
    if path.exists() and not refresh:
        prices = pd.read_csv(path, index_col="Date", parse_dates=True)
    else:
        columns = [
            _download_series(code, data_dir / "raw", force=refresh)
            for code in config.asset_codes
        ]
        prices = pd.concat(columns, axis=1).sort_index()
        prices = prices.loc[config.start_date:]
        # A common business-day calendar; short publication gaps are carried forward.
        prices = prices.ffill(limit=config.max_forward_fill_days).dropna()
        prices.index.name = "Date"
        prices.to_csv(path, float_format="%.8f")
    missing = set(config.asset_codes) - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required data columns: {sorted(missing)}")
    prices = (
        prices[list(config.asset_codes)]
        .sort_index()
        .ffill(limit=config.max_forward_fill_days)
        .dropna()
    )
    if len(prices) < config.minimum_observations:
        raise ValueError("Too few aligned observations for the planned analysis")
    if (prices <= 0).any().any():
        raise ValueError("All price/index levels must be positive")
    return prices


def make_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices).diff().replace([np.inf, -np.inf], np.nan).dropna()
    returns.index.name = "Date"
    return returns
