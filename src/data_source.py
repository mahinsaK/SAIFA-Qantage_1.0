"""Market-data acquisition interfaces and the FRED implementation."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
import time
from typing import Protocol
from urllib.request import Request, urlopen

import pandas as pd


class MarketDataProvider(Protocol):
    """Acquire one named market series from an external or local source."""

    def get_series(self, series: str, refresh: bool = False) -> pd.Series: ...


@dataclass
class FredDataProvider:
    """Download individual FRED series with a raw-response cache."""

    raw_dir: Path
    url_template: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    timeout_seconds: int = 180
    attempts: int = 3

    def get_series(self, series: str, refresh: bool = False) -> pd.Series:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.raw_dir / f"{series}.csv"
        if raw_path.exists() and not refresh:
            return self._parse_series(raw_path.read_text(encoding="utf-8"), series)

        request = Request(
            self.url_template.format(series=series),
            headers={"User-Agent": "Mozilla/5.0 QuantEdge/1.0"},
        )
        last_error: Exception | None = None
        for attempt in range(self.attempts):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    text = response.read().decode("utf-8")
                raw_path.write_text(text, encoding="utf-8")
                return self._parse_series(text, series)
            except Exception as error:  # network failures are retried, then surfaced
                last_error = error
                if raw_path.exists():
                    return self._parse_series(
                        raw_path.read_text(encoding="utf-8"),
                        series,
                    )
                if attempt < self.attempts - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Could not download FRED series {series}") from last_error

    @staticmethod
    def _parse_series(text: str, series: str) -> pd.Series:
        frame = pd.read_csv(StringIO(text))
        if series not in frame.columns:
            raise ValueError(f"FRED response does not contain series {series}")
        date_column = frame.columns[0]
        dates = pd.to_datetime(frame[date_column])
        values = pd.to_numeric(frame[series], errors="coerce")
        return pd.Series(values.to_numpy(), index=dates, name=series).dropna()

