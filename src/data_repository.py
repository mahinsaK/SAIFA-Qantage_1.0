"""Persistence for the aligned market-data snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CsvMarketDataRepository:
    """Read and write the deterministic aligned-price snapshot."""

    snapshot_path: Path

    def exists(self) -> bool:
        return self.snapshot_path.exists()

    def load(self) -> pd.DataFrame:
        if not self.exists():
            raise FileNotFoundError(self.snapshot_path)
        return pd.read_csv(self.snapshot_path, index_col="Date", parse_dates=True)

    def save(self, prices: pd.DataFrame) -> None:
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(self.snapshot_path, float_format="%.8f")

