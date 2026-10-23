"""Pure transformations from acquired market levels to model-ready returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.config import AnalysisConfig


@dataclass(frozen=True)
class MarketDataPreprocessor:
    """Align, clean, order, and validate configured market series."""

    config: AnalysisConfig

    def combine(self, series: list[pd.Series]) -> pd.DataFrame:
        if not series:
            raise ValueError("At least one market series is required")
        return self.prepare(pd.concat(series, axis=1))

    def prepare(self, prices: pd.DataFrame) -> pd.DataFrame:
        missing = set(self.config.asset_codes) - set(prices.columns)
        if missing:
            raise ValueError(f"Missing required data columns: {sorted(missing)}")
        prepared = (
            prices[list(self.config.asset_codes)]
            .sort_index()
            .loc[self.config.start_date :]
            .ffill(limit=self.config.max_forward_fill_days)
            .dropna()
        )
        prepared.index.name = "Date"
        self._validate(prepared)
        return prepared

    def _validate(self, prices: pd.DataFrame) -> None:
        if len(prices) < self.config.minimum_observations:
            raise ValueError("Too few aligned observations for the planned analysis")
        if (prices <= 0).any().any():
            raise ValueError("All price/index levels must be positive")
        if prices.index.has_duplicates:
            raise ValueError("Market-data dates must be unique")


class LogReturnCalculator:
    """Convert positive market levels into aligned daily log returns."""

    @staticmethod
    def calculate(prices: pd.DataFrame) -> pd.DataFrame:
        returns = (
            np.log(prices)
            .diff()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        returns.index.name = "Date"
        return returns
