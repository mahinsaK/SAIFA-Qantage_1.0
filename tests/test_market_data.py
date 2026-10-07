"""Offline tests for separated market-data responsibilities."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.data import MarketDataService
from src.data_preparation import LogReturnCalculator, MarketDataPreprocessor
from src.data_repository import CsvMarketDataRepository


class StubProvider:
    def __init__(self, values: dict[str, pd.Series]) -> None:
        self.values = values
        self.calls: list[tuple[str, bool]] = []

    def get_series(self, series: str, refresh: bool = False) -> pd.Series:
        self.calls.append((series, refresh))
        return self.values[series]


class MarketDataServiceTests(unittest.TestCase):
    def test_snapshot_reuse_and_refresh_are_separate_from_preparation(self) -> None:
        config = replace(DEFAULT_CONFIG, minimum_observations=4)
        dates = pd.date_range("2026-01-01", periods=6, freq="B")
        values = {
            code: pd.Series(
                np.arange(100.0 + index, 106.0 + index),
                index=dates,
                name=code,
            )
            for index, code in enumerate(config.asset_codes)
        }

        with TemporaryDirectory() as directory:
            repository = CsvMarketDataRepository(
                Path(directory) / "market_prices.csv"
            )
            provider = StubProvider(values)
            service = MarketDataService(
                provider=provider,
                repository=repository,
                preprocessor=MarketDataPreprocessor(config),
                return_calculator=LogReturnCalculator(),
            )

            downloaded = service.load_prices()
            self.assertEqual(len(provider.calls), len(config.assets))
            self.assertTrue(repository.exists())

            provider.calls.clear()
            cached = service.load_prices()
            self.assertEqual(provider.calls, [])
            pd.testing.assert_frame_equal(downloaded, cached, check_freq=False)

            refreshed = service.load_prices(refresh=True)
            self.assertEqual(len(provider.calls), len(config.assets))
            self.assertTrue(all(refresh for _, refresh in provider.calls))
            pd.testing.assert_frame_equal(downloaded, refreshed, check_freq=False)

            returns = service.calculate_returns(refreshed)
            self.assertEqual(len(returns), len(refreshed) - 1)
            self.assertTrue(np.isfinite(returns.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
