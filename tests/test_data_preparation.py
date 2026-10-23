"""Focused tests for pure market-data preparation."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from src.data.preparation import LogReturnCalculator, MarketDataPreprocessor
from src.domain.config import DEFAULT_CONFIG


class MarketDataPreprocessorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(
            DEFAULT_CONFIG,
            start_date="2026-01-02",
            minimum_observations=3,
            max_forward_fill_days=1,
        )
        self.preprocessor = MarketDataPreprocessor(self.config)

    def test_prepare_orders_filters_and_fills_configured_series(self) -> None:
        dates = pd.to_datetime(
            ["2026-01-05", "2026-01-01", "2026-01-02", "2026-01-06"]
        )
        prices = pd.DataFrame(
            {
                "EXTRA": [1.0, 1.0, 1.0, 1.0],
                "DEXUSEU": [np.nan, 100.0, 102.0, 104.0],
                "SP500": [103.0, 100.0, 102.0, 104.0],
                "DCOILBRENTEU": [203.0, 200.0, 202.0, 204.0],
                "NASDAQCOM": [303.0, 300.0, 302.0, 304.0],
            },
            index=dates,
        )

        prepared = self.preprocessor.prepare(prices)

        self.assertEqual(tuple(prepared.columns), self.config.asset_codes)
        self.assertEqual(prepared.index.min(), pd.Timestamp("2026-01-02"))
        self.assertTrue(prepared.index.is_monotonic_increasing)
        self.assertEqual(prepared.loc["2026-01-05", "DEXUSEU"], 102.0)
        self.assertEqual(prepared.index.name, "Date")

    def test_prepare_rejects_missing_or_nonpositive_levels(self) -> None:
        dates = pd.date_range("2026-01-02", periods=3, freq="B")
        valid = pd.DataFrame(
            100.0,
            index=dates,
            columns=self.config.asset_codes,
        )

        with self.assertRaisesRegex(ValueError, "Missing required data columns"):
            self.preprocessor.prepare(valid.drop(columns=[self.config.asset_codes[0]]))

        invalid = valid.copy()
        invalid.iloc[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.preprocessor.prepare(invalid)

    def test_log_returns_are_aligned_and_mathematically_correct(self) -> None:
        dates = pd.date_range("2026-01-02", periods=3, freq="B")
        prices = pd.DataFrame(
            {"asset": [100.0, 110.0, 121.0]},
            index=dates,
        )

        returns = LogReturnCalculator.calculate(prices)

        np.testing.assert_allclose(returns["asset"], np.log([1.1, 1.1]))
        self.assertEqual(returns.index.tolist(), dates[1:].tolist())
        self.assertEqual(returns.index.name, "Date")


if __name__ == "__main__":
    unittest.main()
