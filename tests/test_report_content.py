"""Tests for renderer-independent report content construction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.report_content import build_report_content
from src.results import BacktestResult, RiskEstimate, ScaleResult


class ReportContentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = ScaleResult(
            scale="2-4 days",
            empirical_tail=0.28,
            t_tail=0.16,
            t_df=5.5,
            mean_corr=0.25,
            t_risk=RiskEstimate(var=0.02, expected_shortfall=0.03),
            gaussian_risk=RiskEstimate(var=0.018, expected_shortfall=0.025),
            t_backtest=BacktestResult(3, 500, 0.006, 0.001, 0.04, 0.03),
            gaussian_backtest=BacktestResult(4, 500, 0.008, 0.0012, 0.038, 0.025),
        )
        self.prices = pd.DataFrame(
            {"SP500": [100.0, 101.0]},
            index=pd.to_datetime(["2020-01-02", "2024-12-31"]),
        )

    def test_content_builder_derives_prose_and_tables_without_renderer(self) -> None:
        config = replace(DEFAULT_CONFIG, simulations=20_000)
        content = build_report_content(
            [self.result], self.prices, "2023-12-29", "2024-01-02", config,
        )

        self.assertIn("20,000 joint uniform draws", content.method_sections[2].body)
        self.assertIn("2-4 days", content.recommendation)
        self.assertIn("20.0% difference", content.recommendation)
        self.assertEqual(content.risk_table.rows[0][-1], "+20.0%")
        self.assertEqual(content.portfolio_table.rows[0][0], "SP500")
        self.assertEqual(content.report_date, config.report_date)

    def test_report_content_is_immutable(self) -> None:
        content = build_report_content(
            [self.result], self.prices, "2023-12-29", "2024-01-02", DEFAULT_CONFIG,
        )
        with self.assertRaises(FrozenInstanceError):
            content.decision = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
