"""Focused tests for the machine-readable output adapter."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.application.pipeline import OutputPaths, PreparedData
from src.domain.config import DEFAULT_CONFIG
from src.domain.results import (
    AnalysisResult,
    BacktestResult,
    HorizonModelParameters,
    RiskEstimate,
    ScaleResult,
)
from src.infrastructure.result_writer import save_results


class OutputAdapterTests(unittest.TestCase):
    def test_save_results_writes_documented_csv_and_json_contracts(self) -> None:
        scale = ScaleResult(
            scale="2-4 days",
            empirical_tail=0.2,
            t_tail=0.1,
            t_df=5.0,
            mean_corr=0.3,
            t_risk=RiskEstimate(0.02, 0.03),
            gaussian_risk=RiskEstimate(0.018, 0.025),
            t_backtest=BacktestResult(1, 10, 0.1, 0.01, 0.04, 0.03),
            gaussian_backtest=BacktestResult(2, 10, 0.2, 0.02, 0.05, 0.025),
        )
        results = AnalysisResult(
            scales=(scale,),
            benchmark_correlation=((1.0, 0.2), (0.2, 1.0)),
            model_parameters=(
                HorizonModelParameters(
                    "2-4 days", 5.0, 12.0, ((1.0, 0.3), (0.3, 1.0))
                ),
            ),
        )
        index = pd.date_range("2026-01-01", periods=2, freq="B")
        frame = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]}, index=index)
        data = PreparedData(
            prices=frame + 100,
            returns=frame,
            train=frame.iloc[:1],
            test=frame.iloc[1:],
            train_components={},
            test_components={},
            weights=np.array([0.5, 0.5]),
        )
        config = replace(
            DEFAULT_CONFIG,
            simulations=20_000,
            assets=tuple(
                replace(asset, weight=0.5)
                for asset in DEFAULT_CONFIG.assets[:2]
            ),
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = OutputPaths(root / "figures", root / "tables", root / "pdf")
            paths.tables.mkdir()
            save_results(paths, data, results, config)

            csv = pd.read_csv(paths.tables / "risk_results.csv")
            parameters = json.loads(
                (paths.tables / "model_parameters.json").read_text(encoding="utf-8")
            )
            aligned = pd.read_csv(paths.tables / "aligned_log_returns.csv")

        self.assertEqual(csv.loc[0, "scale"], "2-4 days")
        self.assertAlmostEqual(csv.loc[0, "es_change_pct"], 20.0)
        self.assertEqual(parameters["asset_order"], list(config.asset_codes))
        self.assertEqual(parameters["models"]["2-4 days"]["t_df"], 5.0)
        self.assertEqual(parameters["seed"], config.seed)
        self.assertEqual(len(aligned), 2)


if __name__ == "__main__":
    unittest.main()
