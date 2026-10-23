"""Tests for dependency-injected pipeline orchestration."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from src.application.pipeline import PipelineDependencies, run_pipeline
from src.domain.config import DEFAULT_CONFIG
from src.models.copulas import GaussianCopula, TCopula


class FakeMarketData:
    def __init__(self) -> None:
        self.refresh_values: list[bool] = []
        index = pd.date_range("2024-01-01", periods=21, freq="B")
        self.prices = pd.DataFrame(
            {
                code: 100 + offset + np.arange(len(index), dtype=float)
                for offset, code in enumerate(DEFAULT_CONFIG.asset_codes)
            },
            index=index,
        )

    def load_prices(self, refresh: bool = False) -> pd.DataFrame:
        self.refresh_values.append(refresh)
        return self.prices

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        values = np.linspace(-0.02, 0.02, len(prices) - 1)
        return pd.DataFrame(
            {
                code: values + column * 0.0001
                for column, code in enumerate(prices.columns)
            },
            index=prices.index[1:],
        )


class FakeStudentTModel:
    name = "fake_student_t"

    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, data: np.ndarray) -> TCopula:
        self.fit_calls += 1
        return TCopula(5.0, np.eye(data.shape[1]), 10.0)

    def simulate_uniforms(self, fitted, size, rng):
        return rng.uniform(0.01, 0.99, size=(size, fitted.corr.shape[0]))

    def lower_tail_dependence(self, fitted) -> float:
        return 0.15

    def correlation(self, fitted) -> np.ndarray:
        return fitted.corr

    def parameters(self, fitted) -> dict[str, object]:
        return {"df": fitted.df}


class FakeGaussianModel:
    name = "fake_gaussian"

    def __init__(self) -> None:
        self.fit_calls = 0

    def fit(self, data: np.ndarray) -> GaussianCopula:
        self.fit_calls += 1
        return GaussianCopula(np.eye(data.shape[1]))

    def simulate_uniforms(self, fitted, size, rng):
        return rng.uniform(0.01, 0.99, size=(size, fitted.corr.shape[0]))

    def lower_tail_dependence(self, fitted) -> float:
        return 0.0

    def correlation(self, fitted) -> np.ndarray:
        return fitted.corr

    def parameters(self, fitted) -> dict[str, object]:
        return {}


class PipelineDependencyTests(unittest.TestCase):
    def test_pipeline_uses_injected_collaborators(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            wavelet_level=1,
            simulations=50,
            minimum_simulations=10,
        )
        market_data = FakeMarketData()
        primary_model = FakeStudentTModel()
        benchmark_model = FakeGaussianModel()
        decomposer_calls: list[int] = []
        written = []
        published = []

        def decompose(frame, supplied_config):
            decomposer_calls.append(len(frame))
            return {
                scale: frame.copy()
                for scale in supplied_config.ordered_scale_labels
            }

        def write_results(paths, data, results, supplied_config):
            written.append((paths, data, results, supplied_config))

        def publish_artifacts(paths, data, results, supplied_config):
            published.append((paths, data, results, supplied_config))

        dependencies = PipelineDependencies(
            market_data=market_data,
            decomposer=decompose,
            primary_model=primary_model,
            benchmark_model=benchmark_model,
            result_writer=write_results,
            artifact_publisher=publish_artifacts,
        )

        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            run_pipeline(root, config, dependencies, refresh_data=True)
            self.assertTrue((root / "output" / "figures").is_dir())

        self.assertEqual(market_data.refresh_values, [True])
        self.assertEqual(decomposer_calls, [16, 4])
        self.assertEqual(primary_model.fit_calls, 2)
        self.assertEqual(benchmark_model.fit_calls, 1)
        self.assertEqual(len(written), 1)
        self.assertEqual(len(published), 1)
        self.assertIs(written[0][2], published[0][2])
        self.assertEqual(len(written[0][2].scales), 2)


if __name__ == "__main__":
    unittest.main()
