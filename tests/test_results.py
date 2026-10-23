"""Tests for typed analysis results and stable serialization schemas."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from src.domain.results import (
    AnalysisResult,
    BacktestResult,
    HorizonModelParameters,
    RiskEstimate,
    ScaleResult,
)


class TypedResultTests(unittest.TestCase):
    def setUp(self) -> None:
        t_risk = RiskEstimate(var=0.02, expected_shortfall=0.03)
        gaussian_risk = RiskEstimate(var=0.018, expected_shortfall=0.025)
        self.scale = ScaleResult(
            scale="2-4 days",
            empirical_tail=0.28,
            t_tail=0.16,
            t_df=5.5,
            mean_corr=0.25,
            t_risk=t_risk,
            gaussian_risk=gaussian_risk,
            t_backtest=BacktestResult(3, 500, 0.006, 0.001, 0.04, 0.03),
            gaussian_backtest=BacktestResult(
                4,
                500,
                0.008,
                0.0012,
                0.038,
                0.025,
            ),
        )

    def test_scale_result_exposes_stable_csv_schema(self) -> None:
        record = self.scale.to_record()
        self.assertEqual(
            list(record),
            [
                "scale",
                "empirical_tail",
                "t_tail",
                "t_df",
                "mean_corr",
                "t_var",
                "t_es",
                "gaussian_var",
                "gaussian_es",
                "test_n",
                "t_exceptions",
                "t_exception_rate",
                "t_quantile_loss",
                "gaussian_exceptions",
                "gaussian_exception_rate",
                "gaussian_quantile_loss",
                "es_change_pct",
            ],
        )
        self.assertAlmostEqual(record["es_change_pct"], 20.0)

    def test_analysis_result_serializes_only_at_output_boundary(self) -> None:
        parameters = HorizonModelParameters(
            scale="2-4 days",
            degrees_of_freedom=5.5,
            log_likelihood=42.0,
            correlation=((1.0, 0.3), (0.3, 1.0)),
        )
        result = AnalysisResult(
            scales=(self.scale,),
            benchmark_correlation=((1.0, 0.2), (0.2, 1.0)),
            model_parameters=(parameters,),
        )
        self.assertEqual(result.csv_records(), [self.scale.to_record()])
        self.assertEqual(
            result.model_parameter_records()["2-4 days"]["t_df"],
            5.5,
        )

    def test_results_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.scale.scale = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()

