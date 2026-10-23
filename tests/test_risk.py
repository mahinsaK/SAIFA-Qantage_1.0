"""Focused tests for portfolio risk and holdout scoring."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.domain.results import RiskEstimate
from src.domain.risk import holdout_metrics, portfolio_losses, var_es


class RiskCalculationTests(unittest.TestCase):
    def test_portfolio_losses_apply_weights_and_loss_sign(self) -> None:
        returns = np.array([[0.10, -0.10], [-0.20, 0.00]])

        losses = portfolio_losses(returns, np.array([0.5, 0.5]))

        np.testing.assert_allclose(losses, [0.0, 0.10], atol=1e-15)

    def test_var_and_expected_shortfall_use_the_selected_tail(self) -> None:
        estimate = var_es(np.array([1.0, 2.0, 3.0, 4.0]), 0.75)

        self.assertEqual(estimate.var, 3.25)
        self.assertEqual(estimate.expected_shortfall, 4.0)

    def test_holdout_metrics_use_strict_var_exceedances(self) -> None:
        risk = RiskEstimate(var=2.0, expected_shortfall=2.5)
        metrics = holdout_metrics(np.array([1.0, 2.0, 3.0]), risk, 0.90)

        self.assertEqual(metrics.exceptions, 1)
        self.assertEqual(metrics.observations, 3)
        self.assertAlmostEqual(metrics.exception_rate, 1 / 3)
        self.assertEqual(metrics.tail_mean_if_exceeded, 3.0)
        self.assertEqual(metrics.es_forecast, 2.5)

        no_exceptions = holdout_metrics(np.array([1.0, 2.0]), risk, 0.90)
        self.assertTrue(math.isnan(no_exceptions.tail_mean_if_exceeded))


if __name__ == "__main__":
    unittest.main()
