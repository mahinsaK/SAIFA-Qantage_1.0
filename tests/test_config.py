"""Focused tests for centralized configuration invariants."""

from __future__ import annotations

from dataclasses import replace
import unittest

from src.config import DEFAULT_CONFIG


class AnalysisConfigTests(unittest.TestCase):
    def test_rejects_invalid_portfolio_and_simulation_settings(self) -> None:
        invalid_changes = (
            {"assets": DEFAULT_CONFIG.assets[:1]},
            {
                "assets": (
                    DEFAULT_CONFIG.assets[0],
                    replace(DEFAULT_CONFIG.assets[1], code="SP500"),
                )
            },
            {
                "assets": (
                    replace(DEFAULT_CONFIG.assets[0], weight=0.40),
                    *DEFAULT_CONFIG.assets[1:],
                )
            },
            {"simulations": DEFAULT_CONFIG.minimum_simulations - 1},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(DEFAULT_CONFIG, **changes)

    def test_rejects_invalid_probability_and_model_bounds(self) -> None:
        invalid_changes = (
            {"train_fraction": 1.0},
            {"confidence_level": 0.5},
            {"tail_quantile": 0.5},
            {"t_df_bounds": (2.0, 10.0)},
            {"wavelet_level": 0},
        )

        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(DEFAULT_CONFIG, **changes)

    def test_scale_labels_follow_the_configured_wavelet_level(self) -> None:
        config = replace(DEFAULT_CONFIG, wavelet_level=2)

        self.assertEqual(config.mra_scale_labels, (">8 days", "4-8 days", "2-4 days"))
        self.assertEqual(config.ordered_scale_labels, ("2-4 days", "4-8 days", ">8 days"))


if __name__ == "__main__":
    unittest.main()
