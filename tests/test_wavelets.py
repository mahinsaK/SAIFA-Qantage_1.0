"""Focused tests for additive wavelet decomposition."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from src.wavelets import decompose


class WaveletDecompositionTests(unittest.TestCase):
    def test_components_reconstruct_trimmed_input_in_documented_order(self) -> None:
        config = replace(DEFAULT_CONFIG, wavelet_level=2)
        index = pd.date_range("2026-01-01", periods=19, freq="B")
        frame = pd.DataFrame(
            {
                "first": np.sin(np.arange(19) / 3),
                "second": np.cos(np.arange(19) / 5),
            },
            index=index,
        )

        components = decompose(frame, config)
        reconstructed = sum(
            (components[label] for label in config.mra_scale_labels),
            start=pd.DataFrame(0.0, index=index[3:], columns=frame.columns),
        )

        self.assertEqual(tuple(components), config.mra_scale_labels)
        self.assertTrue(all(len(component) == 16 for component in components.values()))
        pd.testing.assert_frame_equal(reconstructed, frame.iloc[3:])


if __name__ == "__main__":
    unittest.main()
