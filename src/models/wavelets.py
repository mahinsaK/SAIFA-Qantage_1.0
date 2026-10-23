"""Additive maximum-overlap wavelet decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

from ..domain.config import AnalysisConfig


def _valid_length(frame: pd.DataFrame, level: int) -> pd.DataFrame:
    block = 2**level
    remainder = len(frame) % block
    return frame.iloc[remainder:].copy() if remainder else frame.copy()


def decompose(frame: pd.DataFrame, config: AnalysisConfig) -> dict[str, pd.DataFrame]:
    """Return additive SWT multiresolution components for all assets.

    PyWavelets returns [A4, D4, D3, D2, D1]. With a daily input, detail
    bands correspond approximately to periods of 16-32, 8-16, 4-8, and
    2-4 trading days. The smooth A4 component represents periods above 32 days.
    """
    aligned = _valid_length(frame, config.wavelet_level)
    labels = config.mra_scale_labels
    arrays: dict[str, list[np.ndarray]] = {label: [] for label in labels}
    for column in aligned.columns:
        components = pywt.mra(
            aligned[column].to_numpy(),
            wavelet=config.wavelet,
            level=config.wavelet_level,
            transform="swt",
        )
        if len(components) != len(labels):
            raise RuntimeError("Unexpected wavelet component count")
        if not np.allclose(np.sum(components, axis=0), aligned[column].to_numpy(), atol=1e-10):
            raise RuntimeError("Wavelet components do not reconstruct the input")
        for label, values in zip(labels, components, strict=True):
            arrays[label].append(values)
    return {
        label: pd.DataFrame(np.column_stack(values), index=aligned.index, columns=aligned.columns)
        for label, values in arrays.items()
    }
