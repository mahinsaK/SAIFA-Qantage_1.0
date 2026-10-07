"""Central, validated configuration for the complete analysis."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose


@dataclass(frozen=True)
class AssetConfig:
    """One market exposure used in the portfolio."""

    code: str
    name: str
    exposure: str
    reason: str
    weight: float


@dataclass(frozen=True)
class AnalysisConfig:
    """All user-selectable assumptions that determine analysis outputs."""

    assets: tuple[AssetConfig, ...]
    start_date: str
    train_fraction: float
    simulations: int
    minimum_simulations: int
    confidence_level: float
    tail_quantile: float
    wavelet: str
    wavelet_level: int
    max_forward_fill_days: int
    minimum_observations: int
    t_df_bounds: tuple[float, float]
    seed: int
    report_date: str
    report_filename: str

    def __post_init__(self) -> None:
        if len(self.assets) < 2:
            raise ValueError("At least two assets are required")
        codes = [asset.code for asset in self.assets]
        if len(codes) != len(set(codes)):
            raise ValueError("Asset codes must be unique")
        if not isclose(sum(asset.weight for asset in self.assets), 1.0, abs_tol=1e-12):
            raise ValueError("Portfolio weights must sum to 1")
        if any(asset.weight < 0 for asset in self.assets):
            raise ValueError("Portfolio weights cannot be negative")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1")
        if self.simulations < self.minimum_simulations:
            raise ValueError(
                f"Use at least {self.minimum_simulations:,} simulations for stable tail estimates"
            )
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if not 0 < self.tail_quantile < 0.5:
            raise ValueError("tail_quantile must be between 0 and 0.5")
        if self.wavelet_level < 1:
            raise ValueError("wavelet_level must be positive")
        if self.max_forward_fill_days < 0:
            raise ValueError("max_forward_fill_days cannot be negative")
        if self.minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if not 2 < self.t_df_bounds[0] < self.t_df_bounds[1]:
            raise ValueError("Student-t degrees-of-freedom bounds must satisfy 2 < lower < upper")

    @property
    def asset_codes(self) -> tuple[str, ...]:
        return tuple(asset.code for asset in self.assets)

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(asset.weight for asset in self.assets)

    @property
    def loss_probability(self) -> float:
        return 1 - self.confidence_level

    @property
    def mra_scale_labels(self) -> tuple[str, ...]:
        """Labels in PyWavelets MRA output order: approximation then Dn...D1."""
        smooth_limit = 2 ** (self.wavelet_level + 1)
        details = tuple(
            f"{2**level}-{2 ** (level + 1)} days"
            for level in range(self.wavelet_level, 0, -1)
        )
        return (f">{smooth_limit} days", *details)

    @property
    def ordered_scale_labels(self) -> tuple[str, ...]:
        """Labels ordered from the shortest to longest investment horizon."""
        return tuple(reversed(self.mra_scale_labels))


DEFAULT_CONFIG = AnalysisConfig(
    assets=(
        AssetConfig(
            code="SP500",
            name="S&P 500",
            exposure="Large-cap US equity",
            reason="Broad equity-market loss factor",
            weight=0.25,
        ),
        AssetConfig(
            code="NASDAQCOM",
            name="Nasdaq Composite Index",
            exposure="Technology-heavy US equity",
            reason="Concentrated growth exposure",
            weight=0.25,
        ),
        AssetConfig(
            code="DCOILBRENTEU",
            name="Crude Oil Prices: Brent - Europe",
            exposure="Brent spot crude oil",
            reason="Commodity and inflation shock factor",
            weight=0.25,
        ),
        AssetConfig(
            code="DEXUSEU",
            name="U.S. Dollars to Euro Spot Exchange Rate",
            exposure="Long EUR / short USD",
            reason="Major liquid currency exposure",
            weight=0.25,
        ),
    ),
    start_date="2016-10-07",
    train_fraction=0.80,
    simulations=100_000,
    minimum_simulations=20_000,
    confidence_level=0.99,
    tail_quantile=0.10,
    wavelet="haar",
    wavelet_level=4,
    max_forward_fill_days=5,
    minimum_observations=1_000,
    t_df_bounds=(2.05, 50.0),
    seed=20261007,
    report_date="7 October 2026",
    report_filename="Quant_Edge_1.0_Report.pdf",
)
