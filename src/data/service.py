"""Application service coordinating market-data acquisition and preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..domain.config import AnalysisConfig
from .preparation import LogReturnCalculator, MarketDataPreprocessor
from .repository import CsvMarketDataRepository
from .source import FredDataProvider, MarketDataProvider


@dataclass
class MarketDataService:
    """Coordinate provider, snapshot repository, and pure preprocessing."""

    provider: MarketDataProvider
    repository: CsvMarketDataRepository
    preprocessor: MarketDataPreprocessor
    return_calculator: LogReturnCalculator

    def load_prices(self, refresh: bool = False) -> pd.DataFrame:
        if self.repository.exists() and not refresh:
            return self.preprocessor.prepare(self.repository.load())

        series = [
            self.provider.get_series(code, refresh=refresh)
            for code in self.preprocessor.config.asset_codes
        ]
        prices = self.preprocessor.combine(series)
        self.repository.save(prices)
        return prices

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.return_calculator.calculate(prices)


def create_market_data_service(
    root: Path,
    config: AnalysisConfig,
) -> MarketDataService:
    """Build the production market-data service for the project workspace."""
    data_dir = root / "data"
    return MarketDataService(
        provider=FredDataProvider(data_dir / "raw"),
        repository=CsvMarketDataRepository(data_dir / "market_prices.csv"),
        preprocessor=MarketDataPreprocessor(config),
        return_calculator=LogReturnCalculator(),
    )
