# Quant Edge 1.0 - Risk Across Tails and Timescales

This project answers the screening question: **Does tail dependence change with the investment horizon, and what does ignoring this do to a portfolio's measured risk?**

It studies an equal-weight portfolio of the S&P 500, Nasdaq Composite, Brent crude oil and a long-EUR/short-USD currency exposure. A four-level stationary Haar wavelet decomposition separates daily log returns into approximate 2-4, 4-8, 8-16, 16-32 and greater-than-32-trading-day components. A Student-t copula is fitted separately at every horizon. The controlled benchmark uses one Gaussian copula estimated from raw daily returns at every horizon while retaining the same horizon-specific empirical margins.

## One-command reproduction

Install [uv](https://docs.astral.sh/uv/) and run this from the project root:

```bash
uv run python run.py
```

That command creates every number, table, figure and the final PDF report. The first run installs the dependencies declared in `pyproject.toml`; later runs use the generated lock file. The included `data/market_prices.csv` snapshot fixes the submitted result and permits offline reproduction.

All analytical assumptions are defined once in `src/config.py`. This includes the assets and weights, sample split, simulation count, confidence level, tail threshold, wavelet settings, Student-t bounds and random seed. The CLI creates a validated copy of this configuration when `--simulations` is supplied.

To redownload the public data before running:

```bash
uv run python run.py --refresh-data
```

The current analysis uses 100,000 simulations per horizon. For a quick code check, the lowest accepted setting is:

```bash
uv run python run.py --simulations 20000
```

Run the offline unit tests with:

```bash
uv run python -m unittest discover -s tests -v
```

## Outputs

- `output/pdf/Quant_Edge_1.0_Report.pdf` - submission report.
- `output/tables/risk_results.csv` - all reported dependence, risk and backtest results.
- `output/tables/model_parameters.json` - fitted copula parameters, correlations and random seed.
- `output/tables/aligned_log_returns.csv` - cleaned model input.
- `output/figures/` - the three report figures.

## Data

The script downloads CSV observations from the Federal Reserve Bank of St. Louis FRED graph endpoint, which does not require an API key:

- `SP500` - S&P 500.
- `NASDAQCOM` - Nasdaq Composite Index.
- `DCOILBRENTEU` - Crude Oil Prices: Brent - Europe.
- `DEXUSEU` - U.S. Dollars to Euro Spot Exchange Rate (long EUR / short USD exposure).

The sample begins on 7 October 2016 because FRED's freely distributed daily S&P 500 history begins there. Series are aligned on a common calendar, and isolated publication gaps of up to five business days are carried forward. The analysis treats these series as market-exposure proxies; it does not include ETF tracking, currency carry, transaction costs or futures rolls.

## Method and validation

Daily log returns are split chronologically: the first 80% estimate the models and the final 20% form an untouched holdout set. The stationary wavelet transform is applied separately to each subset. The main model fits the Student-t copula degrees of freedom by maximum likelihood and estimates its correlation matrix from transformed ranks. Copula draws are mapped through empirical margins, then equal-weight portfolio 99% VaR and Expected Shortfall are calculated.

The benchmark's raw-return Gaussian copula correlation is reused at all wavelet horizons. Because both models share the same band-specific empirical margins, their risk difference isolates dependence and joint-tail assumptions. Holdout VaR exception rates and quantile losses provide the out-of-sample comparison.

The fixed holdout design is distributional validation, not a rolling one-day forecast. Main limitations are index/spot proxies, symmetric t-copula tails, wavelet boundary effects and finite-sample empirical tail estimates. These are discussed in the report.

## Code map

- `run.py` - single entry point.
- `src/config.py` - central, validated analysis configuration.
- `src/data.py` - market-data application service.
- `src/data_source.py` - FRED acquisition and raw-response caching.
- `src/data_repository.py` - aligned CSV snapshot persistence.
- `src/data_preparation.py` - alignment, validation and log-return construction.
- `src/wavelets.py` - additive multiresolution decomposition.
- `src/copulas.py` - copula estimation, simulation and tail dependence.
- `src/risk.py` - VaR, Expected Shortfall and holdout scoring.
- `src/plots.py` - figures.
- `src/report.py` - PDF generation.
- `src/pipeline.py` - complete orchestration.

## AI-use disclosure

OpenAI Codex was materially used for research-design discussion, Python implementation, debugging, report drafting and formatting. The team remains fully responsible for reviewing every methodological choice, line of code, calculation and claim. All numbers in the report are generated from the submitted code and public data.
