"""One-command entry point for the complete Quant Edge analysis."""

from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

from src.application.composition import create_pipeline_dependencies
from src.application.pipeline import run_pipeline
from src.domain.config import DEFAULT_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Redownload public FRED data instead of using the included snapshot.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_CONFIG.simulations,
        help=(
            "Number of copula simulations per wavelet horizon "
            f"(default: {DEFAULT_CONFIG.simulations})."
        ),
    )
    args = parser.parse_args()
    config = replace(DEFAULT_CONFIG, simulations=args.simulations)
    dependencies = create_pipeline_dependencies(PROJECT_ROOT, config)
    run_pipeline(
        PROJECT_ROOT,
        config,
        dependencies,
        refresh_data=args.refresh_data,
    )


if __name__ == "__main__":
    main()
