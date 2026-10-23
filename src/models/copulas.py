"""Small, transparent Gaussian and Student-t copula implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import norm, rankdata, t


EPS = 1e-9
FittedCopulaT = TypeVar("FittedCopulaT")


@runtime_checkable
class CopulaModel(Protocol[FittedCopulaT]):
    """Common contract for fitting, simulating, and inspecting a copula."""

    @property
    def name(self) -> str: ...

    def fit(self, data: np.ndarray) -> FittedCopulaT: ...

    def simulate_uniforms(
        self,
        fitted: FittedCopulaT,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...

    def lower_tail_dependence(self, fitted: FittedCopulaT) -> float: ...

    def correlation(self, fitted: FittedCopulaT) -> np.ndarray: ...

    def parameters(self, fitted: FittedCopulaT) -> dict[str, object]: ...


def pseudo_observations(data: np.ndarray) -> np.ndarray:
    n, d = data.shape
    return np.column_stack([rankdata(data[:, j], method="average") / (n + 1.0) for j in range(d)])


def nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, floor)
    fixed = (vectors * values) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(scale, scale)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def _multivariate_t_logpdf(x: np.ndarray, corr: np.ndarray, df: float) -> np.ndarray:
    n, d = x.shape
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        return np.full(n, -np.inf)
    solved = np.linalg.solve(corr, x.T).T
    quad = np.einsum("ij,ij->i", x, solved)
    constant = gammaln((df + d) / 2) - gammaln(df / 2)
    constant -= 0.5 * (d * np.log(df * np.pi) + logdet)
    return constant - 0.5 * (df + d) * np.log1p(quad / df)


def _t_copula_fit_for_df(u: np.ndarray, df: float) -> tuple[float, np.ndarray]:
    x = t.ppf(np.clip(u, EPS, 1 - EPS), df=df)
    corr = nearest_correlation(np.corrcoef(x, rowvar=False))
    joint = _multivariate_t_logpdf(x, corr, df)
    marginal = np.sum(t.logpdf(x, df=df), axis=1)
    return float(np.sum(joint - marginal)), corr


@dataclass(frozen=True)
class TCopula:
    df: float
    corr: np.ndarray
    loglik: float


@dataclass(frozen=True)
class GaussianCopula:
    corr: np.ndarray


def fit_t_copula(data: np.ndarray, df_bounds: tuple[float, float]) -> TCopula:
    u = pseudo_observations(data)

    def objective(df: float) -> float:
        loglik, _ = _t_copula_fit_for_df(u, df)
        return -loglik

    result = minimize_scalar(objective, bounds=df_bounds, method="bounded", options={"xatol": 1e-3})
    loglik, corr = _t_copula_fit_for_df(u, float(result.x))
    return TCopula(float(result.x), corr, loglik)


def fit_gaussian_correlation(data: np.ndarray) -> np.ndarray:
    u = pseudo_observations(data)
    z = norm.ppf(np.clip(u, EPS, 1 - EPS))
    return nearest_correlation(np.corrcoef(z, rowvar=False))


def simulate_t_uniforms(model: TCopula, size: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.multivariate_normal(np.zeros(model.corr.shape[0]), model.corr, size=size)
    scale = np.sqrt(rng.chisquare(model.df, size=size) / model.df)
    x = z / scale[:, None]
    return np.clip(t.cdf(x, df=model.df), EPS, 1 - EPS)


def simulate_gaussian_uniforms(corr: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.multivariate_normal(np.zeros(corr.shape[0]), corr, size=size)
    return np.clip(norm.cdf(z), EPS, 1 - EPS)


def empirical_lower_tail_dependence(data: np.ndarray, q: float) -> float:
    u = pseudo_observations(data)
    estimates = []
    for i in range(u.shape[1]):
        for j in range(i + 1, u.shape[1]):
            denom = np.sum(u[:, i] <= q)
            estimates.append(np.sum((u[:, i] <= q) & (u[:, j] <= q)) / denom)
    return float(np.mean(estimates))


def t_lower_tail_dependence(model: TCopula) -> float:
    estimates = []
    for i in range(model.corr.shape[0]):
        for j in range(i + 1, model.corr.shape[0]):
            rho = float(np.clip(model.corr[i, j], -0.999, 0.999))
            value = 2 * t.cdf(-np.sqrt((model.df + 1) * (1 - rho) / (1 + rho)), df=model.df + 1)
            estimates.append(value)
    return float(np.mean(estimates))


def empirical_inverse_margins(u: np.ndarray, data: np.ndarray) -> np.ndarray:
    probs = np.clip(u, EPS, 1 - EPS)
    return np.column_stack(
        [np.quantile(data[:, j], probs[:, j], method="linear") for j in range(data.shape[1])]
    )


@dataclass(frozen=True)
class StudentTCopulaModel:
    """Student-t copula implementation of the common model contract."""

    df_bounds: tuple[float, float]
    name: str = "student_t"

    def fit(self, data: np.ndarray) -> TCopula:
        return fit_t_copula(data, self.df_bounds)

    def simulate_uniforms(
        self,
        fitted: TCopula,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return simulate_t_uniforms(fitted, size, rng)

    def lower_tail_dependence(self, fitted: TCopula) -> float:
        return t_lower_tail_dependence(fitted)

    def correlation(self, fitted: TCopula) -> np.ndarray:
        return fitted.corr

    def parameters(self, fitted: TCopula) -> dict[str, object]:
        return {
            "t_df": fitted.df,
            "t_loglik": fitted.loglik,
            "t_correlation": fitted.corr.tolist(),
        }


@dataclass(frozen=True)
class GaussianCopulaModel:
    """Gaussian copula implementation of the common model contract."""

    name: str = "gaussian"

    def fit(self, data: np.ndarray) -> GaussianCopula:
        return GaussianCopula(fit_gaussian_correlation(data))

    def simulate_uniforms(
        self,
        fitted: GaussianCopula,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return simulate_gaussian_uniforms(fitted.corr, size, rng)

    def lower_tail_dependence(self, fitted: GaussianCopula) -> float:
        return 0.0

    def correlation(self, fitted: GaussianCopula) -> np.ndarray:
        return fitted.corr

    def parameters(self, fitted: GaussianCopula) -> dict[str, object]:
        return {"correlation": fitted.corr.tolist()}
