"""Leakage-safe, origin-based forecasting baselines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


DEFAULT_CROSTON_ALPHA = 0.1


def _validated_history(history: np.ndarray | list[float]) -> np.ndarray:
    values = np.asarray(history, dtype="float64")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Forecast history must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecast history must contain finite non-negative demand values.")
    return values


def _validated_horizon(horizon: int) -> int:
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("Forecast horizon must be a positive integer.")
    return horizon


def zero_forecast(history: np.ndarray | list[float], horizon: int = 30) -> np.ndarray:
    """Return the zero-demand sanity baseline."""

    _validated_history(history)
    return np.zeros(_validated_horizon(horizon), dtype="float64")


def last_value_forecast(history: np.ndarray | list[float], horizon: int = 30) -> np.ndarray:
    """Repeat the final pre-origin demand value across the horizon."""

    values = _validated_history(history)
    return np.full(_validated_horizon(horizon), values[-1], dtype="float64")


def seasonal_naive_7_forecast(history: np.ndarray | list[float], horizon: int = 30) -> np.ndarray:
    """Repeat the final seven observed pre-origin demand values."""

    values = _validated_history(history)
    if values.size < 7:
        raise ValueError("The 7-day seasonal naive baseline requires at least seven history values.")
    return np.resize(values[-7:], _validated_horizon(horizon)).astype("float64")


def mean_28_forecast(history: np.ndarray | list[float], horizon: int = 30) -> np.ndarray:
    """Repeat the mean of the final 28 pre-origin observations."""

    values = _validated_history(history)
    if values.size < 28:
        raise ValueError("The 28-day mean baseline requires at least 28 history values.")
    return np.full(_validated_horizon(horizon), values[-28:].mean(), dtype="float64")


def croston_sba_level(history: np.ndarray | list[float], alpha: float = DEFAULT_CROSTON_ALPHA) -> float:
    """Estimate a constant Croston forecast with the SBA adjustment.

    The first positive demand initializes demand size. Its 1-based position
    initializes the interval estimate, so leading active-period zeros contribute
    to intermittency. At every later positive demand, exponential smoothing
    updates demand size and inter-demand interval. SBA returns
    ``(1 - alpha / 2) * demand_size / interval``. All-zero history returns zero.
    """

    values = _validated_history(history)
    if not 0 < alpha <= 1:
        raise ValueError("Croston alpha must be in the interval (0, 1].")
    positive_positions = np.flatnonzero(values > 0)
    if positive_positions.size == 0:
        return 0.0

    first = int(positive_positions[0])
    demand_size = float(values[first])
    interval_estimate = float(first + 1)
    interval_since_positive = 1
    for demand in values[first + 1 :]:
        if demand > 0:
            demand_size += alpha * (float(demand) - demand_size)
            interval_estimate += alpha * (interval_since_positive - interval_estimate)
            interval_since_positive = 1
        else:
            interval_since_positive += 1
    return float((1 - alpha / 2) * demand_size / interval_estimate)


def croston_sba_forecast(
    history: np.ndarray | list[float],
    horizon: int = 30,
    alpha: float = DEFAULT_CROSTON_ALPHA,
) -> np.ndarray:
    """Repeat the pre-origin Croston-SBA level across the horizon."""

    level = croston_sba_level(history, alpha=alpha)
    return np.full(_validated_horizon(horizon), level, dtype="float64")


BASELINE_FUNCTIONS: dict[str, Callable[[np.ndarray | list[float], int], np.ndarray]] = {
    "zero": zero_forecast,
    "last_value": last_value_forecast,
    "seasonal_naive_7": seasonal_naive_7_forecast,
    "mean_28": mean_28_forecast,
    "croston_sba": croston_sba_forecast,
}


def forecast_all_baselines(history: np.ndarray | list[float], horizon: int = 30) -> dict[str, np.ndarray]:
    """Generate all frozen Stage 7 baselines from one pre-origin history."""

    values = _validated_history(history)
    return {name: function(values, horizon) for name, function in BASELINE_FUNCTIONS.items()}
