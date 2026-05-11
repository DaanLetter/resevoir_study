"""Validation helpers for STARFIT/RF bounds and point reservoir runs.

The functions here are intentionally data-frame oriented so they can be used
from exploratory notebooks, Daan's current RF script, or a later cleaned
pipeline without locking the project into one preprocessing workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


STARFIT_PARAMETER_NAMES = [
    "flood_p1",
    "flood_p2",
    "flood_p3",
    "flood_max",
    "flood_min",
    "conserve_p1",
    "conserve_p2",
    "conserve_p3",
    "conserve_max",
    "conserve_min",
]


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _valid_pair(observed: Iterable[float], simulated: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    obs = _as_array(observed)
    sim = _as_array(simulated)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def rmse(observed: Iterable[float], simulated: Iterable[float]) -> float:
    obs, sim = _valid_pair(observed, simulated)
    if obs.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


def mae(observed: Iterable[float], simulated: Iterable[float]) -> float:
    obs, sim = _valid_pair(observed, simulated)
    if obs.size == 0:
        return np.nan
    return float(np.mean(np.abs(sim - obs)))


def bias(observed: Iterable[float], simulated: Iterable[float]) -> float:
    obs, sim = _valid_pair(observed, simulated)
    if obs.size == 0:
        return np.nan
    return float(np.mean(sim - obs))


def nrmse(observed: Iterable[float], simulated: Iterable[float]) -> float:
    """RMSE normalized by observed standard deviation, as in Turner et al."""
    obs, sim = _valid_pair(observed, simulated)
    if obs.size == 0:
        return np.nan
    obs_std = np.std(obs)
    if obs_std == 0:
        return np.nan
    return float(rmse(obs, sim) / obs_std)


def box_cox_transform(values: Iterable[float], lambda_value: float = 0.3, epsilon: float = 1e-6) -> np.ndarray:
    """Box-Cox style transform used for release errors in Turner et al."""
    arr = np.maximum(_as_array(values), 0) + epsilon
    if lambda_value == 0:
        return np.log(arr)
    return (arr**lambda_value - 1) / lambda_value


def transformed_nrmse(
    observed: Iterable[float],
    simulated: Iterable[float],
    lambda_value: float = 0.3,
) -> float:
    """nRMSE after Box-Cox transforming observed and simulated releases."""
    obs, sim = _valid_pair(observed, simulated)
    if obs.size == 0:
        return np.nan
    obs_t = box_cox_transform(obs, lambda_value=lambda_value)
    sim_t = box_cox_transform(sim, lambda_value=lambda_value)
    obs_std = np.std(obs_t)
    if obs_std == 0:
        return np.nan
    return float(rmse(obs_t, sim_t) / obs_std)


def high_storage_nrmse(
    observed_storage: Iterable[float],
    simulated_storage: Iterable[float],
    flood_bound: Iterable[float] | float,
) -> float:
    """nRMSE for periods where observed storage sits above the upper bound."""
    obs = _as_array(observed_storage)
    sim = _as_array(simulated_storage)
    threshold = _as_array(flood_bound)
    if threshold.ndim == 0:
        threshold = np.full(obs.shape, float(threshold))
    mask = np.isfinite(obs) & np.isfinite(sim) & np.isfinite(threshold) & (obs > threshold)
    if not np.any(mask):
        return np.nan
    return nrmse(obs[mask], sim[mask])


def pearson_r(observed: Iterable[float], simulated: Iterable[float]) -> float:
    obs, sim = _valid_pair(observed, simulated)
    if obs.size < 2 or np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    return float(np.corrcoef(obs, sim)[0, 1])


def spearman_r(observed: Iterable[float], simulated: Iterable[float]) -> float:
    obs, sim = _valid_pair(observed, simulated)
    if obs.size < 2:
        return np.nan
    return pearson_r(pd.Series(obs).rank(), pd.Series(sim).rank())


def kge(observed: Iterable[float], simulated: Iterable[float]) -> float:
    """Kling-Gupta efficiency for streamflow or storage time series."""
    obs, sim = _valid_pair(observed, simulated)
    if obs.size < 2 or np.mean(obs) == 0 or np.std(obs) == 0:
        return np.nan
    r = pearson_r(obs, sim)
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metric_summary(observed: Iterable[float], simulated: Iterable[float]) -> dict[str, float]:
    return {
        "rmse": rmse(observed, simulated),
        "mae": mae(observed, simulated),
        "bias": bias(observed, simulated),
        "nrmse": nrmse(observed, simulated),
        "pearson_r": pearson_r(observed, simulated),
        "spearman_r": spearman_r(observed, simulated),
        "kge": kge(observed, simulated),
    }


def starfit_parameter_metrics(
    observed_params: pd.DataFrame | np.ndarray,
    predicted_params: pd.DataFrame | np.ndarray,
    parameter_names: list[str] | None = None,
) -> pd.DataFrame:
    """Validate RF-predicted STARFIT parameters against held-out labels."""
    obs = pd.DataFrame(observed_params)
    pred = pd.DataFrame(predicted_params)
    if parameter_names is None:
        parameter_names = STARFIT_PARAMETER_NAMES[: obs.shape[1]]
    obs.columns = parameter_names
    pred.columns = parameter_names

    rows = []
    for name in parameter_names:
        row = {"parameter": name}
        row.update(metric_summary(obs[name], pred[name]))
        rows.append(row)
    return pd.DataFrame(rows)


def bound_curve_metrics(
    observed_curves: pd.DataFrame,
    predicted_curves: pd.DataFrame,
    observed_flood_col: str = "flood",
    observed_conservation_col: str = "conservation",
    predicted_flood_col: str = "flood",
    predicted_conservation_col: str = "conservation",
) -> pd.DataFrame:
    """Validate reconstructed weekly flood/conservation curves."""
    comparisons = {
        "flood": (
            observed_curves[observed_flood_col],
            predicted_curves[predicted_flood_col],
        ),
        "conservation": (
            observed_curves[observed_conservation_col],
            predicted_curves[predicted_conservation_col],
        ),
    }

    rows = []
    for bound, (obs, pred) in comparisons.items():
        row = {"bound": bound}
        row.update(metric_summary(obs, pred))
        rows.append(row)
    return pd.DataFrame(rows)


def bound_sanity_checks(
    bounds: pd.DataFrame,
    flood_col: str = "flood",
    conservation_col: str = "conservation",
    min_active_zone_fraction: float = 0.05,
) -> Mapping[str, float]:
    """Check physical plausibility of weekly operating bounds.

    Bounds are assumed to be fractions or percentages consistently. For percent
    bounds, pass min_active_zone_fraction=5.
    """
    flood = _as_array(bounds[flood_col])
    conservation = _as_array(bounds[conservation_col])
    active_zone = flood - conservation
    return {
        "weeks": float(len(bounds)),
        "flood_below_conservation_weeks": float(np.sum(active_zone < 0)),
        "too_narrow_active_zone_weeks": float(np.sum(active_zone < min_active_zone_fraction)),
        "min_active_zone": float(np.nanmin(active_zone)) if active_zone.size else np.nan,
        "mean_active_zone": float(np.nanmean(active_zone)) if active_zone.size else np.nan,
    }


def storage_position_counts(
    storage: Iterable[float],
    flood_bound: Iterable[float],
    conservation_bound: Iterable[float],
) -> Mapping[str, float]:
    """Count when storage is above, within, or below STARFIT-like bounds."""
    storage_arr = _as_array(storage)
    flood_arr = _as_array(flood_bound)
    conservation_arr = _as_array(conservation_bound)
    mask = np.isfinite(storage_arr) & np.isfinite(flood_arr) & np.isfinite(conservation_arr)
    storage_arr = storage_arr[mask]
    flood_arr = flood_arr[mask]
    conservation_arr = conservation_arr[mask]
    total = storage_arr.size
    if total == 0:
        return {"n": 0.0, "above": np.nan, "within": np.nan, "below": np.nan}

    above = np.sum(storage_arr > flood_arr)
    below = np.sum(storage_arr < conservation_arr)
    within = total - above - below
    return {
        "n": float(total),
        "above_fraction": float(above / total),
        "within_fraction": float(within / total),
        "below_fraction": float(below / total),
    }


@dataclass(frozen=True)
class PointModelColumns:
    observed_storage: str = "observed_storage"
    simulated_storage: str = "simulated_storage"
    observed_release: str = "observed_release"
    simulated_release: str = "simulated_release"
    inflow: str = "inflow"
    flood_bound: str = "flood"
    conservation_bound: str = "conservation"


def point_model_validation(
    data: pd.DataFrame,
    columns: PointModelColumns = PointModelColumns(),
) -> dict[str, pd.DataFrame | Mapping[str, float]]:
    """Validate point-reservoir storage/release simulation.

    This follows the Turner split between release with observed storage and
    fully simulated storage/release, but leaves the simulation itself outside
    this helper.
    """
    storage_metrics = metric_summary(data[columns.observed_storage], data[columns.simulated_storage])
    release_metrics = metric_summary(data[columns.observed_release], data[columns.simulated_release])
    position = storage_position_counts(
        data[columns.simulated_storage],
        data[columns.flood_bound],
        data[columns.conservation_bound],
    )
    return {
        "storage_metrics": storage_metrics,
        "release_metrics": release_metrics,
        "storage_position": position,
    }


def water_balance_residual(
    previous_storage: Iterable[float],
    inflow: Iterable[float],
    release: Iterable[float],
    next_storage: Iterable[float],
    precipitation: Iterable[float] | float = 0.0,
    evaporation: Iterable[float] | float = 0.0,
) -> np.ndarray:
    """Return residual for next_storage = previous + inflow + precip - evap - release."""
    return (
        _as_array(next_storage)
        - _as_array(previous_storage)
        - _as_array(inflow)
        - _as_array(precipitation)
        + _as_array(evaporation)
        + _as_array(release)
    )
