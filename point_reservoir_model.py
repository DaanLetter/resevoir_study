"""Point-reservoir simulation utilities for model validation.

This module intentionally stays independent of PCRaster/PCR-GLOBWB. It uses the
same ingredients as the HESS 2025 point logic: observed inflow forcing, weekly
storage bounds, a simple environmental-flow proxy, and separate
irrigation-like/hydropower-like release behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86_400
CMS_TO_MCM_PER_DAY = SECONDS_PER_DAY / 1_000_000
BANKFULL_NUMBER = 2.3


@dataclass(frozen=True)
class ReservoirConfig:
    dam_id: int
    capacity_mcm: float
    use_category: str
    bankfull_number: float = BANKFULL_NUMBER
    environmental_flow_fraction: float = 0.10


def cms_to_mcm_per_day(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce") * CMS_TO_MCM_PER_DAY


def add_epiweek(dates: pd.Series) -> pd.Series:
    weeks = pd.to_datetime(dates).dt.isocalendar().week.astype(int)
    return weeks.clip(upper=52)


def normalize_weekly_bounds(bounds: pd.DataFrame, capacity_mcm: float, min_active_zone_fraction: float = 0.05) -> pd.DataFrame:
    """Convert weekly percent bounds to physically valid MCM bounds."""
    required = {"epiweek", "flood_pct", "conservation_pct"}
    missing = required.difference(bounds.columns)
    if missing:
        raise ValueError(f"Missing weekly-bound columns: {sorted(missing)}")
    if capacity_mcm <= 0:
        raise ValueError("capacity_mcm must be positive")

    frame = bounds[["epiweek", "flood_pct", "conservation_pct"]].copy()
    frame["epiweek"] = frame["epiweek"].astype(int)
    if set(frame["epiweek"]) != set(range(1, 53)):
        missing_weeks = sorted(set(range(1, 53)).difference(frame["epiweek"]))
        raise ValueError(f"Weekly bounds must contain epiweeks 1-52; missing {missing_weeks}")

    dead_storage = 0.10 * capacity_mcm
    min_active_zone = min_active_zone_fraction * capacity_mcm
    max_conservation = max(capacity_mcm - min_active_zone, dead_storage)

    frame["conservation_mcm"] = (
        pd.to_numeric(frame["conservation_pct"], errors="coerce") / 100 * capacity_mcm
    ).clip(lower=dead_storage, upper=max_conservation)
    frame["flood_mcm"] = pd.to_numeric(frame["flood_pct"], errors="coerce") / 100 * capacity_mcm
    frame["flood_mcm"] = np.maximum(frame["flood_mcm"], frame["conservation_mcm"] + min_active_zone)
    frame["flood_mcm"] = frame["flood_mcm"].clip(upper=capacity_mcm)
    frame["conservation_mcm"] = np.minimum(frame["conservation_mcm"], frame["flood_mcm"] - min_active_zone)
    frame["conservation_mcm"] = frame["conservation_mcm"].clip(lower=0)
    return frame.sort_values("epiweek").reset_index(drop=True)


def weekly_demand_climatology(training_data: pd.DataFrame, fallback: float) -> pd.Series:
    """Average observed outflow by epiweek, used as the v1 demand proxy."""
    observed = training_data.dropna(subset=["observed_release_mcm_day"]).copy()
    if observed.empty:
        return pd.Series({week: fallback for week in range(1, 53)}, dtype=float)
    climatology = observed.groupby("epiweek")["observed_release_mcm_day"].mean()
    fallback_value = float(climatology.mean()) if np.isfinite(climatology.mean()) else float(fallback)
    return pd.Series({week: float(climatology.get(week, fallback_value)) for week in range(1, 53)})


def release_decision(
    storage_state_mcm: float,
    inflow_mcm_day: float,
    average_inflow_mcm_day: float,
    demand_mcm_day: float,
    environmental_flow_mcm_day: float,
    flood_bound_mcm: float,
    conservation_bound_mcm: float,
    config: ReservoirConfig,
) -> dict[str, float]:
    """One daily release decision and mass-balance update."""
    inflow = max(float(inflow_mcm_day), 0.0)
    current_storage = max(float(storage_state_mcm), 0.0) + inflow
    capacity = float(config.capacity_mcm)
    dead_storage = 0.10 * capacity
    flood = min(max(float(flood_bound_mcm), dead_storage), capacity)
    conservation = min(max(float(conservation_bound_mcm), dead_storage), flood)
    active_zone = max(flood - conservation, 1e-9)

    reduction_factor = max((current_storage - conservation) / active_zone * config.bankfull_number, 0.0)
    if reduction_factor > config.bankfull_number and current_storage > flood:
        reduction_factor = config.bankfull_number

    release = reduction_factor * max(float(average_inflow_mcm_day), 0.0)

    if current_storage - release > capacity:
        release += current_storage - release - capacity

    demand = max(float(demand_mcm_day), 0.0)
    if release < demand and demand > 0:
        if config.use_category == "irrigation_like":
            if current_storage >= conservation:
                release = demand
            elif current_storage > dead_storage:
                demand_factor = (current_storage - dead_storage) / max(conservation - dead_storage, 1e-9)
                release = demand * np.clip(demand_factor, 0.0, 1.0)
        else:
            if current_storage > conservation:
                release = min(demand, current_storage - conservation)

    if release < environmental_flow_mcm_day and current_storage - environmental_flow_mcm_day > 0:
        release = environmental_flow_mcm_day

    if current_storage < dead_storage:
        release = 0.0

    release = min(max(float(release), 0.0), current_storage)
    simulated_storage = max(current_storage - release, 0.0)
    return {
        "current_storage_before_release_mcm": current_storage,
        "simulated_release_mcm_day": release,
        "simulated_storage_mcm": simulated_storage,
        "reduction_factor": reduction_factor,
    }


def simulate_reservoir(
    daily_data: pd.DataFrame,
    weekly_bounds: pd.DataFrame,
    demand_by_week: pd.Series,
    average_inflow_mcm_day: float,
    config: ReservoirConfig,
    mode: str,
) -> pd.DataFrame:
    """Run either RS-OBS or RS-SIM daily point-reservoir simulation."""
    if mode not in {"RS_OBS", "RS_SIM"}:
        raise ValueError("mode must be RS_OBS or RS_SIM")

    bounds = normalize_weekly_bounds(weekly_bounds, config.capacity_mcm)
    data = daily_data.merge(
        bounds[["epiweek", "flood_mcm", "conservation_mcm"]],
        on="epiweek",
        how="left",
    ).sort_values("date")
    env_flow = config.environmental_flow_fraction * max(float(average_inflow_mcm_day), 0.0)

    previous_storage = float(data["observed_storage_mcm"].dropna().iloc[0])
    rows = []
    for row in data.itertuples(index=False):
        if mode == "RS_OBS":
            storage_state = row.observed_storage_mcm
        else:
            storage_state = previous_storage

        decision = release_decision(
            storage_state_mcm=storage_state,
            inflow_mcm_day=row.inflow_mcm_day,
            average_inflow_mcm_day=average_inflow_mcm_day,
            demand_mcm_day=float(demand_by_week.get(int(row.epiweek), 0.0)),
            environmental_flow_mcm_day=env_flow,
            flood_bound_mcm=row.flood_mcm,
            conservation_bound_mcm=row.conservation_mcm,
            config=config,
        )
        if mode == "RS_SIM":
            previous_storage = decision["simulated_storage_mcm"]

        rows.append(
            {
                "date": row.date,
                "epiweek": int(row.epiweek),
                "observed_storage_mcm": row.observed_storage_mcm,
                "observed_release_mcm_day": row.observed_release_mcm_day,
                "inflow_mcm_day": row.inflow_mcm_day,
                "flood_bound_mcm": row.flood_mcm,
                "conservation_bound_mcm": row.conservation_mcm,
                **decision,
            }
        )

    return pd.DataFrame(rows)
