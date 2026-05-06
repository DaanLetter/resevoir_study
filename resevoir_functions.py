import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional


# Based on Steyaert et al. (2025):
# "Data derived reservoir operations simulated in a global hydrologic model"

BANKFULL_NUMBER = 2.3
SECONDS_PER_DAY = 24 * 60 * 60


# ---------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------

def discharge_to_volume(discharge_m3s: float, dt_seconds: float = SECONDS_PER_DAY) -> float:
    """
    Convert discharge [m3/s] to volume [m3 per timestep].

    Use this if your storage is in m3.
    If your storage is in MCM, divide this result by 1e6.
    """
    return discharge_m3s * dt_seconds


def discharge_to_mcm(discharge_m3s: float, dt_seconds: float = SECONDS_PER_DAY) -> float:
    """
    Convert discharge [m3/s] to volume [MCM per timestep].
    """
    return discharge_m3s * dt_seconds / 1e6


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def clip_nonnegative(value: float) -> float:
    return max(float(value), 0.0)


def bankfull_discharge_from_avg(
    avg_discharge: float,
    bankfull_number: float = BANKFULL_NUMBER,
) -> float:
    """
    Bankfull discharge/volume estimate.

    If avg_discharge is in m3/s, result is m3/s.
    If avg_discharge is in MCM/timestep, result is MCM/timestep.
    """
    return bankfull_number * avg_discharge


def validate_storage_bounds(min_storage: float, max_storage: float) -> None:
    if min_storage >= max_storage:
        raise ValueError(
            f"min_storage ({min_storage:.3f}) must be smaller than "
            f"max_storage ({max_storage:.3f})."
        )


def reduction_factor(
    current_storage: float,
    min_storage: float,
    max_storage: float,
    clip: bool = True,
) -> float:
    """
    Eq. 3:
        RF = (Sc - Smin) / (Smax - Smin)

    By default, clipped to [0, 1] because outside the active zone
    the raw equation can become negative or larger than 1.
    """
    validate_storage_bounds(min_storage, max_storage)

    rf = (current_storage - min_storage) / (max_storage - min_storage)

    if clip:
        rf = np.clip(rf, 0.0, 1.0)

    return float(rf)


def water_before_release(
    current_storage: float,
    inflow: float = 0.0,
    precipitation: float = 0.0,
    evaporation: float = 0.0,
) -> float:
    """
    Storage available before release at the current timestep.

    All terms must be in the same unit:
    e.g. MCM/timestep for flows and MCM for storage.
    """
    return max(current_storage + inflow + precipitation - evaporation, 0.0)


def apply_water_balance(
    current_storage: float,
    release: float,
    inflow: float = 0.0,
    precipitation: float = 0.0,
    evaporation: float = 0.0,
    storage_capacity: Optional[float] = None,
) -> tuple[float, float]:
    """
    Eq. 2:
        Sc(t+1) = Sc(t) + I(t) + P(t) - E(t) - R(t)

    Returns:
        next_storage, actual_release

    Includes final physical constraints:
    - release cannot be negative
    - release cannot exceed available water
    - storage cannot exceed storage_capacity if provided
    """
    available = water_before_release(
        current_storage=current_storage,
        inflow=inflow,
        precipitation=precipitation,
        evaporation=evaporation,
    )

    actual_release = min(max(release, 0.0), available)
    next_storage = available - actual_release

    if storage_capacity is not None and next_storage > storage_capacity:
        spill = next_storage - storage_capacity
        actual_release += spill
        next_storage = storage_capacity

    return next_storage, actual_release


# ---------------------------------------------------------------------
# Eq. 1: Initial release
# ---------------------------------------------------------------------

def initial_release(
    current_storage: float,
    min_storage: float,
    max_storage: float,
    avg_outflow: float,
    bankfull_discharge: Optional[float] = None,
    bankfull_number: float = BANKFULL_NUMBER,
) -> float:
    """
    Eq. 1:
        Ri = RF * Ravg

    The paper states that Ri cannot exceed the downstream bankfull
    threshold. Here bankfull_discharge is treated in the same unit as
    avg_outflow.
    """
    rf = reduction_factor(current_storage, min_storage, max_storage, clip=True)

    release = rf * avg_outflow

    if bankfull_discharge is None:
        bankfull_discharge = bankfull_discharge_from_avg(
            avg_discharge=avg_outflow,
            bankfull_number=bankfull_number,
        )

    release = min(release, bankfull_discharge)

    return clip_nonnegative(release)


# ---------------------------------------------------------------------
# Eq. 4: Generic PCR-GLOBWB-style reservoir release
# ---------------------------------------------------------------------

def generic_release(
    decision_storage: float,
    avg_discharge: float,
    storage_capacity: float,
    bankfull_discharge: Optional[float] = None,
    bankfull_number: float = BANKFULL_NUMBER,
    min_fraction: float = 0.10,
    max_fraction: float = 0.75,
    available_before_release: Optional[float] = None,
) -> float:
    """
    Eq. 4: generic reservoir scheme.

    Active zone:
        Smin = 10% of capacity
        Smax = 75% of capacity

    Below Smin:
        R = 0

    Between Smin and Smax:
        R = RF * Qavg

    Above Smax:
        release moves toward bankfull behavior.

    Important correction:
    The paper prints '+ B' in Eq. 4, where B is the bankfull number.
    Since B is dimensionless, this implementation uses a dimensionally
    consistent interpolation between avg_discharge and bankfull_discharge.
    """
    if storage_capacity <= 0:
        raise ValueError("storage_capacity must be positive.")

    min_storage = min_fraction * storage_capacity
    max_storage = max_fraction * storage_capacity
    validate_storage_bounds(min_storage, max_storage)

    if bankfull_discharge is None:
        bankfull_discharge = bankfull_discharge_from_avg(
            avg_discharge=avg_discharge,
            bankfull_number=bankfull_number,
        )

    if decision_storage <= min_storage:
        release = 0.0

    elif decision_storage <= max_storage:
        rf = reduction_factor(decision_storage, min_storage, max_storage, clip=True)
        release = rf * avg_discharge

    else:
        # Dimensionally consistent version of the high-storage branch:
        # smoothly increases from avg_discharge at Smax
        # to bankfull_discharge at storage_capacity.
        high_storage_fraction = (
            (decision_storage - max_storage)
            / (storage_capacity - max_storage)
        )
        high_storage_fraction = np.clip(high_storage_fraction, 0.0, 1.0)

        release = avg_discharge + high_storage_fraction * (
            bankfull_discharge - avg_discharge
        )

    # Flood drawdown condition from the generic scheme:
    # if projected storage remains above Smax, release extra water
    # to bring storage back to Smax.
    source_storage = (
        decision_storage
        if available_before_release is None
        else available_before_release
    )

    projected_storage = source_storage - release

    if projected_storage > max_storage:
        release += projected_storage - max_storage

    return clip_nonnegative(release)


# ---------------------------------------------------------------------
# Eq. 5 / Eq. 10: STARFIT-style seasonal bounds
# ---------------------------------------------------------------------

def starfit_bound(
    week: int,
    mu: float,
    alpha: float,
    beta: float,
    lower_limit: Optional[float] = None,
    upper_limit: Optional[float] = None,
    period_weeks: int = 52,
) -> float:
    """
    Eq. 5:
        St = mu + alpha * sin(2π ω t) + beta * cos(2π ω t)

    This gives one seasonal STARFIT bound, e.g. flood or conservation.
    """
    omega = 1 / period_weeks

    value = (
        mu
        + alpha * np.sin(2 * np.pi * omega * week)
        + beta * np.cos(2 * np.pi * omega * week)
    )

    if lower_limit is not None:
        value = max(value, lower_limit)

    if upper_limit is not None:
        value = min(value, upper_limit)

    return float(value)


def enforce_min_active_zone(
    flood_bound: float,
    conservation_bound: float,
    min_gap: float,
) -> tuple[float, float]:
    """
    Eq. 10 idea:
    Ensure that flood_bound - conservation_bound is at least min_gap.

    Returns:
        flood_bound, adjusted_conservation_bound
    """
    adjusted_conservation = min(conservation_bound, flood_bound - min_gap)
    return flood_bound, adjusted_conservation


# ---------------------------------------------------------------------
# Eq. 7 and Eq. 8: Hydropower-like release
# ---------------------------------------------------------------------

def initial_hydropower_release(
    preliminary_release: float,
    demand: float,
    current_storage: float,
    min_storage: float,
    max_storage: float,
    bankfull_number: float = BANKFULL_NUMBER,
) -> float:
    """
    Eq. 7:
        Rhi = D * RF / B, if R < D
        Rhi = R,          if R > D

    Here:
    - preliminary_release corresponds to R
    - demand corresponds to D
    """
    rf = reduction_factor(current_storage, min_storage, max_storage, clip=True)

    if preliminary_release < demand:
        release = demand * rf / bankfull_number
    else:
        release = preliminary_release

    return clip_nonnegative(release)


def hydropower_release(
    current_storage: float,
    min_storage: float,
    max_storage: float,
    preliminary_release: float,
    demand: float,
    bankfull_number: float = BANKFULL_NUMBER,
) -> float:
    """
    Eq. 8: hydropower-like release.

    Hydropower-like dams try to keep storage stable and avoid dropping
    below the conservation bound.
    """
    r_hi = initial_hydropower_release(
        preliminary_release=preliminary_release,
        demand=demand,
        current_storage=current_storage,
        min_storage=min_storage,
        max_storage=max_storage,
        bankfull_number=bankfull_number,
    )

    if min_storage < current_storage < max_storage:
        if current_storage - r_hi > min_storage:
            release = current_storage - r_hi
        else:
            release = max(current_storage - min_storage, 0.0)
    else:
        release = 0.0

    return clip_nonnegative(release)


# ---------------------------------------------------------------------
# Eq. 9: Irrigation-like release
# ---------------------------------------------------------------------

def irrigation_release(
    current_storage: float,
    min_storage: float,
    max_storage: float,
    preliminary_release: float,
    demand: float,
    storage_capacity: float,
    dead_storage_fraction: float = 0.10,
) -> float:
    """
    Eq. 9: irrigation-like release.

    Irrigation-like reservoirs prioritize downstream demand, but should
    not release water below the dead-storage zone.
    """
    dead_storage = dead_storage_fraction * storage_capacity

    if min_storage < current_storage < max_storage:
        if preliminary_release >= demand:
            release = preliminary_release
        else:
            rf = reduction_factor(current_storage, min_storage, max_storage, clip=True)
            release = rf * demand
    else:
        release = 0.0

    # Dead-storage protection should override the selected release.
    max_possible_without_dead_storage = max(current_storage - dead_storage, 0.0)
    release = min(release, max_possible_without_dead_storage)

    return clip_nonnegative(release)


# ---------------------------------------------------------------------
# Eq. 6: STARFIT / data-derived reservoir release
# ---------------------------------------------------------------------

def starfit_release(
    decision_storage: float,
    storage_capacity: float,
    preliminary_release: float,
    avg_outflow: float,
    flood_bound: float,
    conservation_bound: float,
    demand: float = 0.0,
    env_flow: float = 0.0,
    use: Literal["irrigation", "hydropower"] = "irrigation",
    bankfull_number: float = BANKFULL_NUMBER,
    available_before_release: Optional[float] = None,
) -> float:
    """
    Eq. 6: data-derived STARFIT-style release.

    For hydropower-like dams:
        lower bound = STARFIT conservation bound
        upper bound = STARFIT flood bound

    For irrigation-like dams:
        lower bound = 10% of capacity
        upper bound = STARFIT flood bound

    The paper distinguishes these because irrigation-like dams mainly
    meet downstream demand, while hydropower-like dams try to keep
    storage stable.
    """
    if storage_capacity <= 0:
        raise ValueError("storage_capacity must be positive.")

    # Ensure STARFIT bounds are physically possible.
    flood_bound = np.clip(flood_bound, 0.05 * storage_capacity, storage_capacity)
    conservation_bound = np.clip(conservation_bound, 0.0, storage_capacity)

    # Ensure at least a 5% capacity active zone between conservation and flood.
    flood_bound, conservation_bound = enforce_min_active_zone(
        flood_bound=flood_bound,
        conservation_bound=conservation_bound,
        min_gap=0.05 * storage_capacity,
    )

    if use == "irrigation":
        min_storage = 0.10 * storage_capacity
        max_storage = flood_bound
    elif use == "hydropower":
        min_storage = conservation_bound
        max_storage = flood_bound
    else:
        raise ValueError("use must be either 'irrigation' or 'hydropower'.")

    validate_storage_bounds(min_storage, max_storage)

    if use == "irrigation":
        active_release = irrigation_release(
            current_storage=decision_storage,
            min_storage=min_storage,
            max_storage=max_storage,
            preliminary_release=preliminary_release,
            demand=demand,
            storage_capacity=storage_capacity,
        )
    else:
        active_release = hydropower_release(
            current_storage=decision_storage,
            min_storage=min_storage,
            max_storage=max_storage,
            preliminary_release=preliminary_release,
            demand=demand,
            bankfull_number=bankfull_number,
        )

    source_storage = (
        decision_storage
        if available_before_release is None
        else available_before_release
    )

    # Flood/spill condition:
    # If projected storage exceeds capacity, add enough release to avoid overtopping.
    projected_storage = source_storage - active_release
    flood_release = max(projected_storage - storage_capacity, 0.0)

    if flood_release > 0:
        release = active_release + flood_release

    elif min_storage < decision_storage < max_storage:
        release = active_release

    elif decision_storage <= min_storage:
        # Environmental flow only if there is enough water to release it.
        if active_release < env_flow and source_storage - env_flow > 0:
            release = env_flow
        else:
            release = 0.0

    else:
        release = active_release

    return clip_nonnegative(release)


# ---------------------------------------------------------------------
# One central timestep function
# ---------------------------------------------------------------------

def reservoir_step(
    current_storage: float,
    storage_capacity: float,
    avg_outflow: float,
    operation: Literal["generic", "starfit"] = "generic",
    use: Literal["irrigation", "hydropower"] = "irrigation",
    inflow: float = 0.0,
    precipitation: float = 0.0,
    evaporation: float = 0.0,
    demand: float = 0.0,
    env_flow: float = 0.0,
    flood_bound: Optional[float] = None,
    conservation_bound: Optional[float] = None,
    bankfull_discharge: Optional[float] = None,
    bankfull_number: float = BANKFULL_NUMBER,
) -> tuple[float, float]:
    """
    Complete one-timestep reservoir update.

    Returns:
        next_storage, actual_release

    All storage and flux variables must be in consistent units:
    - storage_capacity, current_storage: e.g. MCM
    - inflow, precipitation, evaporation, avg_outflow, demand, env_flow:
      e.g. MCM per timestep
    """
    if storage_capacity <= 0:
        raise ValueError("storage_capacity must be positive.")

    available = water_before_release(
        current_storage=current_storage,
        inflow=inflow,
        precipitation=precipitation,
        evaporation=evaporation,
    )

    if bankfull_discharge is None:
        bankfull_discharge = bankfull_discharge_from_avg(
            avg_discharge=avg_outflow,
            bankfull_number=bankfull_number,
        )

    # -------------------------------------------------------------
    # Step 1: preliminary/initial release, Eq. 1
    # -------------------------------------------------------------
    if operation == "generic":
        prelim_min_storage = 0.10 * storage_capacity
        prelim_max_storage = 0.75 * storage_capacity

    elif operation == "starfit":
        if flood_bound is None or conservation_bound is None:
            raise ValueError(
                "For operation='starfit', provide flood_bound and conservation_bound."
            )

        if use == "irrigation":
            prelim_min_storage = 0.10 * storage_capacity
            prelim_max_storage = flood_bound
        else:
            prelim_min_storage = conservation_bound
            prelim_max_storage = flood_bound

    else:
        raise ValueError("operation must be either 'generic' or 'starfit'.")

    preliminary_release = initial_release(
        current_storage=current_storage,
        min_storage=prelim_min_storage,
        max_storage=prelim_max_storage,
        avg_outflow=avg_outflow,
        bankfull_discharge=bankfull_discharge,
        bankfull_number=bankfull_number,
    )

    # -------------------------------------------------------------
    # Step 2: intermediate storage after initial release, Eq. 2
    # This is used as the decision storage for the selected scheme.
    # -------------------------------------------------------------
    decision_storage, _ = apply_water_balance(
        current_storage=current_storage,
        release=preliminary_release,
        inflow=inflow,
        precipitation=precipitation,
        evaporation=evaporation,
        storage_capacity=None,
    )

    # -------------------------------------------------------------
    # Step 3: final release from the selected operation scheme
    # -------------------------------------------------------------
    if operation == "generic":
        release = generic_release(
            decision_storage=decision_storage,
            avg_discharge=avg_outflow,
            storage_capacity=storage_capacity,
            bankfull_discharge=bankfull_discharge,
            bankfull_number=bankfull_number,
            available_before_release=available,
        )

    else:
        release = starfit_release(
            decision_storage=decision_storage,
            storage_capacity=storage_capacity,
            preliminary_release=preliminary_release,
            avg_outflow=avg_outflow,
            flood_bound=flood_bound,
            conservation_bound=conservation_bound,
            demand=demand,
            env_flow=env_flow,
            use=use,
            bankfull_number=bankfull_number,
            available_before_release=available,
        )

    # -------------------------------------------------------------
    # Step 4: final water balance using the final release
    # -------------------------------------------------------------
    next_storage, actual_release = apply_water_balance(
        current_storage=current_storage,
        release=release,
        inflow=inflow,
        precipitation=precipitation,
        evaporation=evaporation,
        storage_capacity=storage_capacity,
    )

    return next_storage, actual_release


# ---------------------------------------------------------------------
# Example simulation
# ---------------------------------------------------------------------

def main():
    timesteps = np.arange(0, 100, 1)

    storage_capacity = 100.0
    current_storage = 50.0

    # All values are in "storage units per timestep".
    # For example: MCM and MCM/day.
    avg_outflow = 1.0
    inflow = 0.8
    precipitation = 0.0
    evaporation = 0.05

    storage_list = []
    release_list = []

    for _ in timesteps:
        current_storage, release = reservoir_step(
            current_storage=current_storage,
            storage_capacity=storage_capacity,
            avg_outflow=avg_outflow,
            operation="generic",
            inflow=inflow,
            precipitation=precipitation,
            evaporation=evaporation,
        )

        storage_list.append(current_storage)
        release_list.append(release)

    plt.figure()
    plt.plot(timesteps, storage_list, label="Storage")
    plt.plot(timesteps, release_list, label="Release")
    plt.xlabel("Timestep")
    plt.ylabel("Volume per timestep / Storage")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
