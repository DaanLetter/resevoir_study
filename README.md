# README – Reservoir Operation Functions

This code implements simplified reservoir operation functions based on Steyaert et al. (2025), *Data derived reservoir operations simulated in a global hydrologic model*. The goal is to represent the main reservoir-operation logic from the HESS paper in a clear Python structure.

The code includes:

- A generic PCR-GLOBWB-style reservoir operation scheme.
- A STARFIT-style data-derived reservoir operation scheme.
- Separate logic for irrigation-like and hydropower-like reservoirs.
- A central timestep function for updating storage.
- Physical constraints to prevent impossible storage or release values.

## Main idea

The reservoir water balance is:

`next_storage = current_storage + inflow + precipitation - evaporation - release`

The important correction is that the **final release** should be used in this balance, not only the first/initial release. The initial release is only a starting point. It is later adjusted depending on the reservoir operation scheme.

## Important unit rule

All storage and flow terms must use consistent units.

For example, if storage is in MCM:

- `current_storage`: MCM
- `storage_capacity`: MCM
- `inflow`: MCM per timestep
- `precipitation`: MCM per timestep
- `evaporation`: MCM per timestep
- `avg_outflow`: MCM per timestep
- `release`: MCM per timestep

If discharge is given in m³/s, it must first be converted to volume per timestep:

`MCM_per_day = discharge_m3s * 86400 / 1e6`

Without this conversion, the water balance is wrong.

## Main mistakes fixed

### 1. Bankfull number was used as if it were a discharge

The original code used:

`BANKFULL_NUMBER = 2.3`

and compared release directly to `2.3`.

This is not correct because `2.3` is a dimensionless ratio, not a discharge. The corrected version calculates:

`bankfull_discharge = BANKFULL_NUMBER * avg_discharge`

So the release is compared to an actual discharge or volume-per-timestep value.

### 2. Initial release was treated as the final release

The first version updated storage using only the initial release:

`storage = storage + inflow + precipitation - evaporation - initial_release`

But the paper first calculates an initial release and then modifies it using the selected reservoir scheme. The corrected version calculates:

1. Initial/preliminary release.
2. Intermediate storage.
3. Final release from the generic or STARFIT scheme.
4. Final storage using the final release.

### 3. Reduction factor was not clipped

The reduction factor is:

`RF = (current_storage - min_storage) / (max_storage - min_storage)`

This can become negative below `min_storage` or greater than 1 above `max_storage`. The corrected version clips it to:

`0 <= RF <= 1`

This avoids impossible scaling.

### 4. Storage bounds were not fully checked

The original check only caught:

`min_storage > max_storage`

but not:

`min_storage == max_storage`

That could cause division by zero. The corrected version uses:

`min_storage >= max_storage`

as an error.

### 5. Equality cases were missing

The first version used strict comparisons like:

`min_storage < current_storage < max_storage`

This misses cases where storage is exactly equal to the lower or upper bound. The corrected version uses safer conditions such as:

`<=`

so release is always defined.

### 6. Flood release could become negative

The original flood release calculation could return negative values. The corrected version uses:

`max(projected_storage - threshold, 0)`

so extra flood release is only added when storage is actually too high.

### 7. Generic flood logic and STARFIT spill logic were mixed

In the generic scheme, extra release is added if storage stays above the generic upper operating level:

`Smax = 0.75 * storage_capacity`

In the STARFIT scheme, extra spill release should only occur if storage exceeds the physical reservoir capacity:

`storage_capacity`

These are different thresholds, so the corrected version separates them.

### 8. Irrigation dead-storage protection was applied too late

The original irrigation function checked the dead-storage rule in an `elif`, meaning it could be skipped. The corrected version first calculates the candidate irrigation release and then limits it so storage cannot fall below:

`0.10 * storage_capacity`

This prevents irrigation release from draining the reservoir below dead storage.

### 9. Release could exceed available water

The corrected version prevents release from being larger than the water available:

`available = current_storage + inflow + precipitation - evaporation`

Then:

`actual_release = min(release, available)`

This avoids negative storage.

### 10. Storage could exceed capacity

The corrected version also prevents storage from becoming larger than physical capacity. If storage exceeds capacity, the excess becomes spill/release.

## Generic reservoir scheme

The generic scheme uses fixed bounds:

`Smin = 0.10 * storage_capacity`

`Smax = 0.75 * storage_capacity`

The release logic is:

- If storage is below `Smin`, release is zero.
- If storage is between `Smin` and `Smax`, release is scaled by the reduction factor.
- If storage is above `Smax`, release increases toward bankfull discharge.
- If projected storage is still above `Smax`, extra water is released.

One important note: the high-storage branch of Equation 4 in the paper is dimensionally unclear because it combines discharge terms with the dimensionless bankfull number. The corrected code uses a dimensionally consistent interpretation by interpolating between average discharge and bankfull discharge.

## STARFIT / data-derived scheme

The STARFIT-style scheme uses seasonal operating bounds instead of fixed 10% and 75% storage limits.

The seasonal bound can be calculated as:

`St = mu + alpha * sin(2πwt) + beta * cos(2πwt)`

In practice, this gives two bounds:

- `flood_bound`: upper operating bound
- `conservation_bound`: lower operating bound

These bounds can change through the year.

## Irrigation-like reservoirs

Irrigation-like reservoirs mainly try to meet downstream demand.

For these reservoirs:

`min_storage = 0.10 * storage_capacity`

`max_storage = flood_bound`

The reservoir may release water to meet demand, but it should not release below the dead-storage level.

## Hydropower-like reservoirs

Hydropower-like reservoirs mainly try to keep storage stable.

For these reservoirs:

`min_storage = conservation_bound`

`max_storage = flood_bound`

The reservoir may meet downstream demand, but not if doing so would draw storage below the conservation bound.

## Main function

The main function is:

`reservoir_step()`

It performs one full timestep:

1. Calculate available water.
2. Calculate preliminary release.
3. Calculate intermediate storage.
4. Calculate final release using the selected operation scheme.
5. Apply the final water balance.
6. Return next storage and actual release.

Example:

```python
next_storage, release = reservoir_step(
    current_storage=50.0,
    storage_capacity=100.0,
    avg_outflow=1.0,
    operation="generic",
    inflow=0.8,
    precipitation=0.0,
    evaporation=0.05,
)
```
Remaining limitations

This is still a simplified standalone version. It is not a full PCR-GLOBWB implementation.

Important limitations:

STARFIT parameters are not estimated in this file.
Random forest extrapolation is not included.
Downstream command-area demand calculation is not included.
Precipitation and evaporation must already be converted to volume units.
The generic high-storage equation is interpreted in a dimensionally consistent way, not copied literally.
The hydropower release logic should still be validated against observed reservoir data.
Suggested tests

Before using the code for serious model results, test:

Storage exactly at Smin, Smax, and storage_capacity.
A reservoir with no water.
A reservoir with very high inflow that should spill.
An irrigation reservoir with high demand and low storage.
A hydropower reservoir close to its conservation bound.
Whether the water balance closes at every timestep.
Summary

The original code followed the paper’s equations reasonably well, but several variables were mixed up. The main problems were:

bankfull number was treated as a discharge,
initial release was treated as final release,
units were not handled explicitly,
reduction factor could become invalid,
flood release could become negative,
storage and release were not physically constrained,
irrigation and hydropower logic were not clearly separated.

The corrected version makes the reservoir logic more explicit, safer, and easier to test.
