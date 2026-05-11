# Validation Plan

This project should keep the feature preprocessing flexible, but use a shared
validation layer so model versions can be compared consistently.

## 1. Random forest validation

Question: can static reservoir, hydroclimate, and socioeconomic predictors
recover STARFIT-like seasonal bounds?

Recommended validation:

- Split reservoirs into train/test groups before fitting the random forest.
- Predict all 10 STARFIT parameters for held-out reservoirs.
- Report parameter metrics for all 10 labels: RMSE, MAE, bias, correlation.
- Reconstruct weekly flood and conservation curves from the 10 parameters.
- Report curve-level metrics for flood and conservation bounds separately.
- Run physical checks:
  - flood bound should not be below conservation bound;
  - active zone should not be unrealistically narrow;
  - bounds should remain within plausible storage fractions.

Paper basis:

- Steyaert et al. (2025) used 75 percent of STARFIT-fitted reservoirs for RF
  training and 25 percent for independent validation.
- Their RF evaluation compares STARFIT-derived curves against RF-extrapolated
  curves using bias, correlation, and RMSE for flood and conservation curves.

## 2. Point reservoir model validation

Question: given seasonal bounds, does the point model produce plausible storage
and release dynamics?

Recommended validation:

- Use observed inflow forcing where available.
- Run two simulations following Turner et al. (2021):
  - release simulation with storage fixed to observed values;
  - full storage/release simulation where storage errors can accumulate.
- Compare simulated and observed storage.
- Compare simulated and observed release/outflow where available.
- Report RMSE, normalized RMSE, bias, correlation, and KGE.
- Track whether simulated storage is above, within, or below seasonal bounds.
- Check water balance residuals and physical constraints:
  - no negative storage;
  - release cannot exceed available water;
  - storage should move back toward the active zone after flood/drought events.

Paper basis:

- Turner et al. (2021) validates STARFIT with daily observed inflow forcing and
  compares simulated storage and release against observations.
- Steyaert et al. (2025) emphasizes reservoir storage validation because
  streamflow gauges are often not close enough to reservoirs to isolate the
  effect of reservoir operations.

## 3. Larger hydrologic validation

Question: does the reservoir scheme improve downstream hydrologic behavior?

This is useful later, but it is not the first priority for the point model.

Recommended validation:

- Compare model scenarios against downstream streamflow gauges where available.
- Use KGE and its components, plus correlation and bias.
- Treat this as a weaker validation of reservoir operations than direct storage
  validation, because downstream discharge mixes many other hydrologic errors.

Paper basis:

- PCR-GLOBWB 2 evaluation uses GRDC discharge, KGE, correlation, anomaly
  correlation, and GRACE total water storage.
- Steyaert et al. (2025) uses GRDC for streamflow validation but notes that
  storage observations are more sensitive to reservoir-operation changes.

## Data priorities

Best immediate validation data:

- ResOpsUS: daily storage, outflow/release, and inflow for US reservoirs.
- GloLakes: storage dynamics for broader/global storage validation.

For the current thesis workflow, ResOpsUS is the cleanest starting point for
point-reservoir validation because it includes daily operations data and was the
original STARFIT validation setting.

