# Model Score Report

This report summarizes the current validation outputs for the ResOpsUS-first
reservoir-operation experiments. Large generated files and datasets are kept out
of Git; this file records the headline scores that should travel with the code.

## RF STARFIT Parameter Validation

Validation target: predict the 10 STARFIT seasonal-bound parameters from static
reservoir, hydroclimate, and use features. Scores compare reconstructed weekly
flood/conservation curves against held-out STARFIT-derived curves.

| Dataset | Train reservoirs | Test reservoirs | Flood RMSE | Flood r | Conservation RMSE | Conservation r |
|---|---:|---:|---:|---:|---:|---:|
| ResOpsUS | 462 | 154 | 22.48 | 0.66 | 21.91 | 0.43 |
| GloLakes Sentinel-2 | 1311 | 438 | 23.79 | 0.50 | 23.16 | 0.47 |
| GloLakes ICESat-2 | 1821 | 608 | 23.61 | 0.56 | 23.50 | 0.60 |

Interpretation: the RF learns seasonal structure, but parameter/curve errors
remain high. It is useful as a baseline extrapolation model, not as a perfect
replacement for observed STARFIT bounds.

## Point-Model Validation: RF vs STARFIT vs Generic

Validation target: plug predicted bounds into the point-reservoir simulator and
compare simulated storage/release against ResOpsUS observations. The most useful
comparison is post-2010 `RS-SIM`, where storage evolves forward through the
model.

| Bound source | Release nRMSE | Storage nRMSE | Storage fraction RMSE | Storage KGE |
|---|---:|---:|---:|---:|
| Observed STARFIT | 0.821 | 1.551 | 0.118 | 0.386 |
| RF STARFIT | 0.885 | 2.731 | 0.261 | -0.048 |
| Generic 10/75 | 0.904 | 2.893 | 0.300 | -0.230 |

Interpretation: the RF baseline improves on generic 10/75 bounds, especially
for storage fraction RMSE, but observed STARFIT bounds are still clearly better.

## Chronos-2 CPU Baseline

Chronos-2 run: `amazon/chronos-2` through `chronos-forecasting 2.2.2`, CPU-only.
For each held-out reservoir, pre-2010 weekly storage fraction was used as
context. The 0.9 forecast quantile became the flood bound and the 0.1 forecast
quantile became the conservation bound.

Post-2010 `RS-SIM` comparison:

| Bound source | Release nRMSE | Storage nRMSE | Storage fraction RMSE | Storage KGE |
|---|---:|---:|---:|---:|
| Chronos-2 storage quantiles | 0.823 | 1.420 | 0.114 | 0.430 |
| RF STARFIT | 0.885 | 2.731 | 0.261 | -0.048 |
| Observed STARFIT | 0.821 | 1.551 | 0.118 | 0.386 |
| Generic 10/75 | 0.904 | 2.893 | 0.300 | -0.230 |

Interpretation: Chronos-2 strongly outperforms the RF baseline in this
ResOpsUS point-model benchmark and is close to observed STARFIT. This is not a
fully fair global extrapolation comparison, because Chronos-2 uses each
reservoir's own historical storage context, while RF uses static features only.
Chronos-2 is therefore best interpreted as a strong benchmark for reservoirs
with historical storage records, not yet as a replacement for RF on totally
unobserved dams.

## Current Caveats

- The point-model validation uses ResOpsUS observed inflow, outflow, and storage.
- Downstream demand is approximated with training-period weekly observed outflow
  climatology.
- Environmental flow is approximated as 10 percent of long-term mean inflow.
- `POINTDATA.zip` was not extracted or used.
- Large generated files such as `daily_simulations.csv.gz` are intentionally not
  committed.
