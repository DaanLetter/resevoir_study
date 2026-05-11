# Chronos-2 With Covariates: Reservoir Bound Prediction Idea

This note describes the next Chronos experiment: using Chronos-2 with covariates
to predict reservoir operating bounds, then validating those bounds through the
same point-reservoir model used for the RF baseline.

## Why This Is Interesting

The current RF baseline is a static-feature model. It learns a direct mapping:

```text
reservoir features -> 10 STARFIT parameters -> weekly flood/conservation bounds
```

That is useful for reservoirs with no observed storage history, but it throws
away temporal information. Reservoir operations are not just a function of
capacity, area, purpose, climate, and demand; they also depend on the seasonal
shape and recent state of the reservoir.

Chronos-2 gives us a different modelling route:

```text
historical storage + covariates -> future storage distribution -> operating bounds
```

The current zero-shot Chronos-2 test already used historical storage only. It
forecasted weekly storage fractions from pre-2010 ResOpsUS storage and used:

- forecast `q0.9` as the flood / upper operating bound;
- forecast `q0.1` as the conservation / lower operating bound.

That storage-only Chronos-2 baseline outperformed the RF baseline inside the
point model on the post-2010 ResOpsUS validation:

| Bound source | Release nRMSE | Storage nRMSE | Storage fraction RMSE | Storage KGE |
|---|---:|---:|---:|---:|
| Chronos-2 storage quantiles | 0.823 | 1.420 | 0.114 | 0.430 |
| RF STARFIT | 0.885 | 2.731 | 0.261 | -0.048 |
| Observed STARFIT | 0.821 | 1.551 | 0.118 | 0.386 |
| Generic 10/75 | 0.904 | 2.893 | 0.300 | -0.230 |

The caveat is important: this was not a fully fair global extrapolation
comparison. Chronos used each reservoir's own historical storage context, while
RF used static features only. That makes Chronos a strong benchmark for
reservoirs with observed storage history, not automatically a solution for
totally unobserved reservoirs.

## Why Chronos-2 Specifically

Chronos-2 is the right Chronos variant for this idea because it supports
covariate-informed forecasting natively. The older Chronos and Chronos-Bolt
models are useful univariate forecasters, but Chronos-2 adds:

- multivariate forecasting;
- past-only real and categorical covariates;
- known-future real and categorical covariates;
- cross-learning across related time series;
- longer context and prediction lengths.

Useful references:

- Chronos-2 model card: https://huggingface.co/amazon/chronos-2
- Chronos GitHub repository: https://github.com/amazon-science/chronos-forecasting
- Amazon Science blog: https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
- Technical report: https://arxiv.org/abs/2510.15821

The local installed API exposes this relevant interface:

```python
Chronos2Pipeline.predict_df(
    df,
    future_df=None,
    id_column="item_id",
    timestamp_column="timestamp",
    target="target",
    prediction_length=None,
    quantile_levels=[0.1, 0.2, ..., 0.9],
    batch_size=256,
    context_length=None,
    cross_learning=False,
)
```

That means the model can receive a long-format historical dataframe plus an
optional future dataframe containing covariates known over the forecast horizon.

## Core Modelling Idea

The target should be weekly reservoir storage fraction:

```text
target = observed_storage_mcm / capacity_mcm * 100
```

Chronos-2 then predicts a probabilistic future storage distribution. The
operating bounds are derived from forecast quantiles:

```text
conservation_bound_pct = q0.1 forecast
flood_bound_pct        = q0.9 forecast
median_storage_pct     = q0.5 forecast, optional diagnostic
```

This is not exactly STARFIT, because STARFIT fits smooth seasonal harmonic
bounds. Chronos produces empirical future quantile bounds. That is fine for the
point-model validation layer because the point model only needs weekly flood and
conservation bounds; it does not require the bounds to come from 10 STARFIT
parameters.

In the existing infrastructure, Chronos would be a `bound_timeseries` model:

```text
model_name,prediction_type,dam_id,epiweek,flood_pct,conservation_pct
chronos2_covariate,bound_timeseries,100,1,82.1,47.3
chronos2_covariate,bound_timeseries,100,2,81.8,46.9
...
```

Then `run_point_model_validation.py` can compare it against:

- `rf_starfit`;
- `observed_starfit`;
- `generic_10_75`;
- future models.

## Candidate Covariates

The covariates should be split into three groups, because Chronos-2 can use both
past-only and known-future covariates.

### 1. Historical Dynamic Covariates

These are observed over the historical context window.

| Covariate | Source | Notes |
|---|---|---|
| `inflow_mcm_day` | ResOpsUS | Strong physical driver; can be used historically and, in validation, as known forcing. |
| `release_mcm_day` | ResOpsUS outflow | Useful historical signal, but should not be used as a future covariate because release is what the point model tries to simulate. |
| `evaporation_mcm_day` | ResOpsUS where available | Often sparse; include only if coverage is good enough. |
| `storage_change_mcm` | Derived from storage | Helps signal filling/drawdown regimes; avoid using future values. |
| `previous_storage_pct` | Lagged target | Usually implicit in the target history, but can be useful as an explicit covariate if multivariate layout is used. |

### 2. Known-Future Dynamic Covariates

These can be supplied in `future_df` because they are known or can be produced by
the hydrologic forcing before the reservoir release decision is made.

| Covariate | Realistic future source | Validation source |
|---|---|---|
| `epiweek_sin`, `epiweek_cos` | Calendar | Calendar |
| `month_sin`, `month_cos` | Calendar | Calendar |
| `inflow_mcm_day` | PCR-GLOBWB simulated inflow or forecast forcing | Observed ResOpsUS inflow, for Turner-style observed-inflow validation |
| `demand_proxy_mcm_day` | PCR-GLOBWB downstream demand / command-area demand | Weekly observed-outflow climatology |
| `environmental_flow_mcm_day` | PCR-GLOBWB environmental flow estimate | 10% long-term mean inflow proxy |
| `seasonal_precip_climatology` | Hydroclimate features | Static seasonal climatology repeated by week |
| `seasonal_pet_climatology` | Hydroclimate features | Static seasonal climatology repeated by week |

The most important design choice is whether future inflow is allowed. For
paper-style validation it is defensible because Turner-style point simulations
use observed inflow forcing. For global deployment it must come from the
hydrologic model, not observations.

### 3. Static Reservoir Covariates Repeated Across Time

Chronos expects time-indexed rows, so static features can be repeated at every
timestamp for the reservoir.

| Covariate | Source | Notes |
|---|---|---|
| `capacity_mcm` / `log_capacity_mcm` | RF feature table / GeoDAR | Use log transform to reduce scale problems. |
| `area_km2` / `log_area_km2` | RF feature table / GeoDAR | Repeated through time. |
| `use_Irrigation` | RF feature table | Helps Chronos learn irrigation-like seasonal drawdown. |
| `use_Water Supply` | RF feature table | Grouped as irrigation-like in the point model. |
| `use_Hydroelectricity` | RF feature table | Helps represent stable-storage operations. |
| `use_Flood Control` | RF feature table | Useful for upper-bound timing. |
| `dem_m` | RF feature table | Socioeconomic pressure proxy. |
| `winter/spring/summer/autumn_inflow` | RF feature table | Seasonal climatology, repeated or mapped to week. |
| `winter/spring/summer/autumn_precip` | RF feature table | Seasonal climatology, mapped to week. |
| `winter/spring/summer/autumn_pet` | RF feature table | Seasonal climatology, mapped to week. |
| `winter/spring/summer/autumn_aridity` | RF feature table | Seasonal climatology, mapped to week. |

## Proposed Dataframe Shape

Historical context dataframe:

```text
dam_id,timestamp,target_storage_pct,inflow_mcm_day,release_mcm_day,
demand_proxy_mcm_day,epiweek_sin,epiweek_cos,log_capacity_mcm,
use_Irrigation,use_Water Supply,use_Hydroelectricity,...
```

Future covariate dataframe:

```text
dam_id,timestamp,inflow_mcm_day,demand_proxy_mcm_day,epiweek_sin,
epiweek_cos,log_capacity_mcm,use_Irrigation,use_Water Supply,...
```

The target column should not appear in `future_df`. Past-only covariates that are
not known in the future, such as observed release, should also be excluded from
`future_df`.

Example call:

```python
from chronos import Chronos2Pipeline

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=52,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="dam_id",
    timestamp_column="timestamp",
    target="target_storage_pct",
    batch_size=8,
    context_length=512,
    cross_learning=True,
)
```

The first experiment can keep `cross_learning=False` to isolate each reservoir.
The second experiment should set `cross_learning=True` so Chronos can share
information across reservoirs in the same batch/group.

## Validation Experiments

### Experiment A: Storage-History Benchmark

Question:

```text
If we have observed storage history for a reservoir, can Chronos-2 forecast
useful future operating bounds?
```

Setup:

- Context: pre-2010 weekly storage fraction.
- Future covariates: calendar, observed inflow, demand climatology, static
  reservoir metadata.
- Validation: post-2010 point-model `RS-OBS` and `RS-SIM`.
- Comparison: RF STARFIT, observed STARFIT, generic 10/75.

This is the strongest use case for Chronos and the closest to the first
storage-only Chronos test.

### Experiment B: Covariate Ablation

Question:

```text
Which covariates actually help?
```

Run these variants:

1. Chronos storage-only.
2. Storage + calendar.
3. Storage + calendar + inflow.
4. Storage + calendar + inflow + demand proxy.
5. Storage + calendar + inflow + demand proxy + static metadata.
6. Full covariate set.

Score all variants through the point model, not only through forecast error.
The thesis-relevant question is whether the covariates improve simulated
storage/release behavior.

### Experiment C: Cross-Reservoir Learning

Question:

```text
Can Chronos learn shared reservoir-operation patterns across related reservoirs?
```

Setup:

- Use `cross_learning=True`.
- Batch reservoirs by similar use class or basin/region if available.
- Keep target histories separate but allow Chronos to attend across the group.
- Compare grouped Chronos against independent Chronos.

This is important because Chronos-2's group attention is designed to share
information across related series and covariates.

### Experiment D: Cold-Start / RF-Replacement Stress Test

Question:

```text
Can Chronos help when a reservoir has little or no storage history?
```

Possible setups:

- Short-context test: give only 4, 13, 26, or 52 weeks of storage history.
- Leave-one-reservoir-out style: hide most target history for the target dam but
  provide covariates and related reservoir histories with `cross_learning=True`.
- Static-only future: use static metadata and seasonal climatologies, no observed
  target history.

This is the fairer comparison against RF. If Chronos needs long storage history,
it is a complement to RF. If Chronos works with short/no history plus covariates,
then it becomes a real RF alternative for extrapolation.

## Metrics

Use the existing validation layer:

- direct bound sanity checks;
- point-model `RS-OBS` release validation;
- point-model `RS-SIM` storage/release validation;
- RMSE, nRMSE, transformed release nRMSE, high-storage nRMSE;
- KGE, Pearson, Spearman;
- storage fraction RMSE;
- above/within/below-bound fractions.

Primary table for thesis comparison:

```text
post-2010 RS-SIM:
model_name, release_nrmSE, storage_nRMSE, storage_fraction_RMSE, storage_KGE
```

Secondary table:

```text
post-2010 RS-OBS:
model_name, release_nRMSE, transformed_release_nRMSE, release_KGE
```

## Expected Benefits

Chronos with covariates could improve on RF because it can use:

- each reservoir's own observed storage dynamics;
- seasonal recurrence;
- inflow forcing;
- demand proxy;
- reservoir purpose;
- cross-reservoir patterns.

It may especially help with:

- reservoirs whose STARFIT parameters are hard to fit smoothly;
- reservoirs with non-sinusoidal or irregular operating patterns;
- reservoirs where static features alone do not explain seasonal drawdown;
- cases where the RF predicts physically plausible but too-generic bounds.

## Expected Risks

Chronos may also fail or be misleading if:

- it overfits to historical storage and does not generalize to changed
  operations;
- future inflow is supplied from observations during validation but would not be
  available in real deployment;
- covariates leak information from the validation period;
- sparse or irregular observations are interpolated too aggressively;
- q0.1/q0.9 storage forecasts are not equivalent to operational bounds;
- the model performs well for observed reservoirs but cannot extrapolate to
  unobserved dams.

The biggest scientific caveat is conceptual: Chronos predicts storage
distributions, while STARFIT tries to infer operating policy bounds. These are
related but not identical. The point-model validation is therefore essential.

## Implementation Path

1. Add a covariate-building script:

   ```text
   build_chronos2_covariate_dataset.py
   ```

   It should output:

   ```text
   chronos2_context_covariates.parquet
   chronos2_future_covariates.parquet
   chronos2_covariate_manifest.csv
   ```

2. Add a covariate Chronos runner:

   ```text
   run_chronos2_covariate_bounds.py
   ```

   It should output:

   ```text
   chronos2_covariate_weekly_bound_predictions.csv
   ```

3. Pass the output into the existing validator:

   ```powershell
   py run_point_model_validation.py `
     --run-id chronos2_covariates_vs_rf `
     --extra-bounds-csv validation_outputs\point_model_validation\chronos2_covariates\chronos2_covariate_weekly_bound_predictions.csv `
     --model-order rf_starfit,chronos2_storage_quantile,chronos2_covariate,observed_starfit,generic_10_75
   ```

4. Update `MODEL_SCORE_REPORT.md` with:

   - storage-only Chronos;
   - Chronos + calendar;
   - Chronos + inflow;
   - Chronos + inflow + demand;
   - Chronos + full covariates;
   - RF STARFIT;
   - observed STARFIT;
   - generic 10/75.

## Recommended First Version

The first covariate-informed Chronos run should be deliberately conservative:

- target: weekly storage fraction;
- context: pre-2010 weekly ResOpsUS storage;
- horizon: 52 weeks;
- covariates:
  - `epiweek_sin`, `epiweek_cos`;
  - weekly observed inflow for validation;
  - weekly demand climatology from pre-2010 outflow;
  - capacity, area, and use flags repeated through time;
- `cross_learning=False` first;
- then repeat with `cross_learning=True`.

This keeps the first experiment interpretable. If it beats storage-only Chronos,
then the covariates are actually adding value. If it does not, the simpler
storage-only Chronos baseline remains a strong result.
