"""Generate Chronos-2 weekly storage-bound predictions for ResOpsUS dams.

Chronos-2 forecasts storage fraction from pre-2010 weekly storage history. The
0.9 forecast quantile is used as a flood/upper bound and the 0.1 forecast
quantile as a conservation/lower bound, giving a direct bound-timeseries model
that can be passed into run_point_model_validation.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bound_model_adapters import build_resopsus_baseline_bundle
from point_reservoir_model import add_epiweek
from run_point_model_validation import CUTOFF_DATE, OUTPUT_ROOT, read_resopsus_timeseries


MODEL_NAME = "chronos2_storage_quantile"


def weekly_context_for_dam(dam_id: int, capacity_mcm: float, context_length: int | None) -> pd.DataFrame | None:
    daily, reason = read_resopsus_timeseries(dam_id)
    if reason is not None or daily is None or capacity_mcm <= 0:
        return None
    train = daily[(daily["date"] < CUTOFF_DATE) & daily["observed_storage_mcm"].notna()].copy()
    if train.empty:
        return None
    weekly = (
        train.set_index("date")["observed_storage_mcm"]
        .resample("W-SUN")
        .mean()
    )
    if weekly.dropna().shape[0] < 52:
        return None
    full_index = pd.date_range(weekly.dropna().index.min(), weekly.dropna().index.max(), freq="W-SUN")
    weekly = weekly.reindex(full_index).interpolate(limit_direction="both")
    weekly = weekly.reset_index().rename(columns={"index": "timestamp", "observed_storage_mcm": "storage_mcm"})
    if context_length is not None:
        weekly = weekly.tail(context_length)
    weekly["dam_id"] = int(dam_id)
    weekly["storage_pct"] = (weekly["storage_mcm"] / capacity_mcm * 100).clip(lower=0, upper=100)
    return weekly[["dam_id", "timestamp", "storage_pct"]]


def build_context(max_reservoirs: int | None, context_length: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    bundle = build_resopsus_baseline_bundle()
    metadata = bundle.reservoir_metadata.sort_values("dam_id").reset_index(drop=True)
    if max_reservoirs is not None:
        metadata = metadata.head(max_reservoirs).copy()

    frames = []
    skipped = []
    for row in metadata.itertuples(index=False):
        frame = weekly_context_for_dam(int(row.dam_id), float(row.capacity_mcm), context_length)
        if frame is None:
            skipped.append({"dam_id": int(row.dam_id), "reason": "insufficient_pre2010_weekly_storage"})
        else:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No reservoirs had enough pre-2010 weekly storage for Chronos-2 context.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(skipped)


def forecast_bounds(
    context_df: pd.DataFrame,
    prediction_length: int,
    batch_size: int,
    context_length: int | None,
) -> pd.DataFrame:
    from chronos import Chronos2Pipeline

    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="dam_id",
        timestamp_column="timestamp",
        target="storage_pct",
        batch_size=batch_size,
        context_length=context_length,
    )

    q10_col = "0.1" if "0.1" in pred_df.columns else 0.1
    q90_col = "0.9" if "0.9" in pred_df.columns else 0.9
    bounds = pred_df[["dam_id", "timestamp", q10_col, q90_col]].copy()
    bounds["epiweek"] = add_epiweek(bounds["timestamp"])
    bounds["conservation_pct"] = pd.to_numeric(bounds[q10_col], errors="coerce").clip(lower=0, upper=100)
    bounds["flood_pct"] = pd.to_numeric(bounds[q90_col], errors="coerce").clip(lower=0, upper=100)
    bounds = (
        bounds.groupby(["dam_id", "epiweek"], as_index=False)
        .agg({"flood_pct": "max", "conservation_pct": "min"})
        .sort_values(["dam_id", "epiweek"])
    )

    complete = []
    for dam_id, group in bounds.groupby("dam_id"):
        group = group.set_index("epiweek").reindex(range(1, 53))
        group["dam_id"] = int(dam_id)
        group[["flood_pct", "conservation_pct"]] = group[["flood_pct", "conservation_pct"]].interpolate(
            limit_direction="both"
        )
        group = group.reset_index().rename(columns={"index": "epiweek"})
        complete.append(group)
    bounds = pd.concat(complete, ignore_index=True)
    bounds["model_name"] = MODEL_NAME
    bounds["prediction_type"] = "bound_timeseries"
    bounds["flood_pct"] = np.maximum(bounds["flood_pct"], bounds["conservation_pct"])
    return bounds[["model_name", "prediction_type", "dam_id", "epiweek", "flood_pct", "conservation_pct"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="chronos2_cpu")
    parser.add_argument("--max-reservoirs", type=int, default=None)
    parser.add_argument("--prediction-length", type=int, default=52)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    context_df, skipped = build_context(args.max_reservoirs, args.context_length)
    context_df.to_csv(output_dir / "chronos2_context_weekly_storage.csv", index=False)
    skipped.to_csv(output_dir / "chronos2_skipped_context.csv", index=False)

    bounds = forecast_bounds(
        context_df,
        prediction_length=args.prediction_length,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )
    bounds_path = output_dir / "chronos2_weekly_bound_predictions.csv"
    bounds.to_csv(bounds_path, index=False)
    print(f"Chronos-2 context series: {context_df['dam_id'].nunique()}")
    print(f"Chronos-2 skipped context reservoirs: {len(skipped)}")
    print(f"Chronos-2 bounds written to: {bounds_path}")


if __name__ == "__main__":
    main()
