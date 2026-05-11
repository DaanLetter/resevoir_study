"""Run ResOpsUS point-model validation for bound-prediction baselines."""

from __future__ import annotations

import argparse
import gzip
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bound_model_adapters import build_resopsus_baseline_bundle
from point_reservoir_model import (
    ReservoirConfig,
    add_epiweek,
    cms_to_mcm_per_day,
    simulate_reservoir,
    weekly_demand_climatology,
)
from validation_layer import (
    high_storage_nrmse,
    metric_summary,
    storage_position_counts,
    transformed_nrmse,
)


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTIE_DIR = PROJECT_DIR.parent
RESOPSUS_DIR = SCRIPTIE_DIR / "ResOpsUS"
OUTPUT_ROOT = PROJECT_DIR / "validation_outputs" / "point_model_validation"
CUTOFF_DATE = pd.Timestamp("2010-01-01")
MIN_VALIDATION_DAYS = 365
MIN_DEMAND_DAYS = 180
MODEL_ORDER = ["rf_starfit", "observed_starfit", "generic_10_75"]
SPLITS = ["post2010", "full_overlap"]
SIMULATION_MODES = ["RS_OBS", "RS_SIM"]
PLOT_COLORS = {
    "rf_starfit": "#2563eb",
    "chronos2_storage_quantile": "#7c3aed",
    "observed_starfit": "#059669",
    "generic_10_75": "#dc2626",
}


def read_resopsus_timeseries(dam_id: int) -> tuple[pd.DataFrame | None, str | None]:
    path = RESOPSUS_DIR / "time_series_all" / f"ResOpsUS_{int(dam_id)}.csv"
    if not path.exists():
        return None, "missing_resopsus_file"
    try:
        data = pd.read_csv(
            path,
            usecols=["date", "storage", "inflow", "outflow"],
            na_values=["NA", ""],
        )
    except ValueError as exc:
        return None, f"missing_required_columns: {exc}"

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["epiweek"] = add_epiweek(data["date"])
    data["observed_storage_mcm"] = pd.to_numeric(data["storage"], errors="coerce")
    data["inflow_mcm_day"] = cms_to_mcm_per_day(data["inflow"])
    data["observed_release_mcm_day"] = cms_to_mcm_per_day(data["outflow"])
    data = data.dropna(subset=["date"])
    data = data[data["observed_storage_mcm"].isna() | (data["observed_storage_mcm"] >= 0)]
    data = data[data["observed_release_mcm_day"].isna() | (data["observed_release_mcm_day"] >= 0)]
    return data[["date", "epiweek", "observed_storage_mcm", "inflow_mcm_day", "observed_release_mcm_day"]], None


def validation_context(daily: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame | None, pd.Series | None, float | None, str | None]:
    complete = daily.dropna(subset=["observed_storage_mcm", "inflow_mcm_day", "observed_release_mcm_day"]).copy()
    if complete.empty:
        return None, None, None, "no_complete_storage_inflow_release_days"
    complete["inflow_mcm_day"] = complete["inflow_mcm_day"].clip(lower=0)
    average_inflow = float(complete["inflow_mcm_day"].mean())
    if not np.isfinite(average_inflow):
        return None, None, None, "missing_average_inflow"

    if split_name == "post2010":
        demand_training = daily[daily["date"] < CUTOFF_DATE].copy()
        validation = complete[complete["date"] >= CUTOFF_DATE].copy()
    elif split_name == "full_overlap":
        demand_training = complete.copy()
        validation = complete.copy()
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    demand_days = int(demand_training["observed_release_mcm_day"].notna().sum())
    if demand_days < MIN_DEMAND_DAYS:
        return None, None, None, f"insufficient_demand_climatology_days:{demand_days}"
    if len(validation) < MIN_VALIDATION_DAYS:
        return None, None, None, f"insufficient_validation_days:{len(validation)}"

    demand = weekly_demand_climatology(demand_training, fallback=average_inflow)
    return validation, demand, average_inflow, None


def reservoir_config(meta_row: pd.Series) -> ReservoirConfig | None:
    capacity = float(meta_row["capacity_mcm"])
    if not np.isfinite(capacity) or capacity <= 0:
        return None
    return ReservoirConfig(
        dam_id=int(meta_row["dam_id"]),
        capacity_mcm=capacity,
        use_category=str(meta_row["use_category"]),
    )


def weekly_bounds_for(bounds: pd.DataFrame, dam_id: int, model_name: str) -> pd.DataFrame:
    return bounds[(bounds["dam_id"] == int(dam_id)) & (bounds["model_name"] == model_name)][
        ["epiweek", "flood_pct", "conservation_pct"]
    ].copy()


def compute_metric_row(
    sim: pd.DataFrame,
    config: ReservoirConfig,
    model_name: str,
    split_name: str,
    mode: str,
) -> dict[str, float | int | str]:
    release = metric_summary(sim["observed_release_mcm_day"], sim["simulated_release_mcm_day"])
    row: dict[str, float | int | str] = {
        "dam_id": config.dam_id,
        "model_name": model_name,
        "split_name": split_name,
        "simulation_mode": mode,
        "validation_days": len(sim),
        "release_rmse_mcm_day": release["rmse"],
        "release_nrmse": release["nrmse"],
        "release_mae_mcm_day": release["mae"],
        "release_bias_mcm_day": release["bias"],
        "release_pearson_r": release["pearson_r"],
        "release_spearman_r": release["spearman_r"],
        "release_kge": release["kge"],
        "release_transformed_nrmse": transformed_nrmse(
            sim["observed_release_mcm_day"],
            sim["simulated_release_mcm_day"],
        ),
    }

    position_storage = sim["simulated_storage_mcm"] if mode == "RS_SIM" else sim["observed_storage_mcm"]
    position = storage_position_counts(position_storage, sim["flood_bound_mcm"], sim["conservation_bound_mcm"])
    row.update(
        {
            "above_bound_fraction": position["above_fraction"],
            "within_bound_fraction": position["within_fraction"],
            "below_bound_fraction": position["below_fraction"],
        }
    )

    if mode == "RS_SIM":
        storage = metric_summary(sim["observed_storage_mcm"], sim["simulated_storage_mcm"])
        storage_fraction = metric_summary(
            sim["observed_storage_mcm"] / config.capacity_mcm,
            sim["simulated_storage_mcm"] / config.capacity_mcm,
        )
        row.update(
            {
                "storage_rmse_mcm": storage["rmse"],
                "storage_nrmse": storage["nrmse"],
                "storage_mae_mcm": storage["mae"],
                "storage_bias_mcm": storage["bias"],
                "storage_pearson_r": storage["pearson_r"],
                "storage_spearman_r": storage["spearman_r"],
                "storage_kge": storage["kge"],
                "storage_fraction_rmse": storage_fraction["rmse"],
                "high_storage_nrmse": high_storage_nrmse(
                    sim["observed_storage_mcm"],
                    sim["simulated_storage_mcm"],
                    sim["flood_bound_mcm"],
                ),
            }
        )
    else:
        row.update(
            {
                "storage_rmse_mcm": np.nan,
                "storage_nrmse": np.nan,
                "storage_mae_mcm": np.nan,
                "storage_bias_mcm": np.nan,
                "storage_pearson_r": np.nan,
                "storage_spearman_r": np.nan,
                "storage_kge": np.nan,
                "storage_fraction_rmse": np.nan,
                "high_storage_nrmse": np.nan,
            }
        )
    return row


def append_daily_simulation(
    handle,
    sim: pd.DataFrame,
    config: ReservoirConfig,
    model_name: str,
    split_name: str,
    mode: str,
    write_header: bool,
) -> None:
    frame = sim.copy()
    frame.insert(0, "dam_id", config.dam_id)
    frame.insert(1, "model_name", model_name)
    frame.insert(2, "split_name", split_name)
    frame.insert(3, "simulation_mode", mode)
    frame["capacity_mcm"] = config.capacity_mcm
    frame["observed_storage_fraction"] = frame["observed_storage_mcm"] / config.capacity_mcm
    frame["simulated_storage_fraction"] = frame["simulated_storage_mcm"] / config.capacity_mcm
    frame.to_csv(handle, index=False, header=write_header)


def svg_polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{text}" />'


def write_cdf_svg(metrics: pd.DataFrame, value_col: str, path: Path, title: str, split_name: str, mode: str) -> None:
    data = metrics[(metrics["split_name"] == split_name) & (metrics["simulation_mode"] == mode)].copy()
    data = data[np.isfinite(data[value_col])]
    width, height = 860, 520
    left, right, top, bottom = 74, 24, 52, 58
    inner_w = width - left - right
    inner_h = height - top - bottom
    if data.empty:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>", encoding="utf-8")
        return
    xmax = float(data[value_col].quantile(0.95))
    xmax = xmax if xmax > 0 else float(data[value_col].max())
    xmax = max(xmax, 1e-9)
    lines = []
    legends = []
    for idx, model_name in enumerate(MODEL_ORDER):
        values = np.sort(data.loc[data["model_name"] == model_name, value_col].dropna().to_numpy(dtype=float))
        if values.size == 0:
            continue
        yvals = np.arange(1, len(values) + 1) / len(values)
        points = []
        for value, prob in zip(values, yvals):
            x = left + min(value, xmax) / xmax * inner_w
            y = top + (1 - prob) * inner_h
            points.append((x, y))
        color = PLOT_COLORS.get(model_name, "#374151")
        lines.append(svg_polyline(points, color, 2.5))
        legends.append(
            f'<line x1="{left+20+idx*210}" x2="{left+52+idx*210}" y1="36" y2="36" stroke="{color}" stroke-width="3"/>'
            f'<text x="{left+60+idx*210}" y="40" font-size="12" fill="#374151">{model_name}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="24" font-size="20" font-weight="700" fill="#111827">{title}</text>
{''.join(legends)}
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
{''.join(lines)}
<text x="{left}" y="{height-bottom+22}" font-size="12" fill="#4b5563">0</text>
<text x="{width-right}" y="{height-bottom+22}" text-anchor="end" font-size="12" fill="#4b5563">{xmax:.2f}</text>
<text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="13" fill="#4b5563">{value_col}</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">CDF</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_storage_scatter_svg(points: pd.DataFrame, path: Path) -> None:
    width, height = 720, 720
    left, right, top, bottom = 70, 30, 44, 62
    inner_w = width - left - right
    inner_h = height - top - bottom
    max_axis = max(1.0, float(points[["observed", "simulated"]].quantile(0.98).max())) if not points.empty else 1.0
    dots = []
    for row in points.itertuples(index=False):
        x = left + min(max(row.observed, 0), max_axis) / max_axis * inner_w
        y = top + (1 - min(max(row.simulated, 0), max_axis) / max_axis) * inner_h
        color = PLOT_COLORS.get(row.model_name, "#374151")
        dots.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{color}" opacity="0.32"/>')
    diagonal = f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{top}" stroke="#111827" stroke-width="1.2" stroke-dasharray="5 5"/>'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="26" font-size="20" font-weight="700" fill="#111827">Observed vs Simulated Storage Fraction</text>
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
{diagonal}
{''.join(dots)}
<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="13" fill="#4b5563">observed storage fraction</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">simulated storage fraction</text>
<text x="{left}" y="{height-bottom+22}" font-size="12" fill="#4b5563">0</text>
<text x="{width-right}" y="{height-bottom+22}" text-anchor="end" font-size="12" fill="#4b5563">{max_axis:.2f}</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_timeseries_svg(sim: pd.DataFrame, path: Path, title: str) -> None:
    width, height = 980, 460
    left, right, top, bottom = 70, 28, 44, 54
    inner_w = width - left - right
    inner_h = height - top - bottom
    plot = sim.reset_index(drop=True).copy()
    if len(plot) > 900:
        plot = plot.iloc[:: max(1, len(plot) // 900)].copy()
    cols = ["observed_storage_mcm", "simulated_storage_mcm", "flood_bound_mcm", "conservation_bound_mcm"]
    ymin = float(plot[cols].min().min())
    ymax = float(plot[cols].max().max())
    if ymax <= ymin:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.08
    ymin -= pad
    ymax += pad

    def sx(index_values):
        return left + np.asarray(index_values) / max(len(plot) - 1, 1) * inner_w

    def sy(values):
        return top + (ymax - np.asarray(values, dtype=float)) / (ymax - ymin) * inner_h

    lines = []
    for col, color in [
        ("observed_storage_mcm", "#111827"),
        ("simulated_storage_mcm", "#2563eb"),
        ("flood_bound_mcm", "#dc2626"),
        ("conservation_bound_mcm", "#059669"),
    ]:
        lines.append(svg_polyline(list(zip(sx(range(len(plot))), sy(plot[col]))), color, 2.1))
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="26" font-size="19" font-weight="700" fill="#111827">{title}</text>
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
{''.join(lines)}
<line x1="610" x2="640" y1="58" y2="58" stroke="#111827" stroke-width="3"/><text x="648" y="62" font-size="12" fill="#374151">observed storage</text>
<line x1="610" x2="640" y1="80" y2="80" stroke="#2563eb" stroke-width="3"/><text x="648" y="84" font-size="12" fill="#374151">simulated storage</text>
<line x1="760" x2="790" y1="58" y2="58" stroke="#dc2626" stroke-width="3"/><text x="798" y="62" font-size="12" fill="#374151">flood bound</text>
<line x1="760" x2="790" y1="80" y2="80" stroke="#059669" stroke-width="3"/><text x="798" y="84" font-size="12" fill="#374151">conservation bound</text>
<text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="13" fill="#4b5563">validation period</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">storage (MCM)</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def simulate_one(
    meta_row: pd.Series,
    bounds: pd.DataFrame,
    daily: pd.DataFrame,
    model_name: str,
    split_name: str,
    mode: str,
) -> tuple[pd.DataFrame | None, ReservoirConfig | None, str | None]:
    config = reservoir_config(meta_row)
    if config is None:
        return None, None, "missing_or_invalid_capacity"
    validation, demand, average_inflow, reason = validation_context(daily, split_name)
    if reason is not None:
        return None, config, reason
    weekly_bounds = weekly_bounds_for(bounds, config.dam_id, model_name)
    if len(weekly_bounds) != 52:
        return None, config, f"missing_weekly_bounds:{model_name}"
    sim = simulate_reservoir(
        validation,
        weekly_bounds,
        demand,
        average_inflow,
        config,
        mode,
    )
    return sim, config, None


def write_selected_timeseries(
    metrics: pd.DataFrame,
    metadata: pd.DataFrame,
    bounds: pd.DataFrame,
    daily_cache: dict[int, pd.DataFrame],
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "timeseries_examples"
    plot_dir.mkdir(parents=True, exist_ok=True)
    target = metrics[
        (metrics["split_name"] == "post2010")
        & (metrics["simulation_mode"] == "RS_SIM")
        & np.isfinite(metrics["storage_nrmse"])
    ].copy()
    for model_name in MODEL_ORDER:
        model_metrics = target[target["model_name"] == model_name].sort_values("storage_nrmse")
        if model_metrics.empty:
            continue
        picks = {
            "best": model_metrics.iloc[0],
            "median": model_metrics.iloc[len(model_metrics) // 2],
            "worst": model_metrics.iloc[-1],
        }
        for label, metric_row in picks.items():
            dam_id = int(metric_row["dam_id"])
            meta_row = metadata[metadata["dam_id"] == dam_id].iloc[0]
            sim, _, reason = simulate_one(meta_row, bounds, daily_cache[dam_id], model_name, "post2010", "RS_SIM")
            if reason is not None or sim is None:
                continue
            write_timeseries_svg(
                sim,
                plot_dir / f"{model_name}_{label}_dam_{dam_id}.svg",
                f"{model_name} {label} dam {dam_id} (storage nRMSE={metric_row['storage_nrmse']:.2f})",
            )


def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "release_nrmse",
        "release_transformed_nrmse",
        "release_kge",
        "storage_nrmse",
        "storage_fraction_rmse",
        "storage_kge",
        "high_storage_nrmse",
        "within_bound_fraction",
    ]
    rows = []
    for keys, group in metrics.groupby(["split_name", "model_name", "simulation_mode"]):
        split_name, model_name, mode = keys
        row: dict[str, float | int | str] = {
            "split_name": split_name,
            "model_name": model_name,
            "simulation_mode": mode,
            "reservoirs": group["dam_id"].nunique(),
            "median_validation_days": float(group["validation_days"].median()),
        }
        for col in numeric_cols:
            values = group[col].dropna()
            row[f"median_{col}"] = float(values.median()) if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--max-reservoirs", type=int, default=None)
    parser.add_argument("--no-daily-output", action="store_true")
    parser.add_argument(
        "--extra-bounds-csv",
        action="append",
        default=[],
        help="Additional weekly bound predictions with model_name, dam_id, epiweek, flood_pct, conservation_pct.",
    )
    parser.add_argument(
        "--model-order",
        default=None,
        help="Comma-separated model order. Defaults to RF, any extra models, observed STARFIT, generic 10/75.",
    )
    return parser.parse_args()


def main() -> None:
    global MODEL_ORDER

    args = parse_args()
    output_dir = OUTPUT_ROOT / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_resopsus_baseline_bundle()
    metadata = bundle.reservoir_metadata.sort_values("dam_id").reset_index(drop=True)
    if args.max_reservoirs is not None:
        metadata = metadata.head(args.max_reservoirs).copy()
        keep_ids = set(metadata["dam_id"].astype(int))
        bounds = bundle.bounds[bundle.bounds["dam_id"].astype(int).isin(keep_ids)].copy()
    else:
        bounds = bundle.bounds.copy()

    extra_models: list[str] = []
    for csv_path in args.extra_bounds_csv:
        extra = pd.read_csv(csv_path)
        required = {"model_name", "dam_id", "epiweek", "flood_pct", "conservation_pct"}
        missing = required.difference(extra.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        if "prediction_type" not in extra.columns:
            extra.insert(1, "prediction_type", "bound_timeseries")
        if args.max_reservoirs is not None:
            extra = extra[extra["dam_id"].astype(int).isin(keep_ids)].copy()
        extra_models.extend([m for m in extra["model_name"].dropna().unique().tolist() if m not in extra_models])
        bounds = pd.concat(
            [
                bounds,
                extra[["model_name", "prediction_type", "dam_id", "epiweek", "flood_pct", "conservation_pct"]],
            ],
            ignore_index=True,
        )

    if args.model_order:
        MODEL_ORDER = [item.strip() for item in args.model_order.split(",") if item.strip()]
    elif extra_models:
        MODEL_ORDER = ["rf_starfit"] + extra_models + ["observed_starfit", "generic_10_75"]

    bundle.split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    bundle.rf_parameter_predictions.to_csv(output_dir / "rf_parameter_predictions.csv", index=False)
    bundle.observed_parameter_labels.to_csv(output_dir / "observed_starfit_parameter_labels.csv", index=False)
    bounds.to_csv(output_dir / "weekly_bound_predictions.csv", index=False)
    metadata.to_csv(output_dir / "reservoir_metadata.csv", index=False)

    metrics_rows = []
    skipped_rows = []
    scatter_rows = []
    daily_cache: dict[int, pd.DataFrame] = {}
    daily_path = output_dir / "daily_simulations.csv.gz"
    write_header = True

    daily_handle = gzip.open(daily_path, "wt", newline="") if not args.no_daily_output else None
    try:
        for meta_row in metadata.itertuples(index=False):
            dam_id = int(meta_row.dam_id)
            daily, reason = read_resopsus_timeseries(dam_id)
            if reason is not None or daily is None:
                skipped_rows.append({"dam_id": dam_id, "split_name": "all", "reason": reason})
                continue
            daily_cache[dam_id] = daily
            meta_series = pd.Series(meta_row._asdict())
            for split_name in SPLITS:
                validation, demand, average_inflow, split_reason = validation_context(daily, split_name)
                if split_reason is not None:
                    skipped_rows.append({"dam_id": dam_id, "split_name": split_name, "reason": split_reason})
                    continue
                config = reservoir_config(meta_series)
                if config is None:
                    skipped_rows.append({"dam_id": dam_id, "split_name": split_name, "reason": "missing_or_invalid_capacity"})
                    continue
                for model_name in MODEL_ORDER:
                    weekly_bounds = weekly_bounds_for(bounds, dam_id, model_name)
                    if len(weekly_bounds) != 52:
                        skipped_rows.append(
                            {"dam_id": dam_id, "split_name": split_name, "reason": f"missing_weekly_bounds:{model_name}"}
                        )
                        continue
                    for mode in SIMULATION_MODES:
                        sim = simulate_reservoir(
                            validation,
                            weekly_bounds,
                            demand,
                            average_inflow,
                            config,
                            mode,
                        )
                        metrics_rows.append(compute_metric_row(sim, config, model_name, split_name, mode))
                        if daily_handle is not None:
                            append_daily_simulation(daily_handle, sim, config, model_name, split_name, mode, write_header)
                            write_header = False
                        if split_name == "post2010" and mode == "RS_SIM":
                            step = max(1, len(sim) // 120)
                            sample = sim.iloc[::step].head(120)
                            for row in sample.itertuples(index=False):
                                scatter_rows.append(
                                    {
                                        "model_name": model_name,
                                        "observed": row.observed_storage_mcm / config.capacity_mcm,
                                        "simulated": row.simulated_storage_mcm / config.capacity_mcm,
                                    }
                                )
    finally:
        if daily_handle is not None:
            daily_handle.close()

    metrics = pd.DataFrame(metrics_rows)
    skipped = pd.DataFrame(skipped_rows)
    summary = build_summary(metrics) if not metrics.empty else pd.DataFrame()
    metrics.to_csv(output_dir / "reservoir_metrics.csv", index=False)
    skipped.to_csv(output_dir / "skipped_reservoirs.csv", index=False)
    summary.to_csv(output_dir / "model_comparison_summary.csv", index=False)

    write_cdf_svg(
        metrics,
        "storage_nrmse",
        output_dir / "cdf_storage_nrmse_post2010_rs_sim.svg",
        "Storage nRMSE CDF (post-2010 RS-SIM)",
        "post2010",
        "RS_SIM",
    )
    write_cdf_svg(
        metrics,
        "release_nrmse",
        output_dir / "cdf_release_nrmse_post2010_rs_sim.svg",
        "Release nRMSE CDF (post-2010 RS-SIM)",
        "post2010",
        "RS_SIM",
    )
    write_cdf_svg(
        metrics,
        "release_transformed_nrmse",
        output_dir / "cdf_release_transformed_nrmse_post2010_rs_obs.svg",
        "Transformed Release nRMSE CDF (post-2010 RS-OBS)",
        "post2010",
        "RS_OBS",
    )
    if scatter_rows:
        write_storage_scatter_svg(pd.DataFrame(scatter_rows), output_dir / "storage_fraction_scatter_post2010_rs_sim.svg")
    if not metrics.empty:
        write_selected_timeseries(metrics, metadata, bounds, daily_cache, output_dir)

    print(f"Point-model validation outputs written to: {output_dir}")
    print(f"Reservoir metric rows: {len(metrics)}")
    print(f"Skipped reservoir/split rows: {len(skipped)}")
    if not summary.empty:
        print("\nModel comparison summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
