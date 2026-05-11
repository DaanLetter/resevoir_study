"""Run a first validation-layer pass on ResOpsUS storage records.

This is not yet the RF-vs-STARFIT validation because the RF input/output CSVs
used by Daan's script are not present in this checkout. Instead, this runner
uses real ResOpsUS storage data to exercise the validation layer:

1. learn weekly p05/median/p95 storage bounds from the pre-2010 record;
2. validate those seasonal bounds on 2010 onward held-out observations;
3. write metric tables and SVG plots.

The seasonal median is a simple benchmark for a future point-reservoir model:
the real model should beat this on storage dynamics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from validation_layer import (
    bound_curve_metrics,
    bound_sanity_checks,
    metric_summary,
    storage_position_counts,
)


PROJECT_DIR = Path(__file__).resolve().parent
SCRIPTIE_DIR = PROJECT_DIR.parent
RESOPSUS_DIR = SCRIPTIE_DIR / "ResOpsUS"
TIME_SERIES_DIR = RESOPSUS_DIR / "time_series_all"
OUTPUT_DIR = PROJECT_DIR / "validation_outputs" / "resopsus_storage_bounds_demo"
CUTOFF_DATE = pd.Timestamp("2010-01-01")
MIN_TRAIN_DAYS = 365 * 5
MIN_VALIDATION_DAYS = 365 * 2


def epiweek(dates: pd.Series) -> pd.Series:
    week = dates.dt.isocalendar().week.astype(int)
    return week.clip(upper=52)


def read_storage_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["date", "storage"], na_values=["NA", ""])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["storage"] = pd.to_numeric(df["storage"], errors="coerce")
    df = df.dropna(subset=["date", "storage"])
    df = df[df["storage"] >= 0].copy()
    df["epiweek"] = epiweek(df["date"])
    return df.sort_values("date")


def weekly_bounds(train: pd.DataFrame) -> pd.DataFrame:
    grouped = train.groupby("epiweek")["storage"]
    bounds = grouped.quantile([0.05, 0.50, 0.95]).unstack()
    bounds.columns = ["conservation", "median", "flood"]
    bounds = bounds.reindex(range(1, 53)).interpolate(limit_direction="both")
    bounds.index.name = "epiweek"
    return bounds.reset_index()


def validation_weekly_quantiles(validation: pd.DataFrame) -> pd.DataFrame:
    grouped = validation.groupby("epiweek")["storage"]
    bounds = grouped.quantile([0.05, 0.50, 0.95]).unstack()
    bounds.columns = ["conservation", "median", "flood"]
    bounds = bounds.reindex(range(1, 53)).interpolate(limit_direction="both")
    bounds.index.name = "epiweek"
    return bounds.reset_index()


def validate_one(path: Path) -> tuple[dict[str, float | str] | None, pd.DataFrame | None, pd.DataFrame | None]:
    dam_id = path.stem.replace("ResOpsUS_", "")
    df = read_storage_file(path)
    if df.empty or df["storage"].nunique() < 10:
        return None, None, None

    train = df[df["date"] < CUTOFF_DATE].copy()
    validation = df[df["date"] >= CUTOFF_DATE].copy()
    if len(train) < MIN_TRAIN_DAYS or len(validation) < MIN_VALIDATION_DAYS:
        return None, None, None

    predicted_bounds = weekly_bounds(train)
    observed_bounds = validation_weekly_quantiles(validation)
    joined = validation.merge(predicted_bounds, on="epiweek", how="left")

    storage_metrics = metric_summary(joined["storage"], joined["median"])
    curve_metrics = bound_curve_metrics(
        observed_curves=observed_bounds,
        predicted_curves=predicted_bounds,
        observed_flood_col="flood",
        observed_conservation_col="conservation",
        predicted_flood_col="flood",
        predicted_conservation_col="conservation",
    )
    curve_by_bound = curve_metrics.set_index("bound")
    position = storage_position_counts(joined["storage"], joined["flood"], joined["conservation"])
    sanity = bound_sanity_checks(predicted_bounds, min_active_zone_fraction=0.0)

    summary = {
        "dam_id": dam_id,
        "train_days": float(len(train)),
        "validation_days": float(len(validation)),
        "train_start": str(train["date"].min().date()),
        "train_end": str(train["date"].max().date()),
        "validation_start": str(validation["date"].min().date()),
        "validation_end": str(validation["date"].max().date()),
        "storage_rmse_mcm": storage_metrics["rmse"],
        "storage_nrmse": storage_metrics["nrmse"],
        "storage_bias_mcm": storage_metrics["bias"],
        "storage_pearson_r": storage_metrics["pearson_r"],
        "storage_kge": storage_metrics["kge"],
        "flood_curve_rmse_mcm": curve_by_bound.loc["flood", "rmse"],
        "conservation_curve_rmse_mcm": curve_by_bound.loc["conservation", "rmse"],
        "above_flood_fraction": position["above_fraction"],
        "within_bounds_fraction": position["within_fraction"],
        "below_conservation_fraction": position["below_fraction"],
        "min_active_zone_mcm": sanity["min_active_zone"],
        "mean_active_zone_mcm": sanity["mean_active_zone"],
    }

    plot_df = joined[["date", "storage", "conservation", "median", "flood"]].copy()
    plot_df["dam_id"] = dam_id
    bounds_df = predicted_bounds.copy()
    bounds_df["dam_id"] = dam_id
    return summary, plot_df, bounds_df


def svg_polyline(points: list[tuple[float, float]], color: str, width: float = 2.0) -> str:
    if not points:
        return ""
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


def write_timeseries_svg(df: pd.DataFrame, title: str, path: Path, max_points: int = 900) -> None:
    width, height = 1100, 520
    left, right, top, bottom = 72, 30, 44, 56
    inner_w = width - left - right
    inner_h = height - top - bottom
    df = df.sort_values("date").copy()
    if len(df) > max_points:
        df = df.iloc[np.linspace(0, len(df) - 1, max_points).astype(int)].copy()

    x_vals = df["date"].map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    y_cols = ["storage", "flood", "median", "conservation"]
    y_vals = df[y_cols].to_numpy(dtype=float)
    x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
    y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    y_min -= y_pad
    y_max += y_pad

    def sx(x: np.ndarray) -> np.ndarray:
        return left + (x - x_min) / (x_max - x_min) * inner_w

    def sy(y: np.ndarray) -> np.ndarray:
        return top + (y_max - y) / (y_max - y_min) * inner_h

    lines = []
    colors = {
        "storage": "#1f2937",
        "flood": "#2563eb",
        "median": "#f59e0b",
        "conservation": "#dc2626",
    }
    for col in y_cols:
        mask = np.isfinite(df[col].to_numpy(dtype=float))
        points = list(zip(sx(x_vals[mask]), sy(df[col].to_numpy(dtype=float)[mask])))
        lines.append(svg_polyline(points, colors[col], 2.2 if col == "storage" else 1.8))

    y_ticks = np.linspace(y_min, y_max, 5)
    grid = []
    for tick in y_ticks:
        y = sy(np.array([tick]))[0]
        grid.append(f'<line x1="{left}" x2="{width-right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb" />')
        grid.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#4b5563">{tick:.0f}</text>'
        )

    legend = [
        ("observed storage", colors["storage"], 790, 30),
        ("flood p95", colors["flood"], 790, 52),
        ("seasonal median", colors["median"], 910, 30),
        ("conservation p05", colors["conservation"], 910, 52),
    ]
    legend_svg = []
    for label, color, x, y in legend:
        legend_svg.append(f'<line x1="{x}" x2="{x+24}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3" />')
        legend_svg.append(f'<text x="{x+30}" y="{y+4}" font-size="12" fill="#374151">{label}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="24" font-size="20" font-weight="700" fill="#111827">{title}</text>
{''.join(grid)}
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
{''.join(lines)}
{''.join(legend_svg)}
<text x="{width/2}" y="{height-14}" text-anchor="middle" font-size="13" fill="#4b5563">held-out validation period</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">storage (MCM)</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_histogram_svg(values: pd.Series, title: str, path: Path, clip_max: float | None = None) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if clip_max is not None:
        values = values.clip(upper=clip_max)
    width, height = 900, 460
    left, right, top, bottom = 68, 28, 44, 54
    inner_w = width - left - right
    inner_h = height - top - bottom
    counts, edges = np.histogram(values, bins=24)
    max_count = max(counts.max(), 1)
    bars = []
    for i, count in enumerate(counts):
        x = left + i / len(counts) * inner_w
        bar_w = inner_w / len(counts) - 2
        bar_h = count / max_count * inner_h
        y = top + inner_h - bar_h
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2563eb" opacity="0.82"/>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="26" font-size="20" font-weight="700" fill="#111827">{title}</text>
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
{''.join(bars)}
<text x="{width/2}" y="{height-14}" text-anchor="middle" font-size="13" fill="#4b5563">storage nRMSE{" (clipped)" if clip_max is not None else ""}</text>
<text x="{left}" y="{height-bottom+20}" text-anchor="middle" font-size="11" fill="#4b5563">{edges[0]:.2f}</text>
<text x="{width-right}" y="{height-bottom+20}" text-anchor="middle" font-size="11" fill="#4b5563">{edges[-1]:.2f}</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">reservoir count</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    plots = []
    bounds = []

    for path in sorted(TIME_SERIES_DIR.glob("ResOpsUS_*.csv")):
        summary, plot_df, bounds_df = validate_one(path)
        if summary is None:
            continue
        summaries.append(summary)
        plots.append(plot_df)
        bounds.append(bounds_df)

    summary_df = pd.DataFrame(summaries).sort_values("storage_nrmse")
    summary_path = OUTPUT_DIR / "resopsus_storage_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    all_bounds = pd.concat(bounds, ignore_index=True)
    all_bounds.to_csv(OUTPUT_DIR / "learned_weekly_bounds.csv", index=False)

    write_histogram_svg(
        summary_df["storage_nrmse"],
        "Held-Out Storage Validation: Seasonal Median Benchmark",
        OUTPUT_DIR / "storage_nrmse_histogram.svg",
    )
    write_histogram_svg(
        summary_df["storage_nrmse"],
        "Held-Out Storage Validation: Seasonal Median Benchmark",
        OUTPUT_DIR / "storage_nrmse_histogram_clipped_p95.svg",
        clip_max=float(summary_df["storage_nrmse"].quantile(0.95)),
    )

    compact_summary = pd.DataFrame(
        {
            "metric": [
                "validated_reservoirs",
                "median_storage_rmse_mcm",
                "median_storage_nrmse",
                "median_storage_kge",
                "median_flood_curve_rmse_mcm",
                "median_conservation_curve_rmse_mcm",
                "median_within_bounds_fraction",
                "p95_storage_nrmse",
            ],
            "value": [
                len(summary_df),
                summary_df["storage_rmse_mcm"].median(),
                summary_df["storage_nrmse"].median(),
                summary_df["storage_kge"].median(),
                summary_df["flood_curve_rmse_mcm"].median(),
                summary_df["conservation_curve_rmse_mcm"].median(),
                summary_df["within_bounds_fraction"].median(),
                summary_df["storage_nrmse"].quantile(0.95),
            ],
        }
    )
    compact_summary.to_csv(OUTPUT_DIR / "compact_validation_summary.csv", index=False)

    if not summary_df.empty:
        chosen = {
            "best": summary_df.iloc[0]["dam_id"],
            "median": summary_df.iloc[len(summary_df) // 2]["dam_id"],
            "worst": summary_df.iloc[-1]["dam_id"],
        }
        all_plots = pd.concat(plots, ignore_index=True)
        for label, dam_id in chosen.items():
            one = all_plots[all_plots["dam_id"] == dam_id]
            metric = summary_df[summary_df["dam_id"] == dam_id].iloc[0]
            title = (
                f"Dam {dam_id}: {label} seasonal benchmark "
                f"(RMSE={metric['storage_rmse_mcm']:.1f} MCM, nRMSE={metric['storage_nrmse']:.2f})"
            )
            write_timeseries_svg(one, title, OUTPUT_DIR / f"timeseries_{label}_dam_{dam_id}.svg")

    printed = summary_df[
        [
            "dam_id",
            "validation_days",
            "storage_rmse_mcm",
            "storage_nrmse",
            "storage_pearson_r",
            "storage_kge",
            "flood_curve_rmse_mcm",
            "conservation_curve_rmse_mcm",
            "within_bounds_fraction",
        ]
    ].head(10)
    print(f"Validated reservoirs: {len(summary_df)}")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print()
    print("Top 10 reservoirs by held-out storage nRMSE:")
    print(printed.to_string(index=False))
    print()
    print("Overall summary:")
    print(summary_df[["storage_rmse_mcm", "storage_nrmse", "within_bounds_fraction"]].describe().to_string())


if __name__ == "__main__":
    main()
