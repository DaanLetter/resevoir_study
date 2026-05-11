"""Validate Daan's random-forest STARFIT parameter workflow.

This is a cleaned, reproducible version of the core logic in Run_RF_Model.py.
It follows the HESS 2025 validation idea: compare RF-extrapolated seasonal
bounds against STARFIT-derived bounds for held-out reservoirs. It runs three
available parameter sources:

- ResOpsUS STARFIT parameters;
- GloLakes/Sentinel2 STARFIT parameters;
- GloLakes/ICESat2 STARFIT parameters.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from validation_layer import STARFIT_PARAMETER_NAMES, bound_curve_metrics, starfit_parameter_metrics


PROJECT_DIR = Path(__file__).resolve().parent
RF_DIR = PROJECT_DIR / "Data" / "RF"
OUTPUT_DIR = PROJECT_DIR / "validation_outputs" / "rf_starfit_validation"
RANDOM_STATE = 42

LABEL_COLUMNS = [
    "flood_p1",
    "flood_p2",
    "flood_p3",
    "max_flood",
    "min_flood",
    "conserve_p1",
    "conserve_p2",
    "conserve_p3",
    "conserve_max",
    "conserve_min",
]

VALIDATION_LABEL_COLUMNS = [
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


def create_sinusoidal_curve(alpha: float, beta: float, mu: float, upper: float, lower: float) -> np.ndarray:
    weeks = np.arange(1, 53, dtype=float)
    values = alpha + beta * np.sin(2 * math.pi * weeks / 52) + mu * np.cos(2 * math.pi * weeks / 52)
    return np.clip(values, lower, upper)


def params_to_curves(params: pd.DataFrame | np.ndarray, min_active_zone: float = 5.0) -> pd.DataFrame:
    params = pd.DataFrame(params, columns=VALIDATION_LABEL_COLUMNS)
    rows = []
    for idx, row in params.iterrows():
        flood = create_sinusoidal_curve(
            row["flood_p1"],
            row["flood_p2"],
            row["flood_p3"],
            row["flood_max"],
            row["flood_min"],
        )
        conservation = create_sinusoidal_curve(
            row["conserve_p1"],
            row["conserve_p2"],
            row["conserve_p3"],
            row["conserve_max"],
            row["conserve_min"],
        )
        flood = np.clip(flood, 0, 100)
        conservation = np.clip(conservation, 0, 100)
        conservation = np.minimum(conservation, flood - min_active_zone)
        conservation = np.clip(conservation, 0, 100)
        for week, flood_value, conservation_value in zip(range(1, 53), flood, conservation):
            rows.append(
                {
                    "sample": idx,
                    "epiweek": week,
                    "flood": flood_value,
                    "conservation": conservation_value,
                }
            )
    return pd.DataFrame(rows)


def constrain_bound_parameters(params: pd.DataFrame) -> pd.DataFrame:
    params = params.copy()
    max_cols = ["flood_max", "conserve_max"]
    min_cols = ["flood_min", "conserve_min"]
    for col in max_cols:
        params[col] = params[col].replace(np.inf, 100).replace(-np.inf, np.nan)
        params[col] = params[col].clip(0, 100)
    for col in min_cols:
        params[col] = params[col].replace(-np.inf, 0).replace(np.inf, np.nan)
        params[col] = params[col].clip(0, 100)
    return params


def feature_columns_from_inputs(geodar_inputs: pd.DataFrame, data: pd.DataFrame) -> list[str]:
    drop_columns = {
        "Unnamed: 0",
        "geodar_id",
        "grand_id",
        "grand_id_feature",
        "watershed_area",
        "discharge_anom_top",
        "discharge_anom_bottom",
        "use_Fisheries",
    }
    return [
        c
        for c in geodar_inputs.columns
        if c not in drop_columns and c in data.columns and c not in LABEL_COLUMNS
    ]


def finalize_model_data(
    data: pd.DataFrame,
    geodar_inputs: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    for column in ["dom_autumn", "dom_winter", "dom_spring", "dom_sumnmer", "autumn_inflow"]:
        if column in data.columns:
            data[column] = data[column].fillna(0)
    if "use_Navigation" in data.columns:
        data["use_Navigation"] = data["use_Navigation"].fillna(0)

    feature_columns = feature_columns_from_inputs(geodar_inputs, data)
    labels = data[LABEL_COLUMNS].rename(
        columns={
            "max_flood": "flood_max",
            "min_flood": "flood_min",
        }
    )
    labels = constrain_bound_parameters(labels)
    model_data = pd.concat(
        [data[[id_column]].reset_index(drop=True), labels.reset_index(drop=True), data[feature_columns].reset_index(drop=True)],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=VALIDATION_LABEL_COLUMNS + feature_columns).copy()
    return model_data[feature_columns], model_data[VALIDATION_LABEL_COLUMNS], model_data[id_column]


def prepare_resopsus_rf_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    turner_params = pd.read_csv(RF_DIR / "Turner_ResOpsUS_params.csv")
    geodar_inputs = pd.read_csv(RF_DIR / "random_forest_inputs_geodar_all.csv")
    linkage = pd.read_csv(RF_DIR / "geodar_hydrolakes.csv", usecols=["id_v11", "id_grd_v13"])

    turner_params = turner_params.merge(
        linkage.rename(columns={"id_grd_v13": "grand_id", "id_v11": "geodar_id"}),
        on="grand_id",
        how="left",
    )
    data = turner_params.merge(geodar_inputs, on="geodar_id", how="inner", suffixes=("", "_feature"))
    data["source_id"] = data["grand_id"]
    return finalize_model_data(data, geodar_inputs, "source_id")


def prepare_glolakes_rf_data(params_file: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    params = pd.read_csv(RF_DIR / params_file)
    geodar_inputs = pd.read_csv(RF_DIR / "random_forest_inputs_geodar_all.csv")
    linkage = pd.read_csv(RF_DIR / "geodar_hydrolakes.csv", usecols=["id_v11", "Hylak_id"])
    # Multiple GeoDAR dams can map to one HydroLAKES reservoir. For validating
    # the STARFIT/RF relation itself, keep one representative structure so the
    # sample size matches the parameter dataset rather than duplicating labels.
    linkage = linkage.dropna(subset=["Hylak_id", "id_v11"]).sort_values("id_v11").drop_duplicates("Hylak_id")
    params = params.merge(
        linkage.rename(columns={"Hylak_id": "hydrolakes_id", "id_v11": "geodar_id"}),
        on="hydrolakes_id",
        how="left",
    )
    data = params.merge(geodar_inputs, on="geodar_id", how="inner", suffixes=("", "_feature"))
    data["source_id"] = data["hydrolakes_id"]
    return finalize_model_data(data, geodar_inputs, "source_id")


def svg_scatter_parameter_metrics(param_metrics: pd.DataFrame, path: Path) -> None:
    width, height = 980, 460
    left, right, top, bottom = 72, 24, 44, 82
    inner_w = width - left - right
    inner_h = height - top - bottom
    values = param_metrics["rmse"].astype(float).to_numpy()
    labels = param_metrics["parameter"].tolist()
    vmax = max(np.nanmax(values), 1)
    bars = []
    for i, value in enumerate(values):
        x = left + i / len(values) * inner_w + 8
        bar_w = inner_w / len(values) - 16
        bar_h = value / vmax * inner_h
        y = top + inner_h - bar_h
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2563eb" opacity="0.86"/>')
        bars.append(
            f'<text x="{x + bar_w/2:.2f}" y="{height-bottom+14}" text-anchor="end" transform="rotate(-45 {x + bar_w/2:.2f} {height-bottom+14})" font-size="11" fill="#374151">{labels[i]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="26" font-size="20" font-weight="700" fill="#111827">RF Held-Out STARFIT Parameter RMSE</text>
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
<text x="{left-8}" y="{top+4}" text-anchor="end" font-size="12" fill="#4b5563">{vmax:.1f}</text>
<text x="{left-8}" y="{height-bottom+4}" text-anchor="end" font-size="12" fill="#4b5563">0</text>
{''.join(bars)}
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">RMSE (% storage)</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_curve_example_svg(observed: pd.DataFrame, predicted: pd.DataFrame, sample: int, path: Path, title: str) -> None:
    obs = observed[observed["sample"] == sample]
    pred = predicted[predicted["sample"] == sample]
    width, height = 860, 460
    left, right, top, bottom = 62, 24, 44, 46
    inner_w = width - left - right
    inner_h = height - top - bottom
    y_values = pd.concat([obs[["flood", "conservation"]], pred[["flood", "conservation"]]]).to_numpy(dtype=float)
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)
    y_pad = (y_max - y_min) * 0.10 if y_max > y_min else 1
    y_min -= y_pad
    y_max += y_pad

    def sx(weeks):
        return left + (np.asarray(weeks) - 1) / 51 * inner_w

    def sy(values):
        return top + (y_max - np.asarray(values)) / (y_max - y_min) * inner_h

    def line(df, col, color, dash=""):
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(sx(df["epiweek"]), sy(df[col])))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        return f'<polyline fill="none" stroke="{color}" stroke-width="2.2"{dash_attr} points="{points}" />'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="26" font-size="19" font-weight="700" fill="#111827">{title}</text>
<line x1="{left}" x2="{width-right}" y1="{height-bottom}" y2="{height-bottom}" stroke="#9ca3af"/>
<line x1="{left}" x2="{left}" y1="{top}" y2="{height-bottom}" stroke="#9ca3af"/>
{line(obs, "flood", "#1d4ed8")}
{line(obs, "conservation", "#b91c1c")}
{line(pred, "flood", "#60a5fa", "6 4")}
{line(pred, "conservation", "#fca5a5", "6 4")}
<line x1="570" x2="600" y1="54" y2="54" stroke="#1d4ed8" stroke-width="3"/><text x="608" y="58" font-size="12" fill="#374151">STARFIT flood</text>
<line x1="570" x2="600" y1="76" y2="76" stroke="#60a5fa" stroke-width="3" stroke-dasharray="6 4"/><text x="608" y="80" font-size="12" fill="#374151">RF flood</text>
<line x1="700" x2="730" y1="54" y2="54" stroke="#b91c1c" stroke-width="3"/><text x="738" y="58" font-size="12" fill="#374151">STARFIT conservation</text>
<line x1="700" x2="730" y1="76" y2="76" stroke="#fca5a5" stroke-width="3" stroke-dasharray="6 4"/><text x="738" y="80" font-size="12" fill="#374151">RF conservation</text>
<text x="{width/2}" y="{height-14}" text-anchor="middle" font-size="13" fill="#4b5563">epiweek</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" fill="#4b5563">storage bound (%)</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")


def run_one_dataset(name: str, features: pd.DataFrame, labels: pd.DataFrame, ids: pd.Series) -> dict[str, float | str]:
    dataset_dir = OUTPUT_DIR / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split = train_test_split(
        features,
        labels,
        ids,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )
    train_features, test_features, train_labels, test_labels, train_ids, test_ids = split
    split_manifest = pd.concat(
        [
            pd.DataFrame({"source_id": train_ids.reset_index(drop=True).astype(int), "split": "train"}),
            pd.DataFrame({"source_id": test_ids.reset_index(drop=True).astype(int), "split": "test"}),
        ],
        ignore_index=True,
    )
    split_manifest["dataset"] = name
    split_manifest["random_state"] = RANDOM_STATE
    split_manifest["test_size"] = 0.25
    split_manifest.to_csv(dataset_dir / "rf_split_manifest.csv", index=False)

    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, oob_score=True)
    model.fit(train_features, train_labels)
    predictions = pd.DataFrame(model.predict(test_features), columns=VALIDATION_LABEL_COLUMNS, index=test_labels.index)
    prediction_table = predictions.reset_index(drop=True).copy()
    prediction_table.insert(0, "source_id", test_ids.reset_index(drop=True).astype(int))
    prediction_table.to_csv(dataset_dir / "rf_parameter_predictions.csv", index=False)

    param_metrics = starfit_parameter_metrics(test_labels, predictions, VALIDATION_LABEL_COLUMNS)
    param_metrics.to_csv(dataset_dir / "rf_parameter_metrics.csv", index=False)

    observed_curves = params_to_curves(test_labels.to_numpy())
    predicted_curves = params_to_curves(predictions.to_numpy())
    curve_metrics = bound_curve_metrics(observed_curves, predicted_curves)
    curve_metrics.to_csv(dataset_dir / "rf_curve_metrics_overall.csv", index=False)

    per_sample_rows = []
    for sample in range(len(test_labels)):
        one = bound_curve_metrics(
            observed_curves[observed_curves["sample"] == sample],
            predicted_curves[predicted_curves["sample"] == sample],
        )
        row = {"sample": sample, "grand_id": test_ids.iloc[sample]}
        for _, metric_row in one.iterrows():
            row[f"{metric_row['bound']}_rmse"] = metric_row["rmse"]
            row[f"{metric_row['bound']}_bias"] = metric_row["bias"]
            row[f"{metric_row['bound']}_pearson_r"] = metric_row["pearson_r"]
        per_sample_rows.append(row)
    per_sample = pd.DataFrame(per_sample_rows)
    per_sample.to_csv(dataset_dir / "rf_curve_metrics_by_reservoir.csv", index=False)

    feature_importance = pd.DataFrame(
        {
            "feature": train_features.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(dataset_dir / "rf_feature_importance.csv", index=False)

    compact = pd.DataFrame(
        {
            "metric": [
                "training_reservoirs",
                "test_reservoirs",
                "features",
                "oob_score",
                "parameter_mean_rmse",
                "parameter_median_rmse",
                "flood_curve_rmse",
                "conservation_curve_rmse",
                "flood_curve_correlation",
                "conservation_curve_correlation",
                "flood_curve_bias",
                "conservation_curve_bias",
            ],
            "value": [
                len(train_features),
                len(test_features),
                len(train_features.columns),
                model.oob_score_,
                param_metrics["rmse"].mean(),
                param_metrics["rmse"].median(),
                curve_metrics.loc[curve_metrics["bound"] == "flood", "rmse"].iloc[0],
                curve_metrics.loc[curve_metrics["bound"] == "conservation", "rmse"].iloc[0],
                curve_metrics.loc[curve_metrics["bound"] == "flood", "pearson_r"].iloc[0],
                curve_metrics.loc[curve_metrics["bound"] == "conservation", "pearson_r"].iloc[0],
                curve_metrics.loc[curve_metrics["bound"] == "flood", "bias"].iloc[0],
                curve_metrics.loc[curve_metrics["bound"] == "conservation", "bias"].iloc[0],
            ],
        }
    )
    compact.insert(0, "dataset", name)
    compact.to_csv(dataset_dir / "compact_rf_validation_summary.csv", index=False)

    svg_scatter_parameter_metrics(param_metrics, dataset_dir / "parameter_rmse.svg")
    per_sample["mean_curve_rmse"] = per_sample[["flood_rmse", "conservation_rmse"]].mean(axis=1)
    best = int(per_sample.sort_values("mean_curve_rmse").iloc[0]["sample"])
    median = int(per_sample.sort_values("mean_curve_rmse").iloc[len(per_sample) // 2]["sample"])
    worst = int(per_sample.sort_values("mean_curve_rmse").iloc[-1]["sample"])
    for label, sample in [("best", best), ("median", median), ("worst", worst)]:
        grand_id = int(per_sample.loc[per_sample["sample"] == sample, "grand_id"].iloc[0])
        write_curve_example_svg(
            observed_curves,
            predicted_curves,
            sample,
            dataset_dir / f"curve_example_{label}_id_{grand_id}.svg",
            f"{label.title()} Held-Out Curve Match: ID {grand_id}",
        )

    flood = curve_metrics.loc[curve_metrics["bound"] == "flood"].iloc[0]
    conservation = curve_metrics.loc[curve_metrics["bound"] == "conservation"].iloc[0]
    print(f"Training reservoirs: {len(train_features)}")
    print(f"Test reservoirs: {len(test_features)}")
    print(f"Features: {len(train_features.columns)}")
    print(f"OOB score: {model.oob_score_:.3f}")
    print(f"Outputs written to: {dataset_dir}")
    print()
    print("Parameter metrics:")
    print(param_metrics[["parameter", "rmse", "bias", "pearson_r"]].to_string(index=False))
    print()
    print("Curve metrics:")
    print(curve_metrics[["bound", "rmse", "bias", "pearson_r"]].to_string(index=False))
    print()
    print("Top feature importances:")
    print(feature_importance.head(10).to_string(index=False))
    return {
        "dataset": name,
        "training_reservoirs": len(train_features),
        "test_reservoirs": len(test_features),
        "features": len(train_features.columns),
        "oob_score": model.oob_score_,
        "parameter_mean_rmse": param_metrics["rmse"].mean(),
        "flood_curve_rmse": flood["rmse"],
        "flood_curve_bias": flood["bias"],
        "flood_curve_correlation": flood["pearson_r"],
        "conservation_curve_rmse": conservation["rmse"],
        "conservation_curve_bias": conservation["bias"],
        "conservation_curve_correlation": conservation["pearson_r"],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "resopsus": prepare_resopsus_rf_data(),
        "glolakes_sentinel2": prepare_glolakes_rf_data("params_glolakes_no_inf.csv"),
        "glolakes_icesat2": prepare_glolakes_rf_data("params_glolakes_no_inf_iceSAT.csv"),
    }

    rows = []
    for name, (features, labels, ids) in datasets.items():
        print("\n" + "=" * 80)
        print(name)
        print("=" * 80)
        rows.append(run_one_dataset(name, features, labels, ids))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "all_datasets_rf_validation_summary.csv", index=False)
    print("\nCombined summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
