"""Model-output adapters for reservoir bound prediction experiments.

The point-model validation runner consumes one common weekly-bound table. These
helpers convert the current random-forest STARFIT-parameter workflow, observed
STARFIT labels, generic bounds, and future direct bound time series into that
common shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from run_rf_validation import (
    RANDOM_STATE,
    VALIDATION_LABEL_COLUMNS,
    params_to_curves,
    prepare_resopsus_rf_data,
)


PredictionType = Literal["starfit_parameters", "bound_timeseries"]


@dataclass(frozen=True)
class BoundPredictionSet:
    """Weekly flood/conservation bounds from one model."""

    model_name: str
    prediction_type: PredictionType
    bounds: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        frame = self.bounds.copy()
        frame.insert(0, "model_name", self.model_name)
        frame.insert(1, "prediction_type", self.prediction_type)
        return frame


@dataclass(frozen=True)
class BaselineBundle:
    """Current ResOpsUS baseline predictions and metadata."""

    bounds: pd.DataFrame
    reservoir_metadata: pd.DataFrame
    split_manifest: pd.DataFrame
    rf_parameter_predictions: pd.DataFrame
    observed_parameter_labels: pd.DataFrame
    feature_columns: list[str]
    oob_score: float


def starfit_parameters_to_weekly_bounds(
    model_name: str,
    dam_ids: pd.Series,
    parameters: pd.DataFrame,
) -> BoundPredictionSet:
    """Convert ten STARFIT parameters into 52 weekly bound rows per dam."""
    curves = params_to_curves(parameters[VALIDATION_LABEL_COLUMNS].to_numpy())
    dam_ids = dam_ids.reset_index(drop=True).astype(int)
    curves["dam_id"] = curves["sample"].map(lambda sample: int(dam_ids.iloc[int(sample)]))
    curves = curves.rename(columns={"flood": "flood_pct", "conservation": "conservation_pct"})
    return BoundPredictionSet(
        model_name=model_name,
        prediction_type="starfit_parameters",
        bounds=curves[["dam_id", "epiweek", "flood_pct", "conservation_pct"]],
    )


def direct_timeseries_to_weekly_bounds(
    model_name: str,
    bounds: pd.DataFrame,
    id_col: str = "dam_id",
    week_col: str = "epiweek",
    flood_col: str = "flood_pct",
    conservation_col: str = "conservation_pct",
) -> BoundPredictionSet:
    """Adapt direct weekly-bound model outputs to the shared table shape."""
    required = [id_col, week_col, flood_col, conservation_col]
    missing = [col for col in required if col not in bounds.columns]
    if missing:
        raise ValueError(f"Missing bound output columns: {missing}")

    frame = bounds[required].rename(
        columns={
            id_col: "dam_id",
            week_col: "epiweek",
            flood_col: "flood_pct",
            conservation_col: "conservation_pct",
        }
    )
    frame = frame.copy()
    frame["dam_id"] = frame["dam_id"].astype(int)
    frame["epiweek"] = frame["epiweek"].astype(int)
    return BoundPredictionSet(
        model_name=model_name,
        prediction_type="bound_timeseries",
        bounds=frame[["dam_id", "epiweek", "flood_pct", "conservation_pct"]],
    )


def generic_weekly_bounds(
    model_name: str,
    dam_ids: pd.Series,
    flood_pct: float = 75.0,
    conservation_pct: float = 10.0,
) -> BoundPredictionSet:
    """Create PCR-GLOBWB-style fixed 10/75 storage bounds."""
    rows = []
    for dam_id in dam_ids.astype(int):
        for week in range(1, 53):
            rows.append(
                {
                    "dam_id": int(dam_id),
                    "epiweek": week,
                    "flood_pct": flood_pct,
                    "conservation_pct": conservation_pct,
                }
            )
    return BoundPredictionSet(
        model_name=model_name,
        prediction_type="bound_timeseries",
        bounds=pd.DataFrame(rows),
    )


def classify_reservoir_use(row: pd.Series) -> str:
    """Group GeoDAR use flags into the two HESS 2025 point-model types."""
    irrigation_like_flags = ["use_Irrigation", "use_Water Supply"]
    if any(float(row.get(flag, 0) or 0) > 0 for flag in irrigation_like_flags):
        return "irrigation_like"
    return "hydropower_like"


def build_resopsus_baseline_bundle(
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
    n_estimators: int = 100,
) -> BaselineBundle:
    """Train the current RF baseline and return comparable weekly bounds."""
    features, labels, ids = prepare_resopsus_rf_data()
    split = train_test_split(
        features,
        labels,
        ids,
        test_size=test_size,
        random_state=random_state,
    )
    train_features, test_features, train_labels, test_labels, train_ids, test_ids = split

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, oob_score=True)
    model.fit(train_features, train_labels)
    rf_predictions = pd.DataFrame(model.predict(test_features), columns=VALIDATION_LABEL_COLUMNS)
    test_labels = test_labels.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True).astype(int)

    observed_bounds = starfit_parameters_to_weekly_bounds("observed_starfit", test_ids, test_labels)
    rf_bounds = starfit_parameters_to_weekly_bounds("rf_starfit", test_ids, rf_predictions)
    generic_bounds = generic_weekly_bounds("generic_10_75", test_ids)
    bounds = pd.concat(
        [rf_bounds.to_frame(), observed_bounds.to_frame(), generic_bounds.to_frame()],
        ignore_index=True,
    )

    test_meta = test_features.reset_index(drop=True).copy()
    test_meta.insert(0, "dam_id", test_ids)
    test_meta = test_meta.rename(columns={"cap": "capacity_mcm"})
    test_meta["use_category"] = test_meta.apply(classify_reservoir_use, axis=1)

    split_manifest = pd.concat(
        [
            pd.DataFrame({"dam_id": train_ids.reset_index(drop=True).astype(int), "split": "train"}),
            pd.DataFrame({"dam_id": test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    split_manifest["random_state"] = random_state
    split_manifest["test_size"] = test_size

    rf_parameter_predictions = rf_predictions.copy()
    rf_parameter_predictions.insert(0, "dam_id", test_ids)
    observed_parameter_labels = test_labels.copy()
    observed_parameter_labels.insert(0, "dam_id", test_ids)

    return BaselineBundle(
        bounds=bounds,
        reservoir_metadata=test_meta,
        split_manifest=split_manifest,
        rf_parameter_predictions=rf_parameter_predictions,
        observed_parameter_labels=observed_parameter_labels,
        feature_columns=list(features.columns),
        oob_score=float(model.oob_score_),
    )
