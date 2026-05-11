import unittest

import numpy as np
import pandas as pd

from bound_model_adapters import (
    direct_timeseries_to_weekly_bounds,
    generic_weekly_bounds,
    starfit_parameters_to_weekly_bounds,
)
from point_reservoir_model import (
    ReservoirConfig,
    normalize_weekly_bounds,
    release_decision,
    simulate_reservoir,
)
from run_rf_validation import VALIDATION_LABEL_COLUMNS


class BoundAdapterTests(unittest.TestCase):
    def test_starfit_parameters_convert_to_weekly_bounds(self):
        params = pd.DataFrame(
            [[75, 0, 0, 100, 0, 20, 0, 0, 100, 0]],
            columns=VALIDATION_LABEL_COLUMNS,
        )
        result = starfit_parameters_to_weekly_bounds("constant_starfit", pd.Series([123]), params)
        frame = result.to_frame()

        self.assertEqual(len(frame), 52)
        self.assertEqual(set(frame["dam_id"]), {123})
        self.assertTrue(np.allclose(frame["flood_pct"], 75))
        self.assertTrue(np.allclose(frame["conservation_pct"], 20))
        self.assertEqual(result.prediction_type, "starfit_parameters")

    def test_generic_bounds_are_10_75_for_each_week(self):
        result = generic_weekly_bounds("generic", pd.Series([1, 2]))
        frame = result.to_frame()

        self.assertEqual(len(frame), 104)
        self.assertTrue(np.allclose(frame["flood_pct"], 75))
        self.assertTrue(np.allclose(frame["conservation_pct"], 10))

    def test_direct_timeseries_adapter_shape(self):
        source = pd.DataFrame(
            {
                "reservoir": [5, 5],
                "week": [1, 2],
                "upper": [70, 72],
                "lower": [30, 31],
            }
        )
        result = direct_timeseries_to_weekly_bounds(
            "chronos_placeholder",
            source,
            id_col="reservoir",
            week_col="week",
            flood_col="upper",
            conservation_col="lower",
        )

        self.assertEqual(result.prediction_type, "bound_timeseries")
        self.assertEqual(result.bounds.columns.tolist(), ["dam_id", "epiweek", "flood_pct", "conservation_pct"])


class PointReservoirTests(unittest.TestCase):
    def weekly_bounds(self, flood_pct=75, conservation_pct=25):
        return pd.DataFrame(
            {
                "epiweek": range(1, 53),
                "flood_pct": flood_pct,
                "conservation_pct": conservation_pct,
            }
        )

    def test_normalized_bounds_remain_physical(self):
        bounds = self.weekly_bounds(flood_pct=20, conservation_pct=90)
        normalized = normalize_weekly_bounds(bounds, capacity_mcm=100)

        self.assertTrue((normalized["flood_mcm"] <= 100).all())
        self.assertTrue((normalized["conservation_mcm"] >= 0).all())
        self.assertTrue(((normalized["flood_mcm"] - normalized["conservation_mcm"]) >= 5 - 1e-9).all())

    def test_release_decision_conserves_non_negative_storage(self):
        config = ReservoirConfig(dam_id=1, capacity_mcm=100, use_category="irrigation_like")
        decision = release_decision(
            storage_state_mcm=5,
            inflow_mcm_day=1,
            average_inflow_mcm_day=20,
            demand_mcm_day=100,
            environmental_flow_mcm_day=2,
            flood_bound_mcm=75,
            conservation_bound_mcm=25,
            config=config,
        )

        self.assertGreaterEqual(decision["simulated_storage_mcm"], 0)
        self.assertLessEqual(decision["simulated_release_mcm_day"], 6)

    def test_simulation_one_step_water_balance(self):
        config = ReservoirConfig(dam_id=1, capacity_mcm=100, use_category="hydropower_like")
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01"]),
                "epiweek": [1],
                "observed_storage_mcm": [50.0],
                "observed_release_mcm_day": [3.0],
                "inflow_mcm_day": [4.0],
            }
        )
        demand = pd.Series({week: 2.0 for week in range(1, 53)})
        sim = simulate_reservoir(
            daily,
            self.weekly_bounds(),
            demand,
            average_inflow_mcm_day=3.0,
            config=config,
            mode="RS_SIM",
        )

        before = sim.loc[0, "current_storage_before_release_mcm"]
        release = sim.loc[0, "simulated_release_mcm_day"]
        after = sim.loc[0, "simulated_storage_mcm"]
        self.assertAlmostEqual(after, before - release)


if __name__ == "__main__":
    unittest.main()
