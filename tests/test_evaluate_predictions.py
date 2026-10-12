"""Check M3 aggregation, reference edge cases, and saved-forecast alignment."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utilities.evaluate_predictions import (evaluate, load_forecasts, load_observations,
                                           naive2_forecasts, symmetric_ape)
from utilities.load_data import HORIZON, ROOT


def observations():
    frames = []
    for series_id, level in [("A", 10.), ("B", 20.)]:
        frames.append(pd.DataFrame({
            "series_id": series_id, "month_index": np.arange(1, 25 + HORIZON),
            "date": pd.date_range("2000-01-01", periods=24 + HORIZON, freq="MS"),
            "value": [level] * 24 + [level + 2] * HORIZON,
            "split": ["train"] * 24 + ["test"] * HORIZON,
        }))
    return pd.concat(frames, ignore_index=True)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data_path = self.root / "observations.csv"
        observations().to_csv(self.data_path, index=False)
        self.histories, self.expected = load_observations(self.data_path)
        self.prediction = self.expected[["series_id", "month_index", "date", "step"]].copy()
        self.prediction["prediction"] = self.expected["value"]

    def save_prediction(self, frame):
        path = self.root / "model.csv"
        frame.to_csv(path, index=False)
        return path

    def test_alignment_uses_keys_not_row_order(self):
        path = self.save_prediction(self.prediction.sample(frac=1, random_state=42))
        actual = self.expected["value"].to_numpy().reshape(-1, HORIZON)
        np.testing.assert_array_equal(load_forecasts(path, self.expected), actual)

    def test_rejects_incomplete_duplicate_wrong_step_date_and_nonfinite(self):
        cases = [self.prediction.iloc[:-1],
                 pd.concat([self.prediction, self.prediction.iloc[:1]])]
        for column, value in [("step", 2), ("date", "1990-01-01"),
                              ("prediction", np.inf), ("month_index", 24),
                              ("month_index", 25.5)]:
            frame = self.prediction.copy()
            if column == "month_index":
                frame[column] = frame[column].astype(float)
            frame.loc[0, column] = value
            cases.append(frame)
        for frame in cases:
            with self.subTest(frame=frame.head(1).to_dict()):
                with self.assertRaises(ValueError):
                    load_forecasts(self.save_prediction(frame), self.expected)

    def test_five_metrics_and_average_rank_ties(self):
        actual = self.expected["value"].to_numpy().reshape(-1, HORIZON)
        reference = actual - 2
        tables = evaluate(self.expected, {"perfect": actual, "duplicate": reference,
                                         "naive2": reference}, "naive2", "m3")
        scores = tables["summary"].query("horizons == '1-18'").set_index("model")
        self.assertEqual(scores.loc["perfect", "smape"], 0)
        self.assertEqual(scores.loc["perfect", "median_sape"], 0)
        self.assertEqual(scores.loc["perfect", "average_rank"], 1)
        self.assertEqual(scores.loc["perfect", "percentage_better"], 100)
        self.assertEqual(scores.loc["perfect", "median_rae"], 0)
        self.assertEqual(scores.loc["duplicate", "average_rank"], 2.5)
        self.assertEqual(scores.loc["duplicate", "percentage_better"], 0)
        self.assertEqual(scores.loc["duplicate", "median_rae"], 1)

    def test_summary_averages_horizon_medians_not_series_means(self):
        expected = pd.concat([self.expected, self.expected.iloc[:HORIZON].assign(series_id="C")])
        expected = expected.reset_index(drop=True)
        actual = expected["value"].to_numpy().reshape(-1, HORIZON)
        forecast = actual.copy()
        # Rotating the one bad series leaves each horizon's median at zero,
        # while all three per-series mean sAPEs are strictly positive.
        for step in range(HORIZON):
            forecast[step % 3, step] = 0
        tables = evaluate(expected, {"rotating": forecast, "naive2": actual - 1}, "naive2", "m3")
        score = tables["summary"].query("model == 'rotating' and horizons == '1-18'").iloc[0]
        self.assertEqual(score["median_sape"], 0)
        self.assertAlmostEqual(score["percentage_better"], 100 * 2 / 3)
        self.assertAlmostEqual(score["smape"], 200 / 3)

    def test_nonzero_metrics_match_hand_calculation(self):
        actual = self.expected["value"].to_numpy().reshape(-1, HORIZON)
        forecast = actual - np.array([[4.], [6.]])
        tables = evaluate(self.expected, {"model": forecast, "naive2": actual - 2}, "naive2", "m3")
        score = tables["summary"].query("model == 'model' and horizons == '1-18'").iloc[0]
        expected_sape = (200 * 4 / (12 + 8) + 200 * 6 / (22 + 16)) / 2
        self.assertAlmostEqual(score["smape"], expected_sape)
        self.assertAlmostEqual(score["median_sape"], expected_sape)
        self.assertEqual(score["median_rae"], 2.5)
        self.assertEqual(score["average_rank"], 2)
        self.assertEqual(score["percentage_better"], 0)

    def test_zero_reference_errors_are_explicit(self):
        actual = self.expected["value"].to_numpy().reshape(-1, HORIZON)
        tables = evaluate(self.expected, {"wrong": actual + 1, "naive2": actual}, "naive2", "m3")
        scores = tables["summary"].query("horizons == '1-18'").set_index("model")
        self.assertTrue(np.isinf(scores.loc["wrong", "median_rae"]))
        self.assertTrue(np.isnan(scores.loc["naive2", "median_rae"]))
        self.assertEqual(scores.loc["naive2", "rae_defined"], 0)
        self.assertEqual(scores.loc["wrong", "rae_infinite"], 2 * HORIZON)

    def test_smape_conventions(self):
        actual, forecast = np.array([2., 0.]), np.array([-4., 0.])
        np.testing.assert_array_equal(symmetric_ape(actual, forecast, "m3"), [-600., 0.])
        np.testing.assert_array_equal(symmetric_ape(actual, forecast, "absolute"), [200., 0.])
        with self.assertRaises(ValueError):
            symmetric_ape(np.array([2.]), np.array([-2.]), "m3")

    def test_naive2_restores_seasonal_phase_and_short_history_fallback(self):
        seasonal = np.tile(np.arange(1., 13.), 4)
        reference, fallbacks = naive2_forecasts({"seasonal": seasonal, "short": np.arange(1., 9.)})
        np.testing.assert_allclose(reference[0], np.tile(np.arange(1., 13.), 2)[:HORIZON])
        np.testing.assert_array_equal(reference[1], np.full(HORIZON, 8.))
        self.assertEqual(fallbacks, ["short"])

    def test_cli_outputs_and_empty_directory_failure(self):
        predictions = self.root / "predictions"
        predictions.mkdir()
        output = self.root / "evaluation"
        command = [sys.executable, "-m", "utilities.evaluate_predictions", "--input", str(self.data_path),
                   "--predictions-dir", str(predictions), "--output-dir", str(output)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No prediction CSVs", result.stderr)
        self.assertFalse(output.exists())
        self.prediction.sample(frac=1, random_state=42).to_csv(predictions / "perfect.csv", index=False)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(p.name for p in output.iterdir()),
                         {"summary.csv", "per_horizon.csv", "per_series.csv", "per_forecast.csv", "metadata.json"})
        self.assertEqual(json.loads((output / "metadata.json").read_text())["evaluation_split"], "test")
        summary = pd.read_csv(output / "summary.csv")
        self.assertEqual(summary.query("model == 'perfect' and horizons == '1-18'").iloc[0]["smape"], 0)
        per_horizon = pd.read_csv(output / "per_horizon.csv")
        self.assertEqual(len(per_horizon), 2 * HORIZON)

    def test_cli_rejects_bad_model_before_writing_results(self):
        predictions = self.root / "predictions"
        predictions.mkdir()
        self.prediction.to_csv(predictions / "valid.csv", index=False)
        self.prediction.iloc[:-1].to_csv(predictions / "incomplete.csv", index=False)
        output = self.root / "evaluation"
        result = subprocess.run(
            [sys.executable, "-m", "utilities.evaluate_predictions", "--input", str(self.data_path),
             "--predictions-dir", str(predictions), "--output-dir", str(output)],
            cwd=ROOT, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing=1", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
