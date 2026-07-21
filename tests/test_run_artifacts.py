import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.engine.network_manager import _MetricsManager
from src.lib.run_artifacts import format_duration, write_json


class RunArtifactTests(unittest.TestCase):
    def test_metrics_manager_appends_complete_logging_windows(self) -> None:
        metrics = _MetricsManager(total_epochs=10)
        for epoch in range(10):
            metrics.log(
                epoch,
                loss=float(epoch + 1),
                residual=2e-3,
                boundary=0.0,
                val_kpis=(1e-4, 2e-4, 3e-4) if epoch == 9 else None,
                lr=5e-4,
                grad_norm=4e-2,
                epoch_time=1.0,
            )

        self.assertEqual(len(metrics.rows), 1)
        self.assertEqual(metrics.rows[0]["loss"], 5.5)
        self.assertEqual(metrics.rows[0]["val_kpi_p05"], 1e-4)
        self.assertEqual(metrics.rows[0]["val_kpi_p50"], 2e-4)
        self.assertEqual(metrics.rows[0]["val_kpi_p95"], 3e-4)

    def test_six_hundred_epochs_produce_sixty_cumulative_rows(self) -> None:
        metrics = _MetricsManager(total_epochs=600)
        for epoch in range(600):
            metrics.log(
                epoch,
                loss=1.0,
                residual=2.0,
                boundary=3.0,
                val_kpis=(1e-4, 2e-4, 3e-4) if (epoch + 1) % 50 == 0 else None,
                lr=5e-4,
                grad_norm=4e-2,
                epoch_time=1.0,
            )

        self.assertEqual(len(metrics.rows), 60)
        self.assertEqual(sum(row["val_kpi_p50"] is not None for row in metrics.rows), 12)

    def test_live_dashboard_refresh_is_throttled_and_finishes_current(self) -> None:
        metrics = _MetricsManager(total_epochs=4)
        metrics._live = Mock()

        with patch(
            "src.engine.network_manager.time.monotonic", side_effect=(10.0, 10.4, 11.1, 11.2)
        ):
            for epoch in range(4):
                metrics.log(
                    epoch,
                    loss=1.0,
                    residual=2.0,
                    boundary=3.0,
                    val_kpis=None,
                    lr=5e-4,
                    grad_norm=4e-2,
                    epoch_time=1.0,
                )

        self.assertEqual(metrics._live.update.call_count, 3)
        self.assertTrue(all(call.kwargs["refresh"] for call in metrics._live.update.call_args_list))

    def test_scientific_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            write_json(path, {"lr": [0.0005], "loss": [0.123456], "value": None})

            text = path.read_text()
            self.assertIn("5.0000e-04", text)
            self.assertIn("1.2346e-01", text)
            self.assertEqual(json.loads(text)["lr"], [0.0005])

    def test_duration_format(self) -> None:
        self.assertEqual(format_duration(3723.2), "01h-02min-03sec")


if __name__ == "__main__":
    unittest.main()
