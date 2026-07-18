import json
from pathlib import Path
import tempfile
import unittest

from src.engine.network import _MetricsManager
from src.lib.run_artifacts import format_duration, write_json


class RunArtifactTests(unittest.TestCase):
    def test_metrics_manager_records_validation_percentiles_by_column(self) -> None:
        metrics = _MetricsManager(total_epochs=10)
        for epoch in range(10):
            metrics.log(
                epoch,
                loss=1e-3,
                residual=2e-3,
                boundary=0.0,
                val_kpis=(1e-4, 2e-4, 3e-4) if epoch == 9 else None,
                lr=5e-4,
                grad_norm=4e-2,
                epoch_time=1.0,
            )

        self.assertEqual(len(metrics.rows), 1)
        self.assertEqual(metrics.rows[0]["val_kpi_p05"], 1e-4)
        self.assertEqual(metrics.rows[0]["val_kpi_p50"], 2e-4)
        self.assertEqual(metrics.rows[0]["val_kpi_p95"], 3e-4)

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
