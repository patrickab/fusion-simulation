import unittest

from src.engine.network import _Patience


class PatienceStopperTest(unittest.TestCase):
    def test_stops_after_patience_without_meaningful_rolling_improvement(self) -> None:
        stopper = _Patience(patience=2, min_relative_improvement=0.01, window=2)

        self.assertEqual(stopper.update(1, 10.0), (False, False))
        self.assertEqual(stopper.update(2, 10.0), (False, True))
        self.assertEqual(stopper.update(3, 9.95), (False, False))
        self.assertEqual(stopper.update(4, 9.95), (True, False))
        self.assertEqual(stopper.best_epoch, 2)

    def test_meaningful_rolling_improvement_resets_patience(self) -> None:
        stopper = _Patience(patience=2, min_relative_improvement=0.01, window=2)

        stopper.update(1, 10.0)
        stopper.update(2, 10.0)
        self.assertEqual(stopper.update(3, 9.7), (False, True))
        self.assertEqual(stopper.rounds_without_improvement, 0)
        self.assertEqual(stopper.best_epoch, 3)


if __name__ == "__main__":
    unittest.main()
