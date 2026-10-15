"""History: the account value per minute between runs and the 15-minute candles built from it."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import report  # noqa: E402

MIN = report.MIN


class Candles:
    """1-minute closes for one coin: price 10 + minute index / 100."""
    def candles(self, coin, interval, start, end):
        return [(t, 0, 0, 0, 10 + (t - T0) / MIN / 100, 0) for t in range(start // MIN * MIN, end, MIN)]


T0 = 1_800_000_000_000 // (15 * MIN) * (15 * MIN)


class EquityPath(unittest.TestCase):
    def test_minutes_between_runs_follow_the_held_positions(self):
        with tempfile.TemporaryDirectory() as d:
            h = report.History(os.path.join(d, "h.json"))
            h.extend_equity(Candles(), {}, 1000.0, {}, T0)
            # 20 minutes later: 10 coins held, now worth 10.20 each, account value 1002
            h.extend_equity(Candles(), {"X": 10.0}, 1002.0, {"X": 10.20}, T0 + 20 * MIN)
            v = h.h["m1"]["v"]
            self.assertEqual(len(v), 21)
            self.assertAlmostEqual(v[10], 1002.0 + 10 * (10.10 - 10.20))
            self.assertAlmostEqual(v[-1], 1002.0)
            q0 = h.h["m15"][0]
            self.assertEqual(q0[0], T0)
            self.assertEqual(q0[1], 1000.0)
            self.assertAlmostEqual(q0[4], v[14])
            self.assertAlmostEqual(q0[2], max(v[:15]))
            h.save()
            self.assertTrue(os.path.exists(os.path.join(d, "h.json")))

    def test_events_keep_simulated_and_realised_cost(self):
        with tempfile.TemporaryDirectory() as d:
            h = report.History(os.path.join(d, "h.json"))
            why = {"risk_before": 50, "risk_after": 20, "benefit": 1.0, "cost": 0.4, "hold_risk_cost": 1.2,
                   "full_cost": 0.9, "opening": [], "closing": []}
            fills = [{"coin": "X", "sz": 10.0, "px": 10.01, "fee": 0.015, "crossed": False, "time": T0 / 1000}]
            h.add_event(T0, why, {"X": 100.0}, {"X": 10.0}, {"X": {"bps": 2.6}}, fills, {"fees": 0.015})
            ev = h.h["events"][0]
            self.assertEqual(ev["trades"][0]["kind"], "rebalance")
            self.assertAlmostEqual(ev["trades"][0]["realised"]["price"], 10.0, places=3)
            self.assertEqual(h.traded, {"X"})
            self.assertEqual(h.h["volume"][0][1], 100.1)


if __name__ == "__main__":
    unittest.main()
