"""Hedge, targets and value at risk on synthetic prices with known betas."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import model  # noqa: E402


def synthetic(days=140, seed=3):
    """8h closes where daily returns of AAA = 1.5 x ETH + noise, BBB = 0.8 x ETH + noise, alt CCC = 1.0 x ETH."""
    rnd = random.Random(seed)
    n = days * 3
    coins = {"ETH": 1.0, "AAA": 1.5, "BBB": 0.8, "CCC": 1.0}
    px = {c: 100.0 for c in coins}
    closes = {c: [] for c in coins}
    for _ in range(n):
        m = rnd.gauss(0, 0.02)
        for c, b in coins.items():
            px[c] *= math.exp(b * m + (rnd.gauss(0, 0.01) if c != "ETH" else 0.0))
            closes[c].append(px[c])
    return {"perDay": 3, "t0": 0, "step": 8 * 3600000, "coins": {c: {"closes": v} for c, v in closes.items()}}


class Book(unittest.TestCase):
    def test_hedge_offsets_the_long_beta(self):
        snap = synthetic()
        s = {"weights": {"AAA": 1, "BBB": 1}, "alts": ["CCC"], "window": 90, "ethShare": 40}
        tg, h = model.targets(snap, s, 10000.0)
        self.assertAlmostEqual(h["b_book"], 1.15, delta=0.08)
        b = model.betas(h["rows"], ["AAA", "BBB", "CCC", "ETH"])
        net = sum(b[c] * v for c, v in tg.items())
        self.assertLess(abs(net), 150)                      # beta-neutral within the noise of the estimate
        self.assertAlmostEqual(tg["AAA"] + tg["BBB"], 10000.0)
        self.assertAlmostEqual(-tg["ETH"] / (-tg["ETH"] - tg["CCC"] * h["b_alt"]), 0.4, delta=0.02)

    def test_value_at_risk_matches_the_normal_case(self):
        rnd = random.Random(9)
        rows = [{"X": rnd.gauss(0, 0.03)} for _ in range(20000)]
        var = model.value_at_risk({"X": 10000.0}, rows)
        self.assertAlmostEqual(var["sigma"], 300.0, delta=10)
        self.assertAlmostEqual(var["95"]["param"], 1.645 * var["sigma"], delta=0.5)
        self.assertAlmostEqual(var["95"]["hist"], var["95"]["param"], delta=20)
        self.assertAlmostEqual(var["99"]["es_param"], 2.665 * var["sigma"], delta=1)
        self.assertGreater(var["99"]["es_hist"], var["99"]["hist"])

    def test_quantile_interpolates(self):
        self.assertAlmostEqual(model.quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)


if __name__ == "__main__":
    unittest.main()
