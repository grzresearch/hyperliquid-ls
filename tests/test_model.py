"""Hedge, targets and value at risk on synthetic prices with known betas."""
import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import model  # noqa: E402


def synthetic(days=140, seed=3, extra=None):
    """8h closes where daily returns of AAA = 1.5 x ETH + noise, BBB = 0.8 x ETH + noise, alt CCC = 1.0 x ETH."""
    rnd = random.Random(seed)
    n = days * 3
    coins = {"ETH": 1.0, "AAA": 1.5, "BBB": 0.8, "CCC": 1.0, **(extra or {})}
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

    def test_a_short_book_is_sized_so_the_beta_cancels(self):
        snap = synthetic()
        tg, h = model.targets(snap, {"weights": {"AAA": 1, "BBB": 1}, "shorts": {"CCC": 0.5, "ETH": 0.5},
                                     "window": 90}, 10000.0)
        self.assertEqual(h["mode"], "minvar")
        self.assertAlmostEqual(h["H"] * h["b_short"], h["b_book"])
        self.assertAlmostEqual(tg["CCC"], tg["ETH"])

    def test_without_a_usable_short_book_eth_alone_hedges(self):
        snap = synthetic(extra={"ZZZ": 0.1})            # hardly moves with ETH
        for shorts in ({}, {"ZZZ": 1.0}):
            tg, h = model.targets(snap, {"weights": {"AAA": 1, "BBB": 1}, "shorts": shorts, "window": 90}, 10000.0)
            self.assertEqual(h["mode"], "eth")
            self.assertEqual(set(h["k"]), {"ETH"})
            self.assertAlmostEqual(-tg["ETH"], h["b_book"] * 10000.0)
            self.assertNotIn("ZZZ", tg)

    def test_water_filling_keeps_the_total_and_respects_every_cap(self):
        out = model.water_fill({"A": 500, "B": 300, "C": 200}, {"A": 300})
        self.assertAlmostEqual(out["A"], 300)
        self.assertAlmostEqual(out["B"] / out["C"], 1.5)               # the rest shares in proportion
        self.assertAlmostEqual(sum(out.values()), 1000)
        tight = model.water_fill({"A": 500, "B": 500}, {"A": 100, "B": 200})
        self.assertEqual(tight, {"A": 100, "B": 200})                   # all capped: the total shrinks

    def test_a_capped_long_hands_its_excess_to_the_others(self):
        snap = synthetic()
        s = {"weights": {"AAA": 1, "BBB": 1}, "shorts": {"CCC": 1.0}, "window": 90}
        tg, h = model.targets(snap, s, 10000.0, {"AAA": 3000.0})
        self.assertAlmostEqual(tg["AAA"], 3000.0)
        self.assertAlmostEqual(tg["AAA"] + tg["BBB"], 10000.0)
        self.assertEqual(set(h["capped"]), {"AAA"})
        b = model.betas(h["rows"], ["AAA", "BBB", "CCC", "ETH"])
        self.assertLess(abs(sum(b[c] * v for c, v in tg.items())), 0.002 * 10000)   # still beta-neutral

    def test_a_capped_short_gives_its_beta_to_eth(self):
        snap = synthetic()
        s = {"weights": {"AAA": 1, "BBB": 1}, "shorts": {"CCC": 1.0}, "window": 90}
        free, _ = model.targets(snap, s, 10000.0)
        tg, h = model.targets(snap, s, 10000.0, {"CCC": 4000.0})
        self.assertAlmostEqual(tg["CCC"], -4000.0)
        b = model.betas(h["rows"], ["AAA", "BBB", "CCC", "ETH"])
        self.assertAlmostEqual(tg["ETH"], -(-free["CCC"] - 4000.0) * b["CCC"], places=6)
        self.assertLess(abs(sum(b[c] * v for c, v in tg.items())), 0.002 * 10000)
        self.assertAlmostEqual(h["H"], -(tg["CCC"] + tg["ETH"]) / 10000.0)

    def test_eth_held_long_keeps_its_own_hedge_ratio(self):
        snap = synthetic()
        tg, h = model.targets(snap, {"weights": {"ETH": 1, "AAA": 1}, "shorts": {}, "window": 90}, 10000.0)
        self.assertEqual(h["mode"], "eth")
        self.assertAlmostEqual(h["k"]["ETH"], h["b_book"])            # the hedge, not what is left after netting
        self.assertAlmostEqual(tg["ETH"], 5000.0 - h["b_book"] * 10000.0)

    def test_the_fixed_basket_reports_its_shares_after_the_limits(self):
        snap = synthetic()
        s = {"weights": {"AAA": 1, "BBB": 1}, "alts": ["CCC"], "window": 90, "ethShare": 40}
        tg, h = model.targets(snap, s, 10000.0, {"CCC": 1000.0})
        self.assertAlmostEqual(h["share"], h["k"]["ETH"] / h["H"])
        self.assertAlmostEqual(h["H"] * h["b_short"], h["b_book"])
        self.assertGreater(h["share"], 0.4)

    def test_forced_coins_go_to_their_target_whatever_the_cost(self):
        rows = model.return_rows(model.context(synthetic(), {"weights": {"AAA": 1}, "alts": ["CCC"]}), 420, 270)
        current, target = {"AAA": 10000.0, "ETH": -15000.0}, {"AAA": 5000.0, "ETH": -7500.0}
        cost = {"AAA": 0.01, "ETH": 0.01}                              # far too expensive for the drift rule
        free, _ = model.decide(current, target, rows, cost, 10000.0, 3.0, 25.0)
        forced, why = model.decide(current, target, rows, cost, 10000.0, 3.0, 25.0, force={"AAA", "ETH"})
        self.assertEqual(free, {})
        self.assertEqual(forced, {"AAA": -5000.0, "ETH": 7500.0})
        self.assertEqual(why["resized"], ["AAA", "ETH"])
        self.assertEqual(why["opening"], [])

    def test_drift_trades_never_push_a_coin_past_its_jump_limit(self):
        rnd = random.Random(4)
        rows = []
        for _ in range(300):                       # AAA and BBB share a factor of their own
            f, e = rnd.gauss(0, 0.04), rnd.gauss(0, 0.03)
            rows.append({"AAA": e + f + rnd.gauss(0, 0.005), "BBB": e + f + rnd.gauss(0, 0.005), "ETH": e})
        current = {"AAA": 1000.0, "BBB": 1000.0, "ETH": -4000.0}
        target = {"AAA": 1000.0, "BBB": 3000.0, "ETH": -4000.0}   # BBB under its target, its book expensive
        cost = {"AAA": 0.0002, "BBB": 0.05, "ETH": 0.0002}
        free, _ = model.decide(current, target, rows, cost, 10000.0, 3.0, 25.0)
        self.assertGreater(free.get("AAA", 0.0), 100)                      # AAA stands in for BBB ...
        held, _ = model.decide(current, target, rows, cost, 10000.0, 3.0, 25.0, caps={"AAA": 1000.0})
        self.assertLessEqual(current["AAA"] + held.get("AAA", 0.0), 1000.0 + 1e-6)   # ... but not past its cap

    def test_a_riskier_market_makes_the_bot_trade_back_sooner(self):
        rows = model.return_rows(model.context(synthetic(), {"weights": {"AAA": 1}, "alts": ["CCC"]}), 420, 270)
        current, target = {"AAA": 10600.0, "ETH": -15000.0}, {"AAA": 10000.0, "ETH": -15000.0}
        cost = {"AAA": 0.0004, "ETH": 0.0002}
        calm, _ = model.decide(current, target, rows, cost, 10000.0, 3.0, 25.0, risk_scale=0.25)
        wild, _ = model.decide(current, target, rows, cost, 10000.0, 3.0, 25.0, risk_scale=4.0)
        self.assertGreater(abs(sum(wild.values())), abs(sum(calm.values())))


if __name__ == "__main__":
    unittest.main()
