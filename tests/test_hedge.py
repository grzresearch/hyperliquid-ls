"""The short side on a synthetic market: one ETH factor, two sectors, and noise."""
import os
import random
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import hedge  # noqa: E402

DEFI = ["D1", "D2", "D3", "D4", "D5", "D6"]
L1 = ["L1", "L2", "L3", "L4", "L5"]


def market(days=270, seed=3):
    """ETH drives everything; DeFi names share a second factor the L1s do not have."""
    rnd = random.Random(seed)
    rows = []
    for _ in range(days):
        e, d, l = rnd.gauss(0, 0.03), rnd.gauss(0, 0.02), rnd.gauss(0, 0.02)
        r = {"ETH": e}
        for i, c in enumerate(DEFI):
            r[c] = 1.1 * e + 0.9 * d + rnd.gauss(0, 0.015 + 0.002 * i)
        for c in L1:
            r[c] = 1.0 * e + 0.8 * l + rnd.gauss(0, 0.015)
        rows.append(r)
    return rows


LONGS = {"D1": 0.4, "D2": 0.35, "D3": 0.25}
CANDS = ["ETH"] + DEFI[3:] + L1


class Projection(unittest.TestCase):
    def test_weights_sum_to_one_within_their_caps(self):
        w = hedge.project([0.9, 0.5, -0.2, 0.1], [0.35] * 4)
        self.assertAlmostEqual(sum(w), 1.0)
        self.assertTrue(all(0 <= x <= 0.35 + 1e-12 for x in w))
        self.assertAlmostEqual(w[0], 0.35)


class Choice(unittest.TestCase):
    def test_the_hedge_comes_from_the_same_sector_and_is_beta_neutral(self):
        rows = market()
        res = hedge.choose(rows, LONGS, CANDS)
        w, d = res["weights"], hedge.DEFAULTS
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertTrue(d["min_names"] <= len(w) <= d["max_names"])
        self.assertTrue(all(d["min_share"] - 1e-9 <= x <= d["max_share"] + 1e-9 for x in w.values()))
        defi = sum(x for c, x in w.items() if c in DEFI)
        self.assertGreater(defi, 0.5)                      # the DeFi factor can only be hedged with DeFi names
        pr = hedge.Problem(rows, LONGS, list(w), d["shrink"])
        self.assertAlmostEqual(res["H"] * sum(x * pr.beta[c] for c, x in w.items()), pr.beta_y, places=6)
        self.assertLess(res["resid_vol"], 0.85 * res["eth_only"])
        self.assertLess(res["oos"]["hedged"], res["oos"]["eth_only"])   # also on days it was not fitted on

    def test_without_a_tilt_the_short_score_changes_nothing(self):
        rows = market()
        score = {c: (2.0 if c in L1 else 0.0) for c in CANDS}
        res = hedge.choose(rows, LONGS, CANDS, score=score)
        self.assertEqual(res["weights"], hedge.choose(rows, LONGS, CANDS)["weights"])
        self.assertEqual(res["mu"], 0.0)

    def test_the_tilt_buys_a_better_short_score_for_at_most_ten_percent_more_variance(self):
        rows = market()
        plain = hedge.choose(rows, LONGS, CANDS)
        score = {c: (2.0 if c in L1 else 0.0) for c in CANDS}
        tilted = hedge.choose(rows, LONGS, CANDS, {"tilt_slack": 0.10}, score=score)
        lift = lambda r: sum(x * score.get(c, 0) for c, x in r["weights"].items())   # noqa: E731
        self.assertGreater(lift(tilted), lift(plain))
        self.assertLessEqual(tilted["V"], 1.10 * tilted["V_best"] + 1e-12)     # the optimiser's (shrunk) variance
        self.assertAlmostEqual(tilted["V_best"], plain["V"])

    def test_the_current_shorts_stay_while_they_hedge_almost_as_well(self):
        rows = market()
        first = hedge.choose(rows, LONGS, CANDS)
        again = hedge.choose(market(seed=4), LONGS, CANDS, held=first["weights"])
        self.assertTrue(again["kept"])
        self.assertEqual(set(again["weights"]), set(first["weights"]))

    def test_a_held_short_whose_funding_got_too_expensive_is_dropped_without_an_error(self):
        rows = market()
        first = hedge.choose(rows, LONGS, CANDS)
        top = max(first["weights"], key=first["weights"].get)
        for price in (1.0, 1.5):                          # cap 0 at funding_max and beyond
            res = hedge.choose(rows, LONGS, CANDS, held=first["weights"], cost={top: price})
            self.assertNotIn(top, res["weights"])

    def test_a_kept_short_outside_the_shortlist_is_still_listed(self):
        rows = market()
        first = hedge.choose(rows, LONGS, CANDS)
        again = hedge.choose(rows, LONGS, CANDS, {"shortlist": 6}, held=first["weights"])
        self.assertTrue(again["kept"])
        listed = {c["coin"] for c in again["candidates"]}
        self.assertLessEqual(set(again["weights"]), listed)
        self.assertGreater(len(listed), 6)                 # the outsiders came on top of the shortlist

    def test_a_name_far_more_volatile_than_the_others_gets_a_smaller_cap(self):
        rows = market()
        rnd = random.Random(9)
        for r in rows:                      # D4 with a lot of extra noise of its own
            r["D4"] = r["D4"] + rnd.gauss(0, 0.06)
        res = hedge.choose(rows, LONGS, CANDS)
        caps = {x["coin"]: x["cap"] for x in res["candidates"]}
        vols = sorted(statistics.stdev([r[c] for r in rows]) for c in CANDS)
        median = vols[len(vols) // 2]
        top = hedge.DEFAULTS["max_share"]
        self.assertAlmostEqual(caps["D4"], top * median / statistics.stdev([r["D4"] for r in rows]), places=6)
        self.assertLess(caps["D4"], 0.6 * top)
        self.assertLessEqual(res["weights"].get("D4", 0.0), caps["D4"] + 1e-9)

    def test_expensive_funding_shrinks_and_then_removes_a_short(self):
        self.assertEqual(hedge.funding_multiplier(0.10), 1.0)
        self.assertAlmostEqual(hedge.funding_multiplier(0.625), 0.5)
        self.assertEqual(hedge.funding_multiplier(1.2), 0.0)
        rows = market()
        best = max(hedge.choose(rows, LONGS, CANDS)["weights"].items(), key=lambda kv: kv[1])[0]
        res = hedge.choose(rows, LONGS, CANDS, cost={best: 1.5})
        self.assertNotIn(best, res["weights"])


if __name__ == "__main__":
    unittest.main()
