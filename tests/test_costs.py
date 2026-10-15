"""Implementation shortfall: book walk and decomposition against hand calculations."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import costs  # noqa: E402

BIDS = [(99.9, 10.0), (99.7, 10.0)]
ASKS = [(100.1, 1.0), (100.3, 5.0)]
EX = {"taker_share": 0.3, "maker_fee": 0.00015, "taker_fee": 0.00045, "unfilled_bps": 50.0}


class BookWalk(unittest.TestCase):
    def test_walk_stops_inside_the_level_that_completes_the_order(self):
        fills, left, levels = costs.walk_book([(100.0, 1.0), (101.0, 2.0)], 250.0)
        self.assertEqual(levels, 2)
        self.assertEqual(left, 0.0)
        self.assertAlmostEqual(fills[1][1], 150.0 / 101.0)

    def test_thin_book_reports_what_it_could_not_fill(self):
        fills, left, levels = costs.walk_book([(100.0, 1.0)], 250.0)
        self.assertAlmostEqual(left, 150.0)
        self.assertEqual(levels, 1)


class Shortfall(unittest.TestCase):
    def test_buy_through_two_levels(self):
        r = costs.taker_shortfall(BIDS, ASKS, True, 200.0, 0.00045)
        vwap = 200.0 / (1.0 + 99.9 / 100.3)
        self.assertAlmostEqual(r["spread"], 10.0)                       # (100.1 - 100) / 100
        self.assertAlmostEqual(r["impact"], (vwap - 100.1) / 100 * 1e4)
        self.assertAlmostEqual(r["fee"], 4.5)
        self.assertAlmostEqual(r["total"], r["spread"] + r["impact"] + r["fee"])
        self.assertEqual(r["fill"], 1.0)

    def test_small_sell_pays_half_the_spread_and_the_fee(self):
        r = costs.taker_shortfall(BIDS, ASKS, False, 50.0, 0.00045)
        self.assertAlmostEqual(r["spread"], 10.0)
        self.assertAlmostEqual(r["impact"], 0.0)

    def test_expected_cost_mixes_maker_and_taker(self):
        rate, d = costs.expected_cost(BIDS, ASKS, True, 300.0, EX)
        taker = costs.taker_shortfall(BIDS, ASKS, True, 90.0, 0.00045)["total"]
        self.assertAlmostEqual(rate * 1e4, 0.7 * 1.5 + 0.3 * taker)
        self.assertAlmostEqual(d["bps"], rate * 1e4)

    def test_book_too_thin_for_the_taker_share_costs_extra(self):
        thin = [(100.1, 0.1)]
        rate, d = costs.expected_cost(BIDS, thin, True, 1000.0, EX)
        self.assertLess(d["fill"], 1.0)
        self.assertGreater(rate * 1e4, 0.3 * 0.7 * 50.0)

    def test_realised_cost_of_passive_and_aggressive_fills(self):
        fills = [{"sz": 6.0, "px": 99.95, "fee": 0.09, "crossed": False},
                 {"sz": 4.0, "px": 100.08, "fee": 0.18, "crossed": True}]
        r = costs.realised_shortfall(fills, 100.0, 1000.0)
        vwap = (6 * 99.95 + 4 * 100.08) / 10
        self.assertAlmostEqual(r["vwap"], vwap)
        self.assertAlmostEqual(r["price"], (vwap - 100.0) * 10 / 1000 * 1e4)
        self.assertAlmostEqual(r["fee"], 0.27 / 1000 * 1e4)
        self.assertAlmostEqual(r["maker"], 6 * 99.95 / (6 * 99.95 + 4 * 100.08))

    def test_a_sell_above_the_arrival_mid_is_a_gain(self):
        r = costs.realised_shortfall([{"sz": -5.0, "px": 100.05, "fee": 0.075, "crossed": False}], 100.0, -500.0)
        self.assertLess(r["price"], 0)


if __name__ == "__main__":
    unittest.main()
