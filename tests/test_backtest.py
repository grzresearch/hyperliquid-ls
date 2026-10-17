"""The arithmetic that turns the backtest's measurements into weights."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest import __main__ as B  # noqa: E402


class Ranks(unittest.TestCase):
    def test_ties_share_their_rank_and_order_is_what_counts(self):
        self.assertEqual(B.ranks([3.0, 1.0, 3.0, 2.0]), [2.5, 0.0, 2.5, 1.0])
        self.assertAlmostEqual(B.spearman([1, 2, 3, 4], [10, 20, 30, 400]), 1.0)
        self.assertAlmostEqual(B.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(B.spearman([1, 1, 1], [1, 2, 3]))      # no spread, no correlation


class Weights(unittest.TestCase):
    def test_a_negative_factor_keeps_only_its_prior_half(self):
        prior = {"a": 0.5, "b": 0.5}
        summary = {"a": {"ir": 0.6}, "b": {"ir": -0.4}}
        w, measured = B.Backtest.weights_from(summary, prior)
        self.assertEqual(measured, {"a": 1.0, "b": 0.0})
        self.assertEqual(w, {"a": 0.75, "b": 0.25})

    def test_without_any_positive_factor_the_prior_stays(self):
        prior = {"a": 0.7, "b": 0.3}
        w, _ = B.Backtest.weights_from({"a": {"ir": -0.1}, "b": None}, prior)
        self.assertEqual(w, prior)

    def test_the_t_value_uses_every_other_date_because_the_windows_overlap(self):
        rows = [{"t": i, "ic": {"f": v}} for i, v in enumerate([0.3, -0.9, 0.1, -0.9, 0.2, -0.9])]
        s = B.Backtest.summary(rows, ["f"])["f"]
        self.assertAlmostEqual(s["ic"], -0.35)
        self.assertGreater(s["t"], 0)                 # only 0.3, 0.1, 0.2 count for t
        self.assertAlmostEqual(s["hit"], 0.5)


class Variants(unittest.TestCase):
    def test_every_hedge_variant_spells_out_all_its_settings(self):
        for name, v in B.HEDGE_VARIANTS.items():
            self.assertLessEqual(set(B.hedge.DEFAULTS), set(v), name)
        live = {k: v for k, v in B.hedge.DEFAULTS.items() if k != "holdout"}
        self.assertEqual({k: B.WIDE[k] for k in live}, live)        # the live defaults are the variant "wide"


if __name__ == "__main__":
    unittest.main()
