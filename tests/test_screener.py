"""Screener pieces on small, hand-made inputs shaped like DefiLlama, GitHub and Snapshot responses."""
import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import screener as S  # noqa: E402

DAY = S.DAY_MS
NOW = 1_800_000_000_000
CFG = {"n_longs": 3, "buffer": 2, "hold_floor": -0.25, "max_weight": 0.5, "min_weight": 0.1}


class Mapping(unittest.TestCase):
    parents = [{"id": "parent#lighter", "name": "Lighter", "symbol": "LIT", "gecko_id": "lighter", "mcap": 7.5e8,
                "github": None, "governanceID": None},
               {"id": "parent#aerodrome", "name": "Aerodrome", "symbol": "AERO", "gecko_id": "aerodrome-finance",
                "mcap": 8.5e8, "github": ["aerodrome-finance"], "governanceID": None}]
    protocols = [{"id": "1933", "name": "Timeless Finance", "symbol": "LIT", "gecko_id": "timeless", "mcap": 5e5},
                 {"id": "4524", "name": "Aerodrome Slipstream", "symbol": "AERO", "parentProtocol": "parent#aerodrome"},
                 {"id": "900", "name": "Aerodrome V1", "symbol": "AERO", "parentProtocol": "parent#aerodrome"},
                 {"id": "382", "name": "Pendle V2", "symbol": "PENDLE", "parentProtocol": "parent#pendle", "mcap": 0}]

    def test_symbol_collisions_go_to_the_largest_market_cap(self):
        ent, owner = S.llama_entities(self.protocols, self.parents)
        m = S.map_coins(["LIT", "AERO", "XYZ"], ent, 3e7)
        self.assertEqual(m["LIT"]["name"], "Lighter")
        self.assertEqual(m["AERO"]["id"], "parent#aerodrome")
        self.assertNotIn("XYZ", m)
        self.assertEqual(owner["4524"], "parent#aerodrome")

    def test_fees_of_child_protocols_add_up_and_growth_needs_both_windows(self):
        ent, owner = S.llama_entities(self.protocols, self.parents)
        overview = {"protocols": [
            {"defillamaId": "4524", "parentProtocol": "parent#aerodrome", "total30d": 9e6, "total60dto30d": 6e6,
             "total7d": 2.4e6, "total14dto7d": 2e6},
            {"defillamaId": "900", "parentProtocol": "parent#aerodrome", "total30d": 1e6, "total60dto30d": None}]}
        a = S.aggregate(overview, owner)["parent#aerodrome"]
        self.assertEqual(a["d30"], 10e6)
        self.assertAlmostEqual(S.growth(a), __import__("math").log(9 / 6))
        self.assertAlmostEqual(S.growth(a, short=True), __import__("math").log(1.2))


class Catalysts(unittest.TestCase):
    def test_release_kinds_and_fading(self):
        rel = [{"tag_name": "v1.4.2", "published_at": "2027-01-01T00:00:00Z"},
               {"tag_name": "v2.0.0", "published_at": "2027-01-10T00:00:00Z"},
               {"tag_name": "v2.1.0", "published_at": "2027-01-12T00:00:00Z"},
               {"tag_name": "v2.1.1", "published_at": "2027-01-13T00:00:00Z", "prerelease": True}]
        now = S.calendar.timegm((2027, 1, 14, 0, 0, 0)) * 1000
        ev = S.release_events(rel, now)
        self.assertEqual([e["kind"] for e in ev], ["Major", "Minor"])   # the oldest has no predecessor: left out
        score = S.catalyst_score(ev, now)
        self.assertAlmostEqual(score, 1.0 * 0.5 ** (4 / 14) + 0.5 * 0.5 ** (2 / 14))
        # with the whole history at hand the oldest one is the first release and counts as a launch
        self.assertEqual([e["kind"] for e in S.release_events(rel, now, first_known=True)], ["Major", "Major", "Minor"])

    def test_snapshot_counts_only_passed_proposals(self):
        res = {"data": {"proposals": [
            {"title": "Activate fee switch on v3", "end": (NOW - 2 * DAY) / 1000, "choices": ["For", "Against"],
             "scores": [900, 10], "space": {"id": "x.eth"}},
            {"title": "Fund a grant", "end": (NOW - 3 * DAY) / 1000, "choices": ["For", "Against"],
             "scores": [5, 900], "space": {"id": "x.eth"}},
            {"title": "Elect council", "end": (NOW - 3 * DAY) / 1000, "choices": ["Alice", "Bob"],
             "scores": [10, 5], "space": {"id": "x.eth"}}]}}
        with mock.patch.object(S, "fetch_json", return_value=res):
            ev = S.snapshot_events(["x.eth"], NOW)
        self.assertEqual(len(ev["x.eth"]), 1)
        self.assertEqual(ev["x.eth"][0]["weight"], 0.6)


class Selection(unittest.TestCase):
    def cands(self, scores, vol=None):
        return {c: {"score": s, "eligible": True, "vol": (vol or {}).get(c, 0.04)} for c, s in scores.items()}

    def test_normal_scores_are_rank_based_and_centered(self):
        z = S.normal_scores({"a": 1.0, "b": 100.0, "c": 2.0, "d": None})
        self.assertEqual(sorted(z, key=z.get), ["a", "c", "b"])
        self.assertAlmostEqual(z["c"], 0.0)

    def test_a_held_name_stays_within_the_buffer(self):
        c = self.cands({"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5, "F": 0.4})
        self.assertEqual(S.select(c, held={"E"}, cfg=CFG), ["E", "A", "B"])
        self.assertEqual(S.select(c, held={"F"}, cfg=CFG), ["A", "B", "C"])     # rank 6 > 3 + 2

    def test_negative_scores_only_fill_up_to_three(self):
        c = self.cands({"A": 0.5, "B": -0.1, "C": -0.4, "D": -0.6})
        self.assertEqual(S.select(c, held=set(), cfg=CFG), ["A", "B", "C"])
        c = self.cands({"A": 0.5, "B": 0.3, "C": 0.1, "D": -0.2, "E": -0.3})
        self.assertEqual(S.select(c, held=set(), cfg=dict(CFG, n_longs=5)), ["A", "B", "C"])

    def test_one_failing_organisation_does_not_end_github(self):
        def fake(url, headers=None, tries=2, **kw):
            if "/orgs/gone/" in url or "/users/gone/" in url:
                raise type("E", (Exception,), {"code": 404})()
            if url.endswith("/repos?sort=pushed&per_page=6"):
                return [{"name": "core", "full_name": "live/core"}]
            return [{"tag_name": "v1.0.0", "published_at": "2027-01-01T00:00:00Z"},
                     {"tag_name": "v2.0.0", "published_at": "2027-01-10T00:00:00Z"}]
        now = S.calendar.timegm((2027, 1, 14, 0, 0, 0)) * 1000
        with mock.patch.object(S, "fetch_json", side_effect=fake):
            ev = S.github_events(["gone", "live"], "token", now)
        self.assertEqual([(e["repo"], e["tag"]) for e in ev], [("core", "v1.0.0"), ("core", "v2.0.0")])
        def refused(url, **kw):
            raise type("E", (Exception,), {"code": 403})()
        with mock.patch.object(S, "fetch_json", side_effect=refused):
            with self.assertRaises(Exception):
                S.github_events(["live"], "token", now)

    def test_weights_are_capped_and_sum_to_one(self):
        c = self.cands({"A": 2.0, "B": 0.2, "C": 0.2}, vol={"A": 0.02})
        w = S.weights_for(["A", "B", "C"], c, CFG)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertLessEqual(max(w.values()), 0.5 + 1e-9)
        self.assertGreaterEqual(min(w.values()), 0.1 - 1e-9)

    def test_relative_momentum_removes_the_eth_part(self):
        import math
        import random
        rnd = random.Random(5)
        log_eth = [0.0]
        for _ in range(39):
            log_eth.append(log_eth[-1] + rnd.gauss(0.002, 0.03))
        eth = [100 * math.exp(v) for v in log_eth]
        coin = [50 * math.exp(2 * v) for v in log_eth]     # exactly twice ETH's log return: beta 2, nothing extra
        self.assertAlmostEqual(S.relative_momentum(coin, eth, 30), 0.0, places=9)
        lifted = [c * math.exp(0.01 * i) for i, c in enumerate(coin)]    # plus 1 % a day of its own
        self.assertAlmostEqual(S.relative_momentum(lifted, eth, 30), 0.30, places=6)


class Factors(unittest.TestCase):
    def test_window_sums_use_only_finished_days(self):
        series = [[NOW - i * DAY, 1.0] for i in range(90, -1, -1)]          # one per day up to and including today
        w = S.window_sums(series, NOW)
        self.assertEqual(w["d30"], 30.0)                                     # today's partial day is not counted
        self.assertEqual(w["h30"], 30.0)
        self.assertEqual((w["g7"], w["h7"]), (7.0, 7.0))
        self.assertIsNone(S.window_sums([[NOW - 10 * DAY, 1.0]], NOW))

    def test_fees_and_revenue_growth_count_once_and_buybacks_can_be_added(self):
        fees = {"d30": 2e6, "g30": 2e6, "h30": 1e6, "g7": 5e5, "h7": 5e5}
        rev = {"d30": 1e6, "g30": 1e6, "h30": 1e6}
        f = S.factor_values(fees, rev, None, None, 1e8, [], [], None, NOW)
        self.assertAlmostEqual(f["cashflow_growth"], (math.log(2) + 0.0) / 2)
        self.assertAlmostEqual(f["revenue_yield"], 1e6 * 365 / 30 / 1e8)
        self.assertEqual(f["holder_yield"], 0.0)
        self.assertIsNone(f["catalysts"])
        g = S.factor_values(fees, rev, {"d30": 1e5}, None, 1e8, [], [], [], NOW, holder_extra=5e6)
        self.assertAlmostEqual(g["holder_yield"], 0.05)                       # the configured buybacks win
        self.assertEqual(g["catalysts"], 0.0)

    def test_short_scores_turn_the_long_factors_around(self):
        rows = {"WEAK": {"momentum": -0.4, "cashflow_growth": -0.5}, "MID": {"momentum": 0.0},
                "STRONG": {"momentum": 0.5, "cashflow_growth": 0.6}}
        sc = S.short_scores(rows, {"momentum": 0.5, "cashflow_growth": 0.5})
        self.assertGreater(sc["WEAK"], sc["MID"])
        self.assertGreater(sc["MID"], sc["STRONG"])

    def test_expensive_funding_shrinks_a_long(self):
        c = {"A": {"score": 0.5, "vol": 0.04, "funding_mult": 1.0}, "B": {"score": 0.5, "vol": 0.04, "funding_mult": 0.5},
             "C": {"score": 0.5, "vol": 0.04, "funding_mult": 1.0}}
        w = S.weights_for(["A", "B", "C"], c, dict(CFG, max_weight=0.6, min_weight=0.0))
        self.assertAlmostEqual(w["B"], w["A"] / 2)


if __name__ == "__main__":
    unittest.main()
