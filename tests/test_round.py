"""Whole rounds against an offline exchange, and what keeps the public paper account safe between rounds:
restoring from the published page, retrying a failed screener, and a coin that disappears from the exchange."""
import io
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
import unittest
import urllib.error
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bot.__main__ as M  # noqa: E402

MIN, DAY = 60000, 86400000
LONGS = ["AAA", "BBB", "CCC"]
ALTS = ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO"]


class FakeInfo:
    """Hyperliquid stand-in: prices that only move when told to, a two-level book, flat candles, and 8-hour closes
    that follow one common ETH path with a different beta per coin."""

    def __init__(self):
        coins = ["ETH"] + ALTS + LONGS + [f"FILL{i}" for i in range(14)]
        self.px = {c: 2500.0 if c == "ETH" else 10.0 * (i + 1) for i, c in enumerate(coins)}
        self.gone = set()
        self.jumps = {}          # coin -> (8h steps before the end, size): a move of its own

    def markets(self):
        return {c: {"mark": p, "mid": p, "funding": 1e-5, "maxLev": 10, "szDec": 2, "volume": 5e7, "oi": 1e7,
                    "prevDay": p} for c, p in self.px.items() if c not in self.gone}

    def book(self, c):
        p = self.px[c]
        return ([(p * (1 - 2e-4), 1e6 / p), (p * (1 - 6e-4), 1e6 / p)],
                [(p * (1 + 2e-4), 1e6 / p), (p * (1 + 6e-4), 1e6 / p)])

    def candles(self, coin, interval, start, end):
        step = {"1m": MIN, "1d": DAY}[interval]
        p = self.px[coin]
        return [(t, p, p, p, p, 0.0) for t in range(start // step * step, end + 1, step)]

    def funding_history(self, coin, start):
        return []

    def closes(self, coins, days):
        n, r = days * 3, random.Random(42)
        eth = [0.0]
        for _ in range(n - 1):
            eth.append(eth[-1] + r.gauss(0, 0.02))
        out = {}
        for k, c in enumerate(sorted(coins)):
            b = 0.8 + 0.04 * k
            lp = math.log(self.px[c])
            v = [math.exp(lp + b * (eth[i] - eth[-1]) + r.gauss(0, 0.01)) for i in range(n)]
            if c in self.jumps:
                back, size = self.jumps[c]
                v = [x / (1 + size) if i < n - back else x for i, x in enumerate(v)]
            out[c] = {"closes": v}
        return {"perDay": 3, "t0": 0, "step": 8 * 3600000, "coins": out}


class Quiet:
    def __init__(self):
        self.lines = []

    def say(self, text):
        self.lines.append(text)

    def record(self, row):
        pass


def picks(coins, t):
    return {"t": t, "tried": t, "picks": [{"coin": c, "weight": 1 / len(coins), "score": 0.5} for c in coins],
            "candidates": [], "sources": {}, "n_longs": 7, "pool": 0}


class Rounds(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state, self.site = os.path.join(self.tmp.name, "state"), os.path.join(self.tmp.name, "site")
        os.makedirs(self.state)
        with open(os.path.join(self.state, "screener.json"), "w") as f:
            json.dump(picks(LONGS, int(time.time() * 1000)), f)
        self.fake = FakeInfo()
        self.cfg = M.load_config(os.path.join(ROOT, "bot.paper.toml"))
        self.cfg["execution"].update(maker_wait_seconds=0.2, requote_seconds=0.05)

    def tearDown(self):
        self.tmp.cleanup()

    def bot(self):
        self.log = Quiet()
        with mock.patch.object(M, "Info", lambda *a, **k: self.fake):
            return M.Bot(self.cfg, "paper", self.state, self.log, site=self.site)

    def published(self):
        with open(os.path.join(self.site, "bot.json")) as f:
            return json.load(f)

    def test_a_delisted_long_is_settled_and_the_hedge_shrinks_with_the_book(self):
        self.bot().round()
        first = self.published()
        self.assertLess(abs(first["book"]["net_beta"]), 0.05 * first["book"]["L"])
        self.fake.px["AAA"] *= 1.10
        self.bot().round()                              # the account sees AAA's new price
        before = self.published()["account"]["equity"]
        self.fake.gone.add("AAA")
        b = self.bot()
        b.round()
        after = self.published()
        self.assertNotIn("AAA", b.broker.positions())
        self.assertTrue(any(line.startswith("AAA wird auf Hyperliquid nicht mehr gehandelt") for line in self.log.lines))
        self.assertGreater(b.broker.s["realized"], 0)    # the gain on AAA is booked, not dropped
        self.assertAlmostEqual(after["account"]["equity"], before, delta=0.002 * before)
        aimed = sum((p["beta"] or 0.0) * p["target"] for p in after["positions"])
        self.assertLess(abs(aimed), 0.02 * after["book"]["L"])        # the target hedge shrank with the book
        self.assertLess(abs(after["book"]["net_beta"]), 0.1 * after["book"]["L"])   # and most of it was traded
        self.assertNotIn("AAA", [p["coin"] for p in after["positions"] if p["target"]])


    def test_the_short_book_is_chosen_for_the_longs_and_stays_beta_neutral(self):
        self.bot().round()
        book = self.published()["book"]
        self.assertEqual(book["mode"], "minvar")
        shorts, settings = book["hedge"]["weights"], book["hedge"]["settings"]
        self.assertTrue(settings["min_names"] <= len(shorts) <= settings["max_names"])
        self.assertFalse(set(shorts) & set(LONGS))
        self.assertTrue(all(c in book["k"] for c in shorts))
        self.assertLess(abs(book["net_beta"]), 0.05 * book["L"])
        self.assertLess(book["hedge"]["oos"]["hedged"], book["hedge"]["oos"]["unhedged"])
        oos = book["hedge"]["oos"]
        self.assertEqual(oos["days_fit"] + oos["days_test"], book["hedge"]["days"])     # days, not 8-hour steps
        self.assertTrue(any(line.startswith("Shorts: ") for line in self.log.lines))

    def test_new_hedge_settings_take_effect_at_the_next_round_without_the_old_shorts(self):
        b = self.bot()
        b.round()
        first = self.published()["book"]["hedge"]
        top = max(first["weights"], key=first["weights"].get)
        self.cfg["hedge"]["exclude"] = ["HYPE", top]
        b.round()
        second = self.published()["book"]["hedge"]
        self.assertNotIn(top, second["weights"])
        self.assertIn(top, second["settings"]["exclude"])
        self.assertFalse(second["kept"])
        self.assertNotIn(top, [p["coin"] for p in self.published()["positions"] if p["target"]])

    def test_a_failed_new_choice_keeps_the_old_shorts_and_waits_before_trying_again(self):
        b = self.bot()
        b.round()
        old = b.shorts()
        self.cfg["hedge"]["max_names"] = 7
        now = int(time.time() * 1000)
        with mock.patch.object(b, "run_hedge", return_value=None) as run:
            b.refresh_hedge(now)
            b.refresh_hedge(now + 3600000)
            self.assertEqual(run.call_count, 1)
            b.refresh_hedge(now + (M.RETRY_HOURS + 1) * 3600000)
            self.assertEqual(run.call_count, 2)
        self.assertEqual(b.shorts(), old)

    def test_a_failed_choice_on_a_screener_run_keeps_the_last_book_and_retries_later(self):
        b = self.bot()
        b.round()
        old = b.shorts()
        now = int(time.time() * 1000) + 3600000
        fresh = picks(LONGS[:2] + [max(old, key=old.get)], now)      # one of the shorts is now a long
        fresh["factor_weights"] = self.cfg["screener"]["factors"]
        b.screener_mode = "now"
        with mock.patch.object(M.screener, "run", return_value=fresh), \
                mock.patch.object(b, "run_hedge", return_value=None) as run:
            b.run_screener([], now)
            self.assertEqual(b.shorts(), old)               # not the fixed basket, not ETH alone
            self.assertEqual(b.scr["hedge_tried"], now)
            b.refresh_hedge(now + 3600000)
            self.assertEqual(run.call_count, 1)             # the screener run's own attempt
            b.refresh_hedge(now + (M.RETRY_HOURS + 1) * 3600000)
            self.assertEqual(run.call_count, 2)             # the book was chosen for earlier longs: try again
        b.refresh_hedge(now + (2 * M.RETRY_HOURS + 2) * 3600000)
        self.assertGreaterEqual(b.scr["hedge"]["t"], b.scr["t"])
        self.assertNotIn(fresh["picks"][2]["coin"], b.shorts())
        with mock.patch.object(b, "run_hedge") as run:
            b.refresh_hedge(now + (3 * M.RETRY_HOURS + 3) * 3600000)
            self.assertEqual(run.call_count, 0)             # up to date: nothing more to do

    def test_without_any_short_book_eth_alone_hedges(self):
        b = self.bot()
        with mock.patch.object(b, "run_hedge", return_value=None):
            b.round()
        book = self.published()["book"]
        self.assertEqual(book["mode"], "eth")
        self.assertEqual(set(book["k"]), {"ETH"})
        self.assertFalse(set(ALTS) & {p["coin"] for p in self.published()["positions"] if p["target"]})
        self.assertLess(abs(book["net_beta"]), 0.05 * book["L"])

    def test_fixed_longs_are_hedged_with_eth_alone(self):
        scr = picks(LONGS, int(time.time() * 1000))
        scr["hedge"] = {"weights": {"FILL1": 0.5, "FILL2": 0.5}, "t": scr["t"]}     # chosen for the screener's longs
        with open(os.path.join(self.state, "screener.json"), "w") as f:
            json.dump(scr, f)
        self.cfg["screener"]["enabled"] = False
        self.cfg["strategy"]["weights"] = {"AAA": 1, "BBB": 2}
        self.bot().round()
        book = self.published()["book"]
        self.assertEqual(book["mode"], "eth")
        self.assertEqual(set(book["k"]), {"ETH"})

    def test_a_page_opened_before_the_short_book_still_finds_its_fields(self):
        self.bot().round()
        book = self.published()["book"]
        self.assertAlmostEqual(book["share"], book["k"].get("ETH", 0.0) / book["H"])
        self.assertEqual(book["b_alt"], book["b_short"])
        self.assertEqual(sorted(book["alts"]), sorted(c for c in book["k"] if c != "ETH"))

    def test_short_scores_are_only_computed_for_a_tilt(self):
        b = self.bot()
        b.round()
        sel = self.published()["book"]["hedge"]
        self.assertEqual(sel["mu"], 0.0)
        self.assertTrue(all(c["score"] is None for c in sel["candidates"]))
        self.cfg["hedge"]["tilt_slack"] = 0.10
        b.round()
        self.assertTrue(any(c["score"] is not None for c in self.published()["book"]["hedge"]["candidates"]))

    def test_a_long_that_jumped_on_its_own_is_capped_and_the_risk_is_published(self):
        self.fake.jumps["AAA"] = (300, -0.45)               # AAA fell 45 % in a day, three months ago
        self.bot().round()
        pub = self.published()
        book, r = pub["book"], pub["risk"]
        self.assertGreater(book["jumps"]["AAA"]["down"], 0.4)
        self.assertIn("AAA", book["capped"])
        aaa = next(p for p in pub["positions"] if p["coin"] == "AAA")
        cap = book["jump_loss"] * 10000.0 / book["jumps"]["AAA"]["down"]       # the account before the first trades
        self.assertLessEqual(aaa["target"], cap * 1.0001)
        self.assertTrue(any(line.startswith("Sprunglimits: AAA") for line in self.log.lines))
        for key in ("95", "99", "regime", "size", "risk_scale", "ref_sigma", "samples"):
            self.assertIn(key, r)
        self.assertGreaterEqual(r["99"]["es_hist"], r["99"]["hist"])
        self.assertTrue(0.5 <= r["size"] <= 1.0)
        self.assertTrue(0.0 <= r["regime"]["turbulent"] <= 1.0)

    def test_a_held_position_above_its_new_jump_limit_is_cut_back_at_once(self):
        self.bot().round()
        before = next(p for p in self.published()["positions"] if p["coin"] == "AAA")["value"]
        self.fake.jumps["AAA"] = (300, -0.45)               # the next snapshot shows a fall of its own
        self.bot().round()
        pub = self.published()
        aaa = next(p for p in pub["positions"] if p["coin"] == "AAA")
        cap = pub["book"]["jump_loss"] * pub["account"]["equity"] / pub["book"]["jumps"]["AAA"]["down"]
        self.assertGreater(before, 1.5 * cap)
        self.assertLess(aaa["value"], 1.05 * cap + 25)      # traded down now, not left to the drift rule
        self.assertIn("AAA", pub["decision"]["resized"])

    def test_a_book_far_above_its_size_is_shrunk_as_a_whole(self):
        self.bot().round()
        full = self.published()["account"]["long"]
        restless = lambda pnl, *rest: {"var": 4.0 * statistics.pvariance(pnl), "ref": statistics.pvariance(pnl),  # noqa: E731
                                "days": len(pnl), "tails": None, "garch": None, "regime": None}
        with mock.patch.object(M.risk, "outlook", side_effect=restless):
            self.bot().round()
        pub = self.published()
        self.assertAlmostEqual(pub["account"]["long"] / full, 0.5, delta=0.05)
        self.assertTrue(pub["decision"]["resized"])

    def test_without_a_forecast_the_book_keeps_its_last_size(self):
        restless = lambda pnl, *rest: {"var": 4.0 * statistics.pvariance(pnl), "ref": statistics.pvariance(pnl),  # noqa: E731
                                       "days": len(pnl), "tails": None, "garch": None, "regime": None}
        with mock.patch.object(M.risk, "outlook", side_effect=restless):
            self.bot().round()
        half = self.published()["account"]["long"]
        with mock.patch.object(M.risk, "book_outlook", side_effect=ZeroDivisionError("boom")):
            out = self.bot().round()
        self.assertEqual(out["trades"], {})                 # no swing back to the full book and down again
        self.assertAlmostEqual(self.published()["book"]["L"], 0.5 * self.cfg["book_per_side"] * 10000.0, delta=50)
        self.assertAlmostEqual(self.published()["account"]["long"], half, delta=0.02 * half)

    def test_a_target_the_margin_cannot_carry_is_made_smaller(self):
        self.cfg["execution"]["max_margin_use"] = 0.3
        self.bot().round()
        pub = self.published()
        self.assertTrue(any(line.startswith("Ziel auf") for line in self.log.lines))
        self.assertLessEqual(pub["account"]["margin"], 0.3 * pub["account"]["equity"] * 1.02)
        self.bot().round()
        self.assertEqual(self.published()["decision"]["resized"], [])     # the band does not fire in vain

    def test_a_failing_risk_layer_leaves_the_book_hedged_at_full_size(self):
        with mock.patch.object(M.risk, "book_outlook", side_effect=ZeroDivisionError("boom")):
            self.bot().round()
        pub = self.published()
        self.assertTrue(any(line.startswith("Risiko-Schicht fehlgeschlagen") for line in self.log.lines))
        self.assertAlmostEqual(pub["book"]["L"], self.cfg["book_per_side"] * 10000.0, delta=1.0)
        self.assertLess(abs(pub["book"]["net_beta"]), 0.05 * pub["book"]["L"])
        self.assertIsNone(pub["risk"])

    def test_positions_the_target_no_longer_holds_are_still_in_the_risk(self):
        b = self.bot()
        b.round()
        with open(os.path.join(self.state, "screener.json")) as f:
            scr = json.load(f)
        scr["picks"] = [p for p in scr["picks"] if p["coin"] != "CCC"]
        with open(os.path.join(self.state, "screener.json"), "w") as f:
            json.dump(scr, f)
        real = os.path.exists
        with mock.patch.object(M.os.path, "exists", side_effect=lambda p: p == "STOP" or real(p)):
            b = self.bot()                                  # a STOP file: no orders, CCC stays held
            b.round()
        self.assertIn("CCC", b.broker.positions())
        self.assertIn("CCC", b.snap["coins"])

    def test_a_restless_book_is_made_smaller_but_never_bigger(self):
        b = self.bot()
        with mock.patch.object(M.risk, "outlook", side_effect=lambda pnl, *rest: {
                "var": 4.0 * statistics.pvariance(pnl), "ref": statistics.pvariance(pnl), "days": len(pnl),
                "tails": {}, "garch": {}, "regime": None}):
            b.round()
        book = self.published()["book"]
        self.assertAlmostEqual(book["L"], 0.5 * self.cfg["book_per_side"] * 10000.0, delta=1.0)
        self.assertTrue(any("Buch 50 %" in line for line in self.log.lines))
        with mock.patch.object(M.risk, "outlook", side_effect=lambda pnl, *rest: {
                "var": 0.25 * statistics.pvariance(pnl), "ref": statistics.pvariance(pnl), "days": len(pnl),
                "tails": {}, "garch": {}, "regime": None}):
            view = b.risk_view(b.snap, {"AAA": 1000.0, "ETH": -800.0}, {"rows": [{"AAA": 0.01, "ETH": 0.01}] * 3}, 365)
        self.assertEqual(view["size"], 1.0)

    def test_the_fixed_basket_is_still_there(self):
        self.cfg["hedge"]["method"] = "fixed"
        self.cfg["strategy"]["alts"] = ALTS
        self.bot().round()
        book = self.published()["book"]
        self.assertEqual(book["mode"], "fixed")
        self.assertEqual(sorted(c for c in book["k"] if c != "ETH"), sorted(ALTS))
        self.assertLess(abs(book["net_beta"]), 0.05 * book["L"])


class ScreenerRetry(unittest.TestCase):
    def bot(self, d, scr=None):
        if scr:
            with open(os.path.join(d, "screener.json"), "w") as f:
                json.dump(scr, f)
        with mock.patch.object(M, "Info", lambda *a, **k: FakeInfo()):
            return M.Bot(M.load_config(os.path.join(ROOT, "bot.paper.toml")), "paper", d, Quiet())

    def test_a_failed_run_keeps_the_old_picks_and_waits_before_trying_again(self):
        now = 1_800_000_000_000
        with tempfile.TemporaryDirectory() as d:
            b = self.bot(d, picks(["A", "B", "C"], now - 30 * 3600000))
            with mock.patch.object(M.screener, "run", side_effect=RuntimeError("DefiLlama down")) as run:
                b.run_screener([], now)
                b.run_screener([], now + 3600000)
                self.assertEqual(run.call_count, 1)
                b.run_screener([], now + (M.RETRY_HOURS + 1) * 3600000)
                self.assertEqual(run.call_count, 2)
            self.assertEqual(sorted(b.longs()), ["A", "B", "C"])

    def test_fewer_than_three_longs_are_never_taken(self):
        with tempfile.TemporaryDirectory() as d:
            b = self.bot(d)
            with mock.patch.object(M.screener, "run", return_value=picks(["A"], 1)):
                b.run_screener([], 1_800_000_000_000)
            self.assertEqual(b.longs(), {})
            self.assertFalse(os.path.exists(os.path.join(d, "screener.json")))


class Restore(unittest.TestCase):
    def serve(self, files):
        def urlopen(url, timeout=None):
            body = files.get(url.rsplit("/", 1)[1], 404)
            if isinstance(body, int):
                raise urllib.error.HTTPError(url, body, "status", {}, None)
            return io.BytesIO(json.dumps(body).encode())
        return mock.patch("urllib.request.urlopen", urlopen)

    def test_only_a_page_without_data_starts_a_new_account(self):
        with tempfile.TemporaryDirectory() as d, self.serve({}):
            M.restore("https://x.github.io/r/data", d, lambda s: None)
            self.assertEqual(os.listdir(d), [])

    def test_any_other_failure_stops_the_run(self):
        with tempfile.TemporaryDirectory() as d, self.serve({"bot.json": 503}):
            with self.assertRaises(SystemExit):
                M.restore("https://x.github.io/r/data", d, lambda s: None)
            self.assertEqual(os.listdir(d), [])

    def test_account_and_screener_come_back_even_without_history(self):
        published = {"paper": {"start": 10000, "cash": 9990, "pos": {}}, "screener": picks(["A", "B", "C"], 1),
                     "account": {"equity": 9990}}
        with tempfile.TemporaryDirectory() as d, self.serve({"bot.json": published}):
            M.restore("https://x.github.io/r/data", d, lambda s: None)
            self.assertEqual(sorted(os.listdir(d)), ["paper.json", "screener.json"])


if __name__ == "__main__":
    unittest.main()
