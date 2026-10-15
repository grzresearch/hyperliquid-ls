"""Whole rounds against an offline exchange, and what keeps the public paper account safe between rounds:
restoring from the published page, retrying a failed screener, and a coin that disappears from the exchange."""
import io
import json
import math
import os
import random
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
            out[c] = {"closes": [math.exp(lp + b * (eth[i] - eth[-1]) + r.gauss(0, 0.01)) for i in range(n)]}
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
        self.assertLess(abs(after["book"]["net_beta"]), 0.05 * after["book"]["L"])
        self.assertNotIn("AAA", [p["coin"] for p in after["positions"] if p["target"]])


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
