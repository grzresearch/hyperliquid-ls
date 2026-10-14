"""The shared maths in refresh.py, on the seed snapshot: python3 -m unittest discover -s tests"""
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import refresh as R  # noqa: E402
from bot.__main__ import DEFAULTS as BOT  # noqa: E402
from bot.model import decide as bot_decide  # noqa: E402

SNAP = R.load(os.path.join(ROOT, "seed", "market.json"))
AS_OF = R.to_ms(SNAP["asOf"])


def settings(**kw):
    s = R.settings({"weights": {"AERO": 20, "PENDLE": 20, "LIT": 7.5, "SYRUP": 20, "ETHFI": 13.5, "ZRO": 9, "ENA": 10}})
    s.update(kw)
    return s


class Plan(unittest.TestCase):
    def test_an_empty_account_opens_every_target_at_the_expected_cost(self):
        p = R.plan(SNAP, settings())
        self.assertEqual(sorted(t for t, *_ in p["orders"]), sorted(p["targets"]))
        self.assertEqual(p["why"]["opening"], sorted(p["targets"]))
        self.assertAlmostEqual(p["why"]["cost"], sum(abs(v) for v in p["targets"].values()) * R.COST, places=6)
        self.assertAlmostEqual(sum(v for v in p["targets"].values() if v > 0), 1.5 * 10000, places=6)

    def test_a_book_on_target_is_left_alone(self):
        p = R.plan(SNAP, settings())
        on_target = {t: v / SNAP["coins"][t]["mark"] for t, v in p["targets"].items()}
        q = R.plan(SNAP, settings(pos=on_target))
        self.assertEqual(q["orders"], [])
        self.assertLess(q["why"]["risk_before"], 1e-6)

    def test_positions_outside_the_strategy_are_left_alone(self):
        p = R.plan(SNAP, settings())
        pos = {t: v / SNAP["coins"][t]["mark"] for t, v in p["targets"].items()}
        pos["HYPE"] = 50.0                       # a HYPE perp: the strategy only uses HYPE as spot
        q = R.plan(SNAP, settings(pos=pos))
        self.assertEqual(q["orders"], [])

    def test_an_alt_taken_out_of_the_basket_is_closed(self):
        p = R.plan(SNAP, settings())
        pos = {t: v / SNAP["coins"][t]["mark"] for t, v in p["targets"].items()}
        alts = [t for t in settings()["alts"] if t != "XPL"]
        q = R.plan(SNAP, settings(pos=pos, alts=alts))
        xpl = [o for o in q["orders"] if o[0] == "XPL"]
        self.assertEqual(len(xpl), 1)
        self.assertEqual(xpl[0][3], "Position schließen")
        self.assertIn("XPL", q["why"]["closing"])

    def test_without_a_minimum_order_new_positions_still_open_in_full(self):
        p = R.plan(SNAP, settings(minTrade=0))
        self.assertEqual(p["why"]["opening"], sorted(p["targets"]))
        for t, dq, dusd, *_ in p["orders"]:
            self.assertAlmostEqual(dusd, p["targets"][t], delta=SNAP["coins"][t]["mark"])

    def test_settings_saved_with_a_book_in_usd_keep_their_size(self):
        s = R.settings({"L": 3000, "equity": 10000, "band": 10})
        self.assertAlmostEqual(R.book_usd(s), 3000)  # 0,3 × Kontowert
        self.assertIn("kein Zielbuch", R.summary(SNAP, R.settings({"equity": 0})))

    def test_a_long_set_to_zero_is_closed_in_full(self):
        p = R.plan(SNAP, settings())
        pos = {t: v / SNAP["coins"][t]["mark"] for t, v in p["targets"].items()}
        q = R.plan(SNAP, settings(pos=pos, weights={**settings()["weights"], "ZRO": 0}))
        zro = [o for o in q["orders"] if o[0] == "ZRO"]
        self.assertEqual(len(zro), 1)
        self.assertAlmostEqual(zro[0][1], -round(pos["ZRO"], SNAP["coins"]["ZRO"]["szDec"]))
        self.assertEqual(q["why"]["closing"], ["ZRO"])

    def test_the_bot_decides_like_the_page(self):
        s = settings()
        p = R.plan(SNAP, s)
        pos = {t: v / SNAP["coins"][t]["mark"] * (1 + 0.3 * math.sin(i)) for i, (t, v) in enumerate(p["targets"].items())}
        current = {t: q * SNAP["coins"][t]["mark"] for t, q in pos.items()}
        cfg = {**BOT, "strategy": {**BOT["strategy"], "weights": s["weights"], "alts": s["alts"]}}
        mine, _ = bot_decide(current, p["targets"], p["rows"], {}, 10000.0, cfg)
        page, _ = R.decide(current, p["targets"], p["rows"], {}, 10000.0, s["riskAversion"], s["minTrade"])
        self.assertEqual(mine, page)
        self.assertTrue(page)                    # 30 % off is far enough out to trade something


class Simulation(unittest.TestCase):
    def test_without_risk_aversion_it_is_buy_and_hold(self):
        """Nothing but the opening trades, so the value must match buy and hold computed from the closes."""
        s = settings(riskAversion=0, alts=[])
        r = R.simulate(SNAP, s, AS_OF - 60 * R.DAY)
        k0, C = r["k0"], SNAP["coins"]
        cx = R.context(SNAP, s)
        h = R.hedge_at(cx, k0 + 1, int(s["window"]) * cx["per"])
        tg = R.targets_for(cx, h, dict(s, L=R.book_usd(s)), C["HYPE"]["closes"][k0])
        self.assertEqual(r["book"]["trades"], len(tg))
        n, total = len(C["ETH"]["closes"]), 0.0
        for t, usd in tg.items():
            q = usd / C[t]["closes"][k0]
            total += q * (C[t]["mark"] - C[t]["closes"][k0]) - abs(usd) * R.COST
            for k in range(k0, n):
                begin = SNAP["t0"] + (k + 1) * SNAP["step"]
                end = begin + SNAP["step"] if k < n - 1 else AS_OF
                f = C[t]["fund"][(begin - SNAP["d0"]) // R.DAY] or 0.0
                total -= q * C[t]["closes"][k] * f / 1e4 * (end - begin) / R.HOUR / 24
        self.assertAlmostEqual(r["ls"][-1], total, places=6)

    def test_higher_risk_aversion_trades_more_often(self):
        trades = [R.simulate(SNAP, settings(riskAversion=ra), AS_OF - 60 * R.DAY)["book"]["trades"] for ra in (0.3, 3, 30)]
        self.assertLess(trades[0], trades[1])
        self.assertLess(trades[1], trades[2])


if __name__ == "__main__":
    unittest.main()
