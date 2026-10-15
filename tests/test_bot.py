"""python3 -m unittest discover -s tests"""
import math
import os
import random
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot.broker import PaperBroker, execute, maker_price  # noqa: E402
from bot.hl import round_price, tick  # noqa: E402
from bot.model import decide, optimal_trades  # noqa: E402
from bot.__main__ import DEFAULTS  # noqa: E402


def objective(t, delta, cov, cost, gamma):
    d = [a + b for a, b in zip(delta, t)]
    return gamma / 2 * sum(d[i] * cov[i][j] * d[j] for i in range(len(d)) for j in range(len(d))) \
        + sum(c * abs(x) for c, x in zip(cost, t))


class Optimiser(unittest.TestCase):
    def test_single_position_stops_at_the_edge_of_its_zone(self):
        var, c, gamma = 0.05 ** 2, 0.0003, 3e-4
        edge = c / (gamma * var)                      # 400 USD
        self.assertEqual(optimal_trades([300.0], [[var]], [c], gamma), [0.0])
        t = optimal_trades([1000.0], [[var]], [c], gamma)[0]
        self.assertAlmostEqual(1000.0 + t, edge, places=6)
        t = optimal_trades([-1000.0], [[var]], [c], gamma)[0]
        self.assertAlmostEqual(-1000.0 + t, -edge, places=6)

    def test_offsetting_deviations_are_left_alone(self):
        v = 0.04 ** 2
        cov = [[v, v], [v, v]]                        # perfectly correlated
        self.assertEqual(optimal_trades([500.0, -500.0], cov, [0.0003, 0.0003], 3e-4), [0.0, 0.0])

    def test_solution_meets_the_optimality_conditions(self):
        rnd = random.Random(7)
        for _ in range(50):
            k = 5
            f = [[rnd.gauss(0, 0.03) for _ in range(k)] for _ in range(8)]
            cov = [[sum(f[r][i] * f[r][j] for r in range(8)) / 8 + (0.0004 if i == j else 0) for j in range(k)]
                   for i in range(k)]
            delta = [rnd.uniform(-3000, 3000) for _ in range(k)]
            cost = [rnd.uniform(0.0001, 0.001) for _ in range(k)]
            gamma = 3e-4
            t = optimal_trades(delta, cov, cost, gamma)
            g = [gamma * sum(cov[i][j] * (delta[j] + t[j]) for j in range(k)) for i in range(k)]
            for i in range(k):
                if t[i]:
                    self.assertAlmostEqual(g[i] + math.copysign(cost[i], t[i]), 0.0, delta=1e-6)
                else:
                    self.assertLessEqual(abs(g[i]), cost[i] + 1e-9)
            for _ in range(20):                       # and nothing nearby is cheaper
                probe = [x + rnd.uniform(-5, 5) for x in t]
                self.assertLessEqual(objective(t, delta, cov, cost, gamma),
                                     objective(probe, delta, cov, cost, gamma) + 1e-9)


class Decision(unittest.TestCase):
    rows = [{"ETH": math.sin(i) * 0.03, "AERO": math.sin(i) * 0.04 + math.cos(3 * i) * 0.02,
             "SUI": math.sin(i) * 0.035 + math.cos(5 * i) * 0.015} for i in range(270)]
    target = {"AERO": 3000.0, "ETH": -1200.0, "SUI": -1800.0}
    cost = {c: 0.0003 for c in target}

    def test_new_positions_open_in_full(self):
        trades, why = decide({}, self.target, self.rows, self.cost, 10000.0, 3.0, 25.0)
        self.assertEqual(trades, self.target)
        self.assertEqual(why["opening"], sorted(self.target))

    def test_small_drift_is_not_traded(self):
        current = {"AERO": 3060.0, "ETH": -1190.0, "SUI": -1815.0}
        trades, why = decide(current, self.target, self.rows, self.cost, 10000.0, 3.0, 25.0)
        self.assertEqual(trades, {})
        self.assertGreater(why["risk_before"], 0)
        self.assertAlmostEqual(why["hold_risk_cost"], why["after_risk_cost"])

    def test_a_position_dropped_from_the_book_is_closed_in_full(self):
        trades, why = decide({**self.target, "LIT": 800.0}, self.target, self.rows, self.cost, 10000.0, 3.0, 25.0)
        self.assertEqual(trades, {"LIT": -800.0})
        self.assertEqual(why["closing"], ["LIT"])

    def test_large_drift_trades_back_to_the_edge_and_pays_for_itself(self):
        current = {"AERO": 4500.0, "ETH": -1200.0, "SUI": -1800.0}
        trades, why = decide(current, self.target, self.rows, self.cost, 10000.0, 3.0, 25.0)
        self.assertLess(trades["AERO"], 0)
        self.assertGreater(why["benefit"], why["cost"])
        self.assertLess(why["after_risk_cost"] + why["drift_cost"], why["hold_risk_cost"])

    def test_it_trades_exactly_when_a_dollar_saves_more_risk_cost_than_it_costs(self):
        for current, expect in (({"AERO": 3060.0, "ETH": -1190.0, "SUI": -1815.0}, False),
                                ({"AERO": 4500.0, "ETH": -1200.0, "SUI": -1800.0}, True)):
            trades, why = decide(current, self.target, self.rows, self.cost, 10000.0, 3.0, 0.0)
            trigger = max(r["marginal"] / r["rate"] for r in why["by_coin"])
            self.assertEqual(bool(trades), expect)
            self.assertEqual(trigger > 1, expect)


class Prices(unittest.TestCase):
    def test_tick_and_rounding(self):
        self.assertEqual(round_price(2511.34, 4), 2511.3)
        self.assertAlmostEqual(tick(2511.3, 4), 0.1)
        self.assertAlmostEqual(tick(0.8901, 0), 1e-5)
        self.assertEqual(round_price(0.890123, 0), 0.89012)
        self.assertAlmostEqual(tick(0.08107, 0), 1e-6)

    def test_maker_price_stays_passive(self):
        self.assertEqual(maker_price(True, [(100.0, 1)], [(100.2, 1)], 2), 100.01)
        self.assertEqual(maker_price(False, [(100.0, 1)], [(100.2, 1)], 2), 100.19)
        self.assertEqual(maker_price(True, [(100.0, 1)], [(100.01, 1)], 2), 100.0)
        self.assertEqual(maker_price(False, [(100.0, 1)], [(100.01, 1)], 2), 100.01)


class Paper(unittest.TestCase):
    def test_accounting_matches_a_hand_calculation(self):
        with tempfile.TemporaryDirectory() as d:
            b = PaperBroker(None, os.path.join(d, "p.json"), 1000.0, {"maker": 0.0001, "taker": 0.0005})
            b._fill("X", 10, 100.0, maker=True)      # long 10 @ 100, fee 0.10
            b._fill("X", -4, 110.0, maker=False)     # +40 realised, fee 0.22
            b._fill("X", -10, 90.0, maker=False)     # close 6: -60, short 4 @ 90, fee 0.45
            acc = b.account({"X": {"mark": 95.0}})
            self.assertAlmostEqual(b.s["realized"], -20.0)
            self.assertAlmostEqual(b.s["fees"], 0.77)
            self.assertAlmostEqual(acc["positions"]["X"], -4)
            self.assertAlmostEqual(acc["equity"], 1000 - 20 - 0.77 + (-4) * (95 - 90))
            b.save()
            self.assertTrue(os.path.exists(os.path.join(d, "p.json")))


if __name__ == "__main__":
    unittest.main()


class FakeBroker:
    """An exchange double: our resting orders appear in the book (as on Hyperliquid), maker fills happen on a
    scripted poll, IOC orders fill in full."""
    quotes_in_book = True

    def __init__(self, books, fill_at=None, fail_book_after=None):
        self.books, self.fill_at, self.fail_book_after = books, fill_at or {}, fail_book_after
        self.pos, self.orders, self.fills, self.quotes = {}, {}, [], []
        self.next, self.polls, self.book_calls = 1, 0, 0

    def positions(self):
        return dict(self.pos)

    def open_orders(self):
        return {oid: o["sz"] for oid, o in self.orders.items()}

    def book(self, coin):
        self.book_calls += 1
        if self.fail_book_after and self.book_calls > self.fail_book_after:
            raise ConnectionError("Verbindung weg")
        bids, asks = dict(self.books[coin][0]), dict(self.books[coin][1])
        for o in self.orders.values():
            if o["coin"] == coin:
                side = bids if o["is_buy"] else asks
                side[o["px"]] = side.get(o["px"], 0.0) + o["sz"]
        return sorted(bids.items(), reverse=True), sorted(asks.items())

    def post_only(self, coin, is_buy, sz, px, reduce_only):
        oid, self.next = self.next, self.next + 1
        self.orders[oid] = {"coin": coin, "is_buy": is_buy, "sz": sz, "px": px}
        self.quotes.append((coin, px))
        return oid, None

    def modify(self, oid, coin, is_buy, sz, px, reduce_only):
        self.orders[oid].update(sz=sz, px=px)
        self.quotes.append((coin, px))
        return oid, None

    def cancel(self, coin, oid):
        self.orders.pop(oid, None)

    def cancel_all(self, coins):
        for oid in [o for o, v in self.orders.items() if v["coin"] in coins]:
            del self.orders[oid]

    def poll(self):
        self.polls += 1
        for oid, o in list(self.orders.items()):
            if self.fill_at.get(o["coin"]) is not None and self.polls >= self.fill_at[o["coin"]]:
                self._fill(o["coin"], o["sz"] if o["is_buy"] else -o["sz"], o["px"], False)
                del self.orders[oid]

    def ioc(self, coin, is_buy, sz, cap, reduce_only):
        self._fill(coin, sz if is_buy else -sz, cap, True)
        return sz

    def _fill(self, coin, q, px, crossed):
        self.pos[coin] = round(self.pos.get(coin, 0.0) + q, 10)
        self.fills.append({"coin": coin, "sz": q, "px": px, "fee": 0.0, "crossed": crossed, "time": time.time()})

    def fills_since(self, t0):
        return self.fills


class Execution(unittest.TestCase):
    markets = {"AERO": {"mid": 1.0, "szDec": 0}, "PENDLE": {"mid": 2.0, "szDec": 0},
               "ETH": {"mid": 2500.0, "szDec": 4}, "SUI": {"mid": 1.0, "szDec": 1}}
    books = {"AERO": ([(0.99, 5e4)], [(1.01, 5e4)]), "PENDLE": ([(1.99, 5e4)], [(2.01, 5e4)]),
             "ETH": ([(2499.0, 50.0)], [(2501.0, 50.0)]), "SUI": ([(0.99, 5e4)], [(1.01, 5e4)])}
    cfg = {**DEFAULTS, "execution": {**DEFAULTS["execution"], "maker_wait_seconds": 0.3, "requote_seconds": 0.0}}
    betas = {"AERO": 1.0, "PENDLE": 1.0, "ETH": 1.0, "SUI": 1.0}

    def run_execute(self, broker, trades, book_usd=6000.0):
        return execute(broker, trades, self.markets, self.betas, self.cfg, book_usd, lambda text: None)

    def test_ends_exactly_on_the_wanted_positions(self):
        b = FakeBroker(self.books, fill_at={"AERO": 2, "ETH": 4})
        res = self.run_execute(b, {"AERO": 500.0, "ETH": -500.0})
        self.assertEqual(b.pos, {"AERO": 500.0, "ETH": -0.2})
        self.assertEqual(res["left"], {})
        self.assertEqual(b.orders, {})

    def test_lagging_side_is_hurried_only_as_far_as_needed(self):
        trades = {"AERO": 3000.0, "PENDLE": 3000.0, "ETH": -3000.0, "SUI": -3000.0}
        b = FakeBroker(self.books, fill_at={"AERO": 1, "PENDLE": 1})      # longs fill at once, shorts never
        self.run_execute(b, trades)
        self.assertEqual(b.pos, {"AERO": 3000.0, "PENDLE": 1500.0, "ETH": -1.2, "SUI": -3000.0})
        hurried = [f for f in b.fills if f["crossed"]]
        first = sum(-f["sz"] * self.markets[f["coin"]]["mid"] for f in hurried[:2])
        self.assertAlmostEqual(first, 6000 - 1500 / 2)                     # back to half the tolerance, no further
        self.assertFalse(any(f["sz"] > 0 for f in hurried))                # the long side never went taker

    def test_own_quote_is_not_outbid(self):
        b = FakeBroker(self.books)                                         # nothing ever fills as maker
        self.run_execute(b, {"AERO": 500.0})
        prices = {px for coin, px in b.quotes}
        self.assertEqual(prices, {0.99001})                               # one tick above the others, all along

    def test_resting_orders_are_cancelled_when_something_fails(self):
        b = FakeBroker(self.books, fail_book_after=3)
        with self.assertRaises(ConnectionError):
            self.run_execute(b, {"AERO": 500.0, "ETH": -500.0})
        self.assertEqual(b.orders, {})
