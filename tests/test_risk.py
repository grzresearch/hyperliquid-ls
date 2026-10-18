"""The risk layer on simulated data whose truth is known: GARCH, the hidden Markov model, tails and jumps."""
import math
import os
import random
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bot import risk  # noqa: E402


def garch_series(n, a=0.10, b=0.85, v=0.0004, seed=1):
    rnd = random.Random(seed)
    w, h, x, hs = v * (1 - a - b), v, [], []
    for _ in range(n):
        hs.append(h)
        e = rnd.gauss(0, math.sqrt(h))
        x.append(e)
        h = w + a * e * e + b * h
    return x, hs, h


def regime_series(n, sds=(0.01, 0.03), stay=(0.98, 0.95), seed=2):
    rnd = random.Random(seed)
    s, xs, states = 0, [], []
    for _ in range(n):
        states.append(s)
        xs.append(rnd.gauss(0, sds[s]))
        if rnd.random() > stay[s]:
            s = 1 - s
    return xs, states


class Garch(unittest.TestCase):
    def test_it_finds_the_parameters_of_a_garch_process(self):
        x, _, h_next = garch_series(4000)
        fit = risk.garch_fit(x)
        self.assertAlmostEqual(fit["a"], 0.10, delta=0.04)
        self.assertAlmostEqual(fit["b"], 0.85, delta=0.06)
        self.assertAlmostEqual(fit["next"] / h_next, 1.0, delta=0.35)

    def test_without_clustering_tomorrow_is_the_long_run_variance(self):
        rnd = random.Random(5)
        x = [rnd.gauss(0, 0.02) for _ in range(800)]
        fit = risk.garch_fit(x)
        self.assertAlmostEqual(fit["next"] / fit["v_long"], 1.0, delta=0.25)
        self.assertAlmostEqual(risk.garch_ahead(fit, 30) / fit["v_long"], 1.0, delta=0.15)

    def test_the_forecast_decays_toward_the_long_run_variance(self):
        fit = {"a": 0.1, "b": 0.85, "v_long": 1.0, "next": 4.0}
        self.assertAlmostEqual(risk.garch_ahead(fit, 1), 4.0)
        self.assertAlmostEqual(risk.garch_ahead(fit, 60), 1 + 3 * (1 - 0.95 ** 60) / (0.05 * 60))
        self.assertLess(risk.garch_ahead(fit, 60), risk.garch_ahead(fit, 10))


class Tails(unittest.TestCase):
    def test_fat_tails_give_a_larger_expected_shortfall_than_the_normal(self):
        rnd = random.Random(7)

        def t3():
            z = rnd.gauss(0, 1)
            c = sum(rnd.gauss(0, 1) ** 2 for _ in range(3))
            return z / math.sqrt(c / 3) / math.sqrt(3)      # unit variance

        x = [0.02 * t3() for _ in range(3000)]
        fit = risk.garch_fit(x)
        r = risk.tail_risk(x, fit)
        self.assertGreater(r["99"]["es_hist"], 1.15 * r["99"]["es_param"])
        self.assertAlmostEqual(r["sigma"], math.sqrt(fit["next"]))
        self.assertEqual(len(r["samples"]), len(x))

    def test_a_normal_series_matches_the_normal_numbers(self):
        rnd = random.Random(8)
        x = [rnd.gauss(0, 0.02) for _ in range(5000)]
        r = risk.tail_risk(x, risk.garch_fit(x))
        self.assertAlmostEqual(r["95"]["hist"] / r["95"]["param"], 1.0, delta=0.08)
        self.assertAlmostEqual(r["99"]["es_hist"] / r["99"]["es_param"], 1.0, delta=0.12)


def t3(rnd):
    """Student t with 3 degrees of freedom, scaled to unit variance: fat tails."""
    z = rnd.gauss(0, 1)
    c = sum(rnd.gauss(0, 1) ** 2 for _ in range(3))
    return z / math.sqrt(c / 3) / math.sqrt(3)


class Shape(unittest.TestCase):
    def test_a_normal_sample_looks_normal(self):
        rnd = random.Random(21)
        s = risk.shape([rnd.gauss(0, 1) for _ in range(20000)])
        self.assertAlmostEqual(s["skew"], 0.0, delta=0.05)
        self.assertAlmostEqual(s["kurt"], 3.0, delta=0.15)
        self.assertAlmostEqual(s["normal"]["3"], 20000 * 0.0013499, delta=0.01)
        for k in ("-3", "3"):
            self.assertAlmostEqual(s["beyond"][k], s["normal"]["3"], delta=4 * math.sqrt(s["normal"]["3"]))

    def test_fat_tails_and_skew_show(self):
        rnd = random.Random(22)
        fat = risk.shape([t3(rnd) for _ in range(20000)])
        self.assertGreater(fat["kurt"], 5.0)
        self.assertGreater(fat["beyond"]["3"] + fat["beyond"]["-3"], 2 * fat["normal"]["3"])
        right = risk.shape([math.exp(rnd.gauss(0, 0.5)) for _ in range(20000)])
        self.assertGreater(right["skew"], 1.0)
        self.assertGreater(right["beyond"]["2"], right["beyond"]["-2"])


class Smile(unittest.TestCase):
    def test_the_normal_volatility_comes_back_from_its_price(self):
        for F, K, s in ((0.0, -2.0, 1.5), (100.0, 130.0, 12.0), (5.0, 5.0, 0.3)):
            for call in (True, False):
                self.assertAlmostEqual(risk.implied_normal(F, K, risk.bachelier(F, K, s, call), call), s, places=7)
        self.assertIsNone(risk.implied_normal(0.0, -3.0, 0.0, False))      # nothing below the strike

    def test_a_normal_book_without_clustering_has_no_smile(self):
        rnd = random.Random(23)
        z = [rnd.gauss(0, 1) for _ in range(5000)]
        sm = risk.smile(z, 0.0, 0.0, 1.0, {"flat": 1.0}, paths=6000)
        ph, atm = sm["phases"]["flat"], sm["m"].index(0.0)
        self.assertAlmostEqual(sm["long"], 1.0, delta=0.04)
        for d, sd, row in zip(sm["days"], ph["sd"], ph["iv"]):
            self.assertAlmostEqual(sd, 1.0, delta=0.04)
            self.assertAlmostEqual(row[atm] / sd, 1.0, delta=0.03, msg=d)   # the paths agree with the forecast
            near = [iv for k, iv in zip(sm["m"], row) if abs(k) <= (3 if d == 1 else 2)]
            self.assertLess(max(near) / min(near), 1.08, d)              # flat, within the noise of the sample

    def test_a_strike_with_too_few_paths_beyond_it_gets_no_price(self):
        rnd = random.Random(28)
        z = [rnd.gauss(0, 1) for _ in range(1000)]
        sm = risk.smile(z, 0.0, 0.0, 1.0, {"x": 1.0}, days=(1, 7), paths=300)
        at = {k: i for i, k in enumerate(sm["m"])}
        one, week = sm["phases"]["x"]["iv"]
        self.assertIsNone(week[at[3.0]])
        self.assertIsNone(week[at[-3.0]])
        self.assertIsNotNone(week[at[0.0]])
        self.assertIsNotNone(one[at[3.0]])                               # one day is exact, no paths needed

    def test_fat_tails_bend_the_line_into_a_smile_that_flattens_with_the_horizon(self):
        rnd = random.Random(24)
        z = [t3(rnd) for _ in range(3000)]
        sm = risk.smile(z, 0.0, 0.0, 1.0, {"x": 1.0}, days=(1, 30), paths=6000)
        one, month = sm["phases"]["x"]["iv"]
        at = {k: i for i, k in enumerate(sm["m"])}
        wings = lambda row: (row[at[-2.0]] + row[at[2.0]]) / 2 / row[at[0.0]]     # noqa: E731
        self.assertGreater(wings(one), 1.08)
        self.assertLess(wings(month), wings(one))

    def test_skew_tilts_the_smile(self):
        rnd = random.Random(25)
        z = [math.exp(rnd.gauss(0, 0.6)) for _ in range(3000)]
        m = statistics.mean(z)
        sm = risk.smile([v - m for v in z], 0.0, 0.0, 1.0, {"x": 1.0}, days=(1,))
        row, at = sm["phases"]["x"]["iv"][0], {k: i for i, k in enumerate(sm["m"])}
        self.assertGreater(row[at[2.0]], 1.2 * row[at[0.0]])             # big gains: the right wing is higher
        self.assertLess(row[at[-1.0]], row[at[0.0]])                     # losses are bounded: the left side falls
        self.assertIsNone(row[at[-3.0]])                                 # and nothing lies 3 sd below the mean

    def test_the_level_returns_to_the_long_run_across_horizons(self):
        rnd = random.Random(26)
        z = [rnd.gauss(0, 1) for _ in range(2000)]
        sm = risk.smile(z, 0.1, 0.85, 1.0, {"calm": 0.25, "turbulent": 4.0}, paths=4000)
        calm, wild = sm["phases"]["calm"]["sd"], sm["phases"]["turbulent"]["sd"]
        self.assertLess(calm[0], calm[1])
        self.assertLess(calm[1], calm[2])
        self.assertGreater(wild[0], wild[1])
        self.assertGreater(wild[1], wild[2])
        fit = {"a": 0.1, "b": 0.85, "v_long": 1.0, "next": 4.0}
        self.assertAlmostEqual(wild[2] ** 2 / risk.garch_ahead(fit, 30), 1.0, delta=0.05)
        self.assertAlmostEqual(sm["phases"]["calm"]["start"], 0.5)
        # the paths agree with the forecast: the month's at-the-money volatility near its sd
        atm = sm["m"].index(0.0)
        self.assertAlmostEqual(sm["phases"]["turbulent"]["iv"][2][atm] / wild[2], 1.0, delta=0.06)

    def test_the_same_inputs_give_the_same_smile(self):
        rnd = random.Random(27)
        z = [t3(rnd) for _ in range(400)]
        args = (z, 0.06, 0.9, 1.0, {"today": 1.3})
        self.assertEqual(risk.smile(*args, paths=500), risk.smile(*args, paths=500))
        self.assertIsNone(risk.smile(z[:20], 0.06, 0.9, 1.0, {"today": 1.0}))


class Regimes(unittest.TestCase):
    def test_what_the_model_expected_each_morning_comes_from_the_day_before(self):
        x, _ = regime_series(800)
        fit = risk.hmm_fit(x)
        P, f, pred = fit["P"], fit["filtered"], fit["pred"]
        self.assertEqual(len(pred), len(x))
        self.assertAlmostEqual(pred[0], 0.5 * P[0][1] + 0.5 * P[1][1])
        for t in (1, 100, 799):
            self.assertAlmostEqual(pred[t], (1 - f[t - 1]) * P[0][1] + f[t - 1] * P[1][1])

    def test_in_hindsight_the_regimes_are_clearer_and_today_agrees(self):
        x, states = regime_series(1500, seed=9)
        fit = risk.hmm_fit(x)
        f, sm = fit["filtered"], fit["smoothed"]
        self.assertAlmostEqual(sm[-1], f[-1], places=12)                # the last day knows nothing more
        right = lambda ps: sum((p > 0.5) == (s == 1) for p, s in zip(ps, states)) / len(states)  # noqa: E731
        switches = lambda ps: sum((a > 0.5) != (b > 0.5) for a, b in zip(ps, ps[1:]))       # noqa: E731
        self.assertGreaterEqual(right(sm), right(f))
        self.assertGreater(right(sm), 0.9)
        self.assertLess(switches(sm), switches(f))

    def test_it_finds_the_calm_and_the_turbulent_state(self):
        x, states = regime_series(2500)
        fit = risk.hmm_fit(x)
        self.assertAlmostEqual(fit["sd"][0], 0.01, delta=0.0015)
        self.assertAlmostEqual(fit["sd"][1], 0.03, delta=0.0045)
        self.assertAlmostEqual(fit["stay"][0], 0.98, delta=0.02)
        self.assertAlmostEqual(fit["stay"][1], 0.95, delta=0.04)
        right = sum((p > 0.5) == (s == 1) for p, s in zip(fit["filtered"], states)) / len(states)
        self.assertGreater(right, 0.85)

    def test_a_day_far_beyond_both_states_counts_as_turbulent(self):
        x, _ = regime_series(1500)
        fit = risk.hmm_fit(x)
        for v in (0.5, 5.0, -50.0):                         # 17 to 1700 turbulent sds: no underflow to "calm"
            self.assertGreater(risk.hmm_step(fit, 0.5, v), 0.99)
        fit2 = risk.hmm_fit(x[:-1] + [-50.0])
        self.assertGreater(fit2["today"][1], 0.99)

    def test_one_more_day_updates_the_state_without_refitting(self):
        x, _ = regime_series(1500)
        fit = risk.hmm_fit(x)
        calm = risk.hmm_step(fit, 0.5, 0.0)
        wild = risk.hmm_step(fit, 0.5, 0.09)
        self.assertLess(calm, 0.5)
        self.assertGreater(wild, 0.95)
        self.assertGreater(risk.hmm_variance(fit, wild), risk.hmm_variance(fit, calm))


def snap_with(coins, days=400, per=3, seed=4, jump=None):
    """8h closes: ETH a random walk, every other coin beta 1.2 to it plus noise; jump = (coin, step, size) adds one
    move of its own."""
    rnd = random.Random(seed)
    n = days * per
    eth = [0.0]
    for _ in range(n - 1):
        eth.append(eth[-1] + rnd.gauss(0, 0.012))
    out = {"ETH": {"closes": [2500 * math.exp(v) for v in eth]}}
    for c in coins:
        lp, path = 0.0, []
        for i in range(n):
            if i:
                lp += 1.2 * (eth[i] - eth[i - 1]) + rnd.gauss(0, 0.008)
                if jump and jump[0] == c and i == jump[1]:
                    lp += math.log(1 + jump[2])
            path.append(10 * math.exp(lp))
        out[c] = {"closes": path}
    return {"perDay": per, "t0": 0, "step": 8 * 3600000, "coins": out}


class Jumps(unittest.TestCase):
    def test_a_move_of_its_own_is_found_and_the_market_part_is_not(self):
        snap = snap_with(["AAA", "BBB"], jump=("AAA", 900, 0.40))
        j = risk.jumps(snap, ["AAA", "BBB", "ETH"])
        self.assertAlmostEqual(j["AAA"]["up"], 0.40, delta=0.06)
        self.assertLess(j["BBB"]["up"], 0.12)              # its beta moves with ETH are not its own
        self.assertEqual(j["ETH"]["up"], 0.0)

    def test_a_young_coin_gets_at_least_the_floor(self):
        snap = snap_with(["NEW"])
        snap["coins"]["NEW"]["closes"][:-60 * 3] = [None] * (len(snap["coins"]["NEW"]["closes"]) - 60 * 3)
        j = risk.jumps(snap, ["NEW"], floor_down=0.16, floor_up=0.31)["NEW"]
        self.assertTrue(j["young"])
        self.assertGreaterEqual(j["down"], 0.16)
        self.assertGreaterEqual(j["up"], 0.31)

    def test_a_limit_of_zero_switches_the_caps_off(self):
        j = {"L": {"down": 0.2, "up": 0.5}}
        self.assertEqual(risk.position_caps(j, 10000.0, 0.0), {})

    def test_caps_follow_the_side(self):
        j = {"L": {"down": 0.2, "up": 0.5}, "ETH": {"down": 0.0, "up": 0.0}}
        caps = risk.position_caps(j, 10000.0, 0.05, side={"L": 1})
        self.assertAlmostEqual(caps["L"], 2500.0)
        self.assertAlmostEqual(risk.position_caps(j, 10000.0, 0.05, side={"L": -1})["L"], 1000.0)
        self.assertEqual(caps["ETH"], math.inf)


class Book(unittest.TestCase):
    def test_too_little_or_flat_history_gives_a_neutral_outlook(self):
        for pnl in ([], [1.0] * 5, [2.0] * 100):
            o = risk.outlook(pnl)
            self.assertEqual(o["var"], 0.0)
            self.assertIsNone(o["tails"])

    def test_the_usual_level_counts_the_days_with_real_prices(self):
        snap = snap_with(["NEW"], days=400)
        snap["coins"]["NEW"]["closes"][:-90 * 3] = [None] * (400 * 3 - 90 * 3)
        pnl, mask = risk.pro_forma(snap, {"NEW": 1000.0, "ETH": -1200.0}, days=365, span=True)
        self.assertEqual(len(pnl), 365)
        self.assertEqual(sum(mask), 89)                              # 90 days of closes, 89 daily returns
        self.assertTrue(all(mask[-89:]))
        g = risk.garch_fit(pnl, ref_mask=mask)
        self.assertAlmostEqual(g["v_ref"], statistics.mean(g["h"][-89:]))

    def test_a_day_left_out_keeps_the_dates_of_the_others(self):
        snap = snap_with(["AAA"], days=400)
        snap["coins"]["ETH"]["closes"][-(10 * 3 + 1)] = None              # the close that ended the day 10 days ago
        pnl, _, ages = risk.pro_forma(snap, {"AAA": 1000.0, "ETH": -1200.0}, days=365, span=True, ages=True)
        self.assertEqual(len(pnl), len(ages))
        self.assertEqual(len(pnl), 363)                                    # that day and the next are gone
        self.assertEqual(ages[-1], 0)
        self.assertEqual(ages[0], 364)                                     # the oldest day is still a year back
        self.assertNotIn(10, ages)
        self.assertNotIn(9, ages)

    def test_one_missing_close_does_not_shorten_the_real_days(self):
        snap = snap_with(["NEW"], days=400)
        v = snap["coins"]["NEW"]["closes"]
        v[:-100 * 3] = [None] * (400 * 3 - 100 * 3)
        v[-30 * 3] = None                                            # one 8h close missing a month ago
        _, mask = risk.pro_forma(snap, {"NEW": 1000.0, "ETH": -1200.0}, days=365, span=True)
        self.assertGreater(sum(mask), 95)

    def test_a_small_young_position_does_not_decide_the_usual_level(self):
        snap = snap_with(["OLD", "NEW"], days=400)
        snap["coins"]["NEW"]["closes"][:-40 * 3] = [None] * (400 * 3 - 40 * 3)
        _, mask = risk.pro_forma(snap, {"OLD": 1000.0, "NEW": 30.0, "ETH": -1300.0}, days=365, span=True)
        self.assertEqual(sum(mask), 365)

    def test_the_usual_level_does_not_jump_when_a_coin_turns_60_days_old(self):
        values = [1.0] * 300 + [4.0] * 65
        levels = [risk.usual(values, [False] * (365 - k) + [True] * k) for k in (58, 59, 60, 61)]
        self.assertLess(max(b - a for a, b in zip(levels, levels[1:])), 0.1)
        self.assertAlmostEqual(levels[2], 4.0)

    def test_a_young_coin_does_not_shrink_the_book(self):
        snap = snap_with(["NEW"], days=400, seed=11)
        snap["coins"]["NEW"]["closes"][:-120 * 3] = [None] * (400 * 3 - 120 * 3)
        o = risk.book_outlook(snap, {"NEW": 1000.0, "ETH": -1200.0}, days=365)
        self.assertGreater(math.sqrt(o["ref"] / o["var"]), 0.8)

    def test_the_outlook_carries_what_the_page_draws(self):
        snap = snap_with(["AAA"], days=400, seed=12)
        o = risk.book_outlook(snap, {"AAA": 1000.0, "ETH": -1200.0}, days=365)
        self.assertEqual(o["t_end"], 400 * 3 * 8 * 3600000)               # the last close of the grid that ends now
        self.assertEqual(len(o["pnl"]), 365)
        self.assertEqual(len(o["regime"]["pred"]), 365)
        self.assertEqual(o["ages"], list(range(364, -1, -1)))
        lv = o["levels"]
        self.assertLessEqual(lv["calm"], lv["turbulent"])
        sm = risk.book_smile(o, days=(1, 7), paths=800)
        self.assertEqual(sorted(sm["phases"]), ["calm", "today", "turbulent"])
        self.assertAlmostEqual(sm["phases"]["today"]["start"], math.sqrt(o["var"]))
        self.assertEqual(len(sm["phases"]["today"]["iv"][1]), len(risk.STRIKES))

    def test_a_coin_listed_later_moves_with_its_beta_before(self):
        snap = snap_with(["AAA", "NEW"])
        snap["coins"]["NEW"]["closes"][:600] = [None] * 600
        pnl = risk.pro_forma(snap, {"AAA": 1000.0, "NEW": -500.0}, days=365)
        self.assertEqual(len(pnl), 365)
        eth = risk.daily_returns(snap["coins"]["ETH"]["closes"], 3, 365)
        self.assertLess(statistics.pstdev(pnl[:100]), 0.02 * 1500)      # early days: beta moves only, mostly offset
        self.assertAlmostEqual(len(eth), 365)


if __name__ == "__main__":
    unittest.main()
