"""Backtest of the risk layer (bot/risk.py): does it predict the book's risk better, and does acting on it pay?

    python3 -m backtest.risk --cache backtest/cache        report in backtest/risk.json

On the dates of the main backtest (every 14 days) it builds the book as the bot does: the 7 best longs by the
starting weights, equal weights, and the short book of bot/hedge.py ("wide"), fitted on the 90 days before. The book
is held for 14 days, so the dates chain into one daily return series. It is built twice: as it is, and with the jump
limits (no position may lose more than jump_loss of the account on its worst move of the year before beyond ETH; the
account is the long book divided by book_per_side; what a short may not carry goes to ETH).

For every day it forecasts the book's variance from what was known the evening before:
  window   the variance of the book over the 90 days before (what the bot did so far)
  ewma     exponentially weighted, lambda 0.94 (RiskMetrics)
  garch    GARCH(1,1) fitted on the book's year before the date, then updated day by day
  hmm      the two-state hidden Markov model on the same year, its state updated day by day
and scores them: QLIKE loss (lower is better), realised over forecast variance (1 is right), and how often the
day's loss went beyond the 99 % VaR with a normal distribution and with filtered historical simulation.

Then it scales the book with the forecast (volatility targeting: exposure = the book's own long-run volatility over
tomorrow's, between 0.5 and a cap) or with the regime (exposure falls by half the probability of the turbulent
state), and compares return, volatility, Sharpe ratio, worst day and maximum drawdown after the cost of trading the
changes in size.
"""
import argparse
import json
import math
import statistics
import sys
import time

from bot import hedge, risk, screener
from . import __main__ as B

DAY = B.DAY
HOLD = 14
COST = 0.00035          # per dollar traded when the book changes size (mostly maker, Hyperliquid base fees)
BOOK = 1.5              # long book per dollar of account, as in bot.paper.toml
JUMP_LOSS = 0.05


def day_returns(m, coins, t):
    """Simple returns of the day that ended at midnight t; 0 for a coin without a price (gone or not yet there)."""
    out = {}
    for c in coins:
        a, b = m.price(c, t - DAY), m.price(c, t)
        out[c] = b / a - 1 if a and b else 0.0
    return out


def year_returns(m, coins, t, days=365):
    """Daily returns of the `days` days before t per coin, None where a price is missing, plus ETH's."""
    ts = list(range(t - (days - 1) * DAY, t + DAY, DAY))
    out = {}
    for c in set(coins) | {"ETH"}:
        row = []
        for d in ts:
            a, b = m.price(c, d - DAY), m.price(c, d)
            row.append(b / a - 1 if a and b else None)
        out[c] = row
    return out


def jump_sizes(yr, coins, young_days=180, floor_down=0.16, floor_up=0.31):
    """Worst daily fall and rise beyond beta times ETH over the year, with the floors for young coins (as in
    bot/risk.jumps, on daily closes)."""
    eth = yr["ETH"]
    out = {}
    for c in coins:
        if c == "ETH":
            out[c] = {"down": 0.0, "up": 0.0}
            continue
        pairs = [(a, e) for a, e in zip(yr[c], eth) if a is not None and e is not None]
        down = up = 0.0
        if len(pairs) >= 10:
            b = risk._beta([a for a, _ in pairs], [e for _, e in pairs])
            idio = [a - b * e for a, e in pairs]
            down, up = max(0.0, -min(idio)), max(0.0, max(idio))
        if len(pairs) < young_days:
            down, up = max(down, floor_down), max(up, floor_up)
        out[c] = {"down": down, "up": up}
    return out


def water_fill(weights, caps):
    """Weights summing to at most 1 with w_i <= caps_i: a capped name keeps its cap, the rest share what is left in
    proportion. If everything is capped, the remainder stays uninvested."""
    w = {c: v / sum(weights.values()) for c, v in weights.items()}
    fixed = {}
    for _ in range(len(w) + 1):
        free = {c: v for c, v in w.items() if c not in fixed}
        mass = 1 - sum(fixed.values())
        if not free or mass <= 0:
            break
        tot = sum(free.values())
        trial = {c: mass * v / tot for c, v in free.items()}
        over = {c for c, v in trial.items() if v > caps.get(c, math.inf)}
        if not over:
            fixed.update(trial)
            break
        for c in over:
            fixed[c] = caps[c]
    return fixed


def build(bt, t, prior):
    """The book at t per dollar of long book, without and with jump limits, and what was capped."""
    uni = bt.universe(t)
    if len(uni) < 8:
        return None
    cands = {c: {"factors": f} for c, f in uni.items()}
    screener.score(cands, prior)
    ranked = sorted((c for c in cands if cands[c]["score"] > 0), key=lambda c: -cands[c]["score"])[:7]
    if len(ranked) < 3:
        return None
    longs = {c: 1 / len(ranked) for c in ranked}
    pool = [c for c in bt.coins if c not in longs and c not in B.EXCLUDE_SHORT and bt.m.volume30(c, t) >= 2e6
            and bt.m.closes(c, t, 90)]
    fit = bt.m.returns(list(longs) + pool + ["ETH"], t - 90 * DAY, t)
    if len(fit) < 60:
        return None
    h = hedge.choose(fit, longs, pool, B.WIDE)
    shares = h["weights"]
    eth = [r["ETH"] for r in fit]
    beta = {c: (risk._beta([r[c] for r in fit], eth) if c != "ETH" else 1.0) for c in set(longs) | set(shares)}

    def book(long_w):
        """Long weights -> long and short dollars per dollar of long book, beta-neutral to ETH."""
        tot = sum(long_w.values())
        b_long = sum(w * beta[c] for c, w in long_w.items())
        b_short = sum(s * beta[c] for c, s in shares.items())
        H = b_long / b_short if b_short > 0 else 0.0
        x = dict(long_w)
        for c, s in shares.items():
            x[c] = x.get(c, 0.0) - H * s
        return x, tot

    plain, _ = book(longs)
    yr = year_returns(bt.m, set(longs) | set(shares), t)
    jump = jump_sizes(yr, set(longs) | set(shares))
    budget = JUMP_LOSS / BOOK                       # per dollar of long book
    lw = water_fill(longs, {c: budget / jump[c]["down"] for c in longs if jump[c]["down"] > 0})
    limited, _ = book(lw)
    capped = {"long": sorted(c for c in longs if lw.get(c, 0) < longs[c] - 1e-9), "short": []}
    for c, v in list(limited.items()):
        if v < 0 and c != "ETH" and jump[c]["up"] > 0 and -v > budget / jump[c]["up"]:
            extra = -v - budget / jump[c]["up"]
            limited[c] = -budget / jump[c]["up"]
            limited["ETH"] = limited.get("ETH", 0.0) - extra * beta[c]
            capped["short"].append(c)
    return {"t": t, "plain": plain, "limited": limited, "capped": capped, "yr": yr}


def pnl_history(yr, x):
    """The book's daily return over the year before, as the bot builds it (bot/risk.pro_forma: before a coin was
    listed it moves with its beta to ETH), and per day whether every coin with 5 % of the book had its own price."""
    eth = yr["ETH"]
    gross = sum(abs(v) for v in x.values()) or 1.0
    own = [[a is not None for a in yr[c]] for c, v in x.items() if abs(v) >= 0.05 * gross]
    filled = {c: risk.filled(yr[c], eth) for c in x}
    out, mask = [], []
    for i, e in enumerate(eth):
        if e is None or any(filled[c][i] is None for c in x):
            continue
        out.append(sum(v * filled[c][i] for c, v in x.items()))
        mask.append(all(o[i] for o in own))
    return out, mask


def forecasts(hist, realised, mask=None):
    """Day-by-day variance forecasts for the holding days: each uses the history and the realised days before it.
    GARCH and the HMM twice: fitted with jumps clipped (as the bot does) and on the data as it is ("_raw")."""
    gs = {"garch": risk.garch_fit(hist, ref_mask=mask), "garch_raw": risk.garch_fit(hist, clip=None)}
    hs = {"hmm": risk.hmm_fit(hist, ref_mask=mask), "hmm_raw": risk.hmm_fit(hist, clip=None)}
    g = gs["garch"]
    z = sorted((v - g["mean"]) / math.sqrt(h) for v, h in zip(hist, g["h"]) if h > 0)
    q99 = z[max(0, round(0.01 * len(z)) - 1)]
    seen = list(hist)
    ew = statistics.pvariance(hist[:30])
    path = []
    for v in hist[30:]:
        path.append(ew)
        ew = 0.94 * ew + 0.06 * v * v
    ew_ref = risk.usual(path, mask[30:] if mask else None) if path else ew
    hv = {k: f["next"] for k, f in gs.items()}
    prob = {k: (f["today"][1] if f else None) for k, f in hs.items()}
    out = []
    for r in realised:
        row = {"r": r, "window": statistics.pvariance(seen[-90:]), "ewma": ew, **hv, "q99": q99,
               "v_ref": g["v_ref"], "v_ref_hmm": hs["hmm"]["v_ref"] if hs["hmm"] else g["v_ref"], "v_ref_ewma": ew_ref,
               "p_turb": prob["hmm"]}
        for k, f in hs.items():
            row[k] = risk.hmm_variance(f, prob[k]) if f else row["window"]
        row["combo"] = (row["garch"] + row["hmm"]) / 2          # what the bot uses (bot/risk.outlook)
        row["v_ref_combo"] = (row["v_ref"] + row["v_ref_hmm"]) / 2
        out.append(row)
        for k, f in gs.items():
            e = r - f["mean"]
            hv[k] = f["v_long"] * (1 - f["a"] - f["b"]) + f["a"] * e * e + f["b"] * hv[k]
        ew = 0.94 * ew + 0.06 * r * r
        for k, f in hs.items():
            if f:
                prob[k] = risk.hmm_step(f, prob[k], r)
        seen.append(r)
    hm = hs["hmm"]
    return out, {"a": g["a"], "b": g["b"], "hmm_sd": hm["sd"] if hm else None, "hmm_stay": hm["stay"] if hm else None}


def stats(rets, exposure=None):
    """Annualised mean, volatility and Sharpe of daily account returns, worst day, ES 99 % and maximum drawdown."""
    n = len(rets)
    m, sd = statistics.mean(rets), statistics.pstdev(rets)
    eq, peak, dd = 0.0, 0.0, 0.0
    for v in rets:
        eq += v
        peak = max(peak, eq)
        dd = min(dd, eq - peak)
    srt = sorted(rets)
    k = max(1, round(0.01 * n))
    mm = statistics.mean(rets)
    kurt = sum((v - mm) ** 4 for v in rets) / n / sd ** 4 if sd > 0 else None
    return {"ret": 365 * m, "vol": math.sqrt(365) * sd, "sharpe": math.sqrt(365) * m / sd if sd > 0 else None,
            "worst": srt[0], "es99": statistics.mean(srt[:k]), "max_dd": dd, "kurtosis": kurt, "days": n,
            "exposure": statistics.mean(exposure) if exposure else 1.0}


def hac_t(diff, lag=5):
    """t-value of the mean of diff with Newey-West standard errors: neighbouring days' losses are not independent."""
    n, m = len(diff), statistics.mean(diff)
    d = [v - m for v in diff]
    var = sum(v * v for v in d) / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * sum(d[i] * d[i - k] for i in range(k, n)) / n
    return m / math.sqrt(var / n) if var > 0 else None


def run(cache, out_path, say=print):
    rep = json.load(open("backtest/report.json"))
    end = rep["t"]
    bt = B.Backtest(cache, B.calendar.timegm((2024, 8, 1, 0, 0, 0)) * 1000, end, say, offline=True)
    dates = [r["t"] for r in rep["ic_rows"] if r["t"] + HOLD * DAY <= end]
    days = {"plain": [], "limited": []}
    params, caps = [], []
    skipped = []
    for t in dates:
        b = build(bt, t, B.PRIOR)
        if not b:                         # a long without 60 days of complete closes for the hedge: no book that date
            skipped.append(t)
            continue
        caps.append({"t": t, **b["capped"], "eth_extra": b["limited"].get("ETH", 0) - b["plain"].get("ETH", 0)})
        for name in ("plain", "limited"):
            x = b[name]
            hist, mask = pnl_history(b["yr"], x)
            real = []
            for d in range(1, HOLD + 1):
                r = day_returns(bt.m, x, t + d * DAY)
                real.append(sum(v * r[c] for c, v in x.items()))
            rows, p = forecasts(hist, real, mask)
            gross = sum(abs(v) for v in x.values())
            for i, row in enumerate(rows):
                row.update(t=t + (i + 1) * DAY, gross=gross)
            days[name] += rows
            if name == "plain":
                params.append({"t": t, **p})
        say(f"  {time.strftime('%Y-%m-%d', time.gmtime(t / 1000))}: Longs gekappt {b['capped']['long']},"
            f" Shorts gekappt {b['capped']['short']}")

    out = {"t": end, "dates": len(params), "hold_days": HOLD, "params": params, "caps": caps, "skipped": skipped}
    # ---- forecasts of the plain book
    rows = days["plain"]
    keys = ("window", "ewma", "garch", "garch_raw", "hmm", "hmm_raw", "combo")
    loss = {k: [math.log(r[k]) + r["r"] ** 2 / r[k] for r in rows] for k in keys}
    base = loss["window"]
    out["forecast"] = {}
    for k in keys:
        diff = [a - b for a, b in zip(loss[k], base)]
        out["forecast"][k] = {"qlike": statistics.mean(loss[k]),
                              "vs_window": statistics.mean(diff),
                              "t_vs_window": hac_t(diff) if k != "window" else None,
                              "ratio": statistics.mean(r["r"] ** 2 / r[k] for r in rows),
                              "breach99_normal": sum(r["r"] < -2.3263 * math.sqrt(r[k]) for r in rows) / len(rows)}
    for k in ("garch", "hmm", "combo"):         # fat tails: the year's standardised days at the forecast volatility
        out["forecast"][k]["breach99_fhs"] = sum(r["r"] < r["q99"] * math.sqrt(r[k]) for r in rows) / len(rows)
    for a_, b_ in (("hmm", "garch"), ("combo", "garch"), ("hmm", "combo"), ("combo", "ewma")):
        diff = [x - y for x, y in zip(loss[a_], loss[b_])]
        out["forecast"][f"{a_}_vs_{b_}"] = {"qlike": statistics.mean(diff), "t": hac_t(diff)}
    out["forecast"]["days"] = len(rows)
    out["days"] = [{"t": r["t"], "r": round(r["r"], 6), **{k: round(math.sqrt(r[k]), 6) for k in keys},
                    "p_turb": round(r["p_turb"], 4) if r["p_turb"] is not None else None} for r in rows]
    # ---- what acting on it would have given: account returns (book per dollar of long book times BOOK)
    out["strategies"] = {}
    for name in ("plain", "limited"):
        r = [BOOK * x["r"] for x in days[name]]
        out["strategies"][name] = stats(r)
    for label, key, ref, lo, hi, band in (("garch_0.5_1", "garch", "v_ref", 0.5, 1.0, 0.0),
                                          ("garch_0.5_1.5", "garch", "v_ref", 0.5, 1.5, 0.0),
                                          ("hmmvol_0.5_1", "hmm", "v_ref_hmm", 0.5, 1.0, 0.0),
                                          ("combo_0.5_1", "combo", "v_ref_combo", 0.5, 1.0, 0.0),
                                          ("combo_0.5_1_band", "combo", "v_ref_combo", 0.5, 1.0, 0.10),
                                          ("ewma_0.5_1", "ewma", "v_ref_ewma", 0.5, 1.0, 0.0)):
        for name in ("plain", "limited"):
            ex, rets, held, day0 = [], [], None, None
            for x in days[name]:
                m = min(hi, max(lo, math.sqrt(x[ref] / x[key])))
                new_book = day0 is None or x["t"] - day0 > HOLD * DAY
                if new_book:                  # a new book is built at its target size
                    day0 = x["t"] - DAY
                if held is None or new_book or abs(held / m - 1) > band:
                    cost = COST * abs(m - held) * x["gross"] * BOOK if held is not None and not new_book else 0.0
                    held = m
                else:
                    cost = 0.0
                rets.append(BOOK * held * x["r"] - cost)
                ex.append(held)
            out["strategies"][f"{name}_{label}"] = stats(rets, ex)
    for name in ("plain", "limited"):
        ex, rets, prev = [], [], None
        for x in days[name]:
            m = 1 - 0.5 * (x["p_turb"] or 0.0)
            cost = COST * abs(m - prev) * x["gross"] * BOOK if prev is not None else 0.0
            rets.append(BOOK * m * x["r"] - cost)
            ex.append(m)
            prev = m
        out["strategies"][f"{name}_hmm"] = stats(rets, ex)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(prog="python3 -m backtest.risk", description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="backtest/cache")
    ap.add_argument("--out", default="backtest/risk.json")
    a = ap.parse_args()
    run(a.cache, a.out, say=lambda s: print(s, flush=True))


if __name__ == "__main__":
    sys.exit(main())
