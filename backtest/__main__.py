"""Backtest of the screener's factors and of the short side.

    python3 -m backtest --cache DIR                   download (once, cached) and run, report in backtest/report.json

Every 14 days over the backtest window it rebuilds the screener's universe as it was then (perps with enough volume,
a protocol on DefiLlama, at least 30 Mio. $ market cap; delisted perps included) and computes each factor exactly as
the live screener does, from data up to that day only. It then measures how well each factor ranked the tokens by
their return against ETH over the next 30 days (beta removed): the rank IC. From the ICs it derives weights, checks
the combined score out of sample (each date weighted only with what earlier dates showed), and compares the short
side: the chosen hedge (bot/hedge.py) against the fixed basket and ETH alone, each fitted on the 90 days before and
measured on the 30 days after.

With some 30 tokens per date the ICs are noisy. The derived weights are therefore a blend of the measured ones and
the prior ones; live paper trading has the last word.
"""
import argparse
import calendar
import json
import math
import multiprocessing
import statistics
import sys
import time

from bot import hedge, screener
from . import data as D

DAY = D.DAY
PRIOR = {"momentum": 0.25, "holder_yield": 0.20, "cashflow_growth": 0.20, "revenue_yield": 0.10, "catalysts": 0.10,
         "fee_yield": 0.05, "volume_growth": 0.05, "acceleration": 0.05}
DIAGNOSTIC = ["funding", "fee_growth", "revenue_growth"]     # measured, not scored
FIXED_ALTS = ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO"]
EXCLUDE_LONG = {"BTC", "ETH", "HYPE"}
EXCLUDE_SHORT = {"HYPE"}


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    try:
        return statistics.correlation(ra, rb)
    except statistics.StatisticsError:
        return None


class Market:
    """Daily closes and notional volumes of every perp, and the price at a given midnight."""

    def __init__(self, d, coins):
        self.close, self.vol = {}, {}
        for c in coins:
            rows = d.daily(c) or []
            self.close[c] = {r[0]: r[4] for r in rows}
            self.vol[c] = {r[0]: r[5] for r in rows}

    def price(self, c, t):
        """Close of the day that ended at t (midnight UTC)."""
        return self.close.get(c, {}).get(t - DAY)

    def last_price(self, c, t0, t1):
        """Price at t1, or the last one before it after t0 (a perp delisted in between)."""
        for t in range(t1, t0, -DAY):
            p = self.price(c, t)
            if p:
                return p
        return None

    def closes(self, c, t, days):
        out = [self.price(c, t - i * DAY) for i in range(days, -1, -1)]
        return out if all(out) else None

    def volume30(self, c, t):
        v = [self.vol.get(c, {}).get(t - i * DAY) for i in range(1, 31)]
        v = [x for x in v if x is not None]
        return sum(v) / 30 if len(v) >= 20 else 0.0

    def returns(self, coins, t0, t1):
        """Daily simple returns of all coins between midnights t0 and t1, complete days only."""
        rows = []
        for t in range(t0 + DAY, t1 + DAY, DAY):
            r = {}
            for c in coins:
                a, b = self.price(c, t - DAY), self.price(c, t)
                if not a or not b:
                    r = None
                    break
                r[c] = b / a - 1
            if r:
                rows.append(r)
        return rows


def beta_to(market, c, t, days=90):
    a, e = market.closes(c, t, days), market.closes("ETH", t, days)
    if not a or not e:
        return None
    ra = [math.log(a[i + 1] / a[i]) for i in range(days)]
    re = [math.log(e[i + 1] / e[i]) for i in range(days)]
    me = sum(re) / days
    ve = sum((x - me) ** 2 for x in re)
    ma = sum(ra) / days
    return sum((x - ma) * (y - me) for x, y in zip(ra, re)) / ve if ve else None


class Backtest:
    def __init__(self, cache, start_ms, end_ms, say=print, offline=False):
        self.say = say
        self.d = D.Data(cache, start_ms, end_ms, say, offline)
        uni = self.d.universe()
        self.coins = [u["coin"] for u in uni]
        self.m = Market(self.d, self.coins)
        ent, _ = self.d.entities()
        self.mapped = screener.map_coins([c for c in self.coins if c not in EXCLUDE_LONG], ent, 5e6)
        self.series, self.mcap = {}, {}
        self.with_funding = False          # funding is measured only where ics() asks for it (rate limit)
        for c, e in self.mapped.items():
            if e.get("slug"):
                self.series[c] = {k: self.d.llama_series(e["slug"], k) for k in ("fees", "revenue", "holders", "dexs")}
            if e.get("gecko"):
                self.mcap[c] = {r[0]: r[1] for r in (self.d.market_caps(e["gecko"], fetch_missing=False) or [])}
        self.now_mcap = {c: e.get("mcap") or 0.0 for c, e in self.mapped.items()}
        self.end = end_ms
        spaces = sorted({g.split(":", 1)[1] for e in self.mapped.values() for g in e["gov"]
                         if g.startswith("snapshot:")} | set(screener.SPACE_ALIASES.values()))
        self.votes = self.d.votes(spaces) or {}
        self.holder_extra = {"AAVE": 50e6}

    def mcap_at(self, c, t):
        """CoinGecko's market cap that day, else today's DefiLlama market cap scaled with the price since then."""
        m = self.mcap.get(c) or {}
        for back in range(0, 4):
            v = m.get(t - back * DAY)
            if v:
                return v
        if m and t >= min(m):
            return None
        p_t, p_now = self.m.price(c, t), self.m.last_price(c, t, self.end)
        return self.now_mcap.get(c, 0.0) * p_t / p_now if p_t and p_now and self.now_mcap.get(c) else None

    def events(self, c, t):
        e = self.mapped[c]
        out = []
        for g in e["gov"]:
            if g.startswith("snapshot:"):
                sp = g.split(":", 1)[1]
                out += [v for v in self.votes.get(screener.SPACE_ALIASES.get(sp, sp), []) if t - 45 * DAY <= v["t"] < t]
        return out

    def factors(self, c, t, need_mcap=True):
        """The live screener's factors for coin c with data up to midnight t, or None outside the universe."""
        s = self.series.get(c)
        if not s or not s["fees"]:
            return None
        fees = screener.window_sums(s["fees"], t)
        if not fees or not fees.get("d30"):
            return None
        mcap = self.mcap_at(c, t)
        if need_mcap and (not mcap or mcap < 3e7):
            return None
        closes = self.m.closes(c, t, 90) or self.m.closes(c, t, 30)
        eth = self.m.closes("ETH", t, 90)
        if not closes or not eth:
            return None
        f = screener.factor_values(fees, screener.window_sums(s["revenue"], t), screener.window_sums(s["holders"], t),
                                   screener.window_sums(s["dexs"], t), mcap, closes, eth, self.events(c, t), t,
                                   self.holder_extra.get(c, 0.0))
        f["funding"] = None
        if self.with_funding:
            fund = [r for _, r in self.d.funding_week(c, t) or []]
            f["funding"] = -sum(fund) / len(fund) * 24 * 365 if fund else None
        rev = screener.window_sums(s["revenue"], t)
        f["fee_growth"], f["revenue_growth"] = screener.growth(fees), screener.growth(rev)
        return f

    def forward(self, c, t, horizon=30):
        """Return against ETH over the next `horizon` days, beta from the 90 days before."""
        p0, e0 = self.m.price(c, t), self.m.price("ETH", t)
        p1 = self.m.last_price(c, t, t + horizon * DAY)
        e1 = self.m.price("ETH", t + horizon * DAY)
        b = beta_to(self.m, c, t)
        if not (p0 and p1 and e0 and e1) or b is None:
            return None
        return math.log(p1 / p0) - b * math.log(e1 / e0)

    def universe(self, t, need_mcap=True):
        out = {}
        for c in self.mapped:
            if self.m.volume30(c, t) < 2e6:
                continue
            f = self.factors(c, t, need_mcap)
            if f is not None:
                out[c] = f
        return out

    # ---------- factor ICs ----------
    def ics(self, dates, factors, need_mcap=True, funding_every=2):
        rows = []
        for i, t in enumerate(dates):
            self.with_funding = "funding" in factors and i % funding_every == 0
            uni = self.universe(t, need_mcap)
            self.with_funding = False
            fwd = {c: self.forward(c, t) for c in uni}
            fwd = {c: v for c, v in fwd.items() if v is not None}
            row = {"t": t, "n": len(fwd), "ic": {}}
            for f in factors:
                pairs = [(uni[c][f], fwd[c]) for c in fwd if uni[c].get(f) is not None]
                if len(pairs) >= 8:
                    row["ic"][f] = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            rows.append(row)
            self.say(f"  {time.strftime('%Y-%m-%d', time.gmtime(t / 1000))}: {len(fwd)} Token")
        return rows

    @staticmethod
    def summary(rows, factors):
        out = {}
        for f in factors:
            v = [r["ic"][f] for r in rows if r["ic"].get(f) is not None]
            mono = [r["ic"][f] for r in rows[::2] if r["ic"].get(f) is not None]     # every 28 days: no overlap
            if len(v) < 4:
                out[f] = None
                continue
            sd = statistics.stdev(v)
            t_stat = (statistics.mean(mono) / (statistics.stdev(mono) / math.sqrt(len(mono)))
                      if len(mono) >= 3 and statistics.stdev(mono) > 0 else None)
            out[f] = {"ic": statistics.mean(v), "sd": sd, "ir": statistics.mean(v) / sd if sd else 0.0,
                      "t": t_stat, "hit": sum(x > 0 for x in v) / len(v), "dates": len(v)}
        return out

    @staticmethod
    def weights_from(summary, prior, blend=0.5):
        """Measured weights (positive information ratios, normalised), blended with the prior."""
        ir = {f: max(0.0, (summary.get(f) or {}).get("ir", 0.0)) for f in prior}
        tot = sum(ir.values())
        measured = {f: ir[f] / tot for f in prior} if tot > 0 else dict(prior)
        w = {f: blend * measured[f] + (1 - blend) * prior[f] for f in prior}
        s = sum(w.values())
        return {f: round(v / s, 3) for f, v in w.items()}, measured

    def composite(self, dates, ic_rows, prior):
        """Out of sample: each date scored with weights from the ICs of dates at least 30 days earlier."""
        out = []
        for i, t in enumerate(dates):
            past = [r for r in ic_rows if r["t"] <= t - 30 * DAY]
            w_live = self.weights_from(self.summary(past, list(prior)), prior)[0] if len(past) >= 6 else prior
            uni = self.universe(t)
            fwd = {c: self.forward(c, t) for c in uni}
            fwd = {c: v for c, v in fwd.items() if v is not None}
            if len(fwd) < 8:
                continue
            res = {"t": t, "n": len(fwd)}
            for name, w in (("prior", prior), ("walk_forward", w_live)):
                cands = {c: {"factors": uni[c]} for c in fwd}
                screener.score(cands, w)
                sc = {c: d["score"] for c, d in cands.items()}
                top = sorted(sc, key=lambda c: -sc[c])[:7]
                res[name] = {"ic": spearman([sc[c] for c in fwd], [fwd[c] for c in fwd]),
                             "top7": statistics.mean(fwd[c] for c in top),
                             "all": statistics.mean(fwd.values()), "picks": top}
            out.append(res)
        return out

    # ---------- short side ----------
    def hedges(self, dates, prior, workers=2):
        """Out of sample: each hedge fitted on the days before t and measured on the 30 days after it."""
        global _BT
        _BT = (self, prior)
        if workers > 1:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_hedge_date, dates)
        else:
            rows = [_hedge_date(t) for t in dates]
        out = [r for r in rows if r]
        for res in out:
            self.say(f"  Hedge {time.strftime('%Y-%m-%d', time.gmtime(res['t'] / 1000))}: " + ", ".join(
                f"{n} {100 * res[n]['vol']:.0f}" for n in ("minvar", "wide", "wide_180", "fixed", "liquid6", "eth_only")
                if n in res) + " % Restvol.")
        return out

    def hedge_date(self, t, prior):
        """One date: the long basket by the prior weights, every hedge variant fitted before t, measured after."""
        uni = self.universe(t)
        if len(uni) < 8:
            return None
        cands = {c: {"factors": f} for c, f in uni.items()}
        screener.score(cands, prior)
        ranked = sorted((c for c in cands if cands[c]["score"] > 0), key=lambda c: -cands[c]["score"])[:7]
        if len(ranked) < 3:
            return None
        longs = {c: 1 / len(ranked) for c in ranked}
        pool = [c for c in self.coins if c not in longs and c not in EXCLUDE_SHORT and self.m.volume30(c, t) >= 2e6
                and self.m.closes(c, t, 90)]
        fit90 = self.m.returns(list(longs) + pool + ["ETH"], t - 90 * DAY, t)
        if len(fit90) < 60:
            return None
        pool180 = [c for c in pool if self.m.closes(c, t, 180)]
        fit180 = self.m.returns(list(longs) + pool180 + ["ETH"], t - 180 * DAY, t) \
            if all(self.m.closes(c, t, 180) for c in list(longs) + ["ETH"]) else []
        frows = {}
        for c in pool:
            f = self.factors(c, t, need_mcap=False) if c in self.mapped else None
            f = dict(f or {})
            if f.get("momentum") is None:
                cl, eth = self.m.closes(c, t, 90), self.m.closes("ETH", t, 90)
                moms = [m for m in (screener.relative_momentum(cl, eth, 30), screener.relative_momentum(cl, eth, 90))
                        if m is not None] if cl and eth else []
                f["momentum"] = sum(moms) / len(moms) if moms else None
            frows[c] = f
        scores = screener.short_scores(frows, prior)
        res = {"t": t, "longs": ranked}
        variants = {}
        for name, v in HEDGE_VARIANTS.items():
            v = dict(v)
            days = v.pop("_days", 90)
            rows, cand = (fit180, pool180) if days == 180 else (fit90, pool)
            if len(rows) < 120 * days // 180:
                continue
            h = hedge.choose(rows, longs, cand, v, score=scores if v["tilt_slack"] else None)
            variants[name] = {c: h["H"] * w for c, w in h["weights"].items()}
            res[name + "_names"] = sorted(h["weights"], key=lambda c: -h["weights"][c])
        pr = hedge.Problem(fit90, longs, ["ETH"], 0.0)
        m_eth = [r["ETH"] for r in fit90]
        vm = statistics.variance(m_eth)
        liquid = sorted((c for c in pool if c not in ("ETH", "BTC")), key=lambda c: -self.m.volume30(c, t))[:6]
        for name, alts in (("fixed", [c for c in FIXED_ALTS if c in pool]), ("liquid6", liquid)):
            if alts:
                a = [sum(r[c] for c in alts) / len(alts) for r in fit90]
                b_alt = statistics.covariance(a, m_eth) / vm
                variants[name] = {"ETH": 0.4 * pr.beta_y, **{c: 0.6 * pr.beta_y / b_alt / len(alts) for c in alts}}
        variants["eth_only"] = {"ETH": pr.beta_y}
        variants["unhedged"] = {}
        need = set(longs) | {c for k in variants.values() for c in k}
        test = self.m.returns(sorted(need | {"ETH"}), t, t + 30 * DAY)
        if len(test) < 20:
            return None
        for name, k in variants.items():
            resid = [sum(w * r[c] for c, w in longs.items()) - sum(h * r[c] for c, h in k.items()) for r in test]
            eth = [r["ETH"] for r in test]
            res[name] = {"vol": statistics.stdev(resid) * math.sqrt(365),
                         "beta": statistics.covariance(resid, eth) / statistics.variance(eth),
                         "ret": sum(resid), "short_ret": -sum(sum(h * r[c] for c, h in k.items()) for r in test),
                         "score": sum(scores.get(c, 0.0) * h for c, h in k.items()) / (sum(k.values()) or 1.0)}
        return res

    @staticmethod
    def hedge_summary(rows):
        names = sorted({k for r in rows for k, v in r.items() if isinstance(v, dict) and "vol" in v})
        out = {}
        for n in names:
            v = [r[n] for r in rows if n in r]
            both = [r for r in rows if n in r and "fixed" in r]
            out[n] = {"vol": statistics.mean(x["vol"] for x in v), "vol_median": statistics.median(x["vol"] for x in v),
                      "abs_beta": statistics.mean(abs(x["beta"]) for x in v), "ret": statistics.mean(x["ret"] for x in v),
                      "short_ret": statistics.mean(x["short_ret"] for x in v),
                      "beats_fixed": sum(r[n]["vol"] < r["fixed"]["vol"] for r in both) / len(both) if both else None,
                      "dates": len(v)}
        return out


_BT = None


def _hedge_date(t):
    bt, prior = _BT
    return bt.hedge_date(t, prior)


BASE = {"min_share": 0.05, "min_beta": 0.3, "shortlist": 16, "vol_caps": True, "keep_slack": 0.15,
        "funding_free": 0.25, "funding_max": 1.0, "holdout": 0.0, "tilt_slack": 0.0}
NARROW = {**BASE, "shrink": 0.3, "min_names": 4, "max_names": 8, "max_share": 0.35}     # first draft
WIDE = {**BASE, "shrink": 0.6, "min_names": 6, "max_names": 10, "max_share": 0.25}      # live default
TILT = {"tilt_slack": 0.10}            # the short score may cost up to 10 % more variance
HEDGE_VARIANTS = {                     # spelled out in full, so a change of hedge.DEFAULTS does not change the test
    "minvar": NARROW,
    "minvar_tilt": {**NARROW, **TILT},
    "shrink60": {**NARROW, "shrink": 0.6},
    "wide": WIDE,
    "wide_tilt": {**WIDE, **TILT},
    "wide_180": {**WIDE, "_days": 180},
}


def run(cache, out_path, say=print, offline=False, end=None):
    end = end or int(time.time() * 1000) // DAY * DAY
    start = calendar.timegm((2024, 8, 1, 0, 0, 0)) * 1000
    bt = Backtest(cache, start, end, say, offline)
    mcap_from = sorted(min(m) for m in bt.mcap.values() if m)[len([m for m in bt.mcap.values() if m]) // 2] + 2 * DAY
    dates = list(range(mcap_from // DAY * DAY + 3 * DAY, end - 30 * DAY, 14 * DAY))
    long_dates = list(range(start + 120 * DAY, end - 30 * DAY, 14 * DAY))
    factors = list(PRIOR) + DIAGNOSTIC
    say(f"Faktoren an {len(dates)} Stichtagen (mit Market Cap)")
    ic_rows = bt.ics(dates, factors)
    summ = bt.summary(ic_rows, factors)
    weights, measured = bt.weights_from(summ, PRIOR)
    say(f"Faktoren ohne Market Cap an {len(long_dates)} Stichtagen seit {time.strftime('%Y-%m', time.gmtime(long_dates[0] / 1000))}")
    nomcap = ["momentum", "cashflow_growth", "volume_growth", "acceleration", "catalysts", "fee_growth",
              "revenue_growth"]
    long_rows = bt.ics(long_dates, [f for f in nomcap if f != "funding"], need_mcap=False)
    long_summ = bt.summary(long_rows, nomcap)
    say("Kombinierter Score, außerhalb der Stichprobe")
    comp = bt.composite(dates, ic_rows, PRIOR)
    say("Short-Seite")
    hedges = bt.hedges(dates, PRIOR)
    report = {"t": end, "dates": len(dates), "from": dates[0], "to": dates[-1], "summary": summ,
              "hedge_summary": bt.hedge_summary(hedges),
              "summary_long": long_summ, "long_from": long_dates[0], "weights": weights, "measured": measured,
              "prior": PRIOR, "ic_rows": ic_rows, "composite": comp, "hedges": hedges,
              "data": {"perps": len(bt.coins), "mapped": len(bt.mapped), "archive": bool(bt.d.archive_key)}}
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    ap = argparse.ArgumentParser(prog="python3 -m backtest", description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="backtest/cache")
    ap.add_argument("--out", default="backtest/report.json")
    ap.add_argument("--offline", action="store_true", help="nur den Cache benutzen, nichts nachladen")
    a = ap.parse_args()
    run(a.cache, a.out, say=lambda s: print(s, flush=True), offline=a.offline)


if __name__ == "__main__":
    sys.exit(main())
