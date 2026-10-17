"""What the bot remembers between runs and publishes for the page.

History (state/history.json, published as data/history.json):
  m1      account value at every minute of the last seven days, rebuilt after each run from 1-minute candles of the
          positions held since the previous run (they do not change between runs)
  m15     15-minute candles of the account value since the start: [open time ms, open, high, low, close]
  volume  traded notional per fill: [time ms, USD]
  events  every rebalancing: the decision, the trades with simulated and realised implementation shortfall
  runs    every check: [time ms, account value, risk before, risk after, trades, risk cost of holding the drift,
          cost of trading all of it back]
  size    the book size the volatility forecast set last (bot/risk.py); kept when a round has no forecast

The page draws 5- and 15-minute candles from m1 and 1-hour to daily candles from m15.
"""
import json
import os
import time

from . import costs

MIN = 60000
Q = 15 * MIN
KEEP_M1 = 7 * 24 * 60          # minutes
KEEP_RUNS = 7 * 24 * 4         # checks
KEEP_EVENTS = 500


def load_json(path, default):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, data, indent=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=indent, separators=(",", ":") if indent is None else None)
    os.replace(tmp, path)


class History:
    def __init__(self, path):
        self.path = path
        self.h = load_json(path, {"m1": {"t0": None, "v": []}, "m15": [], "volume": [], "events": [], "runs": [],
                                  "traded": []})

    def save(self):
        save_json(self.path, self.h)

    @property
    def traded(self):
        return set(self.h["traded"])

    # ---------- account value over time ----------
    def extend_equity(self, info, positions, equity_now, marks, now_ms):
        """Account value at every minute since the last recorded one, from 1-minute closes: the value now minus
        what each position has moved since that minute. Positions are those held since the previous run."""
        m1 = self.h["m1"]
        end = now_ms // MIN * MIN
        if m1["t0"] is None:
            m1["t0"], m1["v"] = end, [round(equity_now, 2)]
            self._rebuild_m15(end, end)
            return
        last = m1["t0"] + (len(m1["v"]) - 1) * MIN
        if end <= last:
            m1["v"][-1] = round(equity_now, 2)
            self._rebuild_m15(last, last)
            return
        start = max(last + MIN, end - 3 * 24 * 60 * MIN)      # a longer gap is bridged flat
        closes = {}
        for coin, q in positions.items():
            if not q or coin not in marks:
                continue
            by = {t: c for t, _, _, _, c, _ in info.candles(coin, "1m", start, now_ms)}
            closes[coin] = by
        prev = {coin: None for coin in closes}
        values = []
        for t in range(last + MIN, end + MIN, MIN):
            if t < start:
                values.append(m1["v"][-1])
                continue
            v = equity_now
            for coin, by in closes.items():
                px = by.get(t) or prev[coin] or marks[coin]
                prev[coin] = px
                v += positions[coin] * (px - marks[coin])
            values.append(round(v, 2))
        values[-1] = round(equity_now, 2)
        m1["v"] += values
        if len(m1["v"]) > KEEP_M1:
            cut = len(m1["v"]) - KEEP_M1
            m1["v"], m1["t0"] = m1["v"][cut:], m1["t0"] + cut * MIN
        self._rebuild_m15(last, end)

    def _rebuild_m15(self, first_ms, last_ms):
        """Recompute the 15-minute candles that contain minutes first_ms .. last_ms."""
        m1, m15 = self.h["m1"], self.h["m15"]
        if not m1["v"]:
            return
        by_t = {row[0]: row for row in m15}
        for q0 in range(first_ms // Q * Q, last_ms // Q * Q + Q, Q):
            i0 = max(0, (q0 - m1["t0"]) // MIN)
            i1 = min(len(m1["v"]), (q0 + Q - m1["t0"]) // MIN)
            vals = m1["v"][i0:i1]
            if not vals or q0 + Q <= m1["t0"]:
                continue
            old = by_t.get(q0)
            o = old[1] if old and q0 < m1["t0"] else vals[0]
            hi, lo = max(vals + ([old[2]] if old and q0 < m1["t0"] else [])), min(vals + ([old[3]] if old and q0 < m1["t0"] else []))
            by_t[q0] = [q0, o, hi, lo, vals[-1]]
        self.h["m15"] = [by_t[t] for t in sorted(by_t)]

    # ---------- what happened ----------
    def add_run(self, now_ms, equity, why, n_trades):
        sig = lambda x: float(f"{x:.4g}")     # noqa: E731  (the risk cost is often a fraction of a cent)
        self.h["runs"].append([now_ms, round(equity, 2), round(why["risk_before"], 2), round(why["risk_after"], 2),
                               n_trades, sig(why["hold_risk_cost"]), sig(why["full_cost"])])
        self.h["runs"] = self.h["runs"][-KEEP_RUNS:]

    def add_event(self, now_ms, why, trades, arrival, expected, fills, result):
        """One rebalancing: per coin the intended trade, the simulated cost and what execution really cost."""
        rows = []
        by_coin = {}
        for f in fills:
            by_coin.setdefault(f["coin"], []).append(f)
        for coin, usd in sorted(trades.items(), key=lambda kv: -abs(kv[1])):
            fl = by_coin.get(coin, [])
            real = costs.realised_shortfall(fl, arrival.get(coin), usd)
            rows.append({"coin": coin, "usd": round(usd, 2), "sz": sum(f["sz"] for f in fl),
                         "vwap": real["vwap"], "maker": round(real["maker"], 3), "filled": round(real["filled"], 3),
                         "expected": expected.get(coin), "realised": {k: round(real[k], 3) for k in ("price", "fee", "total", "usd")},
                         "kind": "open" if coin in why["opening"] else "close" if coin in why["closing"] else "rebalance"})
        self.h["events"].append({
            "t": now_ms, "risk_before": round(why["risk_before"], 2), "risk_after": round(why["risk_after"], 2),
            "benefit": round(why["benefit"], 4), "cost": round(why["cost"], 4),
            "hold_risk_cost": round(why["hold_risk_cost"], 4), "full_cost": round(why["full_cost"], 4),
            "opening": why["opening"], "closing": why["closing"], "resized": why.get("resized", []), "trades": rows,
            "fees": round(result.get("fees", 0.0), 4), "seconds": round(result.get("seconds", 0.0), 1),
            "maker": result.get("maker", 0), "taker": result.get("taker", 0),
            "left": result.get("left", {})})
        self.h["events"] = self.h["events"][-KEEP_EVENTS:]
        for f in fills:
            self.h["volume"].append([int(f["time"] * 1000), round(abs(f["sz"]) * f["px"], 2)])
        cutoff = now_ms - 400 * 86400000
        self.h["volume"] = [v for v in self.h["volume"] if v[0] >= cutoff]
        self.h["traded"] = sorted(self.traded | set(trades))


def site_payload(now_ms, mode, cfg, acct, paper, positions, marks, target, h, decision, var, screener, book,
                 weights, longs_score):
    """The page's view of the bot right now."""
    rows = []
    for coin in sorted(set(positions) | set(target), key=lambda c: (-(target.get(c, 0) > 0), c)):
        q = positions.get(coin, 0.0)
        mark = marks.get(coin)
        pos = paper["pos"].get(coin) if paper else None
        rows.append({"coin": coin, "sz": q, "avg": pos["avg"] if pos else None, "mark": mark,
                     "value": q * mark if mark else 0.0,
                     "upnl": q * (mark - pos["avg"]) if pos and mark else None,
                     "target": target.get(coin, 0.0), "weight": weights.get(coin),
                     "score": longs_score.get(coin), "beta": book["betas"].get(coin)})
    return {"v": 3, "t": now_ms, "mode": mode, "interval_min": cfg["rebalancing"].get("check_minutes", 15),
            "account": acct, "positions": rows, "book": book, "decision": decision, "risk": var,
            "screener": screener,
            "config": {"risk_aversion": cfg["rebalancing"]["risk_aversion"],
                       "min_order": cfg["rebalancing"]["min_order_usd"], "book_per_side": cfg["book_per_side"],
                       "maker_fee": cfg["execution"]["maker_fee"], "taker_fee": cfg["execution"]["taker_fee"],
                       "taker_share": cfg["execution"]["taker_share"], "window_days": cfg["strategy"]["window_days"],
                       "eth_share": cfg["strategy"]["eth_share"], "alts": cfg["strategy"]["alts"],
                       "risk": {k: cfg["risk"][k] for k in ("jump_loss", "vol_target", "min_scale", "history_days",
                                                            "size_band")}},
            "paper": paper}


def write_site(site_dir, payload, history):
    save_json(os.path.join(site_dir, "bot.json"), payload)
    save_json(os.path.join(site_dir, "history.json"), history.h)


def stamp():
    return int(time.time() * 1000)
