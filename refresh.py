#!/usr/bin/env python3
"""Citrini Long/Short: refresh the Hyperliquid snapshot, print today's orders and the paper portfolio.

Usage: python3 refresh.py OUT.json [BOOK.json] [--prev PREV.json] [--funding-days N] [--offline SNAP.json]
  OUT.json        written: marks, max leverage, size decimals, one year of 8h closes and daily funding
  BOOK.json       saved settings and positions (the page's data/users/me/book doc), optional
  --prev          the previous snapshot; its funding history is carried forward
  --funding-days  days of funding fetched fresh (default: 30, or back to --prev if that is older;
                  without --prev the whole year is rebuilt, which takes about 12 minutes)
  --offline       recompute from an existing snapshot without calling Hyperliquid (tests)
Stdlib only. Prints a German summary: hedge, the trades that pay for themselves (with leverage), paper portfolio.

The maths below is the single source for three places: the dashboard runs the same formulas in JavaScript,
and the rebalancing bot imports them from here.
"""
import argparse
import json
import math
import operator
import ssl
import sys
import time
import urllib.request
from datetime import datetime, timezone

URL = "https://api.hyperliquid.xyz/info"
LONGS = ["AERO", "PENDLE", "LIT", "SYRUP", "ETHFI", "ZRO", "ENA"]
ALTS = ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO", "NEAR", "TIA", "XLM"]
COINS = LONGS + ALTS + ["ETH", "HYPE"]
DEFAULTS = {   # a neutral template; real settings come from BOOK.json
    "equity": 10000, "bookPerSide": 1.5, "riskAversion": 3, "minTrade": 25, "window": 90,
    "levLong": 5, "levShort": 5, "levEth": 10,
    "hypeQty": 0, "hedgeHype": False, "ethShare": 40, "paperStart": None,
    "weights": {t: 1 for t in LONGS},
    "alts": ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO"],
    "pos": {},
}
HOUR = 3600000
DAY = 24 * HOUR
STEP = 8 * HOUR       # one candle
STEPS = 3 * 365       # one year of 8h closes
# Expected cost per USD traded with the bot's maker-first execution: the maker fee, and for the share that ends
# up as taker after the maker window the taker fee plus half the spread instead (Hyperliquid base tier; half
# spreads of these markets were 0.1 to 2 bp in October 2026). The bot itself uses the live spread.
MAKER_FEE = 0.00015
TAKER_FEE = 0.00045
TAKER_SHARE = 0.3
HALF_SPREAD = 0.0001
COST = MAKER_FEE + TAKER_SHARE * (TAKER_FEE - MAKER_FEE + HALF_SPREAD)


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_ms(text):
    text = text.strip().replace("Z", "")
    fmt = "%Y-%m-%d" if len(text) == 10 else "%Y-%m-%dT%H:%M:%S"
    return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)


def unwrap(doc):
    """Accept a plain document or the database wrapper {id, data, version}."""
    if isinstance(doc, dict) and "id" in doc and isinstance(doc.get("data"), dict):
        return doc["data"]
    return doc


# ---------- fetching ----------
def post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    ctx = ssl.create_default_context()
    for attempt in range(6):
        try:
            return json.loads(urllib.request.urlopen(req, context=ctx, timeout=60).read())
        except Exception as e:
            if attempt == 5:
                raise
            time.sleep(20 * (attempt + 1) if getattr(e, "code", 0) == 429 else 2 + 3 * attempt)


def funding_rows(coin, start, end, pause):
    rows = []
    while start < end:
        batch = post({"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": end})
        rows += [(r["time"], float(r["fundingRate"])) for r in batch]
        if len(batch) < 500:
            break
        start = batch[-1]["time"] + 1
        time.sleep(pause)
    return rows


def carried_funding(prev):
    """Daily funding (bp per day) from yesterday's snapshot, keyed by coin and UTC day."""
    out = {}
    if isinstance(prev, dict) and prev.get("v") == 2:
        for t, c in prev["coins"].items():
            out[t] = {prev["d0"] + i * DAY: v for i, v in enumerate(c.get("fund") or []) if v is not None}
    return out


def fetch_snapshot(prev, funding_days=None):
    now = int(time.time() * 1000)
    if not funding_days:              # cover the gap since the previous snapshot, or rebuild the year
        v2 = isinstance(prev, dict) and prev.get("v") == 2
        funding_days = max(30, (now - to_ms(prev["asOf"])) // DAY + 2) if v2 else 366
    funding_days = min(366, funding_days)
    end = now // STEP * STEP            # open time of the running candle
    t0 = end - STEPS * STEP             # first of STEPS completed candles
    d0, today = t0 // DAY * DAY, now // DAY * DAY
    days = (today - d0) // DAY + 1
    pause = 0.3 if funding_days <= 31 else 2.2   # stay under the API weight limit on a full rebuild
    old = carried_funding(prev)
    meta, ctxs = post({"type": "metaAndAssetCtxs"})
    info = {u["name"]: (u, c) for u, c in zip(meta["universe"], ctxs)}
    coins = {}
    for t in COINS:
        if t not in info:
            continue
        u, c = info[t]
        candles = post({"type": "candleSnapshot", "req": {"coin": t, "interval": "8h", "startTime": t0, "endTime": now}})
        close = {k["t"]: float(k["c"]) for k in candles if k["T"] < now}
        rows = funding_rows(t, max(d0, today - funding_days * DAY), now, pause)
        fresh = {}
        for ts, r in rows:
            fresh.setdefault(ts // DAY * DAY, []).append(r)
        fund = []
        for i in range(days):
            d = d0 + i * DAY
            # mean hourly rate of that UTC day, expressed per day in basis points
            fund.append(round(sum(fresh[d]) / len(fresh[d]) * 24e4, 2) if d in fresh else old.get(t, {}).get(d))
        last30 = [r for ts, r in rows if ts >= now - 30 * DAY]
        coins[t] = {"mark": float(c["markPx"]), "maxLev": u["maxLeverage"], "szDec": u["szDecimals"],
                    "fundNow": float(c["funding"]) * 24 * 365,
                    "fund30": sum(last30) / len(last30) * 24 * 365 if last30 else 0.0,
                    "closes": [float(f"{close[t0 + i * STEP]:.5g}") if t0 + i * STEP in close else None
                               for i in range(STEPS)],
                    "fund": fund}
        if funding_days > 31:
            print(f"{t}: {len(rows)} Funding-Stunden", file=sys.stderr, flush=True)
        time.sleep(0.3)
    return {"v": 2, "asOf": iso(now), "source": "api.hyperliquid.xyz", "interval": "8h", "perDay": 3,
            "t0": t0, "step": STEP, "d0": d0, "coins": coins}


# ---------- maths (mirrors the page) ----------
def mean(v):
    return sum(v) / len(v)


def std(v):
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def beta(y, x):
    my, mx = mean(y), mean(x)
    return sum((a - my) * (b - mx) for a, b in zip(y, x)) / sum((b - mx) ** 2 for b in x)


def settings(book):
    s = json.loads(json.dumps(DEFAULTS))
    book = dict(book or {})
    # settings saved before the book was sized by the account value carry the long book in USD: keep that size
    if "bookPerSide" not in book and float(book.get("L") or 0) > 0 and float(book.get("equity") or 0) > 0:
        book["bookPerSide"] = round(float(book["L"]) / float(book["equity"]), 2)
    s.update(book)
    return s


def context(snap, s):
    C = snap["coins"]
    longs = [(t, float(w)) for t, w in s["weights"].items() if float(w) > 0 and t in C]
    alts = [t for t in s["alts"] if t in C and t not in dict(longs)]
    need = [t for t, _ in longs] + alts + ["ETH", "HYPE"]
    lc = {t: [math.log(v) if v else None for v in C[t]["closes"]] for t in need}
    return {"C": C, "per": int(snap.get("perDay", 3)), "longs": longs, "wsum": sum(w for _, w in longs),
            "alts": alts, "need": need, "lc": lc, "ethShare": float(s["ethShare"])}


def hedge_at(cx, end, n):
    """Hedge ratios from the overlapping daily log returns at 8h steps end-n .. end-1."""
    per, lc, longs, wsum, alts = cx["per"], cx["lc"], cx["longs"], cx["wsum"], cx["alts"]
    rows = []
    for i in range(max(per, end - n), end):
        r = {}
        for t in cx["need"]:
            a, b = lc[t][i - per], lc[t][i]
            if a is None or b is None:
                r = None
                break
            r[t] = b - a
        if r:
            rows.append(r)
    if not longs or len(rows) < 30:
        return None
    y = [math.log1p(sum(w / wsum * math.expm1(r[t]) for t, w in longs)) for r in rows]
    x = [r["ETH"] for r in rows]
    a = [sum(r[t] for t in alts) / len(alts) for r in rows] if alts else None
    b_book = beta(y, x)
    b_alt = beta(a, x) if alts else 0.0
    share = min(1.0, max(0.0, cx["ethShare"] / 100)) if alts and b_alt > 0.2 else 1.0
    return {"rows": rows, "y": y, "x": x, "a": a, "b_book": b_book, "b_alt": b_alt, "share": share,
            "h_eth": share * b_book, "h_alt": (1 - share) * b_book / b_alt if share < 1 else 0.0,
            "hype_beta": beta([r["HYPE"] for r in rows], x)}


def targets_for(cx, h, s, hype_px):
    L = float(s["L"])
    tg = {t: L * w / cx["wsum"] for t, w in cx["longs"]}
    eth = h["h_eth"] * L + (h["hype_beta"] * float(s["hypeQty"]) * hype_px if s["hedgeHype"] else 0.0)
    if eth > 0:
        tg["ETH"] = -eth
    if h["h_alt"] > 0:
        for t in cx["alts"]:
            tg[t] = tg.get(t, 0.0) - h["h_alt"] * L / len(cx["alts"])
    return tg


def book_usd(s, equity=None):
    """The long book in USD: bookPerSide times the account value, as the bot sizes it."""
    return max(0.0, float(s["bookPerSide"]) * float(s["equity"] if equity is None else equity))


def universe(s):
    """The coins the strategy manages: its longs and alt basket, ETH, and every candidate the dashboard offers, so
    a coin dropped from the basket (weight 0, alt unticked) is closed instead of being left behind unhedged.
    Positions in any other coin are left alone."""
    return set(s["weights"]) | set(s["alts"]) | set(LONGS) | set(ALTS) | {"ETH"}


# ---------- the rebalancing decision (the bot trades with exactly this) ----------
# A deviation d (USD per coin) from the target book adds d'Sd of daily PnL variance, S being the covariance of
# the daily returns in the hedge window. Trades t minimise  gamma/2 (d+t)'S(d+t) + sum c_i |t_i|  with
# gamma = risk aversion / account value and c_i the expected cost per USD traded. The absolute-value term gives
# every position a zone in which trading does not pay; outside it a trade stops at the edge of the zone.
def covariance(rows, names):
    n = len(rows)
    dev = []
    for t in names:
        col = [r[t] for r in rows]
        m = sum(col) / n
        dev.append([v - m for v in col])
    k = len(names)
    cov = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            cov[i][j] = cov[j][i] = sum(map(operator.mul, dev[i], dev[j])) / (n - 1)
    return cov


def quad(v, cov):
    return sum(v[i] * cov[i][j] * v[j] for i in range(len(v)) for j in range(len(v)))


def optimal_trades(delta, cov, cost, gamma, sweeps=1000, tol=1e-7):
    """Coordinate descent on  gamma/2 (d + t)' S (d + t) + sum c_i |t_i|  (convex, so it converges).
    A coin with infinite cost is never moved."""
    k = len(delta)
    t = [0.0] * k
    if not gamma > 0:
        return t
    g = [sum(cov[i][j] * delta[j] for j in range(k)) for i in range(k)]  # S (d + t)
    for _ in range(sweeps):
        moved = 0.0
        for i in range(k):
            if cov[i][i] <= 0:
                continue
            b = gamma * (g[i] - cov[i][i] * t[i])
            x = -math.copysign(max(abs(b) - cost[i], 0.0), b) / (gamma * cov[i][i])
            d = x - t[i]
            if d:
                for j in range(k):
                    g[j] += cov[j][i] * d
                t[i] = x
                moved = max(moved, abs(d))
        if moved < tol:
            break
    return t


def decide(current, target, rows, cost, equity, risk_aversion, min_order):
    """Rebalancing trades in USD per coin (positive = buy) and the numbers behind the decision.

    current, target: USD per coin; current holds only coins the strategy manages (see universe). rows: the daily
    return rows of the hedge window. cost: expected cost per USD traded by coin, COST where missing."""
    names = sorted(c for c in set(target) | set(current) if rows and c in rows[0])
    # Opening a position or dropping one is a strategy decision, not a drift: it happens in full, and the
    # optimiser leaves those coins where the decision puts them. It only weighs the drift of positions that
    # exist and stay; a pure risk view would wrongly prefer staying flat over holding the hedged book.
    forced = {c: -v for c, v in current.items() if c not in names or abs(target.get(c, 0.0)) < 1e-9}
    least = max(float(min_order), 1e-9)       # a position below one minimum order counts as not open
    for c in names:
        if abs(current.get(c, 0.0)) < least and abs(target.get(c, 0.0)) >= least:
            forced[c] = target[c] - current.get(c, 0.0)
    rate = {c: cost.get(c, COST) for c in set(names) | set(forced)}
    cov = covariance(rows, names)
    gamma = max(float(risk_aversion), 0.0) / max(float(equity), 1.0)
    before = [current.get(c, 0.0) - target.get(c, 0.0) for c in names]
    delta = [before[i] + forced.get(c, 0.0) for i, c in enumerate(names)]
    t = optimal_trades(delta, cov, [math.inf if c in forced else rate[c] for c in names], gamma)
    trades = dict(forced)
    for i, c in enumerate(names):
        if t[i]:
            trades[c] = trades.get(c, 0.0) + t[i]
    trades = {c: v for c, v in trades.items() if abs(v) >= min_order}
    after = [before[i] + trades.get(c, 0.0) for i, c in enumerate(names)]
    risk_drift, risk_after = quad(delta, cov), quad(after, cov)
    return trades, {
        "risk_before": math.sqrt(max(quad(before, cov), 0.0)),  # daily 1 sigma of the deviation, USD
        "risk_after": math.sqrt(max(risk_after, 0.0)),
        "benefit": gamma / 2 * (risk_drift - risk_after),      # risk cost the drift trades save, USD
        "cost": sum(abs(v) * rate[c] for c, v in trades.items()),
        "opening": sorted(c for c in forced if c in trades and target.get(c)),
        "closing": sorted(c for c in forced if c in trades and not target.get(c)),
        "names": names, "cov": cov,
    }


def plan(snap, s):
    """Today's book: hedge, targets and the trades that pay for themselves given the positions in s["pos"]."""
    cx = context(snap, s)
    C, per = cx["C"], cx["per"]
    N = len(C["ETH"]["closes"])
    h = hedge_at(cx, N, min(int(s["window"]) * per, N - per))
    if not h:
        return None
    yd, xd = [v - mean(h["y"]) for v in h["y"]], [v - mean(h["x"]) for v in h["x"]]
    ad = [v - mean(h["a"]) for v in h["a"]] if h["a"] else [0.0] * len(yd)
    resid = [v - h["h_eth"] * p - h["h_alt"] * q for v, p, q in zip(yd, xd, ad)]
    L = book_usd(s)
    if not L > 0:
        return None
    targets = targets_for(cx, h, dict(s, L=L), C["HYPE"]["mark"])
    pos = {k: float(v) for k, v in s["pos"].items() if float(v)}
    mine = universe(s)
    current = {t: q * C[t]["mark"] for t, q in pos.items() if t in mine and t in C and C[t]["mark"] > 0}
    trades, why = decide(current, targets, h["rows"], {}, float(s["equity"]), s["riskAversion"], float(s["minTrade"]))
    orders = []
    for t in sorted(trades, key=lambda c: (-abs(trades[c]), c)):
        mark, dec = C[t]["mark"], C[t]["szDec"]
        dq = round(trades[t] / mark, dec)
        if not dq:
            continue
        cur, tgt = pos.get(t, 0.0), targets.get(t, 0.0)
        if not tgt:
            what = "Position schließen"
        elif tgt > 0:
            what = "Long öffnen" if cur <= 0 else ("Long aufstocken" if dq > 0 else "Long reduzieren")
        else:
            what = "Short öffnen" if cur >= 0 else ("Short aufstocken" if dq < 0 else "Short reduzieren")
        lev = min(C[t]["maxLev"], s["levEth"] if t == "ETH" else (s["levLong"] if tgt > 0 else s["levShort"]))
        orders.append((t, dq, dq * mark, what, int(lev) if tgt else None))
    return dict(h, alts=cx["alts"], L=L, n=len(h["rows"]), targets=targets, orders=orders, why=why,
                unhedged=std(h["y"]) * math.sqrt(365), resid=std(resid) * math.sqrt(365))


def simulate(snap, s, start_ms):
    """Paper portfolio run the way the bot runs it: open every target at the first 8h close on or after
    start_ms; after each later close re-estimate the hedge and trade only what decide() says pays for itself.
    The long book is bookPerSide times the account value at that close. Pays funding from the history and the
    expected cost per USD traded. Returns the value path in USD for the long/short book and, for comparison,
    the long basket alone (same rule, its own account)."""
    cx = context(snap, s)
    C, per, lc = cx["C"], cx["per"], cx["lc"]
    t0, step, d0 = snap["t0"], snap["step"], snap["d0"]
    N = len(C["ETH"]["closes"])
    n = int(s["window"]) * per
    first = max(next((i for i, v in enumerate(lc[t]) if v is not None), N) for t in cx["need"])
    wanted = next((k for k in range(N) if t0 + (k + 1) * step >= start_ms), N)
    k0 = max(wanted, first + per + n - 1)
    if k0 >= N:
        return None
    px = {}
    for t in cx["need"]:                      # closes, carried forward over gaps
        last, col = None, []
        for v in C[t]["closes"]:
            last = v if v else last
            col.append(last)
        px[t] = col
    fund = {t: C[t].get("fund") or [] for t in cx["need"]}
    equity0, min_order = float(s["equity"]), float(s["minTrade"])
    hype = float(s["hypeQty"]) if s["hedgeHype"] else 0.0

    def portfolio():
        return {"q": {}, "pnl": {}, "fund": {}, "fees": 0.0, "trades": 0, "turnover": 0.0}

    ls, lo = portfolio(), portfolio()
    hype_pnl = 0.0

    def accrue(pf, k, p_new, hours):
        """Carry pf from the close of step k for `hours`, ending at the prices p_new(t)."""
        di = (t0 + (k + 1) * step - d0) // DAY
        for t, q in pf["q"].items():
            p0 = px[t][k]
            pf["pnl"][t] = pf["pnl"].get(t, 0.0) + q * (p_new(t) - p0)
            f = fund[t][di] if di < len(fund[t]) and fund[t][di] is not None else 0.0
            pf["fund"][t] = pf["fund"].get(t, 0.0) - q * p0 * f / 1e4 * hours / 24

    def value(pf, extra=0.0):
        return sum(pf["pnl"].values()) + sum(pf["fund"].values()) - pf["fees"] + extra

    def rebalance(pf, h, k, longs_only):
        equity = equity0 + value(pf)          # the account the bot would see, HYPE spot not included
        tg = targets_for(cx, h, dict(s, L=book_usd(s, equity)), px["HYPE"][k])
        if longs_only:
            tg = {t: v for t, v in tg.items() if v > 0}
        current = {t: q * px[t][k] for t, q in pf["q"].items()}
        trades, _ = decide(current, tg, h["rows"], {}, equity, s["riskAversion"], min_order)
        for t, usd in trades.items():
            if abs(tg.get(t, 0.0)) < 1e-9:
                pf["q"].pop(t, None)          # dropped from the book: closed in full
            else:
                pf["q"][t] = pf["q"].get(t, 0.0) + usd / px[t][k]
            pf["fees"] += abs(usd) * COST
            pf["turnover"] += abs(usd)
            pf["trades"] += 1

    times, v_ls, v_lo = [], [], []
    for k in range(k0, N):
        if k > k0:
            for pf in (ls, lo):
                accrue(pf, k - 1, lambda t: px[t][k], step / HOUR)
            hype_pnl += hype * (px["HYPE"][k] - px["HYPE"][k - 1])
        h = hedge_at(cx, k + 1, n)
        if h:
            rebalance(ls, h, k, False)
            rebalance(lo, h, k, True)
        times.append(t0 + (k + 1) * step)
        v_ls.append(value(ls, hype_pnl))
        v_lo.append(value(lo))
    as_of = to_ms(snap["asOf"])
    if as_of > times[-1]:                     # mark to the live price
        hours = (as_of - times[-1]) / HOUR
        for pf in (ls, lo):
            accrue(pf, N - 1, lambda t: C[t]["mark"], hours)
        hype_pnl += hype * (C["HYPE"]["mark"] - px["HYPE"][N - 1])
        times.append(as_of)
        v_ls.append(value(ls, hype_pnl))
        v_lo.append(value(lo))
    return {"t": times, "ls": v_ls, "lo": v_lo, "k0": k0, "clamped": k0 > wanted, "book": ls, "longOnly": lo,
            "hypePnl": hype_pnl}


# ---------- output ----------
def de(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def signed(x, d=0):
    return ("+" if x >= 0 else "−") + de(abs(x), d)


def paper_line(snap, s):
    if not s.get("paperStart"):
        return None
    try:
        sim = simulate(snap, s, to_ms(str(s["paperStart"])))
    except (ValueError, KeyError):
        return None
    if not sim:
        return None
    v, eq = sim["ls"][-1], float(s["equity"])
    start = datetime.fromtimestamp(sim["t"][0] / 1000, timezone.utc).strftime("%d.%m.%Y")
    line = f"Paper-Portfolio seit {start}: {signed(v)} $"
    if eq > 0:
        line += f" ({signed(100 * v / eq, 1)} % auf {de(eq)} $)"
    if sim["t"][0] <= sim["t"][-1] - DAY:
        j = next(i for i, t in enumerate(sim["t"]) if t >= sim["t"][-1] - DAY)
        line += f", letzte 24 h {signed(v - sim['ls'][j])} $"
    return line + f" · Long-Basket allein {signed(sim['lo'][-1])} $"


def decision_line(why, orders):
    """One line on why the bot would or would not trade, worded like its log."""
    if not orders:
        if why["risk_before"] < 0.5:
            return "Alles im Ziel – nichts zu tun."
        return (f"Kein Trade: Die Abweichung vom Ziel (1σ ≈ {de(why['risk_before'])} $/Tag) kostet weniger, "
                "als das Zurückhandeln kosten würde.")
    if why["opening"] or why["closing"]:
        parts = ([f"{len(why['opening'])} Positionen eröffnen"] if why["opening"] else []) + \
                ([f"{len(why['closing'])} schließen"] if why["closing"] else [])
        return ", ".join(parts) + f", Kosten ≈ {de(why['cost'], 2)} $"
    return (f"Rebalancing lohnt sich: Risiko {de(why['risk_before'])} → {de(why['risk_after'])} $/Tag, "
            f"Nutzen {de(why['benefit'], 2)} $ > Kosten {de(why['cost'], 2)} $")


def summary(snap, s):
    if not book_usd(s) > 0:
        return "Citrini Long/Short – Kontowert oder Long-Book-Faktor ist 0, es gibt kein Zielbuch."
    p = plan(snap, s)
    if not p:
        return "Citrini Long/Short – zu wenig Kursdaten für den Hedge."
    lines = [f"Citrini Long/Short – Marktdaten Stand {snap['asOf'].replace('T', ' ').replace('Z', ' UTC')}",
             f"Konto {de(float(s['equity']))} $ · Long-Book {de(p['L'])} $ · Beta Long-Basket {de(p['b_book'], 2)} · Alt-Korb {de(p['b_alt'], 2)}",
             f"Hedge: ETH {de(p['h_eth'] * p['L'])} $ ({de(100 * p['share'])} %), Alts je {de(p['h_alt'] * p['L'] / max(1, len(p['alts'])))} $ ({', '.join(p['alts'])})",
             f"Rest-Volatilität {de(100 * p['resid'])} % p. a. (ungehedged {de(100 * p['unhedged'])} %), 1σ pro Tag ≈ {de(p['resid'] * p['L'] / math.sqrt(365))} $"]
    paper = paper_line(snap, s)
    if paper:
        lines.append(paper)
    line = decision_line(p["why"], p["orders"])
    lines.append(line + (" (Hebel pro Position, alles im Cross-Margin):" if p["orders"] else ""))
    for t, dq, dusd, what, lev in p["orders"]:
        side = "KAUFEN" if dq > 0 else "VERKAUFEN"
        hebel = f", Hebel {lev}x" if lev else ""
        lines.append(f"- {side} {de(abs(dq), 4).rstrip('0').rstrip(',')} {t} (≈ {de(abs(dusd))} $) – {what}{hebel}")
    return "\n".join(lines)


def load(path):
    with open(path) as f:
        return unwrap(json.load(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("book", nargs="?")
    ap.add_argument("--prev")
    ap.add_argument("--funding-days", type=int)
    ap.add_argument("--offline")
    a = ap.parse_args()
    s = settings(load(a.book) if a.book else None)
    snap = load(a.offline) if a.offline else fetch_snapshot(load(a.prev) if a.prev else None, a.funding_days)
    with open(a.out, "w") as f:
        json.dump(snap, f, separators=(",", ":"))
    print(summary(snap, s))


if __name__ == "__main__":
    main()
