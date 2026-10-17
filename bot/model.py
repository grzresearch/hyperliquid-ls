"""The book the bot aims for, its risk, and the rule it trades by.

* book: longs from the screener with their weights; a short book whose beta to ETH offsets the long basket's beta.
  The short names and their shares come from bot/hedge.py (the liquid perps that hedge this basket best); its size
  follows the betas of every round. The older fixed basket (a share in ETH, the rest spread over listed alts) is
  still there for comparison. Betas come from overlapping daily log returns measured every 8 hours over the hedge
  window. The long book is book_per_side times the account value. Jump limits (bot/risk.py) cap single positions:
  a long above its cap gives the excess to the other longs, a short gives its beta to ETH.
* risk: the daily covariance S of those returns. A deviation d (USD per coin) from the target book adds d'Sd of
  daily PnL variance.
* decision: trades t minimise  gamma/2 (d + t)'S(d + t) + sum c_i |t_i|  with gamma = risk_aversion / equity and
  c_i the expected cost per USD traded (bot/costs.py). The absolute-value term gives every position a zone in which
  trading does not pay; outside it a trade stops at the edge of that zone. New positions are opened and dropped
  positions closed in full: those are strategy decisions, and a pure risk view would prefer staying flat.
"""
import math
import operator

PER_DAY = 3                   # 8h closes per day
DEFAULT_COST = 0.00027        # per USD traded, used only where no order book was priced


def mean(v):
    return sum(v) / len(v)


def std(v):
    m = mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / (len(v) - 1))


def beta(y, x):
    my, mx = mean(y), mean(x)
    return sum((a - my) * (b - mx) for a, b in zip(y, x)) / sum((b - mx) ** 2 for b in x)


# ---------- the target book ----------
def context(snap, s):
    """Log closes of everything the book needs. s: weights (long -> weight) and either shorts (coin -> share of the
    short book, from bot/hedge.py) or the fixed basket: alts and ethShare."""
    C = snap["coins"]
    longs = [(t, float(w)) for t, w in s["weights"].items() if float(w) > 0 and t in C]
    held = {t for t, _ in longs}
    shorts = [(t, float(w)) for t, w in (s.get("shorts") or {}).items() if float(w) > 0 and t in C and t not in held]
    alts = [] if shorts else [t for t in s.get("alts", []) if t in C and t not in held]
    need = list(dict.fromkeys([t for t, _ in longs] + [t for t, _ in shorts] + alts + ["ETH"]))
    lc = {t: [math.log(v) if v else None for v in C[t]["closes"]] for t in need}
    return {"C": C, "per": int(snap.get("perDay", PER_DAY)), "longs": longs, "wsum": sum(w for _, w in longs),
            "shorts": shorts, "ssum": sum(w for _, w in shorts), "alts": alts, "need": need, "lc": lc,
            "ethShare": float(s.get("ethShare", 40))}


def return_rows(cx, end, n):
    """Overlapping daily log returns at 8h steps end-n .. end-1 of everything in cx["need"], complete rows only."""
    per, lc = cx["per"], cx["lc"]
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
    return rows


def hedge_at(cx, end, n):
    """The hedge from the overlapping daily log returns at 8h steps end-n .. end-1: k, the short in USD per USD of
    longs for each hedge coin, sized so the book's beta to ETH is zero."""
    longs, wsum = cx["longs"], cx["wsum"]
    rows = return_rows(cx, end, n)
    if not longs or len(rows) < 30:
        return None
    y = [math.log1p(sum(w / wsum * math.expm1(r[t]) for t, w in longs)) for r in rows]
    x = [r["ETH"] for r in rows]
    b_book = beta(y, x)
    if cx["shorts"]:
        ssum = cx["ssum"]
        a = [math.log1p(sum(w / ssum * math.expm1(r[t]) for t, w in cx["shorts"])) for r in rows]
        b_short = beta(a, x)
        if b_short > 0.2:
            H = b_book / b_short
            k = {t: H * w / ssum for t, w in cx["shorts"]}
            return {"rows": rows, "y": y, "x": x, "a": a, "b_book": b_book, "b_short": b_short, "H": H, "k": k,
                    "mode": "minvar"}
    if not cx["alts"]:          # no short book yet, or one that hardly moves with ETH: ETH alone
        return {"rows": rows, "y": y, "x": x, "a": x, "b_book": b_book, "b_short": 1.0, "H": b_book,
                "k": {"ETH": b_book}, "mode": "eth"}
    alts = cx["alts"]
    a = [sum(r[t] for t in alts) / len(alts) for r in rows] if alts else None
    b_alt = beta(a, x) if alts else 0.0
    share = min(1.0, max(0.0, cx["ethShare"] / 100)) if alts and b_alt > 0.2 else 1.0
    h_eth, h_alt = share * b_book, (1 - share) * b_book / b_alt if share < 1 else 0.0
    k = {"ETH": h_eth, **{t: h_alt / len(alts) for t in alts if h_alt > 0}}
    return {"rows": rows, "y": y, "x": x, "a": a, "b_book": b_book, "b_short": b_book / sum(k.values()),
            "H": sum(k.values()), "k": k, "mode": "fixed", "share": share, "b_alt": b_alt}


def water_fill(wanted, caps):
    """Amounts as close to `wanted` as the caps allow, keeping the total: a coin above its cap stays at the cap and the
    others share what it gives up, in proportion to what they wanted, up to their own caps. If every coin sits at
    its cap, the total is smaller."""
    total, fixed = sum(wanted.values()), {}
    while True:
        free = {c: v for c, v in wanted.items() if c not in fixed}
        if not free:
            return fixed
        mass, tot = total - sum(fixed.values()), sum(free.values())
        trial = {c: mass * v / tot if tot > 0 else 0.0 for c, v in free.items()}
        over = [c for c, v in trial.items() if v > caps.get(c, math.inf) + 1e-9]
        if not over:
            fixed.update(trial)
            return fixed
        for c in over:
            fixed[c] = caps[c]


def targets(snap, s, L, caps=None):
    """Target positions for a long book of L, the hedge behind them, and the rows they were estimated from.
    caps: the largest position per coin in USD (the jump limits of bot/risk.py). A long above its cap gives the
    excess to the other longs; a short above its cap gives its beta to ETH, so the book stays beta-neutral.
    h["capped"] lists what was cut: {coin: {"want", "cap"}} in signed USD."""
    caps = caps or {}
    longs = {t: float(w) for t, w in s["weights"].items() if float(w) > 0 and t in snap["coins"]}
    total = sum(longs.values())
    want = {t: L * w / total for t, w in longs.items()} if total else {}
    usd = water_fill(want, caps)
    cx = context(snap, dict(s, weights=usd))
    n_steps = len(cx["C"]["ETH"]["closes"])
    h = hedge_at(cx, n_steps, min(int(s["window"]) * cx["per"], n_steps - cx["per"]))
    if not h:
        raise RuntimeError("Zu wenig Kursdaten für die Hedge-Quoten")
    base = cx["wsum"]                       # the long dollars the caps leave
    capped = {t: {"want": want[t], "cap": usd.get(t, 0.0)} for t in want if usd.get(t, 0.0) < want[t] - 1e-6}
    short = {t: ratio * base for t, ratio in h["k"].items() if ratio > 0}      # the hedge, before any netting
    b = None
    for t in sorted(short):
        if t != "ETH" and short[t] > caps.get(t, math.inf):
            b = b or betas(h["rows"], list(h["rows"][0]))
            extra = short[t] - caps[t]
            capped[t] = {"want": -short[t], "cap": -caps[t]}
            short[t] = caps[t]
            short["ETH"] = short.get("ETH", 0.0) + extra * b.get(t, 1.0)
    tg = {t: w for t, w in cx["longs"]}
    for t, v in short.items():
        tg[t] = tg.get(t, 0.0) - v
    if base > 0 and short:
        h["k"] = {t: v / base for t, v in short.items()}
        h["H"] = sum(h["k"].values())
        if h["H"] > 0:                      # the short side's beta and ETH's share after the limits
            h["b_short"] = h["b_book"] / h["H"]
            if h.get("mode") == "fixed":
                h["share"] = h["k"].get("ETH", 0.0) / h["H"]
    h["capped"] = capped
    return tg, h


def residual_vol(h):
    """Annualised volatility of the hedged book's daily return, and of the long basket alone."""
    resid = [math.expm1(y) - sum(k * math.expm1(r[t]) for t, k in h["k"].items()) for y, r in zip(h["y"], h["rows"])]
    return std(resid) * math.sqrt(365), std([math.expm1(v) for v in h["y"]]) * math.sqrt(365)


# ---------- risk ----------
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


def betas(rows, names):
    x = [r["ETH"] for r in rows]
    mx = sum(x) / len(x)
    vx = sum((v - mx) ** 2 for v in x)
    out = {}
    for t in names:
        y = [r[t] for r in rows]
        my = sum(y) / len(y)
        out[t] = sum((a - my) * (b - mx) for a, b in zip(y, x)) / vx
    return out


# ---------- the rebalancing decision ----------
def optimal_trades(delta, cov, cost, gamma, sweeps=1000, tol=1e-7, lo=None, hi=None):
    """Coordinate descent on  gamma/2 (d + t)' S (d + t) + sum c_i |t_i|  (convex, so it converges), each t_i kept
    within [lo_i, hi_i] (which contain 0). A coin with infinite cost is never moved."""
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
            if lo is not None:
                x = min(hi[i], max(lo[i], x))
            d = x - t[i]
            if d:
                for j in range(k):
                    g[j] += cov[j][i] * d
                t[i] = x
                moved = max(moved, abs(d))
        if moved < tol:
            break
    return t


def decide(current, target, rows, cost, equity, risk_aversion, min_order, risk_scale=1.0, force=(), caps=None):
    """Rebalancing trades in USD per coin (positive = buy) and the numbers behind the decision.

    current, target: USD per coin; current holds only coins the bot manages. rows: the daily return rows of the
    hedge window. cost: expected cost per USD traded by coin, DEFAULT_COST where missing. risk_scale: tomorrow's
    variance over the window's (bot/risk.py: GARCH and HMM), so a deviation costs more risk in a restless market.
    force: coins that go to their target in full, like an opening (a position beyond its jump limit, or the whole
    book when its size is off): the covariance cannot see jump risk, and a hedged book that is simply too big looks
    like a small deviation to it. caps: the jump limits in USD; the drift trades never push a position past its
    cap (else drift and force would take turns on a coin that hedges another one well).

    Besides the trades it returns what the dashboard draws: the risk cost of the drift if nothing is traded
    (gamma/2 d'Sd, USD per day), the cost of trading all of it back, and where the chosen trades land."""
    names = sorted(c for c in set(target) | set(current) if rows and c in rows[0])
    # Opening a position or dropping one is a strategy decision, not a drift: it happens in full, and the
    # optimiser leaves those coins where the decision puts them. It only weighs the drift of positions that
    # exist and stay.
    forced = {c: -v for c, v in current.items() if c not in names or abs(target.get(c, 0.0)) < 1e-9}
    least = max(float(min_order), 1e-9)       # a position below one minimum order counts as not open
    for c in names:
        if abs(current.get(c, 0.0)) < least and abs(target.get(c, 0.0)) >= least:
            forced[c] = target[c] - current.get(c, 0.0)
    resized = []
    for c in force:
        if c not in forced and abs(target.get(c, 0.0) - current.get(c, 0.0)) >= least:
            forced[c] = target.get(c, 0.0) - current.get(c, 0.0)
            resized.append(c)
    rate = {c: cost.get(c, DEFAULT_COST) for c in set(names) | set(forced)}
    cov = [[v * risk_scale for v in row] for row in covariance(rows, names)]
    gamma = max(float(risk_aversion), 0.0) / max(float(equity), 1.0)
    before = [current.get(c, 0.0) - target.get(c, 0.0) for c in names]
    delta = [before[i] + forced.get(c, 0.0) for i, c in enumerate(names)]
    # Risk cost one dollar traded back saves, per coin, before anything is traded (gamma S d). Nothing is traded
    # exactly when it is below the coin's cost per dollar everywhere: that is the bot's trigger.
    marginal = [gamma * sum(cov[i][j] * delta[j] for j in range(len(names))) for i in range(len(names))]
    lo, hi = [-math.inf] * len(names), [math.inf] * len(names)
    for i, c in enumerate(names):
        cap, cur = (caps or {}).get(c, math.inf), current.get(c, 0.0)
        if cap < math.inf:
            if (target.get(c, 0.0) or cur) >= 0:
                hi[i] = max(0.0, cap - cur)
            else:
                lo[i] = min(0.0, -cap - cur)
    t = optimal_trades(delta, cov, [math.inf if c in forced else rate[c] for c in names], gamma, lo=lo, hi=hi)
    trades = dict(forced)
    for i, c in enumerate(names):
        if t[i]:
            trades[c] = trades.get(c, 0.0) + t[i]
    trades = {c: v for c, v in trades.items() if abs(v) >= min_order}
    after = [before[i] + trades.get(c, 0.0) for i, c in enumerate(names)]
    risk_drift, risk_after = quad(delta, cov), quad(after, cov)
    drift = [c for c in names if c not in forced]
    return trades, {
        "risk_before": math.sqrt(max(quad(before, cov), 0.0)),  # daily 1 sigma of the deviation, USD
        "risk_after": math.sqrt(max(risk_after, 0.0)),
        "benefit": gamma / 2 * (risk_drift - risk_after),      # risk cost the drift trades save, USD
        "cost": sum(abs(v) * rate[c] for c, v in trades.items()),
        "opening": sorted(c for c in forced if c in trades and target.get(c) and c not in resized),
        "closing": sorted(c for c in forced if c in trades and not target.get(c) and c not in resized),
        "resized": sorted(c for c in resized if c in trades),
        "gamma": gamma,
        "hold_risk_cost": gamma / 2 * risk_drift,               # keep the drift: risk cost per day
        "full_cost": sum(rate[c] * abs(delta[names.index(c)]) for c in drift),   # trade all drift back
        "after_risk_cost": gamma / 2 * risk_after,
        "drift_cost": sum(rate[c] * abs(trades.get(c, 0.0)) for c in drift),
        "by_coin": [{"coin": c, "deviation": delta[i], "trade": trades.get(c, 0.0), "rate": rate[c],
                     "marginal": abs(marginal[i]), "forced": c in forced} for i, c in enumerate(names)],
        "names": names, "cov": cov,
    }
