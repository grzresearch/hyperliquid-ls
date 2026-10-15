"""The book the bot aims for, its risk, and the rule it trades by.

* book: longs from the screener with their weights; a short hedge whose beta to ETH offsets the long basket's beta,
  a fixed share of it in ETH and the rest spread evenly over an alt basket. Betas come from overlapping daily log
  returns measured every 8 hours over the hedge window. The long book is book_per_side times the account value.
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
Z = {0.95: 1.6448536269514722, 0.99: 2.3263478740408408}


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
    """Log closes of everything the hedge needs. s: weights (long -> weight), alts, ethShare."""
    C = snap["coins"]
    longs = [(t, float(w)) for t, w in s["weights"].items() if float(w) > 0 and t in C]
    alts = [t for t in s["alts"] if t in C and t not in dict(longs)]
    need = [t for t, _ in longs] + alts + ["ETH"]
    lc = {t: [math.log(v) if v else None for v in C[t]["closes"]] for t in need}
    return {"C": C, "per": int(snap.get("perDay", PER_DAY)), "longs": longs, "wsum": sum(w for _, w in longs),
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
            "h_eth": share * b_book, "h_alt": (1 - share) * b_book / b_alt if share < 1 else 0.0}


def targets_for(cx, h, L):
    """Target positions in USD (signed) for a long book of L."""
    tg = {t: L * w / cx["wsum"] for t, w in cx["longs"]}
    eth = h["h_eth"] * L
    if eth > 0:
        tg["ETH"] = -eth
    if h["h_alt"] > 0:
        for t in cx["alts"]:
            tg[t] = tg.get(t, 0.0) - h["h_alt"] * L / len(cx["alts"])
    return tg


def targets(snap, s, L):
    """Target positions for a long book of L, the hedge behind them, and the rows they were estimated from."""
    cx = context(snap, s)
    n_steps = len(cx["C"]["ETH"]["closes"])
    h = hedge_at(cx, n_steps, min(int(s["window"]) * cx["per"], n_steps - cx["per"]))
    if not h:
        raise RuntimeError("Zu wenig Kursdaten für die Hedge-Quoten")
    return targets_for(cx, h, L), h


def residual_vol(h):
    """Annualised volatility of the hedged book's daily return, and of the long basket alone."""
    yd, xd = [v - mean(h["y"]) for v in h["y"]], [v - mean(h["x"]) for v in h["x"]]
    ad = [v - mean(h["a"]) for v in h["a"]] if h["a"] else [0.0] * len(yd)
    resid = [v - h["h_eth"] * p - h["h_alt"] * q for v, p, q in zip(yd, xd, ad)]
    return std(resid) * math.sqrt(365), std(h["y"]) * math.sqrt(365)


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


def quantile(sorted_values, p):
    """Linear interpolation between order statistics (the usual definition, type 7)."""
    pos = p * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def value_at_risk(exposure, rows):
    """One-day value at risk and expected shortfall of a book (USD per coin), as positive losses in USD.

    Historical: each day of the window's returns applied to today's book. Parametric: a normal distribution with
    the window's covariance. Also returns the historical PnL samples for a histogram."""
    names = [c for c in exposure if rows and c in rows[0] and exposure[c]]
    if not names:
        return None
    pnl = [sum(exposure[c] * math.expm1(r[c]) for c in names) for r in rows]
    srt = sorted(pnl)
    sigma = math.sqrt(max(quad([exposure[c] for c in names], covariance(rows, names)), 0.0))
    out = {"sigma": sigma, "samples": [round(v, 2) for v in pnl], "n": len(pnl)}
    for p in (0.95, 0.99):
        q = quantile(srt, 1 - p)
        tail = [v for v in srt if v <= q] or [srt[0]]
        z = Z[p]
        density = math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
        out[f"{int(p * 100)}"] = {"hist": -q, "es_hist": -mean(tail),
                                  "param": z * sigma, "es_param": sigma * density / (1 - p)}
    return out


# ---------- the rebalancing decision ----------
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

    current, target: USD per coin; current holds only coins the bot manages. rows: the daily return rows of the
    hedge window. cost: expected cost per USD traded by coin, DEFAULT_COST where missing.

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
    rate = {c: cost.get(c, DEFAULT_COST) for c in set(names) | set(forced)}
    cov = covariance(rows, names)
    gamma = max(float(risk_aversion), 0.0) / max(float(equity), 1.0)
    before = [current.get(c, 0.0) - target.get(c, 0.0) for c in names]
    delta = [before[i] + forced.get(c, 0.0) for i, c in enumerate(names)]
    # Risk cost one dollar traded back saves, per coin, before anything is traded (gamma S d). Nothing is traded
    # exactly when it is below the coin's cost per dollar everywhere: that is the bot's trigger.
    marginal = [gamma * sum(cov[i][j] * delta[j] for j in range(len(names))) for i in range(len(names))]
    t = optimal_trades(delta, cov, [math.inf if c in forced else rate[c] for c in names], gamma)
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
        "opening": sorted(c for c in forced if c in trades and target.get(c)),
        "closing": sorted(c for c in forced if c in trades and not target.get(c)),
        "gamma": gamma,
        "hold_risk_cost": gamma / 2 * risk_drift,               # keep the drift: risk cost per day
        "full_cost": sum(rate[c] * abs(delta[names.index(c)]) for c in drift),   # trade all drift back
        "after_risk_cost": gamma / 2 * risk_after,
        "drift_cost": sum(rate[c] * abs(trades.get(c, 0.0)) for c in drift),
        "by_coin": [{"coin": c, "deviation": delta[i], "trade": trades.get(c, 0.0), "rate": rate[c],
                     "marginal": abs(marginal[i]), "forced": c in forced} for i, c in enumerate(names)],
        "names": names, "cov": cov,
    }
