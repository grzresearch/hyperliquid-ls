"""Targets, risk model and the cost-aware rebalancing decision.

The hedge ratios come from refresh.py, the same maths the dashboard runs, so the bot aims at exactly the book the
page shows. On top of that:

* risk: the daily covariance of the overlapping daily returns in the hedge window. A deviation d (USD per coin)
  from the target book adds d' S d of daily PnL variance.
* cost: per USD traded, the maker fee plus the expected taker share times (taker fee - maker fee + half spread).
* decision: trades t minimise  gamma/2 * (d + t)' S (d + t) + sum c_i |t_i|  with gamma = risk_aversion / equity.
  The absolute-value cost term gives every position a zone in which trading does not pay; outside it the trade stops
  at the edge of that zone. New positions are opened and dropped positions closed in full: those are strategy
  decisions, and a pure risk view would wrongly prefer staying flat over holding the hedged book.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import refresh as R  # noqa: E402  the dashboard's hedge maths


def strategy_settings(cfg, equity):
    st = cfg["strategy"]
    return {"L": cfg["book_per_side"] * equity, "weights": st["weights"], "alts": st["alts"],
            "window": st["window_days"], "ethShare": st["eth_share"], "hedgeHype": False, "hypeQty": 0}


def targets(snap, cfg, equity):
    """Target positions in USD (signed), the hedge ratios and the return rows they were estimated from."""
    s = strategy_settings(cfg, equity)
    cx = R.context(snap, s)
    n_steps = len(cx["C"]["ETH"]["closes"])
    h = R.hedge_at(cx, n_steps, min(int(s["window"]) * cx["per"], n_steps - cx["per"]))
    if not h:
        raise RuntimeError("Zu wenig Kursdaten für die Hedge-Quoten")
    return R.targets_for(cx, h, s, 0.0), h


def covariance(rows, names):
    n = len(rows)
    mean = {t: sum(r[t] for r in rows) / n for t in names}
    dev = {t: [r[t] - mean[t] for r in rows] for t in names}
    return [[sum(a * b for a, b in zip(dev[p], dev[q])) / (n - 1) for q in names] for p in names]


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


def cost_rates(books, ex):
    """Expected cost per USD traded with maker-first execution."""
    out = {}
    for coin, (bids, asks) in books.items():
        half_spread = (asks[0][0] - bids[0][0]) / (asks[0][0] + bids[0][0]) if bids and asks else 0.001
        out[coin] = ex["maker_fee"] + ex["taker_share"] * (ex["taker_fee"] - ex["maker_fee"] + half_spread)
    return out


def quad(v, cov):
    return sum(v[i] * cov[i][j] * v[j] for i in range(len(v)) for j in range(len(v)))


def optimal_trades(delta, cov, cost, gamma, sweeps=1000, tol=1e-7):
    """Coordinate descent on  gamma/2 (d + t)' S (d + t) + sum c_i |t_i|  (convex, so it converges)."""
    k = len(delta)
    t = [0.0] * k
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


def universe(cfg):
    """The coins the bot manages. Positions in any other coin are left alone."""
    st = cfg["strategy"]
    return set(st["weights"]) | set(st["alts"]) | {"ETH"}


def decide(current, target, rows, cost, equity, cfg):
    """Rebalancing trades in USD per coin (positive = buy) and the numbers behind the decision.

    current, target: USD per coin. rows: return rows of the hedge window. cost: expected cost per USD traded."""
    rb = cfg["rebalancing"]
    current = {c: v for c, v in current.items() if c in universe(cfg)}
    names = [c for c in dict.fromkeys(list(target) + list(current)) if rows and c in rows[0]]
    # Opening a position or dropping one is a strategy decision, not a drift: it happens in full. The optimiser
    # only weighs the drift of positions that exist and stay.
    forced = {c: -v for c, v in current.items() if c not in names or abs(target.get(c, 0.0)) < 1e-9}
    for c in names:
        if abs(current.get(c, 0.0)) < rb["min_order_usd"] and abs(target.get(c, 0.0)) >= rb["min_order_usd"]:
            forced[c] = target[c] - current.get(c, 0.0)
    cov = covariance(rows, names)
    gamma = rb["risk_aversion"] / max(equity, 1.0)
    before = [current.get(c, 0.0) - target.get(c, 0.0) for c in names]
    delta = [before[i] + forced.get(c, 0.0) for i, c in enumerate(names)]
    t = optimal_trades(delta, cov, [cost.get(c, 0.001) for c in names], gamma)
    trades = dict(forced)
    for i, c in enumerate(names):
        if t[i]:
            trades[c] = trades.get(c, 0.0) + t[i]
    trades = {c: v for c, v in trades.items() if abs(v) >= rb["min_order_usd"]}
    after = [before[i] + trades.get(c, 0.0) for i, c in enumerate(names)]
    risk_drift, risk_after = quad(delta, cov), quad(after, cov)
    return trades, {
        "risk_before": math.sqrt(max(quad(before, cov), 0.0)),  # daily 1 sigma of the deviation, USD
        "risk_after": math.sqrt(max(risk_after, 0.0)),
        "benefit": gamma / 2 * (risk_drift - risk_after),      # risk cost the drift trades save, USD
        "cost": sum(abs(v) * cost.get(c, 0.001) for c, v in trades.items()),
        "opening": sorted(c for c in forced if abs(trades.get(c, 0.0)) and target.get(c)),
        "closing": sorted(c for c in forced if not target.get(c)),
        "names": names, "cov": cov,
    }
