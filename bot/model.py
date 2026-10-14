"""Targets, risk model and the cost-aware rebalancing decision.

The hedge ratios and the decision come from refresh.py, the same maths the dashboard runs, so the bot aims at
exactly the book the page shows and trades by the same rule:

* risk: the daily covariance of the overlapping daily returns in the hedge window. A deviation d (USD per coin)
  from the target book adds d' S d of daily PnL variance.
* cost: per USD traded, the maker fee plus the expected taker share times (taker fee - maker fee + half spread),
  here with the live spread of each market.
* decision: trades t minimise  gamma/2 * (d + t)' S (d + t) + sum c_i |t_i|  with gamma = risk_aversion / equity.
  The absolute-value cost term gives every position a zone in which trading does not pay; outside it the trade stops
  at the edge of that zone. New positions are opened and dropped positions closed in full: those are strategy
  decisions, and a pure risk view would wrongly prefer staying flat over holding the hedged book.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import refresh as R  # noqa: E402  the dashboard's maths
from refresh import covariance, optimal_trades, quad  # noqa: E402,F401  re-exported for the tests


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


def strategy_coins(cfg):
    """The coins the strategy holds: its longs (weight 0 means close), the alt basket and ETH."""
    st = cfg["strategy"]
    return set(st["weights"]) | set(st["alts"]) | {"ETH"}


def universe(cfg):
    """The coins the bot manages: the strategy's coins plus every candidate the dashboard offers, so a coin dropped
    from the basket is closed instead of being left behind unhedged. Positions in any other coin are left alone."""
    return strategy_coins(cfg) | set(R.LONGS) | set(R.ALTS)


def decide(current, target, rows, cost, equity, cfg):
    """Rebalancing trades in USD per coin (positive = buy) and the numbers behind the decision, by the rule
    the dashboard shows (refresh.decide).

    current, target: USD per coin. rows: return rows of the hedge window. cost: expected cost per USD traded."""
    rb = cfg["rebalancing"]
    mine = {c: v for c, v in current.items() if c in universe(cfg)}
    return R.decide(mine, target, rows, cost, equity, rb["risk_aversion"], rb["min_order_usd"])
