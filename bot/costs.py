"""Execution costs as implementation shortfall, after github.com/gelatotrade/implementation-shortfall-hyperliquid.

The book walk and the decomposition follow that repository's simulator (Modul 4) and decomposer (Modul 5):

    shortfall = spread + impact + fee        (Perold 1988; opportunity cost is reported as the unfilled share)

* spread: crossing the touch, (ask - mid) / mid for a buy, times the filled fraction
* impact: walking past the touch, (VWAP - ask) / mid for a buy, times the filled fraction
* fee:    the exchange fee, times the filled fraction

All figures are basis points of the order's notional; positive is a cost.

Ex ante the bot prices a trade the way it executes it: maker first (post-only, maker fee), and the share it expects
to finish as taker walks the live order book. Ex post it measures every executed trade against the mid price at the
moment it decided to trade, so simulated and realised costs can be compared trade by trade.
"""

BPS = 1e4


def walk_book(levels, target_usd):
    """Walk price levels, best first, until target_usd is filled. Returns the fills [(price, quantity)], the USD
    left unfilled and the number of levels touched (Modul 4 of the IS repository)."""
    remaining = target_usd
    fills = []
    for i, (px, sz) in enumerate(levels):
        if remaining <= 0:
            return fills, 0.0, i
        level_notional = px * sz
        if level_notional >= remaining:
            fills.append((px, remaining / px))
            return fills, 0.0, i + 1
        fills.append((px, sz))
        remaining -= level_notional
    return fills, remaining, len(levels)


def taker_shortfall(bids, asks, is_buy, usd, taker_fee):
    """Cost of sweeping `usd` through the visible book right now, in basis points of `usd` (Modul 5)."""
    bid, ask = bids[0][0], asks[0][0]
    mid = (bid + ask) / 2
    fills, unfilled, levels = walk_book(asks if is_buy else bids, usd)
    filled_usd = sum(px * q for px, q in fills)
    qty = sum(q for _, q in fills)
    frac = filled_usd / usd if usd > 0 else 0.0
    vwap = filled_usd / qty if qty else (ask if is_buy else bid)
    spread = ((ask - mid) if is_buy else (mid - bid)) / mid * BPS * frac
    impact = ((vwap - ask) if is_buy else (bid - vwap)) / mid * BPS * frac
    fee = taker_fee * BPS * frac
    return {"spread": spread, "impact": impact, "fee": fee, "total": spread + impact + fee,
            "fill": frac, "unfilled_usd": unfilled, "levels": levels, "mid": mid, "vwap": vwap}


def expected_cost(bids, asks, is_buy, usd, ex):
    """Expected cost of a maker-first order of `usd`, as a rate per USD traded, and its parts in basis points.

    The share ex["taker_share"] is assumed to end up as taker and walks the live book; the rest fills passively at
    the maker fee. A book too thin for the taker share adds ex["unfilled_bps"] on the part it cannot absorb."""
    share = ex["taker_share"]
    maker_bps = ex["maker_fee"] * BPS
    if not bids or not asks:
        return (maker_bps + share * ex.get("unfilled_bps", 50.0)) / BPS, {"maker": maker_bps, "taker": None}
    taker = taker_shortfall(bids, asks, is_buy, max(usd * share, 1.0), ex["taker_fee"])
    taker_bps = taker["total"] + (1 - taker["fill"]) * ex.get("unfilled_bps", 50.0)
    bps = (1 - share) * maker_bps + share * taker_bps
    return bps / BPS, {"bps": bps, "maker": maker_bps, "spread": taker["spread"], "impact": taker["impact"],
                       "fee": taker["fee"], "fill": taker["fill"], "levels": taker["levels"], "mid": taker["mid"]}


def realised_shortfall(fills, arrival_mid, intended_usd):
    """Cost of what was actually executed for one trade, against the mid when the bot decided to trade.

    fills: [{"sz": signed size, "px": price, "fee": USD, "crossed": taker?}]. intended_usd: signed USD the bot set
    out to trade. The price part is negative when passive fills bought below (or sold above) the arrival mid."""
    side = 1.0 if intended_usd > 0 else -1.0
    qty = sum(abs(f["sz"]) for f in fills)
    notional = sum(abs(f["sz"]) * f["px"] for f in fills)
    fees = sum(f["fee"] for f in fills)
    order = abs(intended_usd)
    if not qty or not order or not arrival_mid:
        return {"price": 0.0, "fee": 0.0, "total": 0.0, "usd": 0.0, "filled": 0.0, "maker": 0.0,
                "vwap": None, "notional": 0.0}
    vwap = notional / qty
    price_usd = side * (vwap - arrival_mid) * qty
    maker = sum(abs(f["sz"]) * f["px"] for f in fills if not f.get("crossed")) / notional
    price_bps, fee_bps = price_usd / order * BPS, fees / order * BPS
    return {"price": price_bps, "fee": fee_bps, "total": price_bps + fee_bps, "usd": price_usd + fees,
            "filled": min(1.0, notional / order), "maker": maker, "vwap": vwap, "notional": notional}
