"""Maker-first execution against Hyperliquid, for real (testnet or mainnet) or on paper.

Each trade starts as a post-only order one tick inside the spread, re-priced only when someone else improves the
touch. If fills leave the book with a large net beta exposure, just enough of the lagging side is completed at once.
Whatever is still open when the maker window ends goes out as an immediate-or-cancel order with a price cap. Resting
orders are cancelled on every exit path, and at the start of each run.
"""
import json
import math
import os
import time

from .hl import round_price, round_size, tick

MIN_NOTIONAL = 10.0  # Hyperliquid's minimum order value in USD


class FatalError(RuntimeError):
    """Something is wrong with the account setup; the bot must stop instead of retrying."""


def maker_price(is_buy, bids, asks, sz_dec):
    """One tick inside the spread, or at the touch when the spread is a single tick."""
    bid, ask = bids[0][0], asks[0][0]
    if is_buy:
        px = round_price(bid + tick(bid, sz_dec), sz_dec)
        return px if px < ask else bid
    px = round_price(ask - tick(ask, sz_dec), sz_dec)
    return px if px > bid else ask


def without_own(levels, px, sz):
    """The book side as the rest of the market sees it: our own resting order taken out."""
    out = []
    for p, q in levels:
        if p == px:
            q -= sz
        if q > 1e-12:
            out.append((p, q))
    return out


class PaperBroker:
    """A simulated account on live Hyperliquid prices. Post-only orders fill once the book trades through their
    price, IOC orders walk the visible book. Fees and hourly funding are charged as on the exchange."""

    quotes_in_book = False   # paper orders never appear in the real order book

    def __init__(self, info, path, start_equity, fees):
        self.info, self.path, self.fees = info, path, fees
        if path and os.path.exists(path):
            with open(path) as f:
                self.s = json.load(f)
        else:
            self.s = {"start": start_equity, "cash": start_equity, "pos": {}, "fees": 0.0, "funding": 0.0,
                      "realized": 0.0, "volume": 0.0, "t": time.time(), "since": time.time()}
        self.orders, self.next_oid, self.fills = {}, 1, []

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.s, f, indent=1)
        os.replace(tmp, self.path)

    def account(self, markets):
        upnl = sum(p["sz"] * (markets[c]["mark"] - p["avg"]) for c, p in self.s["pos"].items() if c in markets)
        return {"equity": self.s["cash"] + upnl, "positions": self.positions()}

    def positions(self):
        return {c: p["sz"] for c, p in self.s["pos"].items()}

    def accrue_funding(self, markets):
        now = time.time()
        hours, self.s["t"] = (now - self.s["t"]) / 3600, now
        for c, p in self.s["pos"].items():
            if c in markets:   # longs pay positive funding, shorts receive it
                paid = p["sz"] * markets[c]["mark"] * markets[c]["funding"] * hours
                self.s["cash"] -= paid
                self.s["funding"] -= paid

    def set_leverage(self, coin, lev):
        return None

    def book(self, coin):
        return self.info.book(coin)

    def _fill(self, coin, q, px, maker):
        fee = abs(q) * px * (self.fees["maker"] if maker else self.fees["taker"])
        p = self.s["pos"].get(coin, {"sz": 0.0, "avg": 0.0})
        sz, avg = p["sz"], p["avg"]
        new = sz + q
        if sz == 0 or (sz > 0) == (q > 0):
            avg = (sz * avg + q * px) / new
        else:
            closed = min(abs(q), abs(sz)) * (1 if sz > 0 else -1)
            pnl = closed * (px - avg)
            self.s["realized"] += pnl
            self.s["cash"] += pnl
            if new and (new > 0) != (sz > 0):
                avg = px
        if abs(new) < 1e-12:
            self.s["pos"].pop(coin, None)
        else:
            self.s["pos"][coin] = {"sz": new, "avg": avg}
        self.s["cash"] -= fee
        self.s["fees"] += fee
        self.s["volume"] += abs(q) * px
        self.fills.append({"coin": coin, "sz": q, "px": px, "fee": fee, "crossed": not maker, "time": time.time()})

    def post_only(self, coin, is_buy, sz, px, reduce_only):
        bids, asks = self.info.book(coin)
        if (is_buy and asks and px >= asks[0][0]) or (not is_buy and bids and px <= bids[0][0]):
            return None, "würde sofort ausführen"
        oid, self.next_oid = self.next_oid, self.next_oid + 1
        self.orders[oid] = {"coin": coin, "is_buy": is_buy, "sz": sz, "px": px}
        return oid, None

    def modify(self, oid, coin, is_buy, sz, px, reduce_only):
        if oid not in self.orders:
            return None, "nicht mehr offen"
        self.orders[oid].update(sz=sz, px=px)
        return oid, None

    def cancel(self, coin, oid):
        self.orders.pop(oid, None)
        return None

    def cancel_all(self, coins):
        for oid in [o for o, v in self.orders.items() if v["coin"] in coins]:
            del self.orders[oid]

    def open_orders(self):
        return {oid: o["sz"] for oid, o in self.orders.items()}

    def poll(self):
        """Fill the resting orders that the book has traded through. The real book does not contain our paper
        orders, so there is no self-quote to strip here."""
        for oid, o in list(self.orders.items()):
            bids, asks = self.info.book(o["coin"])
            if (o["is_buy"] and asks and asks[0][0] <= o["px"]) or (not o["is_buy"] and bids and bids[0][0] >= o["px"]):
                self._fill(o["coin"], o["sz"] if o["is_buy"] else -o["sz"], o["px"], maker=True)
                del self.orders[oid]

    def ioc(self, coin, is_buy, sz, cap, reduce_only):
        bids, asks = self.info.book(coin)
        filled = notional = 0.0
        for px, qty in (asks if is_buy else bids):
            if (is_buy and px > cap) or (not is_buy and px < cap):
                break
            take = min(qty, sz - filled)
            filled += take
            notional += take * px
            if filled >= sz - 1e-12:
                break
        if filled:
            self._fill(coin, filled if is_buy else -filled, notional / filled, maker=False)
        return filled

    def fills_since(self, t0):
        return [f for f in self.fills if f["time"] >= t0]


class LiveBroker:
    """Real orders through Hyperliquid's official Python SDK, signed by an API wallet that can trade but cannot
    withdraw. Positions and fills are read from the account's public address, and at start-up the bot checks that
    the API wallet really trades for that account (a subaccount is addressed as vault)."""

    quotes_in_book = True    # our resting orders are part of the l2Book we read

    def __init__(self, info, address, key):
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        agent = Account.from_key(key)
        self.info, self.address = info, address
        vault = self._vault_for(info, agent.address, address)
        self.ex = Exchange(agent, info.base_url, account_address=address, vault_address=vault, timeout=10)
        self.leverage_set = set()

    @staticmethod
    def _vault_for(info, agent, address):
        if agent.lower() == address.lower():
            return None
        role = info.post({"type": "userRole", "user": agent})
        master = ((role or {}).get("data") or {}).get("user", "") if (role or {}).get("role") == "agent" else ""
        if not master:
            raise FatalError("Diese API-Wallet ist bei keinem Konto freigeschaltet (Hyperliquid → API).")
        if master.lower() == address.lower():
            return None
        sub = info.post({"type": "userRole", "user": address})
        if (sub or {}).get("role") == "subAccount" and ((sub.get("data") or {}).get("master", "")).lower() == master.lower():
            return address
        raise FatalError(f"Die API-Wallet handelt für {master}, der Bot liest aber {address}. "
                         "account.address muss dieses Konto oder eines seiner Subkonten sein.")

    def save(self):
        pass

    def account(self, markets):
        return self.info.account(self.address)

    def positions(self):
        return self.info.account(self.address)["positions"]

    def accrue_funding(self, markets):
        pass

    def set_leverage(self, coin, lev):
        if (coin, lev) in self.leverage_set:
            return None
        r = self.ex.update_leverage(int(lev), coin, is_cross=True)
        if isinstance(r, dict) and r.get("status") == "ok":
            self.leverage_set.add((coin, lev))
            return None
        return str(r)

    def book(self, coin):
        return self.info.book(coin)

    @staticmethod
    def _status(resp):
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            return None, str(resp)
        statuses = ((resp.get("response") or {}).get("data") or {}).get("statuses") or [{}]
        st = statuses[0]
        if isinstance(st, dict) and "error" in st:
            return None, st["error"]
        if isinstance(st, dict) and "resting" in st:
            return st["resting"]["oid"], None
        if isinstance(st, dict) and "filled" in st:
            return st["filled"].get("oid"), None
        return None, None

    def post_only(self, coin, is_buy, sz, px, reduce_only):
        return self._status(self.ex.order(coin, is_buy, sz, px, {"limit": {"tif": "Alo"}}, reduce_only=reduce_only))

    def modify(self, oid, coin, is_buy, sz, px, reduce_only):
        new_oid, err = self._status(self.ex.modify_order(oid, coin, is_buy, sz, px, {"limit": {"tif": "Alo"}},
                                                         reduce_only=reduce_only))
        return (new_oid or oid, None) if not err else (None, err)

    def cancel(self, coin, oid):
        r = self.ex.cancel(coin, oid)
        return None if isinstance(r, dict) and r.get("status") == "ok" else str(r)

    def cancel_all(self, coins):
        for o in self.info.open_orders(self.address):
            if o["coin"] in coins:
                self.cancel(o["coin"], o["oid"])

    def open_orders(self):
        return {o["oid"]: float(o["sz"]) for o in self.info.open_orders(self.address)}

    def poll(self):
        pass

    def ioc(self, coin, is_buy, sz, cap, reduce_only):
        resp = self.ex.order(coin, is_buy, sz, cap, {"limit": {"tif": "Ioc"}}, reduce_only=reduce_only)
        try:
            return float(resp["response"]["data"]["statuses"][0]["filled"]["totalSz"])
        except (KeyError, IndexError, TypeError):
            return 0.0

    def fills_since(self, t0):
        rows = self.info.post({"type": "userFillsByTime", "user": self.address, "startTime": int(t0 * 1000)})
        return [{"coin": f["coin"], "sz": float(f["sz"]) * (1 if f["side"] == "B" else -1), "px": float(f["px"]),
                 "fee": float(f["fee"]), "crossed": bool(f.get("crossed")), "time": f["time"] / 1000} for f in rows]


def execute(broker, trades, markets, betas, cfg, book_usd, say):
    """Carry out the rebalancing trades (USD per coin, positive = buy). Returns a summary of the fills."""
    ex = cfg["execution"]
    t_start = time.time()
    broker.cancel_all(set(trades))           # nothing left over from an earlier run may fill on top
    start = broker.positions()
    want = {}
    for coin, usd in trades.items():
        q = round_size(usd / markets[coin]["mid"], markets[coin]["szDec"])
        if q:
            want[coin] = q
    reduce_only = {c: bool(start.get(c)) and (start[c] > 0) != (q > 0) and abs(q) <= abs(start[c]) + 1e-12
                   for c, q in want.items()}
    tol = ex["beta_tolerance"] * book_usd

    def remaining():
        """What is still to trade, from a fresh read of the positions."""
        broker.poll()
        pos = broker.positions()
        out = {}
        for c, q in want.items():
            r = round_size(q - (pos.get(c, 0.0) - start.get(c, 0.0)), markets[c]["szDec"])
            if (r > 0) == (q > 0) and abs(r) * markets[c]["mid"] >= MIN_NOTIONAL:
                out[c] = r
        return out

    def go_taker(c, r):
        """IOC with a price cap; stops the bot if the fill does not show up on the account it reads."""
        bids, asks = broker.book(c)
        if not bids or not asks:
            return 0.0
        ref = asks[0][0] if r > 0 else bids[0][0]
        cap = round_price(ref * (1 + ex["max_slippage"]) if r > 0 else ref * (1 - ex["max_slippage"]),
                          markets[c]["szDec"])
        before = broker.positions().get(c, 0.0)
        filled = broker.ioc(c, r > 0, abs(r), cap, reduce_only[c])
        if filled:
            moved = broker.positions().get(c, 0.0) - before
            if abs(moved - math.copysign(filled, r)) > max(1e-9, 0.01 * filled):
                raise FatalError(f"{c}: {filled} gefüllt, auf dem gelesenen Konto aber {moved:+} angekommen. "
                                 "Die Orders landen auf einem anderen Konto – Bot gestoppt.")
        return filled

    working = {}
    try:
        deadline = t_start + ex["maker_wait_seconds"]
        while True:
            rem = remaining()
            resting = broker.open_orders()
            for c in list(working):
                if working[c]["oid"] not in resting:      # filled, or gone for another reason
                    working.pop(c)
                elif c not in rem:
                    broker.cancel(c, working.pop(c)["oid"])
            if not rem or time.time() >= deadline:
                break
            # Keep both sides in step: if fills so far left a net beta exposure beyond the tolerance, complete just
            # enough of the lagging side at once to bring it back to half the tolerance, biggest beta first.
            imbalance = sum(betas.get(c, 1.0) * (want[c] - rem.get(c, 0.0)) * markets[c]["mid"] for c in want)
            if abs(imbalance) > tol:
                sign = 1.0 if imbalance > 0 else -1.0
                lagging = sorted((c for c, r in rem.items() if betas.get(c, 1.0) * r * sign < 0),
                                 key=lambda c: -abs(betas.get(c, 1.0) * rem[c] * markets[c]["mid"]))
                for c in lagging:
                    excess = imbalance * sign - tol / 2
                    if excess <= 0:
                        break
                    if c in working:
                        broker.cancel(c, working.pop(c)["oid"])
                    r = remaining().get(c, 0.0)          # fresh, after the cancel
                    beta_usd = abs(betas.get(c, 1.0)) * markets[c]["mid"]
                    q = round_size(min(abs(r), excess / beta_usd if beta_usd else abs(r)), markets[c]["szDec"])
                    if not r or q * markets[c]["mid"] < MIN_NOTIONAL:
                        continue
                    say(f"  {c}: Gegenseite ist schon weiter gefüllt, {q} sofort als Taker")
                    filled = go_taker(c, math.copysign(q, r))
                    imbalance += betas.get(c, 1.0) * math.copysign(filled, r) * markets[c]["mid"]
                rem = remaining()
                resting = broker.open_orders()
            for c, r in rem.items():
                bids, asks = broker.book(c)
                w = working.get(c)
                if w and w["oid"] in resting and broker.quotes_in_book:   # quote against the others, not ourselves
                    if r > 0:
                        bids = without_own(bids, w["px"], resting[w["oid"]])
                    else:
                        asks = without_own(asks, w["px"], resting[w["oid"]])
                if not bids or not asks:
                    continue
                px = maker_price(r > 0, bids, asks, markets[c]["szDec"])
                if w and w["px"] == px and w["sz"] == abs(r):
                    continue
                if w:
                    oid, err = broker.modify(w["oid"], c, r > 0, abs(r), px, reduce_only[c])
                    if err:
                        working.pop(c, None)
                    else:
                        working[c] = {"oid": oid, "px": px, "sz": abs(r)}
                else:
                    oid, err = broker.post_only(c, r > 0, abs(r), px, reduce_only[c])
                    if oid is not None:
                        working[c] = {"oid": oid, "px": px, "sz": abs(r)}
            time.sleep(ex["requote_seconds"])
    finally:
        for c, w in working.items():
            broker.cancel(c, w["oid"])
    for c, r in remaining().items():
        go_taker(c, r)
    left = remaining()
    fills = broker.fills_since(t_start - 1)
    return {"fills": fills, "left": left,
            "maker": sum(1 for f in fills if not f["crossed"]), "taker": sum(1 for f in fills if f["crossed"]),
            "fees": sum(f["fee"] for f in fills), "seconds": time.time() - t_start}
