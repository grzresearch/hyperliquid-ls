"""Small stdlib client for the public Hyperliquid info API."""
import json
import math
import ssl
import time
import urllib.request

MAINNET = "https://api.hyperliquid.xyz"
TESTNET = "https://api.hyperliquid-testnet.xyz"
HOUR = 3600000
DAY = 24 * HOUR
STEP = 8 * HOUR


class Info:
    def __init__(self, base_url=MAINNET, timeout=30):
        self.base_url = base_url
        self.url = base_url + "/info"
        self.timeout = timeout
        self.ctx = ssl.create_default_context()

    def post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1) if getattr(e, "code", 0) == 429 else 1 + 2 * attempt)

    def markets(self):
        """Perp markets by coin: mark and mid price, hourly funding, max leverage, size decimals."""
        meta, ctxs = self.post({"type": "metaAndAssetCtxs"})
        out = {}
        for u, c in zip(meta["universe"], ctxs):
            if u.get("isDelisted"):
                continue
            mark = float(c["markPx"])
            out[u["name"]] = {"mark": mark, "mid": float(c.get("midPx") or mark), "funding": float(c["funding"]),
                              "maxLev": int(u["maxLeverage"]), "szDec": int(u["szDecimals"])}
        return out

    def book(self, coin):
        """Best levels of the order book: (bids high to low, asks low to high) as (price, size)."""
        levels = self.post({"type": "l2Book", "coin": coin})["levels"]
        return ([(float(x["px"]), float(x["sz"])) for x in levels[0]],
                [(float(x["px"]), float(x["sz"])) for x in levels[1]])

    def account(self, address):
        st = self.post({"type": "clearinghouseState", "user": address})
        positions = {}
        for p in st.get("assetPositions", []):
            szi = float(p["position"]["szi"])
            if szi:
                positions[p["position"]["coin"]] = szi
        ms = st["marginSummary"]
        return {"equity": float(ms["accountValue"]), "positions": positions,
                "marginUsed": float(ms["totalMarginUsed"]),
                "maintenance": float(st.get("crossMaintenanceMarginUsed") or 0)}

    def open_orders(self, address):
        return self.post({"type": "openOrders", "user": address})

    def closes(self, coins, days):
        """8h closes of completed candles on a common time grid, shaped like the dashboard's market data."""
        now = int(time.time() * 1000)
        n = days * 3
        t0 = now // STEP * STEP - n * STEP
        out = {}
        for c in coins:
            candles = self.post({"type": "candleSnapshot",
                                 "req": {"coin": c, "interval": "8h", "startTime": t0, "endTime": now}})
            by = {k["t"]: float(k["c"]) for k in candles if k["T"] < now}
            out[c] = {"closes": [by.get(t0 + i * STEP) for i in range(n)]}
            time.sleep(0.2)
        return {"perDay": 3, "t0": t0, "step": STEP, "coins": out}


# ---------- Hyperliquid price and size rules ----------
def round_size(sz, sz_dec):
    return round(sz, sz_dec)


def round_price(px, sz_dec):
    """At most 5 significant figures and 6 - szDecimals decimals (perps)."""
    return round(float(f"{px:.5g}"), 6 - sz_dec)


def tick(px, sz_dec):
    """Smallest price step that the rules above allow at this price."""
    digits = math.floor(math.log10(px)) + 1
    return 10.0 ** -min(6 - sz_dec, max(0, 5 - digits))
