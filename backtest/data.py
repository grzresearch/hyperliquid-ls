"""Historical data for the backtest, cached on disk so a rerun costs nothing.

  Hyperliquid  daily candles of every perp ever listed (delisted ones included, so the test does not only see the
               survivors), and the week of funding before each date for the tokens in the universe
  DefiLlama    daily fees, revenue, holder revenue and DEX volume per protocol
  CoinGecko    daily market caps of the last 365 days (the free API reaches no further, and it throttles: where it
               gave no answer, the market cap is today's DefiLlama figure scaled with the Hyperliquid price)
  Snapshot     passed proposals of the protocols' spaces
  0xArchive    only with a plan that reaches back further than 30 days (Build and above); the free tier cannot
               serve a backtest, so the liquidity filter uses 30-day average volume instead of order books

GitHub releases are not fetched here: they need an API token, and the live screener's catalyst score uses them only
as one of two sources. The backtest measures catalysts with the Snapshot part.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from bot import screener

DAY = 86400000
HL = "https://api.hyperliquid.xyz/info"
LLAMA = "https://api.llama.fi"
GECKO = "https://api.coingecko.com/api/v3"
ARCHIVE = "https://api.0xarchive.io"


class Cache:
    """JSON files under one folder, one per request. A missing answer (404) is cached as null."""

    def __init__(self, folder, offline=False):
        self.folder, self.offline = folder, offline      # offline: a missing answer stays missing (None)
        os.makedirs(folder, exist_ok=True)

    def path(self, *parts):
        name = "_".join(str(p) for p in parts).replace("/", "-").replace("#", "-").replace(" ", "-")
        return os.path.join(self.folder, name + ".json")

    def get(self, parts, fetch):
        p = self.path(*parts)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        if self.offline:
            return None
        data = fetch()
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, p)
        return data


def http_json(url, body=None, headers=None, tries=6, pause=0.0):
    data = json.dumps(body).encode() if body is not None else None
    hdr = {"User-Agent": "hyperliquid-ls-backtest", **({"Content-Type": "application/json"} if data else {}),
           **(headers or {})}
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=hdr)
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            if pause:
                time.sleep(pause)
            return out
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                return None
            if e.code == 429:
                wait = float(e.headers.get("Retry-After") or 0) or 20 * (attempt + 1)
                time.sleep(min(wait, 120))
                continue
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"no answer from {url}")


class Data:
    def __init__(self, folder, start_ms, end_ms, say=print, offline=False):
        self.c = Cache(folder, offline)
        self.start, self.end, self.say = start_ms, end_ms, say
        self.archive_key = os.environ.get("OXARCHIVE_API_KEY", "").strip()

    # ---------- Hyperliquid ----------
    def universe(self):
        meta = self.c.get(["hl", "meta"], lambda: http_json(HL, {"type": "meta"}))
        return [{"coin": u["name"], "delisted": bool(u.get("isDelisted")), "maxLev": u.get("maxLeverage", 0)}
                for u in meta["universe"]]

    def daily(self, coin):
        """[(open time ms, open, high, low, close, notional volume)] of daily candles."""
        def fetch():
            rows = http_json(HL, {"type": "candleSnapshot", "req": {"coin": coin, "interval": "1d",
                                                                    "startTime": self.start, "endTime": self.end}},
                             pause=0.15) or []
            return [[k["t"], float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"]) * float(k["c"])]
                    for k in rows]
        return self.c.get(["hl", "1d", coin, self.end // DAY], fetch)

    def funding_week(self, coin, t):
        """Hourly funding rates [(time ms, rate)] of the 7 days before midnight t. Fetched per date and coin: the
        whole history would cost Hyperliquid's rate limit hours (fundingHistory weighs 20 plus 1 per 20 rows)."""
        def fetch():
            page = http_json(HL, {"type": "fundingHistory", "coin": coin, "startTime": t - 7 * DAY,
                                  "endTime": t - 1}, pause=1.6) or []
            return [[r["time"], float(r["fundingRate"])] for r in page]
        return self.c.get(["hl", "fundweek", coin, t // DAY], fetch)

    # ---------- DefiLlama ----------
    def entities(self):
        protocols = self.c.get(["llama", "protocols"], lambda: http_json(f"{LLAMA}/protocols"))
        parents = self.c.get(["llama", "parents"],
                             lambda: http_json(f"{LLAMA}/lite/protocols2?b=2").get("parentProtocols", []))
        return screener.llama_entities(protocols, parents)

    def llama_series(self, slug, kind):
        """Daily values [(day ms, value)] of fees, revenue, holders (holder revenue) or dexs (DEX volume)."""
        path = {"fees": f"/summary/fees/{slug}?dataType=dailyFees",
                "revenue": f"/summary/fees/{slug}?dataType=dailyRevenue",
                "holders": f"/summary/fees/{slug}?dataType=dailyHoldersRevenue",
                "dexs": f"/summary/dexs/{slug}"}[kind]

        def fetch():
            d = http_json(LLAMA + path, pause=0.2)
            if not d:
                return None
            return [[int(t) * 1000, float(v)] for t, v in d.get("totalDataChart") or [] if v is not None]
        return self.c.get(["llama", kind, slug], fetch)

    # ---------- CoinGecko ----------
    def market_caps(self, gecko_id, fetch_missing=True):
        """Daily [(day ms, market cap, price)] of the last 365 days. fetch_missing=False: cached answers only (the
        public CoinGecko API throttles hard; the backtest falls back to price times today's supply)."""
        def fetch():
            d = http_json(f"{GECKO}/coins/{urllib.parse.quote(gecko_id)}/market_chart?vs_currency=usd&days=365&"
                          "interval=daily", pause=4.0)
            if not d:
                return None
            px = {int(t) // DAY * DAY: p for t, p in d.get("prices", [])}
            return [[int(t) // DAY * DAY, float(m), px.get(int(t) // DAY * DAY)] for t, m in d.get("market_caps", [])
                    if m]
        if not fetch_missing and not os.path.exists(self.c.path("gecko", gecko_id, self.end // DAY)):
            return None
        return self.c.get(["gecko", gecko_id, self.end // DAY], fetch)

    # ---------- Snapshot ----------
    def votes(self, spaces):
        """Passed proposals per space over the whole period, with the live screener's weights."""
        def fetch():
            out, skip = {}, 0
            q = ("query($s: [String], $a: Int, $b: Int, $k: Int) { proposals(first: 1000, skip: $k, where: "
                 "{space_in: $s, state: \"closed\", end_gte: $a, end_lte: $b}, orderBy: \"end\", orderDirection: asc)"
                 " { title end choices scores space { id } } }")
            while True:
                res = http_json("https://hub.snapshot.org/graphql",
                                {"query": q, "variables": {"s": spaces, "a": self.start // 1000 - 60 * 86400,
                                                           "b": self.end // 1000, "k": skip}}, pause=1.0)
                props = ((res or {}).get("data") or {}).get("proposals") or []
                for p in props:
                    scores, choices = p.get("scores") or [], p.get("choices") or []
                    if not scores or not choices or scores[0] <= max(scores[1:] or [0]):
                        continue
                    if not choices[0].strip().lower().startswith(screener.AFFIRMATIVE):
                        continue
                    title = p.get("title") or ""
                    w = 0.6 if any(k in (" " + title.lower()) for k in screener.UPGRADE_WORDS) else 0.2
                    out.setdefault(p["space"]["id"], []).append({"t": p["end"] * 1000, "title": title, "weight": w})
                if len(props) < 1000 or skip >= 5000:
                    break
                skip += 1000
            return out
        return self.c.get(["snapshot", "votes", len(spaces), self.end // DAY], fetch)

    # ---------- 0xArchive ----------
    def archive(self, path, params=None):
        if not self.archive_key:
            return None
        url = ARCHIVE + path + ("?" + urllib.parse.urlencode(params) if params else "")
        key = ["archive", path.strip("/").replace("/", "-")] + [f"{k}-{v}" for k, v in sorted((params or {}).items())]
        return self.c.get(key, lambda: http_json(url, headers={"X-API-Key": self.archive_key}, pause=0.1))


def collect(folder, start_ms, end_ms, min_mcap_now=5e6, say=print, gecko=True):
    """Download everything the backtest needs. Returns the Data object; all answers stay in the cache folder."""
    d = Data(folder, start_ms, end_ms, say)
    uni = d.universe()
    say(f"Hyperliquid: {len(uni)} Perps, davon {sum(u['delisted'] for u in uni)} delistet")
    for i, u in enumerate(uni):
        d.daily(u["coin"])
        if i % 40 == 39:
            say(f"  Tageskerzen {i + 1}/{len(uni)}")
    ent, _ = d.entities()
    coins = [u["coin"] for u in uni if u["coin"] not in ("BTC", "ETH", "HYPE")]
    mapped = screener.map_coins(coins, ent, min_mcap_now)
    say(f"DefiLlama: {len(mapped)} Perps einem Protokoll zugeordnet")
    for i, (coin, e) in enumerate(sorted(mapped.items())):
        if e.get("slug"):
            for kind in ("fees", "revenue", "holders", "dexs"):
                d.llama_series(e["slug"], kind)
        if e.get("gecko") and gecko:
            d.market_caps(e["gecko"])
        if i % 10 == 9:
            say(f"  Protokolle {i + 1}/{len(mapped)}")
    spaces = sorted({g.split(":", 1)[1] for e in mapped.values() for g in e["gov"] if g.startswith("snapshot:")}
                    | set(screener.SPACE_ALIASES.values()))
    d.votes(spaces)
    say("Fertig: alle Daten im Cache")
    return d
