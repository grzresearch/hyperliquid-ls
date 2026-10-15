"""Long screener: which Hyperliquid perps are worth holding long, by protocol cash flows, momentum and catalysts.

A candidate is a token with a liquid perp on Hyperliquid that maps to a DefiLlama protocol with fee data. Factors
are turned into cross-sectional normal scores (rank based, so one outlier cannot dominate) and summed with weights:

  fee_growth      fees of the last 30 days over the 30 days before (DefiLlama)
  revenue_growth  the same for revenue
  volume_growth   the same for DEX volume where DefiLlama reports it for free (perp volume is a paid endpoint there;
                  for perp DEXs the fee growth carries that information)
  acceleration    fees of the last 7 days over the 7 days before
  fee_yield       annualised fees over market cap: how much cash flow the valuation buys
  holder_yield    annualised holder revenue (buybacks, distributions) over market cap
  momentum        30- and 90-day return relative to ETH, beta-adjusted (Hyperliquid daily candles)
  catalysts       recent releases on the protocol's GitHub organisations and passed Snapshot votes, fading with a
                  two-week half-life; an upgrade or a governance decision moves a token for days to weeks, not months
  funding         minus the 7-day average funding rate: crowded longs pay to be long
  liquidity       minus the implementation shortfall of a test order on the live book (bot/costs.py)

The selection keeps a held long while it ranks within n_longs + buffer, so a small reshuffle does not trade. Weights
lean on the score and on low volatility, capped so no name carries the book.
"""
import calendar
import json
import math
import os
import ssl
import statistics
import time
import urllib.request

from . import costs

DAY_MS = 86400000
LLAMA = "https://api.llama.fi"
FACTORS = ["fee_growth", "revenue_growth", "volume_growth", "acceleration", "fee_yield", "holder_yield",
           "momentum", "catalysts", "funding", "liquidity"]
SPACE_ALIASES = {"aave.eth": "aavedao.eth"}   # spaces that moved since DefiLlama recorded them
UPGRADE_WORDS = ("upgrade", "launch", "deploy", "activate", "migration", "fee switch", "buyback", "tokenomics",
                 " v2", " v3", " v4", " v5", "relayer", "mainnet", "integration")
AFFIRMATIVE = ("for", "yes", "yae", "approve", "aye", "in favor", "accept")


# ---------- small HTTP helpers (standard library only) ----------
_CTX = ssl.create_default_context()


def fetch_json(url, body=None, headers=None, timeout=90, tries=4):
    data = json.dumps(body).encode() if body is not None else None
    hdr = {"User-Agent": "hyperliquid-ls-screener", **({"Content-Type": "application/json"} if data else {}),
           **(headers or {})}
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=hdr)
            with urllib.request.urlopen(req, context=_CTX, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            code = getattr(e, "code", 0)
            if attempt == tries - 1 or code in (401, 403, 404, 422):
                raise
            time.sleep(3 * (attempt + 1))


# ---------- DefiLlama ----------
def llama_entities(protocols, parents):
    """Entities a token can belong to: parent protocols (Aerodrome, Pendle …) and protocols without a parent.

    protocols: the /protocols list; parents: parentProtocols from /lite/protocols2. Returns {entity id: entity} and
    {protocol id: entity id}."""
    ent, owner = {}, {}
    for p in parents:
        ent[p["id"]] = {"id": p["id"], "name": p["name"], "symbol": (p.get("symbol") or "").upper(),
                        "gecko": p.get("gecko_id"), "mcap": p.get("mcap") or 0.0, "github": p.get("github") or [],
                        "gov": p.get("governanceID") or []}
    for p in protocols:
        pid = str(p["id"])
        parent = p.get("parentProtocol")
        if parent and parent in ent:
            owner[pid] = parent
            continue
        owner[pid] = pid
        ent[pid] = {"id": pid, "name": p["name"], "symbol": (p.get("symbol") or "").upper(),
                    "gecko": p.get("gecko_id"), "mcap": p.get("mcap") or 0.0, "github": p.get("github") or [],
                    "gov": p.get("governanceID") or []}
    return ent, owner


def aggregate(overview, owner):
    """Sum a DefiLlama overview (fees, revenue, volume) per entity: 30 days, the 30 before, 7 days, the 7 before."""
    out = {}
    for e in overview.get("protocols", []):
        eid = e.get("parentProtocol") or owner.get(str(e.get("defillamaId") or e.get("id")))
        if not eid:
            continue
        a = out.setdefault(eid, {"d30": 0.0})
        d30, p30, d7, p7 = (e.get(k) for k in ("total30d", "total60dto30d", "total7d", "total14dto7d"))
        if d30 is not None:
            a["d30"] += d30
        if d30 is not None and p30 is not None:   # growth only from protocols that report both windows
            a["g30"], a["h30"] = a.get("g30", 0.0) + d30, a.get("h30", 0.0) + p30
        if d7 is not None and p7 is not None:
            a["g7"], a["h7"] = a.get("g7", 0.0) + d7, a.get("h7", 0.0) + p7
    return out


def growth(a, short=False):
    """Log growth of the last window over the one before, or None."""
    if not a:
        return None
    cur, prev = (a.get("g7"), a.get("h7")) if short else (a.get("g30"), a.get("h30"))
    if not cur or not prev or cur <= 0 or prev <= 0:
        return None
    return max(-2.0, min(2.0, math.log(cur / prev)))


def map_coins(coins, entities, min_mcap):
    """Hyperliquid coin -> entity by token symbol, the largest market cap winning (LIT is Lighter, not Timeless)."""
    by_symbol = {}
    for e in entities.values():
        if e["symbol"] and e["mcap"] >= min_mcap:
            best = by_symbol.get(e["symbol"])
            if best is None or e["mcap"] > best["mcap"]:
                by_symbol[e["symbol"]] = e
    return {c: by_symbol[c.upper()] for c in coins if c.upper() in by_symbol}


# ---------- catalysts ----------
def semver(tag):
    digits = []
    for part in tag.lstrip("vV").replace("-", ".").split("."):
        if part.isdigit():
            digits.append(int(part))
        else:
            break
    return tuple(digits[:3]) if digits else None


def release_events(releases, now_ms, horizon_days=60, first_known=False):
    """Score releases of one repository: a major version 1.0, a minor 0.5, a patch 0.15, older ones fade.
    first_known: the list holds the whole history, so its oldest entry is the repository's first release."""
    rows = []
    for r in releases:
        if r.get("draft") or r.get("prerelease") or not r.get("published_at"):
            continue
        t = calendar.timegm(time.strptime(r["published_at"][:19], "%Y-%m-%dT%H:%M:%S")) * 1000
        rows.append((t, r.get("tag_name") or "", r.get("html_url") or ""))
    rows.sort()
    out = []
    for i, (t, tag, url) in enumerate(rows):
        age = (now_ms - t) / DAY_MS
        if (i == 0 and not first_known) or age > horizon_days:   # no predecessor to compare with
            continue
        cur, prev = semver(tag), semver(rows[i - 1][1]) if i else None
        kind, w = "Patch", 0.15
        if cur and (not prev or cur[0] > prev[0]):
            kind, w = "Major", 1.0
        elif cur and prev and len(cur) > 1 and len(prev) > 1 and cur[:2] > prev[:2]:
            kind, w = "Minor", 0.5
        out.append({"t": t, "kind": kind, "tag": tag, "url": url, "weight": w})
    return out


def github_events(orgs, token, now_ms, max_repos=6):
    """Recent releases across the most recently pushed repositories of the protocol's GitHub organisations."""
    headers = {"Accept": "application/vnd.github+json", **({"Authorization": f"Bearer {token}"} if token else {})}
    def get(url):
        """None when this one URL fails; a refused or rate-limited token (401, 403) ends GitHub for the run."""
        try:
            return fetch_json(url, headers=headers, tries=2)
        except Exception as e:
            if getattr(e, "code", 0) in (401, 403):
                raise
            return None

    events = []
    for org in orgs[:2]:
        repos = (get(f"https://api.github.com/orgs/{org}/repos?sort=pushed&per_page={max_repos}")
                 or get(f"https://api.github.com/users/{org}/repos?sort=pushed&per_page={max_repos}") or [])
        for repo in repos[:max_repos]:
            if repo.get("archived") or repo.get("fork"):
                continue
            rel = get(f"https://api.github.com/repos/{repo['full_name']}/releases?per_page=7") or []
            for ev in release_events(rel, now_ms, first_known=len(rel) < 7):
                ev["repo"] = repo["name"]
                events.append(ev)
    return events


def snapshot_events(spaces, now_ms, horizon_days=45):
    """Passed Snapshot proposals of the given spaces: affirmative first choice ahead. Upgrade-type titles weigh more."""
    if not spaces:
        return {}
    q = ("query($s: [String], $t: Int) { proposals(first: 400, where: {space_in: $s, state: \"closed\", end_gte: $t},"
         " orderBy: \"end\", orderDirection: desc) { title end choices scores space { id } } }")
    res = fetch_json("https://hub.snapshot.org/graphql",
                     body={"query": q, "variables": {"s": spaces, "t": int(now_ms / 1000 - horizon_days * 86400)}})
    out = {}
    for p in (res.get("data") or {}).get("proposals") or []:
        scores, choices = p.get("scores") or [], p.get("choices") or []
        if not scores or not choices or scores[0] <= max(scores[1:] or [0]):
            continue
        if not choices[0].strip().lower().startswith(AFFIRMATIVE):
            continue
        title = p.get("title") or ""
        w = 0.6 if any(k in (" " + title.lower()) for k in UPGRADE_WORDS) else 0.2
        out.setdefault(p["space"]["id"], []).append({"t": p["end"] * 1000, "title": title, "weight": w})
    return out


def catalyst_score(events, now_ms, half_life_days=14.0):
    s = sum(e["weight"] * 0.5 ** ((now_ms - e["t"]) / DAY_MS / half_life_days) for e in events)
    return min(3.0, s)


# ---------- market factors ----------
def relative_momentum(closes, eth, days):
    """Log return over `days` minus beta times ETH's, beta from the daily returns in between."""
    if len(closes) <= days or len(eth) <= days:
        return None
    a, b = closes[-days - 1:], eth[-days - 1:]
    ra = [math.log(a[i + 1] / a[i]) for i in range(days)]
    rb = [math.log(b[i + 1] / b[i]) for i in range(days)]
    mb = sum(rb) / days
    vb = sum((x - mb) ** 2 for x in rb)
    bt = sum((x - sum(ra) / days) * (y - mb) for x, y in zip(ra, rb)) / vb if vb else 1.0
    return math.log(a[-1] / a[0]) - bt * math.log(b[-1] / b[0])


def daily_vol(closes, days=90):
    r = [math.log(closes[i + 1] / closes[i]) for i in range(max(0, len(closes) - days - 1), len(closes) - 1)]
    return statistics.stdev(r) if len(r) > 2 else None


# ---------- scoring and selection ----------
def normal_scores(values):
    """Rank-based normal scores of the non-missing values (ties share their average rank)."""
    items = sorted((v, k) for k, v in values.items() if v is not None)
    n = len(items)
    out, i = {}, 0
    nd = statistics.NormalDist()
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][0] == items[i][0]:
            j += 1
        z = nd.inv_cdf(((i + j) / 2 + 0.5) / n) if n > 1 else 0.0
        for k in range(i, j + 1):
            out[items[k][1]] = z
        i = j + 1
    return out


def score(cands, weights):
    """Composite score per coin from the factor values in cands[coin]["factors"]; missing factors count as 0."""
    z = {f: normal_scores({c: d["factors"].get(f) for c, d in cands.items()}) for f in FACTORS}
    total = sum(weights.get(f, 0.0) for f in FACTORS) or 1.0
    for c, d in cands.items():
        d["z"] = {f: z[f].get(c) for f in FACTORS}
        d["score"] = sum(weights.get(f, 0.0) * (z[f].get(c) or 0.0) for f in FACTORS) / total
    return cands


MIN_LONGS = 3


def select(cands, held, cfg):
    """Picks: held names that still rank within n + buffer, then the best new names with a positive score, and if that
    leaves fewer than MIN_LONGS, the next best names."""
    n, buffer, floor = int(cfg["n_longs"]), int(cfg["buffer"]), float(cfg["hold_floor"])
    ranked = sorted((c for c, d in cands.items() if d["eligible"]), key=lambda c: -cands[c]["score"])
    rank = {c: i + 1 for i, c in enumerate(ranked)}
    picks = [c for c in ranked if c in held and rank[c] <= n + buffer and cands[c]["score"] > floor][:n]
    for c in ranked:
        if len(picks) >= n:
            break
        if c not in picks and cands[c]["score"] > 0:
            picks.append(c)
    for c in ranked:                     # never fewer than MIN_LONGS: a book of one or two longs is a bet, not a basket
        if len(picks) >= min(MIN_LONGS, n):
            break
        if c not in picks:
            picks.append(c)
    for c, d in cands.items():
        d["rank"] = rank.get(c)
    return picks


def weights_for(picks, cands, cfg):
    """Score over daily volatility, capped at max_weight and floored at min_weight, summing to 1. Names that hit a
    bound are fixed there and the rest share what is left, in proportion, until nothing is out of bounds."""
    if not picks:
        return {}
    raw = {c: max(cands[c]["score"], 0.1) / max(cands[c].get("vol") or 0.05, 0.01) for c in picks}
    lo = min(float(cfg["min_weight"]), 1 / len(picks))
    hi = max(float(cfg["max_weight"]), 1 / len(picks))
    fixed = {}
    for _ in range(len(raw) + 1):
        free = [c for c in raw if c not in fixed]
        if not free:
            break
        mass = 1 - sum(fixed.values())
        share = {c: mass * raw[c] / sum(raw[x] for x in free) for c in free}
        over = [c for c, v in share.items() if v > hi + 1e-12]
        under = [c for c, v in share.items() if v < lo - 1e-12]
        if not over and not under:
            fixed.update(share)
            break
        for c in over or under:
            fixed[c] = hi if over else lo
    total = sum(fixed.values())
    return {c: fixed[c] / total for c in picks}


# ---------- the whole run ----------
def run(info, cfg, held=(), now_ms=None, say=print, token=None):
    """One screener pass. cfg: the [screener] table plus "exclude" (coins never held long). Returns the result
    the bot stores and the page shows."""
    now_ms = now_ms or int(time.time() * 1000)
    token = token if token is not None else os.environ.get("GITHUB_TOKEN", "")
    markets = info.markets()
    held = set(held)
    pool = [c for c, m in markets.items() if c not in cfg["exclude"] and m["maxLev"] >= cfg["min_leverage"]
            and (m["volume"] >= cfg["min_volume_usd"] or c in held)]
    protocols = fetch_json(f"{LLAMA}/protocols")
    parents = fetch_json(f"{LLAMA}/lite/protocols2?b=2").get("parentProtocols", [])
    entities, owner = llama_entities(protocols, parents)
    q = "excludeTotalDataChart=true&excludeTotalDataChartBreakdown=true"
    fees = aggregate(fetch_json(f"{LLAMA}/overview/fees?{q}&dataType=dailyFees"), owner)
    revenue = aggregate(fetch_json(f"{LLAMA}/overview/fees?{q}&dataType=dailyRevenue"), owner)
    holders = aggregate(fetch_json(f"{LLAMA}/overview/fees?{q}&dataType=dailyHoldersRevenue"), owner)
    volume = aggregate(fetch_json(f"{LLAMA}/overview/dexs?{q}"), owner)
    mapped = {c: e for c, e in map_coins(pool, entities, cfg["min_mcap_usd"]).items()
              if fees.get(e["id"], {}).get("d30", 0) > 0}
    say(f"Screener: {len(pool)} liquide Perps, {len(mapped)} mit Protokoll-Cashflows auf DefiLlama")
    eth = [k[4] for k in info.candles("ETH", "1d", now_ms - 100 * DAY_MS, now_ms)]
    sources = {"defillama": True, "github": bool(token), "snapshot": True}
    def spaces_of(e):
        return [SPACE_ALIASES.get(g.split(":", 1)[1], g.split(":", 1)[1]) for g in e["gov"] if g.startswith("snapshot:")]
    spaces = sorted({sp for e in mapped.values() for sp in spaces_of(e)})
    try:
        votes = snapshot_events(spaces, now_ms)
    except Exception as e:
        votes, sources["snapshot"] = {}, False
        say(f"Screener: Snapshot nicht erreichbar ({type(e).__name__})")
    cands = {}
    for c, e in mapped.items():
        m = markets[c]
        closes = [k[4] for k in info.candles(c, "1d", now_ms - 100 * DAY_MS, now_ms)]
        bids, asks = info.book(c)
        test = cfg["test_order_usd"]
        cost = None
        if bids and asks:
            cost = (costs.taker_shortfall(bids, asks, True, test, cfg["taker_fee"])["total"]
                    + costs.taker_shortfall(bids, asks, False, test, cfg["taker_fee"])["total"]) / 2
        hist = info.funding_history(c, now_ms - 7 * DAY_MS)
        fund = sum(r for _, r in hist) / len(hist) * 24 * 365 if hist else m["funding"] * 24 * 365
        f, r, h, v = fees.get(e["id"]), revenue.get(e["id"]), holders.get(e["id"]), volume.get(e["id"])
        mcap = e["mcap"]
        events, evidence = [], []
        if sources["github"] and e["github"]:
            try:
                gh = github_events(e["github"], token, now_ms)
                events += gh
                for ev in sorted(gh, key=lambda x: -x["weight"] * 0.5 ** ((now_ms - x["t"]) / DAY_MS / 14))[:2]:
                    evidence.append({"kind": "release", "text": f"{ev['repo']} {ev['tag']}", "t": ev["t"],
                                     "url": ev["url"], "level": ev["kind"]})
            except Exception as err:
                sources["github"] = False
                say(f"Screener: GitHub verweigert den Zugriff ({type(err).__name__} {getattr(err, 'code', '')}),"
                    " Releases fließen in diesem Lauf nicht ein")
        for sp in spaces_of(e):
            for ev in votes.get(sp, []):
                events.append(ev)
                if ev["weight"] > 0.2 and len([x for x in evidence if x["kind"] == "vote"]) < 2:
                    evidence.append({"kind": "vote", "text": ev["title"][:90], "t": ev["t"]})
        mom30, mom90 = relative_momentum(closes, eth, 30), relative_momentum(closes, eth, 90)
        moms = [x for x in (mom30, mom90) if x is not None]
        factors = {
            "fee_growth": growth(f), "revenue_growth": growth(r), "volume_growth": growth(v),
            "acceleration": growth(f, short=True),
            "fee_yield": (f["d30"] * 365 / 30 / mcap) if f and mcap else None,
            "holder_yield": (h["d30"] * 365 / 30 / mcap) if h and h["d30"] > 0 and mcap else 0.0,
            "momentum": sum(moms) / len(moms) if moms else None,
            "catalysts": catalyst_score(events, now_ms) if (sources["github"] or votes) else None,
            "funding": -fund,
            "liquidity": -math.log(max(cost, 0.1)) if cost is not None else None,
        }
        reasons = []
        limit = cfg["max_cost_bps"] * (2 if c in held else 1)   # a held name is not dropped for a passing spike
        if cost is None or cost > limit:
            reasons.append(f"Handelskosten {cost:.0f} bp" if cost is not None else "kein Orderbuch")
        if len(closes) < 31:
            reasons.append("weniger als 30 Tage gelistet")
        cands[c] = {"coin": c, "name": e["name"], "mcap": mcap, "volume": m["volume"], "oi": m["oi"],
                    "fees30": f["d30"] if f else None, "revenue30": r["d30"] if r else None,
                    "holders30": h["d30"] if h else None, "funding": fund, "cost_bps": cost,
                    "vol": daily_vol(closes), "factors": factors, "evidence": evidence,
                    "eligible": not reasons, "reason": ", ".join(reasons)}
        time.sleep(0.1)
    score(cands, cfg["factors"])
    picks = select(cands, held, cfg)
    weights = weights_for(picks, cands, cfg)
    for c, d in cands.items():
        if c in weights:
            d["status"] = "gehalten" if c in held else "neu"
        else:
            d["status"] = "fällt raus" if c in held else ("ausgeschlossen" if not d["eligible"] else "Kandidat")
    return {"t": now_ms, "picks": [{"coin": c, "weight": weights[c], "score": cands[c]["score"]} for c in picks],
            "candidates": sorted(cands.values(), key=lambda d: -d["score"]), "sources": sources,
            "factor_weights": cfg["factors"], "n_longs": cfg["n_longs"], "pool": len(pool)}
