"""Long/short rebalancing bot for Hyperliquid.

    python3 -m bot                       laufen lassen (Modus aus der Konfiguration, Standard: paper)
    python3 -m bot --once                eine Runde, dann Ende (so läuft er in GitHub Actions)
    python3 -m bot --plan                nur rechnen und anzeigen, was er handeln würde
    python3 -m bot --mode testnet        Modus überschreiben (paper | testnet | mainnet)
    python3 -m bot --site site/data      Stand für die Seite schreiben (bot.json, history.json)
    python3 -m bot --screener now        den Screener sofort laufen lassen

Die Longs wählt ein Screener (Holder-, Revenue- und Fee-Rendite, Cashflow-Wachstum, Momentum, Katalysatoren; Funding
verkleinert nur die Position). Die Short-Seite sind die liquiden Perps, mit denen der Long-Basket am wenigsten
schwankt, beta-neutral zu ETH (bot/hedge.py). Rebalanciert wird nicht nach Uhrzeit, sondern sobald der Risikoabbau die
erwarteten Handelskosten übersteigt; die Kosten simuliert er vorher auf dem Live-Orderbuch (Implementation Shortfall).
Ausgeführt wird zuerst als Maker (Post-Only), der Rest nach dem Maker-Fenster als IOC mit Preisgrenze. Eine Datei STOP
im Arbeitsordner hält ihn vom Handeln ab. Für testnet/mainnet braucht er die öffentliche Adresse des Kontos
(account.address oder HL_ACCOUNT) und den Schlüssel einer API-Wallet in HL_API_KEY.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

if sys.version_info < (3, 11):
    sys.exit("Der Bot braucht Python 3.11 oder neuer (python3 --version).")
import tomllib  # noqa: E402

from . import costs, hedge, model, report, risk, screener  # noqa: E402
from .broker import FatalError, LiveBroker, PaperBroker, execute  # noqa: E402
from .hl import MAINNET, TESTNET, Info  # noqa: E402

DEFAULTS = {
    "mode": "paper",
    "start_equity": 10000.0,
    "book_per_side": 1.5,
    "strategy": {"window_days": 90, "eth_share": 40,
                 "alts": ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO"],   # nur für hedge.method = "fixed"
                 "weights": {}},          # feste Longs, nur wenn der Screener aus ist
    "screener": {"enabled": True, "every_hours": 24, "n_longs": 7, "buffer": 3, "hold_floor": -0.25,
                 "min_volume_usd": 2e6, "min_mcap_usd": 3e7, "min_leverage": 3, "max_cost_bps": 25,
                 "test_order_usd": 2500, "max_weight": 0.25, "min_weight": 0.05,
                 "funding_free": 0.25, "funding_max": 1.0,
                 "exclude": ["BTC", "ETH", "HYPE"],
                 "holder_revenue": {"AAVE": 50e6},     # Rückkäufe pro Jahr, die DefiLlama nicht erfasst
                 "factors": {"holder_yield": 0.35, "revenue_yield": 0.15, "fee_yield": 0.15, "momentum": 0.10,
                             "cashflow_growth": 0.10, "catalysts": 0.05, "acceleration": 0.05, "volume_growth": 0.05}},
    "hedge": {"method": "minvar", "exclude": ["HYPE"]},     # die übrigen Werte: bot/hedge.py, DEFAULTS
    "risk": {"jump_loss": 0.05, "history_days": 365, "young_days": 180, "floor_down": 0.16, "floor_up": 0.31,
             "vol_target": True, "min_scale": 0.5, "size_band": 0.10, "cap_tolerance": 0.05},
    "rebalancing": {"risk_aversion": 3.0, "min_order_usd": 25.0, "loop_seconds": 60, "check_minutes": 15},
    "execution": {"maker_wait_seconds": 90, "requote_seconds": 10, "max_slippage": 0.005, "taker_share": 0.3,
                  "maker_fee": 0.00015, "taker_fee": 0.00045, "unfilled_bps": 50, "beta_tolerance": 0.25,
                  "max_margin_use": 0.8},
    "leverage": {"long": 5, "short": 5, "eth": 10},
    "account": {"address": ""},
}
RETRY_HOURS = 2      # after a failed or rejected screener run, wait this long before the next attempt


def load_config(path):
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path, "rb") as f:
            user = tomllib.load(f)
        for k, v in user.items():   # one level deep: a [screener.factors] table replaces the default factors
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def de(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def signed(x, d=0):
    return ("+" if x >= 0 else "−") + de(abs(x), d)


class Log:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def say(self, text):
        print(f"[{datetime.now():%H:%M:%S}] {text}" if not text.startswith("  ") else text, flush=True)

    def record(self, row):
        path = os.path.join(self.folder, f"bot-{datetime.now():%Y-%m-%d}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def leverage_for(coin, target_usd, markets, cfg):
    lv = cfg["leverage"]
    want = lv["eth"] if coin == "ETH" else (lv["long"] if target_usd > 0 else lv["short"])
    return max(1, min(int(want), markets[coin]["maxLev"]))


def margin_guard(trades, current, target, markets, equity, cfg):
    """Scale down trades that add exposure if the book after them would tie up too much initial margin."""
    limit = cfg["execution"]["max_margin_use"] * equity

    def margin(scale):
        total = 0.0
        for c in set(current) | set(trades):
            t = trades.get(c, 0.0)
            grows = abs(current.get(c, 0.0) + t) > abs(current.get(c, 0.0))
            pos = current.get(c, 0.0) + (t * scale if grows else t)
            total += abs(pos) / leverage_for(c, target.get(c, pos), markets, cfg)
        return total

    if margin(1.0) <= limit:
        return trades, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if margin(mid) <= limit else (lo, mid)
    out = {}
    for c, t in trades.items():
        grows = abs(current.get(c, 0.0) + t) > abs(current.get(c, 0.0))
        out[c] = t * lo if grows else t
    return {c: t for c, t in out.items() if abs(t) >= cfg["rebalancing"]["min_order_usd"]}, lo


def restore(url, state_dir, say):
    """Fill an empty state folder from the published page data (…/data/), so a lost cache does not reset the
    paper account. Only a page without data (404) starts a new account; any other failure ends the run, so that a
    passing outage never replaces the public record with a fresh account."""
    base = url.rstrip("/")

    def get(name):
        try:
            with urllib.request.urlopen(base + "/" + name, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    try:
        bot = get("bot.json")
        hist = get("history.json") if bot else None
    except Exception as e:
        sys.exit(f"Veröffentlichter Stand unter {base} nicht lesbar ({type(e).__name__}: {e}). "
                 "Abbruch, statt mit einem neuen Konto weiterzumachen.")
    if not bot:
        say(f"Noch kein veröffentlichter Stand unter {base}, starte neu")
        return
    if bot.get("paper"):
        report.save_json(os.path.join(state_dir, "paper.json"), bot["paper"], indent=1)
    if bot.get("screener"):
        report.save_json(os.path.join(state_dir, "screener.json"), bot["screener"])
    if hist:
        report.save_json(os.path.join(state_dir, "history.json"), hist)
    say(f"Stand aus {base} wiederhergestellt: Konto {de(bot.get('account', {}).get('equity', 0))} $")


class Bot:
    def __init__(self, cfg, mode, state_dir, log, plan_only=False, site=None, screener_mode="auto"):
        self.cfg, self.mode, self.log, self.plan_only, self.site = cfg, mode, log, plan_only, site
        self.screener_mode = screener_mode
        self.history_info = Info(MAINNET)    # hedge ratios, risk and candles always from mainnet prices
        self.venue = Info(TESTNET if mode == "testnet" else MAINNET)
        fees = {"maker": cfg["execution"]["maker_fee"], "taker": cfg["execution"]["taker_fee"]}
        if mode == "paper":
            self.broker = PaperBroker(self.venue, os.path.join(state_dir, "paper.json"), float(cfg["start_equity"]),
                                      fees)
            log.say(f"Paper-Konto mit {de(self.broker.s['start'])} $ Start, Kurse live von Hyperliquid")
        else:
            key = os.environ.get("HL_API_KEY", "").strip()
            address = (cfg["account"].get("address") or os.environ.get("HL_ACCOUNT", "")).strip()
            if not key or not address:
                sys.exit("Für testnet/mainnet: öffentliche Adresse in account.address (oder HL_ACCOUNT) und den "
                         "Schlüssel der API-Wallet in der Umgebungsvariable HL_API_KEY setzen.")
            try:
                self.broker = LiveBroker(self.venue, address, key)
            except FatalError as e:
                sys.exit(str(e))
        self.hist = report.History(os.path.join(state_dir, "history.json"))
        self.scr_path = os.path.join(state_dir, "screener.json")
        self.scr = report.load_json(self.scr_path, None)
        if mode != "paper":
            self.broker.cancel_all(self.hist.traded | self.strategy_coins())
            log.say(f"{mode.upper()}: Orders über die API-Wallet, alte offene Orders der Strategie storniert")
        self.snap, self.snap_key, self.snap_time = None, None, 0.0

    # ---------- the book ----------
    def longs(self):
        sc = self.cfg["screener"]
        if sc.get("enabled") and self.scr and self.scr.get("picks"):
            return {p["coin"]: p["weight"] for p in self.scr["picks"]}
        return {c: float(w) for c, w in self.cfg["strategy"]["weights"].items() if float(w) > 0}

    def shorts(self):
        """The short book chosen for the screener's longs (bot/hedge.py). Empty with the fixed basket, and with fixed
        longs from [strategy.weights]: those are hedged with ETH alone."""
        if self.cfg["hedge"].get("method") != "minvar" or not self.cfg["screener"].get("enabled"):
            return {}
        return dict(((self.scr or {}).get("hedge") or {}).get("weights") or {})

    def strategy_coins(self):
        return set(self.longs()) | set(self.shorts()) | set(self.cfg["strategy"]["alts"]) | {"ETH"}

    def run_screener(self, held, now_ms):
        sc = self.cfg["screener"]
        if not sc.get("enabled") or self.screener_mode == "off":
            return
        last = self.scr or {}
        due = (now_ms - last.get("t", 0)) / 3600000 >= sc["every_hours"] or last.get("factor_weights") != sc["factors"]
        cooled = (now_ms - last.get("tried", 0)) / 3600000 >= RETRY_HOURS
        if self.scr and self.screener_mode != "now" and not (due and cooled):
            self.refresh_hedge(now_ms)
            return
        fixed = self.cfg["hedge"].get("method") != "minvar"
        conf = dict(sc, exclude=sorted(set(sc["exclude"]) | (set(self.cfg["strategy"]["alts"]) if fixed else set())),
                    taker_fee=self.cfg["execution"]["taker_fee"])
        keep = ", es bleibt bei der letzten Auswahl" if self.scr else ""
        try:
            result = screener.run(self.history_info, conf, held=held, now_ms=now_ms, say=self.log.say)
        except Exception as e:      # keep the previous picks: an outage at DefiLlama must not close the book
            self.log.say(f"Screener fehlgeschlagen ({type(e).__name__}: {e}){keep}")
            return self.screener_tried(now_ms)
        if len(result["picks"]) < min(screener.MIN_LONGS, int(sc["n_longs"])):   # too few usable candidates
            self.log.say(f"Screener fand nur {len(result['picks'])} handelbare Longs{keep}")
            return self.screener_tried(now_ms)
        result["tried"] = now_ms
        previous = (self.scr or {}).get("hedge")
        result["hedge"] = self.run_hedge(result, previous, now_ms)
        if result["hedge"] is None:     # keep the last short book (a short that became a long drops out in round())
            result["hedge"], result["hedge_tried"] = previous, now_ms
        self.scr = result
        report.save_json(self.scr_path, result)
        self.screener_mode = "auto"
        self.log.say("Screener: " + ", ".join(f"{p['coin']} {de(100 * p['weight'])} %" for p in result["picks"]))

    def hedge_settings(self):
        """Everything that decides the short book. A change starts the choice afresh, without the old shorts."""
        hc, sc = self.cfg["hedge"], self.cfg["screener"]
        out = {k: hc.get(k, v) for k, v in hedge.DEFAULTS.items()}
        out.update(funding_free=sc.get("funding_free", 0.25), funding_max=sc.get("funding_max", 1.0),
                   exclude=sorted(hc.get("exclude", [])), window_days=int(self.cfg["strategy"]["window_days"]))
        return out

    def refresh_hedge(self, now_ms):
        """Between screener runs: choose the short book again when the last screener run could not (none yet, or
        one chosen for earlier longs) or the settings changed. A failed attempt keeps the old book and waits
        RETRY_HOURS."""
        last = self.scr or {}
        if self.cfg["hedge"].get("method") != "minvar" or not last.get("picks"):
            return
        h = last.get("hedge") or {}
        if h.get("settings") == self.hedge_settings() and h.get("t", 0) >= last.get("t", 0):
            return
        if (now_ms - last.get("hedge_tried", 0)) / 3600000 < RETRY_HOURS:
            return
        new = self.run_hedge(last, last.get("hedge"), now_ms)
        if new:
            self.scr["hedge"] = new
            self.scr.pop("hedge_tried", None)
        else:
            self.scr["hedge_tried"] = now_ms
        report.save_json(self.scr_path, self.scr)

    def run_hedge(self, result, previous, now_ms):
        """The short book for the screener's picks: the liquid perps that hedge them best (bot/hedge.py). None when
        it cannot be computed; the bot then keeps the previous one or the fixed basket."""
        hc, sc, ex = self.cfg["hedge"], self.cfg["screener"], self.cfg["execution"]
        if hc.get("method") != "minvar":
            return None
        try:
            settings = self.hedge_settings()
            longs = {p["coin"]: p["weight"] for p in result["picks"]}
            markets = self.history_info.markets()
            pool = sorted(c for c, m in markets.items() if c not in longs and c not in settings["exclude"]
                          and m["maxLev"] >= sc["min_leverage"] and m["volume"] >= sc["min_volume_usd"])
            if self.mode == "testnet":  # history from mainnet, but only what the testnet trades
                venue = self.venue.markets()
                pool = [c for c in pool if c in venue]
            days = settings["window_days"]
            snap = self.history_info.closes(sorted(set(pool) | set(longs) | {"ETH"}), days + 20)
            rows, usable = hedge.window_rows(snap, list(longs), pool, days)
            if not rows:
                raise RuntimeError("zu wenig Kursdaten der Longs")
            per = int(snap.get("perDay", 3))
            scores = self.short_scores(result, snap, usable) if settings["tilt_slack"] > 0 else None
            cost = {}
            for c in usable:            # a short pays negative funding; the 7-day mean decides, as for the longs
                yearly = -markets[c]["funding"] * 24 * 365
                if yearly > settings["funding_free"]:
                    hist = self.history_info.funding_history(c, now_ms - 7 * screener.DAY_MS)
                    yearly = -sum(r for _, r in hist) / len(hist) * 24 * 365 if hist else yearly
                cost[c] = yearly
            held = None                 # the current shorts get their head start only under the same settings
            if previous and previous.get("settings") == settings:
                held = {c: w for c, w in (previous.get("weights") or {}).items() if c in usable} or None

            def thin(c):                # the test order on the live book costs too much, or there is no book
                try:
                    bids, asks = self.venue.book(c)
                except Exception:
                    return True
                if not bids or not asks:
                    return True
                return (costs.taker_shortfall(bids, asks, True, sc["test_order_usd"], ex["taker_fee"])["total"]
                        + costs.taker_shortfall(bids, asks, False, sc["test_order_usd"], ex["taker_fee"])["total"]
                        ) / 2 > sc["max_cost_bps"]
            excluded = set()
            for attempt in range(3):    # a chosen short whose book is too thin is replaced, twice at most
                res = hedge.choose(rows, longs, [c for c in usable if c not in excluded], settings, score=scores,
                                   held=held, cost=cost)
                bad = {c for c in res["weights"] if thin(c)}
                if not bad:
                    break
                if attempt == 2:
                    self.log.say(f"Orderbuch dünn, trotzdem im Short-Buch: {', '.join(sorted(bad))}")
                    break
                excluded |= bad
            res.update(t=now_ms, excluded=sorted(excluded), days=len(rows) // per, settings=settings)
            if res.get("oos"):          # in days, not in 8-hour steps
                res["oos"].update(days_fit=res["oos"]["days_fit"] // per, days_test=res["oos"]["days_test"] // per)
            for cand in res["candidates"]:
                cand["funding"] = -cost.get(cand["coin"], 0.0)
            self.log.say("Shorts: " + ", ".join(f"{c} {de(100 * w)} %" for c, w in
                                                 sorted(res["weights"].items(), key=lambda kv: -kv[1]))
                         + f" · Restvolatilität {de(100 * res['resid_vol'])} % statt {de(100 * res['eth_only'])} % nur"
                           " mit ETH" + (" · bisherige Shorts behalten" if res["kept"] else ""))
            return res
        except Exception as e:
            self.log.say(f"Short-Auswahl fehlgeschlagen ({type(e).__name__}: {e})")
            return None

    def short_scores(self, result, snap, usable):
        """Short score per candidate, only needed with a tilt: the long factors turned around. Coins without protocol
        data get their momentum from the closes."""
        per, C = int(snap.get("perDay", 3)), snap["coins"]
        daily = {c: [v for v in C[c]["closes"][::-1][::per][::-1] if v] for c in usable + ["ETH"]}
        by_coin = {d["coin"]: d for d in result.get("candidates", [])}
        rows = {}
        for c in usable:
            f = dict(by_coin[c]["factors"]) if c in by_coin else {}
            if f.get("momentum") is None:
                moms = [m for m in (screener.relative_momentum(daily[c], daily["ETH"], 30),
                                    screener.relative_momentum(daily[c], daily["ETH"], 90)) if m is not None]
                f["momentum"] = sum(moms) / len(moms) if moms else None
            rows[c] = f
        return screener.short_scores(rows, self.cfg["screener"]["factors"])

    def screener_tried(self, now_ms):
        """Remember a failed attempt, so the next one waits RETRY_HOURS. Without earlier picks there is nothing to
        trade, and the next round simply tries again."""
        if self.scr:
            self.scr["tried"] = now_ms
            report.save_json(self.scr_path, self.scr)

    def closes(self, coins, days):
        key = tuple(sorted(coins))
        if self.snap is None or key != self.snap_key or time.time() - self.snap_time > 8 * 3600:
            self.snap = self.history_info.closes(sorted(coins), days)
            self.snap_key, self.snap_time = key, time.time()
        return self.snap

    # ---------- one round ----------
    def round(self):
        cfg, log, rb, ex = self.cfg, self.log, self.cfg["rebalancing"], self.cfg["execution"]
        now_ms = int(time.time() * 1000)
        markets = self.venue.markets()
        if self.mode == "paper":
            for c, pnl in self.broker.settle_missing(markets).items():
                log.say(f"{c} wird auf Hyperliquid nicht mehr gehandelt: zum letzten Kurs abgerechnet, {signed(pnl, 2)} $")
        self.broker.accrue_funding(markets)
        acct = self.broker.account(markets)
        equity, positions = acct["equity"], dict(acct["positions"])
        mids = {c: m["mid"] for c, m in markets.items()}
        self.hist.extend_equity(self.venue, positions, equity, mids, now_ms)
        self.run_screener([c for c, q in positions.items() if q > 0], now_ms)

        # the book is sized only from what this market trades: a coin that is missing must not leave its hedge behind
        wanted, wanted_shorts = self.longs(), self.shorts()
        fixed = cfg["hedge"].get("method") != "minvar"      # without a short book, minvar hedges with ETH alone
        longs = {c: w for c, w in wanted.items() if c in markets}
        shorts = {c: w for c, w in wanted_shorts.items() if c in markets and c not in longs}
        alts = [a for a in cfg["strategy"]["alts"] if a not in longs and a in markets] if fixed else []
        missing = sorted((set(wanted) | set(wanted_shorts) | (set(cfg["strategy"]["alts"]) if fixed else set()))
                         - set(markets))
        if missing:
            log.say(f"Auf diesem Markt nicht handelbar, wird ausgelassen: {', '.join(missing)}")
        if not longs:
            raise RuntimeError("Keine Longs: Screener ohne Ergebnis und keine festen Gewichte in [strategy.weights]")
        if "ETH" not in markets:
            raise RuntimeError("ETH ist auf diesem Markt nicht handelbar, ohne ETH gibt es keinen Hedge")
        s = {"weights": longs, "shorts": shorts, "alts": alts, "window": cfg["strategy"]["window_days"],
             "ethShare": cfg["strategy"]["eth_share"]}
        rk = cfg["risk"]
        days = int(rk["history_days"])
        snap = self.closes(set(longs) | set(shorts) | set(alts) | {"ETH"} | {c for c in positions if c in markets},
                           max(int(cfg["strategy"]["window_days"]) + 20, days + 5))
        L = cfg["book_per_side"] * equity

        # fat tails: no position may lose more than jump_loss of the account on its worst own move of the year
        side = {**{c: -1 for c in list(shorts) + alts}, **{c: 1 for c in longs}}
        jump = risk.jumps(snap, list(side), days, rk["young_days"], rk["floor_down"], rk["floor_up"])
        caps = risk.position_caps(jump, equity, rk["jump_loss"], side)
        target, h = model.targets(snap, s, L, caps)
        h["jumps"] = jump
        if h["capped"]:
            log.say("Sprunglimits: " + ", ".join(f"{c} {signed(x['cap'])} $ statt {signed(x['want'])} $"
                                                 for c, x in sorted(h["capped"].items()))
                    + ("; was die Shorts nicht tragen, übernimmt ETH" if any(x["want"] < 0 for x in h["capped"].values())
                       else ""))
        # volatility regime: GARCH and HMM on the book's year say how restless tomorrow is likely to be. Without a
        # view (an error, too little history) the book keeps the size it had, so a passing failure moves nothing.
        last = float(self.hist.h.get("size") or 1.0)
        try:
            view = self.risk_view(snap, target, h, days)
        except Exception as e:      # the risk layer must never stop the hedge
            log.say(f"Risiko-Schicht fehlgeschlagen ({type(e).__name__}: {e}), Buchgröße bleibt {de(100 * last)} %")
            view = {"var": 0.0, "ref": 0.0, "days": 0, "risk_scale": 1.0, "size": last}
        if not view["days"]:
            view["size"] = last
        size = view["size"]
        self.hist.h["size"] = round(size, 4)
        if size < 1:
            target = {c: v * size for c, v in target.items()}
        need = sum(abs(v) / leverage_for(c, v, markets, cfg) for c, v in target.items())
        room = cfg["execution"]["max_margin_use"] * equity
        if need > room:             # a target the margin cannot carry would keep the size band firing in vain
            target = {c: v * room / need for c, v in target.items()}
            log.say(f"Ziel auf {de(100 * room / need)} % verkleinert: mehr Margin, als max_margin_use erlaubt")
        managed = set(target) | self.hist.traded | (set(positions) if self.mode == "paper" else set())
        current = {c: q * markets[c]["mid"] for c, q in positions.items() if c in markets and c in managed}

        # forced back to the target in full, whatever the drift rule says: a position beyond its jump limit, and the
        # whole book once its size is more than size_band off (a hedged book that is simply too big hardly shows in
        # the covariance, so the rule alone would never shrink it)
        force = {c for c, v in current.items()
                 if abs(v) > caps.get(c, math.inf) * (1 + float(rk["cap_tolerance"])) + rb["min_order_usd"]}
        held, aim = sum(abs(v) for v in current.values()), sum(abs(v) for v in target.values())
        if held > 0 and aim > 0 and abs(held / aim - 1) > float(rk["size_band"]):
            force |= set(current) | set(target)

        # expected cost of each trade, simulated on the live book (implementation shortfall)
        books, rate, detail = {}, {}, {}
        for c in sorted(set(target) | set(current)):
            books[c] = self.venue.book(c)
            dev = current.get(c, 0.0) - target.get(c, 0.0)
            rate[c], detail[c] = costs.expected_cost(*books[c], dev < 0, max(abs(dev), rb["min_order_usd"]), ex)
        trades, why = model.decide(current, target, h["rows"], rate, equity, rb["risk_aversion"], rb["min_order_usd"],
                                   view["risk_scale"], force, caps)
        trades, scale = margin_guard(trades, current, target, markets, equity, cfg)
        for key in ("opening", "closing", "resized"):      # what margin_guard left of them
            why[key] = [c for c in why[key] if c in trades]
        arrival = {c: (books[c][0][0][0] + books[c][1][0][0]) / 2 for c in trades if books[c][0] and books[c][1]}
        betas = model.betas(h["rows"], [c for c in h["rows"][0]])

        head = (f"Konto {de(equity)} $" + (f" ({signed(100 * (equity / self.broker.s['start'] - 1), 2)} %)"
                                            if self.mode == "paper" else "")
                + f" · Abweichung 1σ {de(why['risk_before'])} $/Tag"
                + (f" · Volatilität {de(100 * math.sqrt(view['var']) * size / equity, 1)} % des Kontos pro Tag"
                   f" ({de(view['risk_scale'] ** 0.5, 2)} × Fenster), Buch {de(100 * size)} %" if view["days"] else ""))
        cut = f" (wegen Margin auf {de(100 * scale)} % gekürzt)" if scale < 1 else ""
        if not trades:
            log.say(head + " · kein Trade")
        elif why["opening"] or why["closing"] or why["resized"]:
            parts = ([f"{len(why['opening'])} eröffnen"] if why["opening"] else []) + \
                    ([f"{len(why['closing'])} schließen"] if why["closing"] else []) + \
                    ([f"{len(why['resized'])} auf Sprunglimit oder Buchgröße bringen"] if why["resized"] else [])
            log.say(head + " · " + ", ".join(parts) + f", Kosten ≈ {de(why['cost'], 2)} $" + cut)
        else:
            log.say(head + f" · Rebalancing lohnt sich: Risiko {de(why['risk_before'])} → {de(why['risk_after'])} $/Tag,"
                    f" Nutzen {de(why['benefit'], 2)} $ > Kosten {de(why['cost'], 2)} $" + cut)
        for c, usd in sorted(trades.items(), key=lambda kv: -abs(kv[1])):
            log.say(f"  {'KAUFEN' if usd > 0 else 'VERKAUFEN'} {c} ≈ {de(abs(usd))} $ (jetzt {signed(current.get(c, 0.0))} $,"
                    f" Ziel {signed(target.get(c, 0.0))} $, Kosten simuliert {de(detail[c].get('bps', 0), 1)} bp)")

        executed, stopped = False, False
        if trades and not self.plan_only:
            if os.path.exists("STOP"):
                stopped = True
                log.say("  STOP-Datei gefunden: keine Orders")
            else:
                for c in trades:
                    err = self.broker.set_leverage(c, leverage_for(c, target.get(c, current.get(c, 0.0)), markets, cfg))
                    if err:
                        log.say(f"  Hebel für {c} nicht gesetzt: {err}")
                res = execute(self.broker, trades, markets, betas, cfg, L * size, log.say)
                executed = True
                log.say(f"  ausgeführt in {de(res['seconds'])} s: {res['maker']} Maker, {res['taker']} Taker,"
                        f" Gebühren {de(res['fees'], 2)} $" + (f", offen geblieben: {', '.join(res['left'])}" if res["left"] else ""))
                expected = {c: {k: round(v, 3) for k, v in detail[c].items() if isinstance(v, (int, float))} for c in trades}
                self.hist.add_event(now_ms, why, trades, arrival, expected, res["fills"], res)

        acct = self.broker.account(markets)
        positions = dict(acct["positions"])
        exposure = {c: q * markets[c]["mid"] for c, q in positions.items() if c in markets}
        try:
            var = self.risk_now(snap, exposure, days, view, h["rows"])
        except Exception as e:
            log.say(f"VaR und Regime nicht berechnet ({type(e).__name__}: {e})")
            var = None
        self.hist.add_run(now_ms, acct["equity"], why, len(trades) if executed else 0)
        self.broker.save()
        self.hist.save()
        if self.site:
            self.publish(now_ms, acct, positions, markets, target, h, why, detail, var, betas, executed, stopped,
                         L * size, longs, exposure)
        return {"t": now_ms, "equity": acct["equity"], "trades": trades, "executed": executed,
                "risk_before": why["risk_before"], "risk_after": why["risk_after"]}

    def risk_view(self, snap, target, h, days):
        """Tomorrow's risk of the target book (bot/risk.py): the variance GARCH and the HMM expect from the book's
        year, against the hedge window's variance (the decision's risk scale) and against the book's usual level
        (with vol_target the book shrinks while the expected volatility is above it, never grows beyond it)."""
        rk = self.cfg["risk"]
        out = risk.book_outlook(snap, target, days)
        out.update(risk_scale=1.0, size=1.0)
        if out["days"] < 60 or out["var"] <= 0:
            out["days"] = 0
            return out
        names = [c for c in target if c in h["rows"][0]]
        window = model.quad([target[c] for c in names], model.covariance(h["rows"], names))
        if window > 0:
            out["risk_scale"] = min(4.0, max(0.25, out["var"] / window))
        if rk.get("vol_target"):
            out["size"] = min(1.0, max(float(rk["min_scale"]), math.sqrt(out["ref"] / out["var"])))
        return out

    def risk_now(self, snap, exposure, days, view, rows):
        """VaR and ES of the book as it stands, with the fat tails of its year at tomorrow's volatility, its usual
        volatility over the year and over the hedge window, and the regime and size the round worked with."""
        held = {c: v for c, v in exposure.items() if v and c in snap["coins"]}
        if not held:
            return None
        o = risk.book_outlook(snap, held, days)
        if o["days"] < 60 or o["var"] <= 0 or not o["tails"]:
            return None
        names = [c for c in held if c in rows[0]]
        window = model.quad([held[c] for c in names], model.covariance(rows, names))
        out = dict(o["tails"])
        out.update(regime=o["regime"], ref_sigma=math.sqrt(o["ref"]), window_sigma=math.sqrt(max(window, 0.0)),
                   garch=o["garch"], days=o["days"], size=view["size"], risk_scale=view["risk_scale"])
        return out

    def publish(self, now_ms, acct, positions, markets, target, h, why, detail, var, betas, executed, stopped, L,
                longs, exposure):
        cfg = self.cfg
        paper = self.broker.s if self.mode == "paper" else None
        lev = {c: leverage_for(c, target.get(c, exposure.get(c, 0.0)), markets, cfg) for c in set(exposure) | set(target)}
        account = {"equity": acct["equity"], "mode": self.mode,
                   "margin": sum(abs(v) / lev[c] for c, v in exposure.items()),
                   "long": sum(v for v in exposure.values() if v > 0),
                   "short": -sum(v for v in exposure.values() if v < 0)}
        if paper:
            account.update(start=paper["start"], since=int(paper["since"] * 1000), cash=paper["cash"],
                           realized=paper["realized"], fees=paper["fees"], funding=paper["funding"],
                           volume=paper["volume"])
        resid, unhedged = model.residual_vol(h)
        jump = h.get("jumps") or {}
        book = {"L": L, "mode": h["mode"], "b_book": h["b_book"], "b_short": h["b_short"], "H": h["H"], "k": h["k"],
                "capped": h.get("capped") or {},
                "jumps": {c: {k: round(v, 4) if isinstance(v, float) else v for k, v in jump[c].items()}
                          for c in set(target) | set(exposure) if c in jump},
                "jump_loss": cfg["risk"]["jump_loss"],
                "resid": resid, "unhedged": unhedged, "betas": betas,
                "net_beta": sum(betas.get(c, 1.0) * v for c, v in exposure.items()), "rows": len(h["rows"]),
                "lev": lev, "funding": {c: markets[c]["funding"] * 24 * 365 for c in set(exposure) | set(target)
                                        if c in markets}}
        if h["mode"] == "fixed":
            book.update(share=h["share"], b_alt=h["b_alt"], alts=[a for a in h["k"] if a != "ETH"])
        else:
            sel = (self.scr or {}).get("hedge") or {}
            book["hedge"] = {k: sel.get(k) for k in ("t", "weights", "oos", "candidates", "mu", "kept", "resid_vol",
                                                     "eth_only", "unhedged", "excluded", "days", "settings")}
            # read by a page opened before the short book existed (it reloads only with the next version)
            book.update(share=h["k"].get("ETH", 0.0) / h["H"] if h["H"] else 1.0, b_alt=h["b_short"],
                        alts=[c for c in h["k"] if c != "ETH"])
        decision = {k: why[k] for k in ("risk_before", "risk_after", "benefit", "cost", "gamma", "hold_risk_cost",
                                        "full_cost", "after_risk_cost", "drift_cost", "opening", "closing",
                                        "resized")}
        decision.update(t=now_ms, executed=executed, stopped=stopped,
                        by_coin=[dict(r, detail={k: round(v, 3) for k, v in detail.get(r["coin"], {}).items()
                                                 if isinstance(v, (int, float))}) for r in why["by_coin"]])
        scores = {p["coin"]: p["score"] for p in (self.scr or {}).get("picks", [])}
        payload = report.site_payload(now_ms, self.mode, cfg, account, paper, positions,
                                      {c: markets[c]["mid"] for c in markets}, target, h, decision, var,
                                      self.scr, book, longs, scores)
        report.write_site(self.site, payload, self.hist)


def main():
    ap = argparse.ArgumentParser(prog="python3 -m bot", description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="bot.toml")
    ap.add_argument("--mode", choices=["paper", "testnet", "mainnet"])
    ap.add_argument("--once", action="store_true", help="eine Runde, dann Ende")
    ap.add_argument("--plan", action="store_true", help="nur rechnen und anzeigen, nichts handeln")
    ap.add_argument("--state-dir", default="state", help="Ordner für Paper-Konto, Verlauf und Screener")
    ap.add_argument("--site", help="Ordner, in den der Stand für die Seite geschrieben wird")
    ap.add_argument("--restore", help="leeren Zustand aus der veröffentlichten Seite holen (URL von …/data/)")
    ap.add_argument("--screener", choices=["auto", "now", "off"], default="auto")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not (a.once or a.plan):     # the page shows how often it checks
        cfg["rebalancing"]["check_minutes"] = cfg["rebalancing"]["loop_seconds"] / 60
    mode = a.mode or cfg["mode"]
    log = Log("logs")
    os.makedirs(a.state_dir, exist_ok=True)
    if a.restore and not os.path.exists(os.path.join(a.state_dir, "paper.json")):
        restore(a.restore, a.state_dir, log.say)
    bot = Bot(cfg, mode, a.state_dir, log, plan_only=a.plan, site=a.site, screener_mode=a.screener)
    while True:
        try:
            log.record(bot.round())
        except (KeyboardInterrupt, FatalError):
            raise
        except Exception as e:  # keep running; the next round starts from fresh data
            log.say(f"Fehler: {type(e).__name__}: {e}")
            if a.once or a.plan:
                raise
        if a.once or a.plan:
            break
        time.sleep(float(cfg["rebalancing"]["loop_seconds"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBeendet. Offene Orders wurden storniert.")
    except FatalError as err:
        sys.exit(f"Gestoppt: {err}")
