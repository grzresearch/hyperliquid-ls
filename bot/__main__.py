"""Long/short rebalancing bot for Hyperliquid.

    python3 -m bot                       laufen lassen (Modus aus der Konfiguration, Standard: paper)
    python3 -m bot --once                eine Runde, dann Ende (so läuft er in GitHub Actions)
    python3 -m bot --plan                nur rechnen und anzeigen, was er handeln würde
    python3 -m bot --mode testnet        Modus überschreiben (paper | testnet | mainnet)
    python3 -m bot --site site/data      Stand für die Seite schreiben (bot.json, history.json)
    python3 -m bot --screener now        den Screener sofort laufen lassen

Die Longs wählt ein Screener (Protokoll-Cashflows, Momentum, Katalysatoren), die Short-Seite ist ein beta-neutraler
Hedge aus ETH und einem Alt-Korb. Rebalanciert wird nicht nach Uhrzeit, sondern sobald der Risikoabbau die erwarteten
Handelskosten übersteigt; die Kosten simuliert er vorher auf dem Live-Orderbuch (Implementation Shortfall).
Ausgeführt wird zuerst als Maker (Post-Only), der Rest nach dem Maker-Fenster als IOC mit Preisgrenze. Eine Datei STOP
im Arbeitsordner hält ihn vom Handeln ab. Für testnet/mainnet braucht er die öffentliche Adresse des Kontos
(account.address oder HL_ACCOUNT) und den Schlüssel einer API-Wallet in HL_API_KEY.
"""
import argparse
import copy
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

if sys.version_info < (3, 11):
    sys.exit("Der Bot braucht Python 3.11 oder neuer (python3 --version).")
import tomllib  # noqa: E402

from . import costs, model, report, screener  # noqa: E402
from .broker import FatalError, LiveBroker, PaperBroker, execute  # noqa: E402
from .hl import MAINNET, TESTNET, Info  # noqa: E402

DEFAULTS = {
    "mode": "paper",
    "start_equity": 10000.0,
    "book_per_side": 1.5,
    "strategy": {"window_days": 90, "eth_share": 40,
                 "alts": ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO"],
                 "weights": {}},          # feste Longs, nur wenn der Screener aus ist
    "screener": {"enabled": True, "every_hours": 24, "n_longs": 7, "buffer": 3, "hold_floor": -0.25,
                 "min_volume_usd": 2e6, "min_mcap_usd": 3e7, "min_leverage": 3, "max_cost_bps": 25,
                 "test_order_usd": 2500, "max_weight": 0.25, "min_weight": 0.05,
                 "exclude": ["BTC", "ETH", "HYPE"],
                 "factors": {"fee_growth": 0.12, "revenue_growth": 0.10, "volume_growth": 0.08, "acceleration": 0.05,
                             "fee_yield": 0.10, "holder_yield": 0.05, "momentum": 0.20, "catalysts": 0.10,
                             "funding": 0.10, "liquidity": 0.10}},
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

    def strategy_coins(self):
        return set(self.longs()) | set(self.cfg["strategy"]["alts"]) | {"ETH"}

    def run_screener(self, held, now_ms):
        sc = self.cfg["screener"]
        if not sc.get("enabled") or self.screener_mode == "off":
            return
        last = self.scr or {}
        due = (now_ms - last.get("t", 0)) / 3600000 >= sc["every_hours"]
        cooled = (now_ms - last.get("tried", 0)) / 3600000 >= RETRY_HOURS
        if self.scr and self.screener_mode != "now" and not (due and cooled):
            return
        conf = dict(sc, exclude=sorted(set(sc["exclude"]) | set(self.cfg["strategy"]["alts"])),
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
        self.scr = result
        report.save_json(self.scr_path, result)
        self.screener_mode = "auto"
        self.log.say("Screener: " + ", ".join(f"{p['coin']} {de(100 * p['weight'])} %" for p in result["picks"]))

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
        wanted = self.longs()
        longs = {c: w for c, w in wanted.items() if c in markets}
        alts = [a for a in cfg["strategy"]["alts"] if a not in longs and a in markets]
        missing = sorted(set(wanted) - set(longs)) + [a for a in cfg["strategy"]["alts"] if a not in markets]
        if missing:
            log.say(f"Auf diesem Markt nicht handelbar, wird ausgelassen: {', '.join(missing)}")
        if not longs:
            raise RuntimeError("Keine Longs: Screener ohne Ergebnis und keine festen Gewichte in [strategy.weights]")
        if "ETH" not in markets:
            raise RuntimeError("ETH ist auf diesem Markt nicht handelbar, ohne ETH gibt es keinen Hedge")
        s = {"weights": longs, "alts": alts, "window": cfg["strategy"]["window_days"],
             "ethShare": cfg["strategy"]["eth_share"]}
        snap = self.closes(set(longs) | set(alts) | {"ETH"}, int(cfg["strategy"]["window_days"]) + 20)
        L = cfg["book_per_side"] * equity
        target, h = model.targets(snap, s, L)
        managed = set(target) | self.hist.traded | (set(positions) if self.mode == "paper" else set())
        current = {c: q * markets[c]["mid"] for c, q in positions.items() if c in markets and c in managed}

        # expected cost of each trade, simulated on the live book (implementation shortfall)
        books, rate, detail = {}, {}, {}
        for c in sorted(set(target) | set(current)):
            books[c] = self.venue.book(c)
            dev = current.get(c, 0.0) - target.get(c, 0.0)
            rate[c], detail[c] = costs.expected_cost(*books[c], dev < 0, max(abs(dev), rb["min_order_usd"]), ex)
        trades, why = model.decide(current, target, h["rows"], rate, equity, rb["risk_aversion"], rb["min_order_usd"])
        trades, scale = margin_guard(trades, current, target, markets, equity, cfg)
        arrival = {c: (books[c][0][0][0] + books[c][1][0][0]) / 2 for c in trades if books[c][0] and books[c][1]}
        betas = model.betas(h["rows"], [c for c in h["rows"][0]])

        head = (f"Konto {de(equity)} $" + (f" ({signed(100 * (equity / self.broker.s['start'] - 1), 2)} %)"
                                            if self.mode == "paper" else "")
                + f" · Abweichung 1σ {de(why['risk_before'])} $/Tag")
        cut = f" (wegen Margin auf {de(100 * scale)} % gekürzt)" if scale < 1 else ""
        if not trades:
            log.say(head + " · kein Trade")
        elif why["opening"] or why["closing"]:
            parts = ([f"{len(why['opening'])} eröffnen"] if why["opening"] else []) + \
                    ([f"{len(why['closing'])} schließen"] if why["closing"] else [])
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
                res = execute(self.broker, trades, markets, betas, cfg, L, log.say)
                executed = True
                log.say(f"  ausgeführt in {de(res['seconds'])} s: {res['maker']} Maker, {res['taker']} Taker,"
                        f" Gebühren {de(res['fees'], 2)} $" + (f", offen geblieben: {', '.join(res['left'])}" if res["left"] else ""))
                expected = {c: {k: round(v, 3) for k, v in detail[c].items() if isinstance(v, (int, float))} for c in trades}
                self.hist.add_event(now_ms, why, trades, arrival, expected, res["fills"], res)

        acct = self.broker.account(markets)
        positions = dict(acct["positions"])
        exposure = {c: q * markets[c]["mid"] for c, q in positions.items() if c in markets}
        var = model.value_at_risk(exposure, h["rows"])
        self.hist.add_run(now_ms, acct["equity"], why, len(trades) if executed else 0)
        self.broker.save()
        self.hist.save()
        if self.site:
            self.publish(now_ms, acct, positions, markets, target, h, why, detail, var, betas, executed, stopped, L,
                         longs, exposure)
        return {"t": now_ms, "equity": acct["equity"], "trades": trades, "executed": executed,
                "risk_before": why["risk_before"], "risk_after": why["risk_after"]}

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
        book = {"L": L, "b_book": h["b_book"], "b_alt": h["b_alt"], "share": h["share"], "h_eth": h["h_eth"],
                "h_alt": h["h_alt"], "resid": resid, "unhedged": unhedged, "betas": betas,
                "net_beta": sum(betas.get(c, 1.0) * v for c, v in exposure.items()),
                "alts": [a for a in cfg["strategy"]["alts"] if a not in longs and a in markets], "rows": len(h["rows"]),
                "lev": lev, "funding": {c: markets[c]["funding"] * 24 * 365 for c in set(exposure) | set(target)
                                        if c in markets}}
        decision = {k: why[k] for k in ("risk_before", "risk_after", "benefit", "cost", "gamma", "hold_risk_cost",
                                        "full_cost", "after_risk_cost", "drift_cost", "opening", "closing")}
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
