"""Citrini Long/Short – rebalancing bot.

    python3 -m bot                 laufen lassen (Modus aus bot.toml, Standard: paper)
    python3 -m bot --plan          nur rechnen und anzeigen, was er handeln würde
    python3 -m bot --once          eine Runde, dann Ende
    python3 -m bot --mode testnet  Modus überschreiben (paper | testnet | mainnet)

Rebalanciert nicht nach Uhrzeit, sondern sobald der Risikoabbau die erwarteten Handelskosten übersteigt.
Ausgeführt wird zuerst als Maker (Post-Only), der Rest nach dem Maker-Fenster als IOC mit Preisgrenze.
Eine Datei namens STOP im Arbeitsordner hält ihn vom Handeln ab. Für testnet/mainnet braucht er die öffentliche
Adresse des Kontos (account.address oder HL_ACCOUNT) und den Schlüssel einer API-Wallet in HL_API_KEY.
"""
import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime

if sys.version_info < (3, 11):
    sys.exit("Der Bot braucht Python 3.11 oder neuer (python3 --version).")
import tomllib  # noqa: E402

from .broker import FatalError, LiveBroker, PaperBroker, execute
from .hl import MAINNET, TESTNET, Info
from .model import betas, cost_rates, decide, targets, universe

DEFAULTS = {
    "mode": "paper",
    "start_equity": 10000.0,
    "book_per_side": 1.5,
    "strategy": {"window_days": 90, "eth_share": 40,
                 "alts": ["SUI", "AVAX", "ADA", "TAO", "XPL", "MORPHO"],
                 "weights": {t: 1 for t in ["AERO", "PENDLE", "LIT", "SYRUP", "ETHFI", "ZRO", "ENA"]}},
    "rebalancing": {"risk_aversion": 3.0, "min_order_usd": 25.0, "loop_seconds": 60},
    "execution": {"maker_wait_seconds": 90, "requote_seconds": 10, "max_slippage": 0.005, "taker_share": 0.3,
                  "maker_fee": 0.00015, "taker_fee": 0.00045, "beta_tolerance": 0.25, "max_margin_use": 0.8},
    "leverage": {"long": 5, "short": 5, "eth": 10},
    "account": {"address": ""},
}


def load_config(path):
    cfg = copy.deepcopy(DEFAULTS)
    if os.path.exists(path):
        with open(path, "rb") as f:
            user = tomllib.load(f)
        for k, v in user.items():   # one level deep: a [strategy.weights] table replaces the default weights
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


def main():
    ap = argparse.ArgumentParser(prog="python3 -m bot", description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="bot.toml")
    ap.add_argument("--mode", choices=["paper", "testnet", "mainnet"])
    ap.add_argument("--once", action="store_true", help="eine Runde, dann Ende")
    ap.add_argument("--plan", action="store_true", help="nur rechnen und anzeigen, nichts handeln")
    ap.add_argument("--state", default="state/paper.json", help="Datei des Paper-Kontos")
    a = ap.parse_args()
    cfg = load_config(a.config)
    mode = a.mode or cfg["mode"]
    log = Log("logs")
    fees = {"maker": cfg["execution"]["maker_fee"], "taker": cfg["execution"]["taker_fee"]}
    history = Info(MAINNET)  # hedge ratios and risk always from mainnet prices
    venue = Info(TESTNET if mode == "testnet" else MAINNET)
    if mode == "paper":
        broker = PaperBroker(venue, a.state, float(cfg["start_equity"]), fees)
        log.say(f"Paper-Konto mit {de(broker.s['start'])} $ Start (Datei {a.state}), Kurse live von Hyperliquid")
    else:
        key = os.environ.get("HL_API_KEY", "").strip()
        address = (cfg["account"].get("address") or os.environ.get("HL_ACCOUNT", "")).strip()
        if not key or not address:
            sys.exit("Für testnet/mainnet: öffentliche Adresse in account.address (oder HL_ACCOUNT) und den "
                     "Schlüssel der API-Wallet in der Umgebungsvariable HL_API_KEY setzen.")
        try:
            broker = LiveBroker(venue, address, key)
        except FatalError as e:
            sys.exit(str(e))
        broker.cancel_all(universe(cfg))
        log.say(f"{mode.upper()}: Konto {address[:6]}…{address[-4:]}, Orders über die API-Wallet, "
                "alte offene Orders der Strategie storniert")
    coins = sorted(universe(cfg) | {"HYPE"})
    snap, snap_time, warned = None, 0.0, set()
    while True:
        try:
            if snap is None or time.time() - snap_time > 8 * 3600:
                snap = history.closes(coins, int(cfg["strategy"]["window_days"]) + 20)
                snap_time = time.time()
            markets = venue.markets()
            missing = [c for c in coins if c not in markets and c != "HYPE"]
            if missing and tuple(missing) not in warned:
                warned.add(tuple(missing))
                log.say(f"Auf diesem Markt nicht handelbar, wird ausgelassen: {', '.join(missing)}")
            broker.accrue_funding(markets)
            acct = broker.account(markets)
            equity = acct["equity"]
            target, h = targets(snap, cfg, equity)
            target = {c: v for c, v in target.items() if c in markets}
            current = {c: q * markets[c]["mid"] for c, q in acct["positions"].items() if c in markets}
            managed = set(target) | {c for c in current if c in universe(cfg)}
            books = {c: venue.book(c) for c in managed}
            cost = cost_rates(books, cfg["execution"])
            trades, why = decide(current, target, h["rows"], cost, equity, cfg)
            trades, scale = margin_guard(trades, current, target, markets, equity, cfg)
            b = betas(h["rows"], [c for c in managed if c in h["rows"][0]])
            net_beta = sum(b.get(c, 1.0) * v for c, v in current.items() if c in managed)
            longs = sum(v for c, v in current.items() if v > 0 and c in managed)
            shorts = -sum(v for c, v in current.items() if v < 0 and c in managed)
            pnl = equity - broker.s["start"] if mode == "paper" else None
            head = (f"Konto {de(equity)} $" + (f" ({signed(100 * pnl / broker.s['start'], 2)} %)" if pnl is not None else "")
                    + f" · Long {de(longs)} $ / Short {de(shorts)} $ · Netto-Beta {signed(net_beta)} $"
                    + f" · Abweichung 1σ {de(why['risk_before'])} $/Tag")
            cut = f" (wegen Margin auf {de(100 * scale)} % gekürzt)" if scale < 1 else ""
            if not trades:
                log.say(head + " · kein Trade")
            elif why["opening"] or why["closing"]:
                parts = ([f"{len(why['opening'])} Positionen eröffnen"] if why["opening"] else []) + \
                        ([f"{len(why['closing'])} schließen"] if why["closing"] else [])
                log.say(head + " · " + ", ".join(parts) + f", Kosten ≈ {de(why['cost'], 2)} $" + cut)
            else:
                log.say(head + f" · Rebalancing lohnt sich: Risiko {de(why['risk_before'])} → {de(why['risk_after'])} $/Tag,"
                        f" Nutzen {de(why['benefit'], 2)} $ > Kosten {de(why['cost'], 2)} $" + cut)
            if trades:
                for c, usd in sorted(trades.items(), key=lambda kv: -abs(kv[1])):
                    log.say(f"  {'KAUFEN' if usd > 0 else 'VERKAUFEN'} {c} ≈ {de(abs(usd))} $"
                            f" (jetzt {signed(current.get(c, 0.0))} $, Ziel {signed(target.get(c, 0.0))} $)")
            row = {"t": time.time(), "mode": mode, "equity": equity, "current": current, "target": target,
                   "trades": trades, "risk_before": why["risk_before"], "risk_after": why["risk_after"],
                   "benefit": why["benefit"], "cost": why["cost"], "net_beta": net_beta}
            if trades and not a.plan:
                if os.path.exists("STOP"):
                    log.say("  STOP-Datei gefunden: keine Orders")
                else:
                    for c in trades:
                        err = broker.set_leverage(c, leverage_for(c, target.get(c, current.get(c, 0.0)), markets, cfg))
                        if err:
                            log.say(f"  Hebel für {c} nicht gesetzt: {err}")
                    res = execute(broker, trades, markets, b, cfg, cfg["book_per_side"] * equity, log.say)
                    log.say(f"  ausgeführt in {de(res['seconds'])} s: {res['maker']} Maker, {res['taker']} Taker,"
                            f" Gebühren {de(res['fees'], 2)} $"
                            + (f", offen geblieben: {', '.join(res['left'])}" if res["left"] else ""))
                    row["result"] = {k: res[k] for k in ("maker", "taker", "fees", "seconds", "left")}
            broker.save()
            log.record(row)
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
