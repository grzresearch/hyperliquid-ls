# Hyperliquid Long/Short-Bot

Ein beta-neutraler Long/Short-Bot für Hyperliquid und eine Seite, die zeigt, was er tut: Kontowert als Kerzen, was er
heute gehandelt hat und warum, was er hält, Hedge und VaR, und wie der Screener die Longs gewählt hat.

- **Longs:** Ein Screener wählt einmal am Tag die Token mit den stärksten Protokoll-Cashflows (Fees, Revenue und deren
  Wachstum, Fee-Rendite), Momentum gegen ETH, Katalysatoren (Releases auf GitHub, angenommene Snapshot-Abstimmungen),
  Funding und Liquidität. Ein gehaltener Long bleibt, solange er knapp hinter den Besten liegt.
- **Shorts:** ETH und ein Alt-Korb, so bemessen, dass das Buch zu ETH beta-neutral ist.
- **Rebalancing:** kein fester Takt. Bei jeder Prüfung vergleicht der Bot die Risikokosten der Abweichung vom Ziel,
  `Risikoaversion ÷ (2 × Kontowert) × σ²`, mit den Handelskosten, die er vorher auf dem Live-Orderbuch simuliert
  (Implementation Shortfall nach Perold, Methode aus
  [implementation-shortfall-hyperliquid](https://github.com/gelatotrade/implementation-shortfall-hyperliquid)).
  Er handelt nur, was mehr Risiko abbaut, als es kostet.
- **Ausführung:** zuerst als Maker (Post-Only), der Rest nach dem Maker-Fenster als IOC mit Preisgrenze. Nach jedem
  Trade vergleicht er die simulierten mit den tatsächlichen Kosten.

## Dateien

- `bot/` – der Bot, im Paper-Modus nur Python-Standardbibliothek (3.11+)
  - `screener.py` – Auswahl der Longs aus Hyperliquid, DefiLlama, GitHub und Snapshot
  - `model.py` – Hedge, Zielbuch, Entscheidungsregel, VaR und Expected Shortfall
  - `costs.py` – Kostensimulation auf dem Orderbuch und realisierte Kosten
  - `broker.py` – Paper- und Live-Konto, Ausführung
  - `report.py` – Verlauf (Kontowert je Minute, Trades, Prüfungen) und Stand für die Seite
- `bot.paper.toml` – Konfiguration des öffentlichen Paper-Bots
- `bot.example.toml` – Vorlage für einen eigenen Lauf
- `index.html` – die Seite; Kurse kommen live per WebSocket von Hyperliquid, der Rest aus `data/bot.json` und `data/history.json`
- `.github/workflows/pages.yml` – lässt den Paper-Bot alle 15 Minuten eine Runde laufen und veröffentlicht danach die Seite

## Auf GitHub

Einmalig unter Settings → Pages als Quelle „GitHub Actions“ wählen. Danach läuft die Action alle 15 Minuten (GitHub
verschiebt geplante Läufe manchmal um ein paar Minuten) und bei jedem Push. Jede Runde holt den Stand des
Paper-Kontos aus dem Actions-Cache (fehlt er, aus der veröffentlichten Seite), lässt den Screener laufen, wenn sein
letzter Lauf mehr als 24 Stunden her ist, prüft das Buch, handelt, wenn es sich lohnt, und veröffentlicht die Seite
mit dem neuen Stand. Den Screener sofort laufen lassen: Actions → „Bot und Seite“ → „Run workflow“.

Das Paper-Konto ist auf der Seite öffentlich sichtbar: Positionen, Trades und Ergebnis.

## Lokal

```sh
python3 -m bot --config bot.paper.toml --plan                # nur rechnen und zeigen, was er handeln würde
python3 -m bot --config bot.paper.toml --once --site site/data
cp index.html manifest.webmanifest *.png site/ && python3 -m http.server -d site
```

Ohne `--once` läuft er weiter und prüft jede Minute. Der Zustand liegt in `state/` (Paper-Konto, Verlauf, Screener).

## Testnet oder Mainnet

Mit einer API-Wallet, die handeln, aber nicht abheben darf:

```sh
pip install -r requirements-bot.txt
cp bot.example.toml bot.toml   # Adresse unter [account]; bot.toml bleibt lokal
export HL_API_KEY=…            # Schlüssel der API-Wallet, nie in eine Datei im Repo
python3 -m bot --mode testnet
```

Eine Datei `STOP` im Arbeitsordner hält ihn vom Handeln ab. Tests: `python3 -m unittest discover -s tests`.

Rechenhilfe, keine Anlageberatung.
