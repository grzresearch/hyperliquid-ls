# Citrini Long/Short – live

Dashboard für den beta-neutralen Long/Short-Basket auf Hyperliquid: Paper-Portfolio mit Verlauf, Orders mit Hebel, Positionen und Hedge.

- `index.html` – die Seite. Kurse kommen live per WebSocket direkt von Hyperliquid, Verlauf und Funding aus `data/market.json`.
- `refresh.py` – baut `data/market.json`: ein Jahr 8h-Kerzen und tägliches Funding (nur Python-Standardbibliothek).
- `.github/workflows/pages.yml` – lädt die Daten alle 8 Stunden neu und veröffentlicht die Seite auf GitHub Pages.
- `seed/market.json` – Startdaten, damit der erste Lauf nicht ein ganzes Jahr Funding nachladen muss.
- `bot/` – Rebalancing-Bot, siehe unten.

Einstellungen und Positionen liegen nur im Browser. Ein persönlicher Link (`…/#s=…`) überträgt sie auf ein anderes Gerät; der Teil nach `#` wird nie an einen Server geschickt.

Lokal testen:

```sh
python3 refresh.py data/market.json --prev seed/market.json
python3 -m http.server
```

## Rebalancing-Bot

`bot/` hält das Buch auf Ziel, aber nicht nach Uhrzeit: Er prüft jede Minute, ob der Abbau des Abweichungsrisikos die
erwarteten Handelskosten übersteigt, und handelt nur dann. Ausgeführt wird zuerst als Maker (Post-Only, nachgezogen),
was nach 90 Sekunden offen ist, als IOC mit Preisgrenze. Füllt sich eine Seite deutlich früher, zieht er die andere sofort nach.

Probemodus: Paper-Konto mit 10.000 $, echte Kurse von Hyperliquid, keine echten Orders (Python 3.11+, keine Pakete nötig).

```sh
cp bot.example.toml bot.toml   # Gewichte und Einstellungen eintragen; bot.toml bleibt lokal
python3 -m bot --plan          # nur anzeigen, was er jetzt handeln würde
python3 -m bot                 # laufen lassen
```

Testnet oder Mainnet mit einer API-Wallet (darf handeln, aber nicht abheben):

```sh
pip install -r requirements-bot.txt
export HL_API_KEY=…            # Schlüssel der API-Wallet, nie in eine Datei im Repo
python3 -m bot --mode testnet  # Adresse des Kontos in bot.toml unter [account]
```

Eine Datei `STOP` im Ordner hält ihn vom Handeln ab. Tests: `python3 -m unittest discover -s tests`.

Rechenhilfe, keine Anlageberatung.
