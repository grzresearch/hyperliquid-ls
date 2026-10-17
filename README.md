# Hyperliquid Long/Short-Bot

Ein beta-neutraler Long/Short-Bot für Hyperliquid und eine Seite, die zeigt, was er tut: Kontowert als Kerzen, was er
heute gehandelt hat und warum, was er hält, Hedge und VaR, und wie der Screener die Longs gewählt hat.

- **Longs:** Ein Screener wählt einmal am Tag die 7 Token mit dem besten Score aus Holder-Rendite (35 %), Revenue- und
  Fee-Rendite (je 15 %), Momentum gegen ETH und Cashflow-Wachstum (je 10 %), Katalysatoren (Releases auf GitHub,
  angenommene Snapshot-Abstimmungen), Beschleunigung und Volumen-Wachstum (je 5 %). Die Gewichte kommen aus einem
  Backtest (`backtest/`). Funding ist kein Kaufgrund: Über 25 % p. a. (7-Tage-Mittel) verkleinert es die Position,
  ab 100 % kauft er nicht. Liquidität ist ein Filter (Testorder von 2.500 $ höchstens 25 bp). Token-Unlocks zählen nicht:
  Ein Unlock allein drückt den Kurs nicht zuverlässig, erst Zuflüsse zu Börsen zeigen Verkaufsdruck, und die gibt es
  nicht frei. Ein gehaltener Long bleibt, solange er knapp hinter den Besten liegt.
- **Shorts:** Mit jedem Screener-Lauf die 6 bis 10 liquiden Perps, mit denen der Long-Basket über 90 Tage am wenigsten
  schwankt (Minimum-Varianz, Kovarianz geschrumpft zu einem Ein-Faktor-Modell), so bemessen, dass das Buch zu ETH
  beta-neutral ist. Höchstens 25 % in einem Token, weniger bei starkem Schwanken (Squeeze-Risiko) oder teurem Funding;
  HYPE nie. Die bisherigen Shorts bleiben, solange sie höchstens 15 % schlechter absichern. Gelingt die Auswahl nicht,
  bleibt die letzte; gibt es keine, sichert ETH allein ab, bis der nächste Versuch klappt. Der alte feste Korb aus ETH
  und Alts ist noch da (`[hedge] method = "fixed"`).
- **Risiko:** Fat Tails, Volatilitäts-Cluster und Regime (`bot/risk.py`, getestet in `backtest/README.md`).
  Keine Position darf an ihrem schlimmsten eigenen Tag des letzten Jahres (über das hinaus, was ETH erklärt) mehr als
  5 % des Kontos verlieren: Ein Long darüber gibt den Rest an die anderen Longs, ein Short sein Beta an ETH; liegt eine
  gehaltene Position mehr als 5 % über ihrem Limit, handelt der Bot sie sofort zurück. GARCH(1,1) und ein
  Hidden-Markov-Modell mit zwei Regimen schätzen aus dem Jahr des heutigen Buchs die Volatilität von morgen; liegt sie
  über der üblichen, wird das Buch kleiner (bis auf die Hälfte, nie größer). Liegt das gehaltene Buch mehr als 10 %
  neben seiner Zielgröße, bringt er es als Ganzes dorthin; darunter entscheidet die Kostenregel. VaR und ES kommen aus
  gefilterter historischer Simulation: jeder Tag des Jahres, auf die Volatilität von morgen umgerechnet.
- **Rebalancing:** kein fester Takt. Bei jeder Prüfung vergleicht der Bot die Risikokosten der Abweichung vom Ziel,
  `Risikoaversion ÷ (2 × Kontowert) × σ²`, mit den Handelskosten, die er vorher auf dem Live-Orderbuch simuliert
  (Implementation Shortfall nach Perold, Methode aus
  [implementation-shortfall-hyperliquid](https://github.com/gelatotrade/implementation-shortfall-hyperliquid)).
  Er handelt nur, was mehr Risiko abbaut, als es kostet. Die Risikokosten rechnet er mit der Volatilität von morgen:
  In unruhigen Phasen handelt er früher zurück, in ruhigen lässt er mehr laufen.
- **Ausführung:** zuerst als Maker (Post-Only), der Rest nach dem Maker-Fenster als IOC mit Preisgrenze. Nach jedem
  Trade vergleicht er die simulierten mit den tatsächlichen Kosten.

## Dateien

- `bot/` – der Bot, im Paper-Modus nur Python-Standardbibliothek (3.11+)
  - `screener.py` – Auswahl der Longs aus Hyperliquid, DefiLlama, GitHub und Snapshot
  - `hedge.py` – Auswahl der Shorts: Minimum-Varianz-Hedge für den Long-Basket
  - `risk.py` – Sprunglimits, GARCH, Hidden-Markov-Modell, VaR und ES mit Fat Tails
  - `model.py` – Zielbuch mit Sprunglimits, Entscheidungsregel
  - `costs.py` – Kostensimulation auf dem Orderbuch und realisierte Kosten
  - `broker.py` – Paper- und Live-Konto, Ausführung
  - `report.py` – Verlauf (Kontowert je Minute, Trades, Prüfungen) und Stand für die Seite
- `bot.paper.toml` – Konfiguration des öffentlichen Paper-Bots
- `bot.example.toml` – Vorlage für einen eigenen Lauf
- `index.html` – die Seite; Kurse kommen live per WebSocket von Hyperliquid, der Rest aus `data/bot.json` und `data/history.json`
- `.github/workflows/pages.yml` – lässt den Paper-Bot alle 15 Minuten eine Runde laufen und veröffentlicht danach die Seite
- `backtest/` – misst die Faktoren, die Varianten der Short-Seite und die Risiko-Schicht auf einem Jahr Daten;
  Ergebnis und Methode in `backtest/README.md`

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
python3 -m backtest --cache backtest/cache                   # Backtest; das erste Mal lädt er rund 15 Minuten Daten
python3 -m backtest.risk --cache backtest/cache              # Backtest der Risiko-Schicht (danach)
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
