# Citrini Long/Short – live

Dashboard für den beta-neutralen Long/Short-Basket auf Hyperliquid: Paper-Portfolio mit Verlauf, Orders mit Hebel, Positionen und Hedge.

- `index.html` – die Seite. Kurse kommen live per WebSocket direkt von Hyperliquid, Verlauf und Funding aus `data/market.json`.
- `refresh.py` – baut `data/market.json`: ein Jahr 8h-Kerzen und tägliches Funding (nur Python-Standardbibliothek).
- `.github/workflows/pages.yml` – lädt die Daten alle 8 Stunden neu und veröffentlicht die Seite auf GitHub Pages.
- `seed/market.json` – Startdaten, damit der erste Lauf nicht ein ganzes Jahr Funding nachladen muss.

Einstellungen und Positionen liegen nur im Browser. Ein persönlicher Link (`…/#s=…`) überträgt sie auf ein anderes Gerät; der Teil nach `#` wird nie an einen Server geschickt.

Lokal testen:

```sh
python3 refresh.py data/market.json --prev seed/market.json
python3 -m http.server
```

Rechenhilfe, keine Anlageberatung.
