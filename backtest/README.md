# Backtest

```sh
python3 -m backtest --cache backtest/cache              # lädt einmal alles (rund 15 Minuten), danach aus dem Cache
python3 -m backtest --cache backtest/cache --offline    # nur aus dem Cache, lädt nichts nach
python3 -m backtest.risk --cache backtest/cache         # die Risiko-Schicht, braucht den Cache und report.json
```

Die Ergebnisse landen in `backtest/report.json` und `backtest/risk.json`; die der Läufe vom 16. und 17.10.2026
liegen bei. Die Gewichte und die Risiko-Einstellungen in `bot.paper.toml` stammen daraus.

## Ergebnis vom 16.10.2026

24 Stichtage vom 22.10.2025 bis 9.9.2026, je Stichtag 17 bis 32 Token im Universum (Median 23). Von 234 Perps, die es
auf Hyperliquid gibt oder gab, ließen sich 120 einem Protokoll auf DefiLlama zuordnen.

### Faktoren

Rank IC: Rangkorrelation zwischen Faktor und Rendite der nächsten 30 Tage gegen ETH, Mittel über die Stichtage.
t aus jedem zweiten Stichtag (28 Tage Abstand, die 30-Tage-Fenster überlappen sonst), deshalb kann es ein anderes
Vorzeichen haben als das Mittel über alle.

| Faktor | Rank IC | t | IC > 0 | Ausgangsgewicht | Gewicht jetzt |
|---|---:|---:|---:|---:|---:|
| Holder-Rendite | +0,153 | +2,0 | 79 % | 20 % | 35 % |
| Fee-Rendite | +0,096 | +1,7 | 67 % | 5 % | 15 % |
| Revenue-Rendite | +0,091 | +1,6 | 54 % | 10 % | 15 % |
| Beschleunigung | +0,024 | −0,1 | 58 % | 5 % | 5 % |
| Katalysatoren | +0,011 | −0,4 | 50 % | 10 % | 5 % |
| Cashflow-Wachstum | −0,021 | −0,1 | 33 % | 20 % | 10 % |
| Volumen-Wachstum (nur 8 Stichtage) | −0,049 | – | 38 % | 5 % | 5 % |
| Momentum | −0,093 | −1,7 | 42 % | 25 % | 10 % |
| Funding, nur gemessen (12 Stichtage) | +0,006 | +0,1 | 42 % | – | – |

Über den längeren Zeitraum seit 29.11.2024 (47 Stichtage, nur Faktoren ohne Market Cap, deshalb ohne den Filter auf
30 Mio. $): Momentum −0,012, Cashflow-Wachstum +0,035, Katalysatoren +0,047, Beschleunigung −0,043. Momentum war im
letzten Jahr also eher ein Signal für Gegenbewegungen, über den längeren Zeitraum neutral; Cashflow-Wachstum und
Katalysatoren schnitten dort etwas besser ab.

Die Ausgangsgewichte sind der Vorschlag, mit dem der Backtest begann. Live liefen bis dahin andere: Funding und
Liquidität zählten je 10 % im Score, Holder-Rendite nur 5 %, Momentum 20 %, Fee- und Revenue-Wachstum zusammen 22 %.
Die neuen Gewichte sind zur Hälfte die gemessenen (proportional zum positiven Verhältnis von IC zu seiner Streuung:
Holder 47 %, Fee 23 %, Revenue 22 %, Beschleunigung 6 %, Katalysatoren 3 %, der Rest 0) und zur Hälfte die
Ausgangsgewichte, gerundet auf 5 %.

Kombiniert und außerhalb der Stichprobe (an jedem Stichtag nur mit Gewichten aus Stichtagen, die mindestens 30 Tage
früher lagen): die besten 7 schlugen alle Token des Universums im Mittel um 3,8 % in 30 Tagen (an 75 % der Stichtage,
t +3,0), mit den Ausgangsgewichten um 2,2 % (71 %, t +1,5). An den ersten 8 Stichtagen fehlte noch der Vorlauf, die
Gewichte waren dieselben; danach waren die gemessenen an 10 von 16 Stichtagen besser und an 5 schlechter, im Mittel
um 2,5 Prozentpunkte in 30 Tagen.

### Short-Seite

An 21 Stichtagen die 7 besten Longs nach den Ausgangsgewichten, gleich gewichtet, dazu jede Absicherung, geschätzt auf
den 90 Tagen davor. Gemessen an den 30 Tagen danach: die Schwankung des abgesicherten Buchs (pro Jahr) und was die
Shorts je 1 $ Long verdient haben.

| Absicherung | Restvolatilität | Median | ruhiger als der feste Korb | Ertrag der Shorts |
|---|---:|---:|---:|---:|
| fester Korb: 40 % ETH, Rest SUI, AVAX, ADA, TAO, XPL, MORPHO | 33,8 % | 31,5 % | – | +1,1 % |
| **wide**: 6 bis 10 Perps, je höchstens 25 %, Schrumpfung 0,6 | 34,1 % | 31,2 % | 48 % | +0,6 % |
| minvar: 4 bis 8 Perps, je höchstens 35 %, Schrumpfung 0,3 | 34,2 % | 33,0 % | 48 % | +0,0 % |
| ETH und die 6 liquidesten Alts | 34,6 % | 34,0 % | 38 % | +0,8 % |
| shrink60: wie minvar, Schrumpfung 0,6 | 34,7 % | 31,3 % | 33 % | +0,7 % |
| wide mit Short-Score (10 % Spielraum) | 37,6 % | 37,6 % | 38 % | −1,3 % |
| minvar mit Short-Score (10 % Spielraum) | 37,9 % | 38,8 % | 38 % | −0,6 % |
| nur ETH | 43,2 % | 40,1 % | 10 % | +1,2 % |
| ohne Absicherung | 68,5 % | 67,1 % | – | – |

wide auf 180 statt 90 Tagen geschätzt: Nur an 10 Stichtagen gab es genug Kurse. An denselben 10 Stichtagen 36,7 %
gegen 37,4 % mit 90 Tagen, besser an 4 der 10; zu wenig für ein Urteil.

Was daraus folgt:

- **Live: wide ohne Short-Score.** Die Minimum-Varianz-Auswahl sichert so gut ab wie der feste Korb (34,1 gegen
  33,8 %, Unterschied im Rauschen, t +0,3) und mit dem niedrigsten Median. Der feste Korb ist dabei mit heutigem Wissen
  zusammengestellt, das begünstigt ihn im Test; die Auswahl dagegen passt sich jeden Tag den Longs an. Im Mittel
  6 Perps je Buch, am häufigsten ETH, SOL, ADA, LINK, SUI und XRP.
- **Short-Score aus.** Er hebt die Restvolatilität um 3,5 bis 3,6 Prozentpunkte, und seine Shorts verdienten weniger
  statt mehr: Die Token mit schwachem Momentum, die er bevorzugt, liefen danach eher besser (passend zum negativen
  Momentum-IC). Einschalten lässt er sich mit `tilt_slack` in `[hedge]`.
- **90 Tage Fenster** bleiben, bis es genug Stichtage für den Vergleich mit 180 Tagen gibt.

### Risiko-Schicht: Fat Tails, Volatilitäts-Cluster, Regime

`backtest/risk.py` baut an jedem Stichtag das Buch wie der Bot (die 7 besten Longs nach den Ausgangsgewichten, gleich
gewichtet, Short-Seite „wide“) und hält es 14 Tage; aneinandergereiht sind das 308 Handelstage vom 22.10.2025 bis
22.9.2026. An 2 der 24 Stichtage (19.11. und 3.12.2025) hatte ein Long noch keine 60 Tage Kurse für den Hedge; dort
fehlen je 14 Tage. Jede Prognose kennt nur die Tage davor. Vor der Listung eines Coins rechnet das Buch wie im Bot
mit seinem Beta zu ETH (`bot/risk.filled`); die übliche Volatilität zählt die Tage, an denen alle Positionen ab 5 %
des Buchs eigene Kurse hatten, gemischt mit dem ganzen Jahr, solange es weniger als 60 sind. Die Dollar-Beträge
bleiben jeweils 14 Tage gleich, als würde täglich ohne Kosten zurückgehandelt; die Sprunglimits halten damit an
jedem Tag.

Wie gut sagen die Modelle die Schwankung des nächsten Tages voraus? QLIKE ist der übliche Verlust für
Varianzprognosen (kleiner ist besser), hier als Abstand zum 90-Tage-Fenster, mit dem der Bot bisher rechnete; t mit
Newey-West-Fehlern, weil benachbarte Tage nicht unabhängig sind. Realisiert durch prognostiziert: 1 heißt im Mittel
richtig. VaR 99 %: an wie vielen Tagen der Verlust darüber lag; richtig wären 1 %, also rund 3 der 308 Tage.

| Prognose | QLIKE gegen Fenster | t | realisiert / Prognose | über VaR 99 %, normal | mit Fat Tails |
|---|---:|---:|---:|---:|---:|
| 90-Tage-Fenster (bisher) | – | – | 1,02 | 1,3 % | – |
| EWMA (λ 0,94) | −0,084 | −1,8 | 0,93 | 1,3 % | – |
| GARCH(1,1) | −0,087 | −1,9 | 1,00 | 1,9 % | 1,6 % |
| Hidden-Markov-Modell, 2 Regime | −0,110 | −2,7 | 0,98 | 1,6 % | 1,3 % |
| **Mittel aus GARCH und HMM** (Bot) | −0,104 | −2,4 | 0,98 | 2,3 % | 1,6 % |

- Alle Modelle sagen das Risiko des nächsten Tages besser voraus als das Fenster; deutlich ist es beim HMM und beim
  Mittel aus beiden. Das Mittel ist besser als GARCH allein (t −2,2) und vom HMM (t −0,8) und von EWMA (t −0,7)
  nicht zu unterscheiden. Der Bot nimmt das Mittel, weil es nicht davon abhängt, welches Modell gerade danebenliegt.
- Beide Modelle werden mit gekappten Ausreißern geschätzt (5 robuste Standardabweichungen) und laufen dann über die
  echten Tage. Für GARCH macht das kaum einen Unterschied (−0,087 gegen −0,090 ungekappt); ungekappt machte das HMM
  aus einzelnen Sprungtagen aber ein eigenes „Regime“ von einem Tag Dauer und sagte schlechter voraus (−0,099).
- Die Tage vor einer Listung mit eigenen, zufällig gezogenen Schwankungen des Coins aufzufüllen, machte die
  Prognosen 12 bis 15 % zu hoch (die ersten Wochen nach einer Listung sind die wildesten). Mit dem Beta zu ETH
  allein stimmen sie im Mittel; dafür zählt die übliche Volatilität vor allem Tage mit echten Kursen.
- Fat Tails: Gefilterte historische Simulation (jeder Tag des Jahres, durch seine damalige Volatilität geteilt, mit
  der von morgen malgenommen) senkt die Überschreitungen des VaR 99 % von 1,9 auf 1,6 % (GARCH) bzw. von 2,3 auf
  1,6 % (Mittel). Bei 3 erwarteten Tagen ist das eine Richtung, kein Beweis.

Was hätte es gebracht, danach zu handeln? Rendite auf das Konto bei einem Long-Book von 1,5 × Konto, vor den Kosten
des 14-täglichen Umbaus, die alle Varianten gleich tragen; jede Größenänderung kostet 3,5 bp je gehandeltem Dollar.
„Täglich“ heißt: die Größe folgt der Prognose jeden Tag; „ab 10 %“: erst wenn das Buch mehr als 10 % neben der
Zielgröße liegt, wie im Bot. Im Bot greift das Band auch, wenn Kursbewegungen das Buch um mehr als 10 % verschieben;
im Backtest bleiben die Beträge gleich.

| Buch | Rendite p. a. | Volatilität | Sharpe | schlechtester Tag | ES 99 % | max. Drawdown | Größe im Mittel |
|---|---:|---:|---:|---:|---:|---:|---:|
| wie bisher | +90 % | 56 % | 1,61 | −11,1 % | −9,7 % | −30,6 % | 100 % |
| mit Sprunglimits | +95 % | 55 % | 1,73 | −11,8 % | −9,5 % | −28,8 % | 100 % |
| **+ Vol-Targeting GARCH und HMM, ab 10 %** (Bot) | +84 % | 49 % | 1,73 | −9,3 % | −7,7 % | −28,2 % | 94 % |
| + Vol-Targeting GARCH und HMM, täglich | +85 % | 49 % | 1,74 | −8,5 % | −7,5 % | −28,9 % | 95 % |
| + Vol-Targeting nur GARCH, täglich | +86 % | 49 % | 1,76 | −8,5 % | −7,5 % | −29,1 % | 94 % |
| + Vol-Targeting nur HMM, täglich | +85 % | 49 % | 1,72 | −8,5 % | −7,5 % | −28,7 % | 95 % |
| + Vol-Targeting EWMA, täglich | +86 % | 48 % | 1,80 | −8,6 % | −7,5 % | −28,7 % | 92 % |
| + Vol-Targeting GARCH bis 1,5 × Buch | +87 % | 53 % | 1,65 | −8,5 % | −7,8 % | −34,3 % | 104 % |
| + Regime-Schalter (HMM unruhig: kleineres Buch) | +60 % | 41 % | 1,47 | −6,9 % | −6,2 % | −25,8 % | 81 % |

- **Sprunglimits** (kein Long und kein Short darf an seinem schlimmsten eigenen Tag des Vorjahres mehr als 5 % des
  Kontos verlieren): griffen an 8 der 22 Stichtage bei einem Long und an 14 bei einem Short; ETH übernahm im Mittel
  4,7 % des Long-Books zusätzlich. Sie kosteten nichts, der Drawdown war etwas kleiner. Gegen den schlechtesten Tag
  halfen sie nicht: Am 4.6.2026 fiel der ganze Long-Korb um fast 13 %, verteilt über mehrere Token, während ETH
  nur 2 % verlor (−11,8 % des Kontos statt −11,1 %). Am zweitschlechtesten Tag, dem 28.1.2026, sprangen zwei
  einzelne Token (RESOLV long, WLD short); da sank der Verlust von −9,2 auf −5,5 % des Kontos. Sie schützen gegen
  Einzelsprünge, nicht gegen einen Absturz des ganzen Korbs. Den milderte das Vol-Targeting: Am 4.6. stand das HMM
  schon auf „unruhig“ und das Buch auf 79 %, der Verlust war −9,3 % statt −11,8 % des Kontos.
- **Vol-Targeting** (das Buch schrumpft, solange die erwartete Volatilität über der üblichen liegt, bis auf die
  Hälfte, nie größer): gleicher Sharpe, rund ein Fünftel kleinere schlechteste Tage und ES 99 %, bei im Mittel 94 %
  Größe. Weil schlechte Tage in unruhigen Phasen gehäuft kommen, ist das Buch dann schon kleiner. Mit dem Band von
  10 % fällt der Vorteil etwas kleiner aus als bei täglichem Anpassen, dafür handelt der Bot nicht wegen jeder
  kleinen Änderung der Prognose.
- **EWMA**, das einfachste Modell, kommt beim Anpassen der Größe auf dasselbe Ergebnis wie GARCH und HMM. Als
  Prognose war es etwas schwächer als ihr Mittel (t −0,7, also im Rauschen).
- **Hochskalieren** in ruhigen Phasen brachte nichts und vergrößerte den Drawdown: Der Bot macht das Buch nie größer.
- **Regime-Schalter:** Das HMM sagt die Volatilität am besten voraus, als harter Schalter kostete es aber Rendite
  und Sharpe. Das Regime fließt deshalb über die Volatilitätsprognose ein, nicht als Schalter.
- Ein Jahr Sharpe hat einen Standardfehler um 1: Unterschiede von 0,1 bis 0,2 sind Rauschen. Belastbar ist die
  kleinere Verlustseite, die alle Varianten mit Vol-Targeting zeigen.

Im Bot misst `bot/risk.py` die Sprünge auf 24 Stunden in 8-Stunden-Schritten statt auf Tagesschlusskursen; das findet
etwas größere Sprünge und ist damit strenger als der Backtest. Die Volatilitätsprognose mittelt er über die drei
Tageseinteilungen, die 8-Stunden-Kurse erlauben, damit ein neuer Kurs die Buchgröße nicht springen lässt.

## Was der Backtest macht

- **Stichtage:** alle 14 Tage über das letzte Jahr, solange es Market Caps gibt (CoinGecko liefert frei nur 365 Tage).
  Ein zweiter Lauf ohne die Faktoren, die eine Market Cap brauchen, reicht bis Ende November 2024 zurück.
- **Universum je Stichtag:** wie im Live-Screener. Perps auf Hyperliquid mit mindestens 2 Mio. $ Tagesvolumen
  (Mittel über 30 Tage, weil es keine historischen Orderbücher gibt), einem Protokoll auf DefiLlama und mindestens
  30 Mio. $ Market Cap. BTC, ETH und HYPE bleiben draußen. Perps, die Hyperliquid später delistet hat, sind dabei,
  sonst sähe der Test nur die Überlebenden.
- **Faktoren:** mit demselben Code wie live (`bot/screener.py`), aus Daten bis zum Stichtag. Fees, Revenue und
  Holder Revenue von DefiLlama, Kurse von Hyperliquid, Abstimmungen von Snapshot. GitHub-Releases fehlen (sie
  brauchen einen Token), Katalysatoren werden deshalb nur mit dem Snapshot-Teil gemessen.
- **Ziel:** die Rendite der nächsten 30 Tage gegen ETH, Beta herausgerechnet. Für jeden Faktor und Stichtag die
  Rangkorrelation zwischen Faktor und Ziel (Rank IC). Mittel, Streuung, Trefferquote, t-Wert aus den Stichtagen im
  Abstand von 28 Tagen.
- **Gewichte:** gemessene Gewichte proportional zum positiven Verhältnis von mittlerem IC zu seiner Streuung, zur
  Hälfte gemischt mit den Ausgangsgewichten. Mit 20 bis 30 Token je Stichtag ist der IC stark verrauscht; die Mischung
  verhindert, dass ein Jahr Zufall die Gewichte bestimmt.
- **Außerhalb der Stichprobe:** der kombinierte Score an jedem Stichtag mit Gewichten nur aus Stichtagen, die
  mindestens 30 Tage früher lagen; dazu die Rendite der besten 7 gegen alle.
- **Short-Seite:** an jedem Stichtag die 7 besten Longs (gleich gewichtet) und die Absicherungen aus der Tabelle oben
  (`HEDGE_VARIANTS` in `__main__.py`), jede auf den 90 (bzw. 180) Tagen davor geschätzt, gemessen an den 30 Tagen
  danach.

Grenzen: DefiLlama korrigiert Daten auch rückwirkend, der Test sieht also teils Werte, die es am Stichtag so noch nicht
gab. Market Caps vor CoinGeckos 365 Tagen und für Token, bei denen CoinGecko nicht antwortete, sind die heutige Market
Cap, skaliert mit dem Kurs. Mit gut 20 Token je Stichtag schwankt der IC von Stichtag zu Stichtag um ±0,2; ein
einzelner Stichtag sagt wenig, und ein Jahr ist ein Marktumfeld. Die Gewichte sind deshalb eine Richtung, keine
Konstante: Entscheiden muss das Paper-Konto, und nach einem weiteren Quartal lohnt ein neuer Lauf.

0xArchive: Mit einem Plan, der weiter als 30 Tage zurückreicht, könnte der Test historische Orderbücher statt des
Volumens als Liquiditätsfilter nehmen (`OXARCHIVE_API_KEY`, `archive()` in `data.py`). Der kostenlose Plan reicht dafür
nicht.
