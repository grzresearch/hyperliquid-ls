"""The risk layer on top of the covariance: fat tails, volatility clustering and regimes.

Everything here works on daily numbers taken from the bot's 8-hour closes and needs nothing but the standard library.

  jumps    the worst move of each coin within 24 hours over the last year, beyond what ETH explains: for a long the
           worst fall, for a short the worst rise. A position may lose at most jump_loss of the account on such a
           move (position_caps). Coins with less than a year of prices get at least a floor taken from the coins
           that have one, so a short, quiet history does not look safe.
  garch    GARCH(1,1) on the book's daily PnL as today's book would have made it over the last year, fitted by
           quasi maximum likelihood with variance targeting. Tomorrow's variance against the window's says how much
           riskier or calmer than usual the book is right now. Before a coin was listed it moves with its beta to
           ETH; the book's usual level is then taken from the days on which every sizeable coin had its own prices
           (blended with the whole year while there are fewer than 60 of them), so a young coin does not make the
           past look calm.
  tails    filtered historical simulation: each day of that year, divided by its GARCH volatility and multiplied by
           tomorrow's. VaR and ES keep the fat tails of the year without assuming a distribution, at today's level.
  regime   a two-state Gaussian hidden Markov model (calm, turbulent) on the same PnL: the probability of each state
           today and tomorrow, and the volatility each state stands for.
"""
import math
import statistics

NORMAL = statistics.NormalDist()


def _mean(v):
    return sum(v) / len(v)


def _beta(y, x):
    mx, my = _mean(x), _mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / vx if vx > 0 else 1.0


# ---------- daily numbers from 8-hour closes ----------
def daily_returns(closes, per, days):
    """Non-overlapping daily simple returns over the last `days` days, the last one ending with the last close.
    None where a close is missing."""
    idx = list(range(len(closes) - 1, -1, -per))[::-1]
    out = [closes[b] / closes[a] - 1 if closes[a] and closes[b] else None for a, b in zip(idx, idx[1:])]
    return out[-days:]


def filled(r, eth):
    """Daily returns r with the days before the coin was listed (None) filled with its beta times ETH's return.
    Drawing its own residuals for those days as well made the forecasts 12 to 15 % too high in the backtest (the
    first weeks after a listing are the wildest); beta times ETH alone kept them right on average. What the fill
    leaves out, the usual level makes up for: it counts the days with real prices (see pro_forma and usual)."""
    pairs = [(a, e) for a, e in zip(r, eth) if a is not None and e is not None]
    b = _beta([a for a, _ in pairs], [e for _, e in pairs]) if len(pairs) >= 20 else 1.0
    return [a if a is not None or e is None else b * e for a, e in zip(r, eth)]


def pro_forma(snap, exposure, days=365, skip=0, span=False):
    """Daily PnL in USD that the book `exposure` (USD per coin) would have made on each of the last `days` days,
    the last day ending `skip` 8h steps before the last close. Days before a coin was listed: see filled().
    span=True also returns, per day, whether every coin with at least 5 % of the book had its own price change."""
    per, C = int(snap.get("perDay", 3)), snap["coins"]
    cut = lambda v: v[:len(v) - skip] if skip else v        # noqa: E731
    eth = daily_returns(cut(C["ETH"]["closes"]), per, days)
    gross = sum(abs(v) for v in exposure.values() if v) or 1.0
    rets, own = {}, []
    for c, usd in exposure.items():
        if not usd or c not in C:
            continue
        r = daily_returns(cut(C[c]["closes"]), per, days)
        r = [None] * (len(eth) - len(r)) + r
        if abs(usd) >= 0.05 * gross:
            own.append([a is not None for a in r])
        rets[c] = filled(r, eth)
    pnl, mask = [], []
    for i, e in enumerate(eth):
        if e is None or any(r[i] is None for r in rets.values()):
            continue
        pnl.append(sum(exposure[c] * r[i] for c, r in rets.items()))
        mask.append(all(o[i] for o in own))
    return (pnl, mask) if span else pnl


def usual(values, mask=None, full=60):
    """The usual level of a series of variances: their mean over the days marked real, blended with the mean over
    all days while there are fewer than `full` real ones (so nothing jumps when a young coin turns 60 days old)."""
    every = _mean(values)
    real = [v for v, m in zip(values, mask or []) if m]
    if not mask or not real:
        return every
    w = min(1.0, len(real) / full)
    return w * _mean(real) + (1 - w) * every


# ---------- fat tails: how far a single coin can jump ----------
def jumps(snap, coins, days=365, young_days=180, floor_down=0.16, floor_up=0.31):
    """Per coin the worst fall and the worst rise within 24 hours (at the snapshot's 8h steps) over the last `days`
    days, net of the coin's beta times ETH's move. With less than young_days of prices the move is at least the floor
    (by default the upper quartile of the liquid perps that traded all year, measured October 2026).
    Returns {coin: {"down", "up", "days", "young"}}; ETH moves 0 against itself."""
    per, C = int(snap.get("perDay", 3)), snap["coins"]
    eth = C["ETH"]["closes"]
    n = len(eth)
    start = max(per, n - days * per)
    out = {}
    for c in coins:
        if c == "ETH":
            out[c] = {"down": 0.0, "up": 0.0, "days": (n - start) / per, "young": False}
            continue
        v = (C.get(c) or {}).get("closes") or []
        r, e = [], []
        for i in range(start, min(n, len(v))):
            if v[i] and v[i - per] and eth[i] and eth[i - per]:
                r.append(v[i] / v[i - per] - 1)
                e.append(eth[i] / eth[i - per] - 1)
        have = len(r) / per
        if len(r) >= 10 * per:
            b = _beta(r, e)
            idio = [a - b * x for a, x in zip(r, e)]
            down, up = max(0.0, -min(idio)), max(0.0, max(idio))
        else:
            down = up = 0.0
        young = have < young_days
        if young:
            down, up = max(down, floor_down), max(up, floor_up)
        out[c] = {"down": down, "up": up, "days": have, "young": young}
    return out


def position_caps(jump, equity, loss=0.05, side=None):
    """Largest position in USD per coin so that its worst move costs at most `loss` of the account. side: coin ->
    +1 long, -1 short (a long fears the fall, a short the rise). No cap where the coin does not move on its own, and
    none at all with loss <= 0 (the limit switched off)."""
    if not loss or loss <= 0 or equity <= 0:
        return {}
    out = {}
    for c, j in jump.items():
        move = j["down"] if (side or {}).get(c, 1) > 0 else j["up"]
        out[c] = loss * equity / move if move > 1e-9 else math.inf
    return out


def clipped(x, k=5.0):
    """x with every value beyond k robust standard deviations (1.4826 x the median absolute deviation) from the
    median pulled back to that bound: one jump must not decide what the calm and the restless days look like."""
    if len(x) < 3:
        return list(x)
    med = statistics.median(x)
    s = 1.4826 * statistics.median(abs(v - med) for v in x)
    if s <= 0:
        return list(x)
    return [min(med + k * s, max(med - k * s, v)) for v in x]


# ---------- volatility clustering: GARCH(1,1) ----------
def garch_fit(x, clip=5.0, ref_mask=None):
    """GARCH(1,1) by quasi maximum likelihood with variance targeting (omega = V (1 - a - b), V the sample variance
    of the demeaned series). The parameters come from x with its jumps clipped (clip robust sds, None: as is); the
    variances then run over x as it is, so a jump still raises the next day's. Returns a, b, the conditional
    variances h of every day, `next` (tomorrow's), `v_ref` (their usual level, see usual(); ref_mask marks the days
    with real prices) and
    `v_long`, the variance the model returns to without jumps (the jumps are the tails' business)."""
    n = len(x)
    m = _mean(x)
    e = [v - m for v in x]
    V = sum(v * v for v in e) / n if n else 0.0
    if n < 30 or V <= 0:
        return {"a": 0.0, "b": 0.0, "v_long": V, "v_ref": V, "mean": m, "h": [V] * n, "next": V, "n": n}
    fit_e = [v - m for v in clipped(x, clip)] if clip else e
    Vf = sum(v * v for v in fit_e) / n

    def nll(a, b):
        w, h, s = Vf * (1 - a - b), Vf, 0.0
        for v in fit_e:
            s += math.log(h) + v * v / h
            h = w + a * v * v + b * h
        return s

    best = (nll(0.0, 0.0), 0.0, 0.0)
    for a in (0.03, 0.06, 0.1, 0.15, 0.2, 0.3):
        for b in (0.5, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97):
            if a + b < 0.995:
                f = nll(a, b)
                if f < best[0]:
                    best = (f, a, b)
    f, a, b = best
    step = 0.04
    while step > 1e-4:                     # pattern search inside a >= 0, b >= 0, a + b <= 0.995
        moved = False
        for da, db in ((step, 0), (-step, 0), (0, step), (0, -step), (step, -step), (-step, step)):
            na, nb = a + da, b + db
            if na < 0 or nb < 0 or na + nb > 0.995:
                continue
            g = nll(na, nb)
            if g < f - 1e-12:
                f, a, b, moved = g, na, nb, True
        if not moved:
            step /= 2
    w, h, hs = Vf * (1 - a - b), Vf, []
    for v in e:
        hs.append(h)
        h = w + a * v * v + b * h
    return {"a": a, "b": b, "v_long": Vf, "v_ref": usual(hs, ref_mask), "mean": m, "h": hs, "next": h, "n": n}


def garch_ahead(fit, days):
    """Mean variance over the next `days` days: the forecast decays toward the long-run variance at rate a + b."""
    p, V, h1 = fit["a"] + fit["b"], fit["v_long"], fit["next"]
    return sum(V + p ** k * (h1 - V) for k in range(days)) / days


# ---------- fat tails at today's volatility: filtered historical simulation ----------
def tail_risk(x, fit, sigma=None, levels=(0.95, 0.99)):
    """VaR and ES of tomorrow as positive losses, from every day of x divided by its GARCH volatility and multiplied
    by tomorrow's, `sigma` (filtered historical simulation), next to what a normal distribution with the same
    volatility says. Also the rescaled days themselves (for a histogram)."""
    sig = sigma if sigma is not None else math.sqrt(fit["next"])
    z = sorted((v - fit["mean"]) / math.sqrt(h) for v, h in zip(x, fit["h"]) if h > 0)
    if not z:
        return None
    out = {"sigma": sig, "samples": [v * sig for v in z], "n": len(z)}
    for p in levels:
        k = max(1, round((1 - p) * len(z)))
        tail = z[:k]
        q = NORMAL.inv_cdf(p)
        out[f"{int(round(p * 100))}"] = {"hist": -tail[-1] * sig, "es_hist": -_mean(tail) * sig,
                                         "param": q * sig, "es_param": sig * NORMAL.pdf(q) / (1 - p)}
    return out


# ---------- regimes: two-state Gaussian hidden Markov model ----------
def hmm_fit(x, iters=300, tol=1e-7, clip=5.0, ref_mask=None):
    """Two-state Gaussian hidden Markov model by Baum-Welch. State 0 is the calm one (lower volatility). The states
    are fitted on x with its jumps clipped (a single jump would otherwise become a state of its own); the
    probabilities then run over x as it is. Returns means, sds, the transition matrix P (P[i][j]: from i to j), the
    probability of the turbulent state for every day given the data up to it (filtered), today's and tomorrow's
    state probabilities and tomorrow's variance."""
    raw = list(x)
    x = clipped(raw, clip) if clip else raw
    n = len(x)
    m, sd = _mean(x), statistics.pstdev(x)
    if n < 30 or sd <= 0:
        return None
    mu, sg = [m, m], [0.7 * sd, 1.6 * sd]
    P = [[0.95, 0.05], [0.10, 0.90]]
    pi = [0.5, 0.5]
    floor = 0.05 * sd
    last = -math.inf

    def scaled(v):
        """Both states' densities divided by the larger one (log space: a day far out in the tails must not
        underflow to 0 in both), and the log of that divisor."""
        lg = [-0.5 * ((v - mu[k]) / sg[k]) ** 2 - math.log(sg[k] * math.sqrt(2 * math.pi)) for k in (0, 1)]
        top = max(lg)
        return (math.exp(lg[0] - top), math.exp(lg[1] - top)), top

    for _ in range(iters):
        both = [scaled(v) for v in x]
        B = [b for b, _ in both]
        al, cs = [], []
        for t in range(n):
            if t == 0:
                a = [pi[k] * B[0][k] for k in (0, 1)]
            else:
                p = al[-1]
                a = [(p[0] * P[0][k] + p[1] * P[1][k]) * B[t][k] for k in (0, 1)]
            c = sum(a) or 1e-300
            al.append([a[0] / c, a[1] / c])
            cs.append(c)
        be = [[1.0, 1.0] for _ in range(n)]
        for t in range(n - 2, -1, -1):
            nb = be[t + 1]
            be[t] = [sum(P[j][k] * B[t + 1][k] * nb[k] for k in (0, 1)) / cs[t + 1] for j in (0, 1)]
        g = []
        for t in range(n):
            u = [al[t][k] * be[t][k] for k in (0, 1)]
            s = sum(u) or 1e-300
            g.append([u[0] / s, u[1] / s])
        xi = [[0.0, 0.0], [0.0, 0.0]]
        for t in range(n - 1):
            for j in (0, 1):
                for k in (0, 1):
                    xi[j][k] += al[t][j] * P[j][k] * B[t + 1][k] * be[t + 1][k] / cs[t + 1]
        pi = g[0]
        for j in (0, 1):
            row = sum(xi[j]) or 1e-300
            P[j] = [xi[j][0] / row, xi[j][1] / row]
        for k in (0, 1):
            w = sum(gt[k] for gt in g) or 1e-300
            mu[k] = sum(gt[k] * v for gt, v in zip(g, x)) / w
            sg[k] = max(floor, math.sqrt(sum(gt[k] * (v - mu[k]) ** 2 for gt, v in zip(g, x)) / w))
        ll = sum(math.log(c) + top for c, (_, top) in zip(cs, both))
        if ll - last < tol * abs(ll):
            break
        last = ll
    if sg[0] > sg[1]:                      # state 0 is the calm one
        mu, sg, P = mu[::-1], sg[::-1], [P[1][::-1], P[0][::-1]]
    fit = {"mu": mu, "sd": sg, "P": P}
    prob, filtered, predicted = 0.5, [], []
    for v in raw:                          # filtered over the data as it is, jumps included
        predicted.append(hmm_variance(fit, prob))
        prob = hmm_step(fit, prob, v)
        filtered.append(prob)
    today = [1 - prob, prob]
    ahead = [today[0] * P[0][k] + today[1] * P[1][k] for k in (0, 1)]
    mean_next = ahead[0] * mu[0] + ahead[1] * mu[1]
    var_next = sum(ahead[k] * (sg[k] ** 2 + mu[k] ** 2) for k in (0, 1)) - mean_next ** 2
    return {"mu": mu, "sd": sg, "P": P, "filtered": filtered, "today": today, "ahead": ahead,
            "next": var_next, "stay": [P[0][0], P[1][1]],
            "v_ref": usual(predicted[30:], ref_mask[30:] if ref_mask else None) if n > 30 else _mean(predicted)}


def hmm_step(fit, prob, v):
    """Update the probability of the turbulent state with one more day's PnL v (filtering without refitting).
    In log space, so a day far beyond both states (a squeeze of several hundred percent) counts as turbulent."""
    mu, sg, P = fit["mu"], fit["sd"], fit["P"]
    prior = [(1 - prob) * P[0][k] + prob * P[1][k] for k in (0, 1)]
    lg = [math.log(max(prior[k], 1e-300)) - 0.5 * ((v - mu[k]) / sg[k]) ** 2 - math.log(sg[k]) for k in (0, 1)]
    top = max(lg)
    w = [math.exp(x - top) for x in lg]
    return w[1] / (w[0] + w[1])


def hmm_variance(fit, prob):
    """Tomorrow's variance given today's probability of the turbulent state."""
    mu, sg, P = fit["mu"], fit["sd"], fit["P"]
    ahead = [(1 - prob) * P[0][k] + prob * P[1][k] for k in (0, 1)]
    m = ahead[0] * mu[0] + ahead[1] * mu[1]
    return sum(ahead[k] * (sg[k] ** 2 + mu[k] ** 2) for k in (0, 1)) - m * m


# ---------- all of it for one book ----------
def outlook(pnl, mask=None):
    """Tomorrow's risk of a book from its pro-forma daily PnL. The variance is the mean of the GARCH and the HMM
    forecasts. `ref` is the same mean over the past year, the book's usual level (mask: the days with real prices,
    see pro_forma and usual). The tails come from the year at tomorrow's volatility. Too short or flat a history
    gives a neutral outlook: var 0, nothing else."""
    if len(pnl) < 30 or statistics.pvariance(pnl) <= 0:
        return {"var": 0.0, "ref": 0.0, "days": len(pnl), "tails": None, "garch": None, "regime": None}
    g = garch_fit(pnl, ref_mask=mask)
    hm = hmm_fit(pnl, ref_mask=mask)
    var = (g["next"] + hm["next"]) / 2 if hm else g["next"]
    ref = (g["v_ref"] + hm["v_ref"]) / 2 if hm else g["v_ref"]
    out = {"var": var, "ref": ref, "days": len(pnl), "tails": tail_risk(pnl, g, sigma=math.sqrt(var)),
           "garch": {"a": g["a"], "b": g["b"], "next": g["next"], "ref": g["v_ref"]}, "regime": None}
    if hm:
        out["regime"] = {"turbulent": hm["today"][1], "tomorrow": hm["ahead"][1], "sd": hm["sd"], "stay": hm["stay"],
                         "next": hm["next"], "recent": [round(p, 3) for p in hm["filtered"][-90:]]}
    return out


def book_outlook(snap, book, days=365):
    """outlook() of a book on each of the daily grids its 8h closes allow (days ending at the last close, 8 and 16
    hours before), with variance and usual level averaged: a new close then moves the forecast by a third of a step,
    not all of it. Regime and tails come from the grid that ends now, scaled to the averaged volatility."""
    per = int(snap.get("perDay", 3))
    views = [outlook(*pro_forma(snap, book, days, skip, span=True)) for skip in range(per)]
    good = [v for v in views if v["var"] > 0]
    if not good:
        return views[0]
    now = good[0]                           # the grid that ends now, unless it had too little to say
    now["var"] = _mean([v["var"] for v in good])
    now["ref"] = _mean([v["ref"] for v in good])
    if now["tails"]:
        k = math.sqrt(now["var"]) / now["tails"]["sigma"]
        t = now["tails"]
        t["sigma"] *= k
        t["samples"] = [v * k for v in t["samples"]]
        for lvl in ("95", "99"):
            t[lvl] = {key: v * k for key, v in t[lvl].items()}
    return now
