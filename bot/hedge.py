"""The short side: which liquid perps hedge the long basket best, and how much of each.

Everything here works on daily simple returns over the hedge window, one dict coin -> return per row.

  basket   y = sum_i wL_i R_i                 the long book's return per dollar of longs
  hedge    H * sum_j w_j R_j                  a short book w (weights summing to 1, each at most max_share), H
                                              dollars of it per dollar of longs
  beta     H makes the book beta-neutral to ETH:  H * sum_j w_j beta_j = beta_y
  choice   names are added one at a time from the liquid perps (not the longs, not the excluded), 6 to 10 of them,
           each time the one that lowers the variance of y - H R w the most. The covariance is shrunk toward a
           one-factor model (ETH), so a few weeks of coincidence do not pick the hedge.
  tilt     off by default. With tilt_slack > 0 a short score (weak momentum, falling cash flows, ...) moves the book
           toward better shorts, as far as the hedged variance stays within tilt_slack of the best hedge. In the
           backtest (backtest/README.md) the tilt hedged worse and its shorts did not pay for it.
  churn    the current shorts stay while they hedge within keep_slack (15 %) of the new choice.
  funding  a short that pays more than funding_free a year (7-day mean) gets a smaller cap, 0 at funding_max.
  squeeze  a name more volatile than the typical candidate gets a smaller cap in proportion (twice the median
           volatility: half of max_share), so the short book does not lean on the coins that can jump the most.

The defaults are the settings that hedged best out of sample in the backtest ("wide" there).
"""
import itertools
import math

DEFAULTS = {"min_names": 6, "max_names": 10, "max_share": 0.25, "min_share": 0.05, "shrink": 0.6, "tilt_slack": 0.0,
            "keep_slack": 0.15, "min_beta": 0.3, "shortlist": 16, "funding_free": 0.25, "funding_max": 1.0,
            "holdout": 1 / 3, "vol_caps": True}


def funding_multiplier(cost, free=0.25, cap=1.0):
    """Size multiplier for a position that pays `cost` a year in funding (7-day mean): 1 up to `free`, then linearly
    down to 0 at `cap`. Receiving funding earns no bonus: it is a running cost, not a reason to hold something."""
    if cost is None or cost <= free:
        return 1.0
    return max(0.0, 1.0 - (cost - free) / (cap - free))


def _mean(v):
    return sum(v) / len(v)


def _cov(a, b):
    ma, mb = _mean(a), _mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (len(a) - 1)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def project(v, caps, floors=None):
    """Closest point to v with floors_i <= w_i <= caps_i and sum(w) = 1 (needs sum(floors) <= 1 <= sum(caps))."""
    floors = floors or [0.0] * len(v)
    lo, hi = min(v) - max(caps), max(v) - min(floors)
    for _ in range(60):
        tau = (lo + hi) / 2
        if sum(min(c, max(f, x - tau)) for x, c, f in zip(v, caps, floors)) > 1:
            lo = tau
        else:
            hi = tau
    w = [min(c, max(f, x - hi)) for x, c, f in zip(v, caps, floors)]
    s = sum(w)
    return [x / s for x in w] if s > 0 else [1 / len(v)] * len(v)


MIN_DAYS = 28       # shortest hedge window; the screener takes no long listed for less than 30 days


def window_rows(snap, longs, pool, days):
    """Daily simple returns at the snapshot's 8h steps over the last `days` days. The longs and ETH must cover the
    window (it shrinks to what they cover, down to MIN_DAYS); candidates without a complete history there drop out.
    Returns (rows, usable candidates)."""
    per, C = int(snap.get("perDay", 3)), snap["coins"]
    n = len(C["ETH"]["closes"])

    def complete(c, a):
        v = (C.get(c) or {}).get("closes") or []
        return len(v) == n and all(v[i] for i in range(a - per, n))
    need = list(longs) + ["ETH"]
    start = max(per, n - days * per)
    while start < n - MIN_DAYS * per and not all(complete(c, start) for c in need):
        start += per
    if not all(complete(c, start) for c in need):
        return [], []
    usable = [c for c in pool if c not in longs and complete(c, start)]
    names = list(dict.fromkeys(need + usable))
    rows = [{c: C[c]["closes"][i] / C[c]["closes"][i - per] - 1 for c in names} for i in range(start, n)]
    return rows, usable


def basket(rows, weights):
    return [sum(w * r[c] for c, w in weights.items()) for r in rows]


def realized_vol(rows, long_w, k):
    """Annualised volatility of the hedged book's daily return over rows: y - sum_j k_j R_j, with k the hedge in
    dollars per dollar of longs."""
    resid = [a - sum(h * r[c] for c, h in k.items()) for a, r in zip(basket(rows, long_w), rows)]
    m = _mean(resid)
    return math.sqrt(sum((x - m) ** 2 for x in resid) / (len(resid) - 1) * 365)


class Problem:
    """Shrunk covariances of the long basket and the candidate shorts over one window."""

    def __init__(self, rows, long_w, cands, shrink, market="ETH", pairs=True):
        y = basket(rows, long_w)
        cols = {c: [r[c] for r in rows] for c in cands}
        m = [r[market] for r in rows]
        vm = _cov(m, m)
        self.vy = _cov(y, y)
        self.beta_y = _cov(y, m) / vm
        self.beta = {c: _cov(cols[c], m) / vm for c in cands}
        dev = {c: [v - _mean(col) for v in col] for c, col in cols.items()}
        n1 = len(rows) - 1
        self.c = {c: (1 - shrink) * _cov(cols[c], y) + shrink * self.beta[c] * self.beta_y * vm for c in cands}
        self.Q = {}
        names = list(cands)
        for i, a in enumerate(names):
            for b in names[i:] if pairs else [a]:          # pairs=False: variances only, enough for beta and corr
                s = sum(x * z for x, z in zip(dev[a], dev[b])) / n1
                q = s if a == b else (1 - shrink) * s + shrink * self.beta[a] * self.beta[b] * vm
                self.Q[(a, b)] = self.Q[(b, a)] = q
        self.corr = {c: self.c[c] / math.sqrt(max(self.Q[(c, c)] * self.vy, 1e-30)) for c in cands}

    def optimize(self, names, caps, score=None, mu=0.0, w0=None, iters=500, floor=0.0):
        """Weights over `names` minimising  V/vy - mu * w.score  (V: variance of the beta-neutral hedged basket) by
        projected gradient steps. Returns (weights, objective, V, H)."""
        k = len(names)
        b = [self.beta[c] for c in names]
        cv = [self.c[c] for c in names]
        Q = [[self.Q[(x, z)] for z in names] for x in names]
        s = [(score or {}).get(c, 0.0) for c in names]
        vy = self.vy

        def evaluate(w):
            p = _dot(w, b)
            H = self.beta_y / p
            cw = _dot(w, cv)
            qv = [_dot(row, w) for row in Q]
            qw = _dot(w, qv)
            V = vy - 2 * H * cw + H * H * qw
            g = [(2 * H * cw / p * b[i] - 2 * H * cv[i] - 2 * H * H * qw / p * b[i] + 2 * H * H * qv[i]) / vy
                 - mu * s[i] for i in range(k)]
            return V / vy - mu * _dot(w, s), g, V, H

        floors = [min(floor, cp) for cp in caps] if floor * k <= 1 else None
        w = project(w0 if w0 else [1 / k] * k, caps, floors)
        f, g, V, H = evaluate(w)
        eta = 1.0
        for _ in range(iters):
            moved = False
            for _ in range(50):
                trial = project([w[i] - eta * g[i] for i in range(k)], caps, floors)
                f2, g2, V2, H2 = evaluate(trial)
                if f2 < f - 1e-14:
                    moved = True
                    break
                eta *= 0.5
            if not moved:
                break
            gain = f - f2
            w, f, g, V, H = trial, f2, g2, V2, H2
            eta = min(eta * 2.0, 100.0)
            if gain < 1e-11:
                break
        return w, f, V, H

    def greedy(self, pool, caps, cfg, score=None, mu=0.0):
        """Add names one at a time, min_names to max_names of them. Start: the best of the smallest sets among the
        eight most correlated names whose caps can hold the whole short book."""
        order = sorted(pool, key=lambda c: -self.corr[c])[:int(cfg["shortlist"])]
        best = None
        floor = float(cfg.get("min_share", 0.0))
        for size in range(3, int(cfg["max_names"]) + 1):
            for start in itertools.combinations(order[:max(8, size)], size):
                if sum(caps[c] for c in start) < 1:
                    continue
                w, f, V, H = self.optimize(list(start), [caps[c] for c in start], score, mu, floor=floor)
                if best is None or f < best[1]:
                    best = (list(start), f, w, V, H)
            if best is not None:
                break
        if best is None:
            return None
        names, f, w, V, H = best
        while len(names) < int(cfg["max_names"]):
            step = None
            for c in order:
                if c in names:
                    continue
                nm = names + [c]
                w2, f2, V2, H2 = self.optimize(nm, [caps[x] for x in nm], score, mu, w0=w + [0.0], floor=floor)
                if step is None or f2 < step[1]:
                    step = (nm, f2, w2, V2, H2)
            if step is None:
                break
            if len(names) >= int(cfg["min_names"]) and f - step[1] < 0.005:   # under half a percent of y's variance
                break
            names, f, w, V, H = step
        return {"names": names, "w": w, "f": f, "V": V, "H": H}


def caps_for(window, pool, cfg, cost):
    """Largest share of the short book per coin: max_share, smaller for expensive funding and for a coin that swings
    more than the median candidate."""
    def vol(c):
        col = [r[c] for r in window]
        m = _mean(col)
        return math.sqrt(sum((x - m) ** 2 for x in col) / (len(col) - 1))
    vols = {c: vol(c) for c in pool}
    ref = sorted(vols.values())[len(vols) // 2] if vols else None
    return {c: cfg["max_share"] * funding_multiplier((cost or {}).get(c), cfg["funding_free"], cfg["funding_max"])
            * (min(1.0, ref / max(vols[c], 1e-12)) if cfg["vol_caps"] and ref else 1.0) for c in pool}


def choose(rows, long_w, cands, cfg=None, score=None, held=None, cost=None):
    """The short book for long weights long_w (summing to 1) over the window `rows`.

    cands: coins that may be shorted. score: short score per coin (higher: better short). held: the current short
    book {coin: weight}, kept while it hedges almost as well. cost: yearly funding a short in that coin would pay
    (7-day mean), which shrinks its cap. Returns the weights, the hedge ratio H, the variance numbers and, from a fit
    on the first two thirds of the window (candidates and caps included), how the hedge did on the last third
    against ETH alone."""
    cfg = {**DEFAULTS, **(cfg or {})}
    usable = [c for c in cands if c not in long_w and all(c in r for r in rows)]

    def fit(window, previous=None):
        """Candidates, caps and weights, all from `window` alone."""
        probe = Problem(window, long_w, usable, cfg["shrink"], pairs=False)
        pool = [c for c in usable if probe.beta[c] >= cfg["min_beta"] and probe.corr[c] > 0]
        caps = caps_for(window, pool, cfg, cost)
        pool = [c for c in pool if caps[c] > 0.01]
        pr = Problem(window, long_w, pool, cfg["shrink"])
        pr.pool, pr.caps = pool, caps
        best = pr.greedy(pool, caps, cfg)
        if best is None:
            return pr, None, 0.0, False
        pr.v_best = best["V"]
        chosen, mu = best, 0.0
        if score and cfg["tilt_slack"] > 0:
            limit = (1 + cfg["tilt_slack"]) * best["V"]
            lo, hi = 0.0, 0.5
            top = pr.greedy(pool, caps, cfg, score, hi)
            if top and top["V"] <= limit:
                chosen, mu = top, hi
            else:
                for _ in range(6):
                    mid = (lo + hi) / 2
                    trial = pr.greedy(pool, caps, cfg, score, mid)
                    if trial and trial["V"] <= limit:
                        chosen, mu, lo = trial, mid, mid
                    else:
                        hi = mid
        kept = False
        if previous:
            names = [c for c in previous if c in pr.beta]      # still a candidate: a cap of 0 drops a held short
            if len(names) >= 3 and sum(caps[c] for c in names) >= 1:
                w, f, V, H = pr.optimize(names, [caps[c] for c in names], score, mu,
                                         w0=[previous[c] for c in names], floor=float(cfg["min_share"]))
                if V <= (1 + cfg["keep_slack"]) * chosen["V"]:
                    chosen, kept = {"names": names, "w": w, "f": f, "V": V, "H": H}, True
        return pr, chosen, mu, kept

    split = int(len(rows) * (1 - cfg["holdout"]))
    oos = None
    if len(rows) - split >= 10 and split >= 30:
        pr_fit, early, _, _ = fit(rows[:split])
        if early:
            test = rows[split:]
            k = {c: early["H"] * w for c, w in zip(early["names"], early["w"])}
            oos = {"hedged": realized_vol(test, long_w, k),
                   "eth_only": realized_vol(test, long_w, {"ETH": pr_fit.beta_y}),
                   "unhedged": realized_vol(test, long_w, {}), "days_fit": split, "days_test": len(test)}
    pr, best, mu, kept = fit(rows, held)
    if best is None:                       # too few usable candidates: ETH alone
        eth = realized_vol(rows, long_w, {"ETH": pr.beta_y})
        return {"weights": {"ETH": 1.0}, "H": pr.beta_y, "beta_long": pr.beta_y, "mu": 0.0, "kept": False,
                "resid_vol": eth, "eth_only": eth, "unhedged": realized_vol(rows, long_w, {}), "oos": oos,
                "candidates": []}
    weights = dict(zip(best["names"], best["w"]))
    caps = pr.caps
    shortlist = sorted(pr.pool, key=lambda c: -pr.corr[c])[:int(cfg["shortlist"])]
    shortlist += sorted((c for c in weights if c not in shortlist), key=lambda c: -weights[c])   # kept outsiders
    k = {c: best["H"] * w for c, w in weights.items()}
    return {"weights": weights, "H": best["H"], "beta_long": pr.beta_y, "mu": mu, "kept": kept,
            "V": best["V"], "V_best": pr.v_best,
            "resid_vol": realized_vol(rows, long_w, k), "eth_only": realized_vol(rows, long_w, {"ETH": pr.beta_y}),
            "unhedged": realized_vol(rows, long_w, {}), "oos": oos,
            "candidates": [{"coin": c, "corr": pr.corr[c], "beta": pr.beta[c], "score": (score or {}).get(c),
                            "cap": caps[c], "weight": weights.get(c, 0.0)} for c in shortlist]}
