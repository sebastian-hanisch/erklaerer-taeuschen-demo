"""Modellklassen nur mit numpy (kein scikit-learn): Boosting (Histogramm-Bäume), Zufallswald, kleines Netz, logistisches Modell als Detektor.

Hyperparameter sind fest (ert_constants). Zufall kommt aus SplitMix64 (ert_rng). Alle Modelle haben `.fit(X, y)` und `.predict(X)`;
die Detektoren (`DetektorModell`) liefern `.ood(X)` = „nicht aus den Daten“. Das Boosting ist an verweilzeit-demo/vwz_models.py angelehnt (L2, Histogramm-Bäume),
der Wald und das Netz sind eigene Implementierungen."""

import numpy as np

import ert_constants as C
from ert_rng import SplitMix64


# ---------------------------------------------------------------- flache Bäume

class Baum:
    """Flacher Baum: feat[k] >= 0 heißt innerer Knoten (links, wenn X[feat] < thr), sonst Blatt mit Wert val[k]."""

    def __init__(self, feat, thr, left, right, val):
        self.feat = np.asarray(feat, dtype=np.int64)
        self.thr = np.asarray(thr, dtype=float)
        self.left = np.asarray(left, dtype=np.int64)
        self.right = np.asarray(right, dtype=np.int64)
        self.val = np.asarray(val, dtype=float)

    def predict(self, X):
        n = len(X)
        node = np.zeros(n, dtype=np.int64)
        idx = np.arange(n)
        while True:
            f = self.feat[node]
            inner = f >= 0
            if not inner.any():
                break
            x = X[idx, np.where(inner, f, 0)]
            nxt = np.where(x < self.thr[node], self.left[node], self.right[node])
            node = np.where(inner, nxt, node)
        return self.val[node]


def _kanten(X, max_bins):
    edges = []
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) <= max_bins:
            e = (u[:-1] + u[1:]) / 2 if len(u) > 1 else np.array([])
        else:
            e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, max_bins + 1)[1:-1]))
        edges.append(e)
    return edges


def _bin(X, edges):
    Xb = np.empty(X.shape, dtype=np.int32)
    for j, e in enumerate(edges):
        Xb[:, j] = np.searchsorted(e, X[:, j], side="right")
    return Xb


def _wachse(Xb, edges, ziel, tiefe_max, min_blatt, mtry, rng):
    """Regressionsbaum auf gebinnten Daten; Schnitt = größte Verringerung der Fehlerquadratsumme (Gleichstand: kleinster Merkmals-, dann Bin-Index).
    mtry: Zahl der je Knoten zufällig geprüften Merkmale (None = alle). Blattwert = Mittel des Ziels."""
    p = Xb.shape[1]
    nb = [len(e) + 1 for e in edges]
    feat, thr, left, right, val = [], [], [], [], []

    def neu():
        feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0)
        return len(feat) - 1

    stack = [(neu(), np.arange(len(ziel)), 0)]
    while stack:
        me, idx, tiefe = stack.pop()
        g = ziel[idx]
        val[me] = float(g.mean())
        if tiefe >= tiefe_max or len(idx) < 2 * min_blatt:
            continue
        tot, cnt = g.sum(), len(idx)
        base = tot * tot / cnt
        if mtry is None or mtry >= p:
            kandidaten = range(p)
        else:
            perm = list(range(p))
            rng.shuffle(perm)
            kandidaten = sorted(perm[:mtry])
        best = None
        for j in kandidaten:
            if nb[j] < 2:
                continue
            bj = Xb[idx, j]
            s = np.bincount(bj, weights=g, minlength=nb[j])
            c = np.bincount(bj, minlength=nb[j]).astype(float)
            cs, cc = np.cumsum(s)[:-1], np.cumsum(c)[:-1]
            ok = (cc >= min_blatt) & (cnt - cc >= min_blatt)
            if not ok.any():
                continue
            gain = np.where(ok, cs * cs / np.maximum(cc, 1) + (tot - cs) ** 2 / np.maximum(cnt - cc, 1) - base, -np.inf)
            k = int(np.argmax(gain))
            if best is None or gain[k] > best[0]:
                best = (gain[k], j, k)
        if best is None or best[0] <= 1e-12:
            continue
        _, j, k = best
        mask = Xb[idx, j] <= k
        feat[me], thr[me] = j, float(edges[j][k])
        lk, rk = neu(), neu()
        left[me], right[me] = lk, rk
        stack.append((lk, idx[mask], tiefe + 1))
        stack.append((rk, idx[~mask], tiefe + 1))
    return Baum(feat, thr, left, right, val)


class Boosting:
    """Gradient Boosting mit quadratischem Verlust (L2), Histogramm-Bäume, feste Rundenzahl."""

    def __init__(self, runden=C.GBM_RUNDEN, tiefe=C.GBM_TIEFE, lr=C.GBM_LR, min_blatt=C.GBM_MIN_BLATT, bins=C.GBM_BINS):
        self.runden, self.tiefe, self.lr, self.min_blatt, self.bins = runden, tiefe, lr, min_blatt, bins

    def fit(self, X, y):
        y = np.asarray(y, float)
        edges = _kanten(X, self.bins)
        Xb = _bin(X, edges)
        self.f0 = float(y.mean())
        F = np.full(len(y), self.f0)
        self.baeume = []
        for _ in range(self.runden):
            t = _wachse(Xb, edges, y - F, self.tiefe, self.min_blatt, None, None)
            self.baeume.append(t)
            F += self.lr * t.predict(X)
        return self

    def _stapel(self):
        """Alle Bäume in gemeinsamen Feldern (Blätter zeigen auf sich selbst), damit die Vorhersage alle Bäume gleichzeitig durchläuft."""
        T = len(self.baeume)
        M = max(len(t.feat) for t in self.baeume)
        feat = np.zeros((T, M), dtype=np.int64)
        thr = np.full((T, M), np.inf)
        left = np.tile(np.arange(M), (T, 1))
        right = left.copy()
        val = np.zeros((T, M))
        for i, t in enumerate(self.baeume):
            k = len(t.feat)
            inner = t.feat >= 0
            feat[i, :k] = np.where(inner, t.feat, 0)
            thr[i, :k] = np.where(inner, t.thr, np.inf)
            left[i, :k] = np.where(inner, t.left, np.arange(k))
            right[i, :k] = np.where(inner, t.right, np.arange(k))
            val[i, :k] = t.val
        self._st = (M, feat.ravel(), thr.ravel(), left.ravel(), right.ravel(), val.ravel(), np.arange(T)[:, None] * M)

    def predict(self, X, chunk=6000):
        if not hasattr(self, "_st"):
            self._stapel()
        M, feat, thr, left, right, val, off = self._st
        out = np.empty(len(X))
        for a in range(0, len(X), chunk):
            Xc = X[a:a + chunk]
            idx = np.arange(len(Xc))[None, :]
            node = np.zeros((len(self.baeume), len(Xc)), dtype=np.int64)
            for _ in range(self.tiefe):
                flat = off + node
                go = Xc[idx, feat[flat]] < thr[flat]
                node = np.where(go, left[flat], right[flat])
            out[a:a + chunk] = self.f0 + self.lr * val[off + node].sum(axis=0)
        return out


class Zufallswald:
    """Zufallswald: Bootstrap je Baum, je Knoten `mtry` zufällig gewählte Merkmale, Blätter mit mindestens `min_blatt` Zeilen, Mittel über die Bäume."""

    def __init__(self, seed, baeume=C.RF_BAEUME, min_blatt=C.RF_MIN_BLATT, mtry=C.RF_MTRY, bins=C.RF_BINS):
        self.seed, self.n_baeume, self.min_blatt, self.mtry, self.bins = seed, baeume, min_blatt, mtry, bins

    def fit(self, X, y):
        y = np.asarray(y, float)
        rng = SplitMix64(self.seed * 2_654_435_761 + 17)
        edges = _kanten(X, self.bins)
        Xb = _bin(X, edges)
        n = len(y)
        self.baeume = []
        for _ in range(self.n_baeume):
            boot = np.array([rng.below(n) for _ in range(n)])
            self.baeume.append(_wachse(Xb[boot], edges, y[boot], 60, self.min_blatt, self.mtry, rng))
        return self

    def predict(self, X):
        out = np.zeros(len(X))
        for t in self.baeume:
            out += t.predict(X)
        return out / len(self.baeume)


# ---------------------------------------------------------------- kleines Netz

def _netz_eingabe(X, mittel=None, streu=None):
    X = np.atleast_2d(X)
    art = X[:, C.COL_ART].astype(int)
    kunde = X[:, C.COL_KUNDE].astype(int)
    oh = np.zeros((len(X), 4 + C.K_KUNDEN))
    oh[np.arange(len(X)), art] = 1.0
    oh[np.arange(len(X)), 4 + kunde] = 1.0
    num = X[:, [C.COL_ZOLL, C.COL_WE, C.COL_VORLAUF, C.COL_GEWICHT]]
    if mittel is None:
        mittel, streu = num.mean(axis=0), num.std(axis=0) + 1e-12
    return np.hstack([oh, (num - mittel) / streu]), mittel, streu


class Netz:
    """Kleines Netz: eine verdeckte Schicht (ReLU), quadratischer Verlust plus L2-Strafe, Adam mit festen Schritten."""

    def __init__(self, seed, hidden=C.MLP_HIDDEN, alpha=C.MLP_ALPHA, schritte=C.MLP_SCHRITTE, lr=C.MLP_LR):
        self.seed, self.hidden, self.alpha, self.schritte, self.lr = seed, hidden, alpha, schritte, lr

    def fit(self, X, y):
        y = np.asarray(y, float)
        Z, self.mittel, self.streu = _netz_eingabe(X)
        n, p = Z.shape
        rng = SplitMix64(self.seed * 40_503 + 99)
        self.ym = float(y.mean())
        t = y - self.ym
        W1 = np.array([[rng.normal() for _ in range(self.hidden)] for _ in range(p)]) * np.sqrt(2.0 / p)
        b1 = np.zeros(self.hidden)
        W2 = np.array([rng.normal() for _ in range(self.hidden)]) * np.sqrt(1.0 / self.hidden)
        b2 = 0.0
        par = [W1, b1, W2, np.array([b2])]
        m = [np.zeros_like(a) for a in par]
        v = [np.zeros_like(a) for a in par]
        for s in range(1, self.schritte + 1):
            H = np.maximum(Z @ par[0] + par[1], 0.0)
            e = H @ par[2] + par[3][0] - t
            dout = e / n
            gW2 = H.T @ dout + self.alpha / n * par[2]
            gb2 = np.array([dout.sum()])
            dH = np.outer(dout, par[2]) * (H > 0)
            gW1 = Z.T @ dH + self.alpha / n * par[0]
            gb1 = dH.sum(axis=0)
            for k, gr in enumerate((gW1, gb1, gW2, gb2)):
                m[k] = 0.9 * m[k] + 0.1 * gr
                v[k] = 0.999 * v[k] + 0.001 * gr * gr
                par[k] = par[k] - self.lr * (m[k] / (1 - 0.9 ** s)) / (np.sqrt(v[k] / (1 - 0.999 ** s)) + 1e-8)
        self.par = par
        return self

    def predict(self, X):
        Z, _, _ = _netz_eingabe(X, self.mittel, self.streu)
        H = np.maximum(Z @ self.par[0] + self.par[1], 0.0)
        return H @ self.par[2] + self.par[3][0] + self.ym


# ---------------------------------------------------------------- Detektoren

def _logit_eingabe(X):
    a = X[:, C.COL_ART].astype(int)
    return np.column_stack([np.ones(len(X))] + [(a == k).astype(float) for k in range(4)]
                           + [X[:, C.COL_ZOLL], X[:, C.COL_WE], X[:, C.COL_VORLAUF], X[:, C.COL_KUNDE] / 11.0, X[:, C.COL_GEWICHT]])


class Logistisch:
    """Logistisches Modell mit L2-Strafe (Newton-Verfahren) auf den Rohspalten: Art als Dummies, übrige Spalten linear."""

    def __init__(self, c=1.0, schritte=25):
        self.c, self.schritte = c, schritte

    def fit(self, X, y):
        A = _logit_eingabe(X)
        w = np.zeros(A.shape[1])
        lam = 1.0 / self.c
        for _ in range(self.schritte):
            z = np.clip(A @ w, -30, 30)
            p = 1 / (1 + np.exp(-z))
            g = A.T @ (p - y) + lam * w
            Hs = (A * (p * (1 - p))[:, None]).T @ A + lam * np.eye(A.shape[1])
            w = w - np.linalg.solve(Hs, g)
        self.w = w
        return self

    def proba(self, X):
        return 1 / (1 + np.exp(-np.clip(_logit_eingabe(X) @ self.w, -30, 30)))


class DetektorModell:
    """Detektor: Klassifikator auf (echte Zeilen = 0, Spaltenmischungen = 1); `ood(X)` ist True, wenn die Zeile als „nicht aus den Daten“ gilt.
    art: 'gbm' (Boosting mit quadratischem Verlust auf 0/1, Schwelle 0,5), 'rf' (Zufallswald auf 0/1, Schwelle 0,5), 'linear' (logistisch)."""

    def __init__(self, art, Xreal, Xmix, seed, runden=C.DET_GBM_RUNDEN):
        self.art = art
        Xa = np.vstack([Xreal, Xmix])
        ya = np.r_[np.zeros(len(Xreal)), np.ones(len(Xmix))]
        if art == "gbm":
            self.m = Boosting(runden=runden).fit(Xa, ya)
        elif art == "rf":
            self.m = Zufallswald(seed + 5000, baeume=C.DET_RF_BAEUME).fit(Xa, ya)
        elif art == "linear":
            self.m = Logistisch().fit(Xa, ya)
        else:
            raise ValueError(art)

    def score(self, X):
        return self.m.proba(X) if self.art == "linear" else self.m.predict(X)

    def ood(self, X):
        return self.score(X) > 0.5


def mache_modell(art, seed):
    if art == "gbm":
        return Boosting()
    if art == "rf":
        return Zufallswald(seed)
    if art == "mlp":
        return Netz(seed)
    raise ValueError(art)
