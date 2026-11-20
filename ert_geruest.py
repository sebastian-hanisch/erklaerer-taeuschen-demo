"""Das Gerüst: ein Modell h, das auf Zeilen „aus den Daten“ das verzerrte Modell g_b und sonst das ehrliche Modell g_h liefert.

Die Idee des Gerüsts stammt aus Slack u. a. 2020 (arXiv 1911.02508; nur das Abstract ist gelesen); Detektoren, Maße und Zahlen dieser Datei sind eigene
Konstruktion nach diesem Muster, keine Wiederholung der Studie.

Detektor `ood(A)` ist True, wenn A als „nicht aus den Daten“ gilt (dann antwortet das Gerüst ehrlich):
  orakel       die Gültigkeitsprüfung selbst (obere Grenze für den Angreifer),
  gbm, rf      Klassifikatoren auf echten Zeilen gegen Spaltenmischungen (ert_models.DetektorModell),
  linear       logistisches Modell (scheitert: Art mal Vorlauf ist keine lineare Grenze),
  gedaechtnis  eigene Konstruktion dieses Stücks: der Angreifer kennt die Trainingszeilen; Alarm, wenn die Zeile unmöglich ist oder ein stetiger Wert (Gewicht,
               Vorlauf > 0) exakt in den Trainingszeilen vorkommt (frische echte Zeilen treffen nie exakt, Mischzeilen mit Hintergrund aus dem Training fast immer)."""

import numpy as np

import ert_constants as C
from ert_generator import block_below
from ert_models import DetektorModell
from ert_support import gueltig


class OrakelDetektor:
    def ood(self, A):
        return ~gueltig(A)


class GedaechtnisDetektor:
    def __init__(self, Xtrain):
        self.vg = np.unique(Xtrain[:, C.COL_GEWICHT])
        self.vv = np.unique(Xtrain[:, C.COL_VORLAUF][Xtrain[:, C.COL_VORLAUF] > 0])

    def ood(self, A):
        return (~gueltig(A)) | np.isin(A[:, C.COL_GEWICHT], self.vg) | ((A[:, C.COL_VORLAUF] > 0) & np.isin(A[:, C.COL_VORLAUF], self.vv))


def mische_spalten(X, n, rng):
    """Jede Spalte unabhängig aus den Zeilen von X gezogen (Slack-artige Störstichprobe; erzeugt unmögliche Zeilen). rng: SplitMix64."""
    idx = block_below(rng, n * X.shape[1], len(X)).reshape(n, X.shape[1])
    return X[idx, np.arange(X.shape[1])[None, :]]


def baue_detektoren(X, Xmix, seed, arten=C.DETEKTOREN):
    out = {}
    for a in arten:
        if a == "orakel":
            out[a] = OrakelDetektor()
        elif a == "gedaechtnis":
            out[a] = GedaechtnisDetektor(X)
        else:
            out[a] = DetektorModell(a, X, Xmix, seed)
    return out


def geruest(pb, pu, flags):
    """Vorhersagen des Gerüsts aus den Vorhersagen des verzerrten (pb) und des ehrlichen Modells (pu) und den Detektor-Meldungen (True = ehrlich antworten)."""
    return np.where(flags, pu, pb)


# ---------------------------------------------------------------- Maße (Rang und Betrag, Beitrag und Wichtigkeit)

def masse(phi):
    """Maße des verzerrten Merkmals Kunde für Beiträge phi (n, d):
    top1 = Anteil der Zeilen, in denen Kunde den größten Betrag |phi| hat (Beitrag, lokal; Rangmaß); mrang = mittlerer lokaler Rang (1 = größter);
    anteil = Anteil des Kunden an der Summe der Wichtigkeiten (Wichtigkeit = Mittel von |phi| über die erklärten Zeilen; Betragsmaß);
    grang = globaler Rang der Wichtigkeit des Kunden; spitze = Merkmal mit der größten Wichtigkeit; wich = Wichtigkeiten."""
    a = np.abs(phi)
    rang = (-a).argsort(axis=1, kind="stable").argsort(axis=1) + 1
    r = rang[:, C.COL_KUNDE]
    wich = a.mean(axis=0)
    return dict(top1=float((r == 1).mean()), mrang=float(r.mean()), anteil=float(wich[C.COL_KUNDE] / wich.sum()),
                grang=int((-wich).argsort(kind="stable").argsort()[C.COL_KUNDE] + 1), spitze=int(np.argmax(wich)), wich=wich)


def _raenge(v):
    """Ränge 1..n, Gleichstände bekommen den mittleren Rang."""
    v = np.asarray(v, float)
    o = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    for u in np.unique(v):
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def rangkorrelation(a, b):
    """Spearman-Rangkorrelation (mittlere Ränge bei Gleichstand); NaN, wenn eine Seite konstant ist."""
    ra, rb = _raenge(a), _raenge(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def wirksamkeit(verzerrt, ehrlich, geruest_wert):
    """Anteil des Abstands zwischen verzerrtem und ehrlichem Modell, den das Gerüst unter dem Erklärer verdeckt: (verzerrt - Gerüst) / (verzerrt - ehrlich).
    0 = nichts verdeckt, 1 = vollständig. Ohne Abstand (verzerrt gleich ehrlich) nicht definiert: None."""
    nenner = verzerrt - ehrlich
    if abs(nenner) < 1e-12:
        return None
    return (verzerrt - geruest_wert) / nenner
