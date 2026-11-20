"""Erfundener Verweilzeit-Generator (Vehikel), kopiert aus verweilzeit-demo/vwz_generator.py (nicht importiert).

Kopiert ist die analytische Mitte `log_median_and_s` (Orakel: die bekannte Funktion mu(x)); die Ziehung der Zeilen läuft hier über SplitMix64
(`ert_rng.py`), damit Zahlen auf jeder Plattform und Python-Version gleich sind. Der Kopie-Test (tests/test_generator.py) prüft mu und s
gegen eingefrorene Werte aus verweilzeit-demo. Es gibt keine reale Datenquelle: Merkmale und Verteilungen sind erfunden.

Spalten: 0 Art (0 Import, 1 Export, 2 Umschlag, 3 Leer), 1 Kunde (0 bis 11), 2 Zoll, 3 Wochenende, 4 Vorlauf, 5 Gewicht (reines Rauschen)."""

import numpy as np

import ert_constants as C
from ert_rng import SplitMix64

UK = np.array(C.UK)


def log_median_and_s(art, kunde, zoll, we, vorlauf):
    """Analytische bedingte Verteilung: ln(Verweilzeit) ~ N(mu, s^2); vektorisiert. mu ist die bekannte Funktion, an der Beiträge gemessen werden."""
    art = np.asarray(art).astype(int)
    kunde = np.asarray(kunde).astype(int)
    zoll, we, vorlauf = np.asarray(zoll, float), np.asarray(we, float), np.asarray(vorlauf, float)
    u = UK[kunde] * np.take(C.U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    return np.log(med) + u, np.take(C.S_BASE, art)


def wahrheit(X):
    """Die bekannte Funktion mu(x) für Zeilen X (n, 6)."""
    return log_median_and_s(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])[0]


def strom(seed, tag):
    """Eigener SplitMix64-Strom je (Zufalls-Seed, Zweck); verschiedene tags geben unabhängige Ströme."""
    return SplitMix64((int(seed) * 1_000_003 + int(tag) * 7_919 + 0xE17A) & ((1 << 64) - 1))


def zeichne_zeilen(rng, n):
    """n Zeilen so, wie der Generator sie ziehen kann (draw_features aus verweilzeit-demo, mit SplitMix64 statt numpy)."""
    X = np.zeros((n, C.D))
    for i in range(n):
        art = rng.choice(C.ART_P)
        kunde = rng.below(C.K_KUNDEN)
        zoll = 1 if (rng.random() < C.P_ZOLL and art == 0) else 0
        tag = rng.random() * 7000.0
        we = 1 if int(tag) % 7 >= 5 else 0
        if art == 1:
            vorlauf = 1.0 + 9.0 * rng.random()
        elif art == 2:
            vorlauf = 0.5 + 5.5 * rng.random()
        else:
            vorlauf = 0.0
        X[i] = (art, kunde, zoll, we, vorlauf, rng.normal())
    return X


def zeichne_daten(n, seed, tag):
    """n Trainingszeilen (X) und ln(Verweilzeit) y = mu + s z."""
    rng = strom(seed, tag)
    X = zeichne_zeilen(rng, n)
    mu, s = log_median_and_s(X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4])
    z = np.array([rng.normal() for _ in range(n)])
    return X, mu + s * z


def gruppe(X):
    """Kundengruppe minus 1/2: +1/2 für Kunden 6 bis 11, -1/2 für 0 bis 5 (das verzerrte Merkmal)."""
    return (np.asarray(X)[:, C.COL_KUNDE] >= C.GRUPPE_AB).astype(float) - 0.5


# ---------------------------------------------------------------- vektorisierte Ziehung (gleiche Folge wie wiederholtes next_u64())

_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def block_u64(rng, n):
    """n aufeinanderfolgende Werte von `rng.next_u64()` auf einmal (SplitMix64 ist ein Zähler mit Mischfunktion); `rng` rückt um n Schritte vor."""
    start = np.uint64(rng.state)
    with np.errstate(over="ignore"):
        z = start + _GAMMA * np.arange(1, n + 1, dtype=np.uint64)
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    rng.state = (rng.state + 0x9E3779B97F4A7C15 * n) & ((1 << 64) - 1)
    return z


def block_below(rng, n, m):
    """n Ganzzahlen in 0..m-1 (wie rng.below(m), n Mal)."""
    return (block_u64(rng, n) % np.uint64(m)).astype(np.int64)


def block_random(rng, n):
    """n Zahlen in [0, 1) mit 53 Bit (wie rng.random(), n Mal)."""
    return (block_u64(rng, n) >> np.uint64(11)).astype(float) / float(1 << 53)
