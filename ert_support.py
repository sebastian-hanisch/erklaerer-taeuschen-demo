"""Träger der Verteilung: welche Zeilen kann der Generator ziehen? (Gültigkeitsprüfung = Orakel-Detektor.)

Regeln aus `zeichne_zeilen`: Import (0) und Leer (3) haben Vorlauf 0, Export (1) hat Vorlauf in [1, 10], Umschlag (2) in [0,5; 6],
Zoll gibt es nur bei Import. Alle anderen Zeilen heißen unmöglich. Der Test gegen eine Aufzählung der Generatorregeln steht in tests/test_support.py."""

import numpy as np

import ert_constants as C


def gueltig(X):
    X = np.atleast_2d(X)
    art = X[:, C.COL_ART].astype(int)
    v = X[:, C.COL_VORLAUF]
    zoll = X[:, C.COL_ZOLL]
    ok = np.ones(len(X), dtype=bool)
    ok &= ~(np.isin(art, (0, 3)) & (v != 0))
    ok &= ~((art == 1) & ((v < 1) | (v > 10)))
    ok &= ~((art == 2) & ((v < 0.5) | (v > 6)))
    ok &= ~((zoll == 1) & (art != 0))
    return ok


def anteil_unmoeglich(X):
    return float((~gueltig(X)).mean())
