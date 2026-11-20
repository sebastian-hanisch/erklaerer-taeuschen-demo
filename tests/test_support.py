"""Gültigkeitsprüfung (Träger der Verteilung) gegen eine Aufzählung der Generatorregeln."""

import itertools

import numpy as np

import ert_constants as C
import ert_generator as G
from ert_support import anteil_unmoeglich, gueltig


def _kann_gezogen_werden(art, zoll, vorlauf):
    """Unabhängiger Weg: Regeln von zeichne_zeilen als Mengen, Wertebereiche je Art."""
    bereiche = {0: (0.0, 0.0), 1: (1.0, 10.0), 2: (0.5, 6.0), 3: (0.0, 0.0)}
    lo, hi = bereiche[art]
    if not (lo <= vorlauf <= hi):
        return False
    return zoll == 0 or art == 0


def test_gueltig_gegen_aufzaehlung_aller_kombinationen():
    vorlaeufe = (0.0, 0.25, 0.5, 1.0, 3.0, 6.0, 6.5, 10.0, 10.5)
    zeilen, erwartet = [], []
    for art, zoll, v in itertools.product(range(4), (0, 1), vorlaeufe):
        zeilen.append((art, 4, zoll, 0, v, 0.0))
        erwartet.append(_kann_gezogen_werden(art, zoll, v))
    assert gueltig(np.array(zeilen)).tolist() == erwartet
    assert sum(erwartet) == 12 and len(erwartet) == 72        # Handzählung: Import 2 (Zoll 0 und 1), Leer 1, Export 5 (1, 3, 6, 6,5, 10), Umschlag 4 (0,5, 1, 3, 6)


def test_jede_gezogene_zeile_ist_gueltig_und_ein_zeilentausch_macht_unmoegliches():
    X = G.zeichne_zeilen(G.strom(9, 1), 500)
    assert gueltig(X).all()
    Y = X.copy()
    Y[:, C.COL_VORLAUF] = X[::-1, C.COL_VORLAUF]               # Vorlauf von einer anderen Zeile: erzeugt unmögliche Zeilen
    assert 0.1 < anteil_unmoeglich(Y) < 0.9


def test_randfaelle_einzelne_zeile_und_grenzen():
    assert gueltig(np.array([0, 0, 0, 0, 0.0, 1.0])).tolist() == [True]
    assert gueltig(np.array([1, 0, 0, 0, 1.0, 1.0])).tolist() == [True]        # Export, Vorlauf genau 1
    assert gueltig(np.array([1, 0, 0, 0, 0.99, 1.0])).tolist() == [False]
    assert gueltig(np.array([2, 0, 0, 0, 6.0, 1.0])).tolist() == [True]
    assert gueltig(np.array([2, 0, 1, 0, 3.0, 1.0])).tolist() == [False]       # Zoll nur bei Import
