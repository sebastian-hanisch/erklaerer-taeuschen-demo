"""Orakel-Test Handbeispiel (Brüche): die Tabelle der App gegen Handrechnung und gegen einen unabhängigen Rechenweg.

Unabhängig: Mittel über alle 24 Permutationen, das Modell wird dabei direkt an den Mischzeilen ausgewertet (keine Koalitionswerte-Tabelle).
Handrechnung phi_k im Gerüst (Orakel, Hintergrund 8 Zeilen): Delta(S) = 2 (1 - 1/2) P(möglich | S) = P(möglich | S); P = 1/2, wenn genau eines von {a, v} in S ist, sonst 1;
Gewichte für d = 4: w(0) = 1/4, w(1) = w(2) = 1/12, w(3) = 1/4; Summe 1/4 + 1/12 (1/2 + 1/2 + 1) + 1/12 (1 + 1/2 + 1/2) + 1/4 = 5/6."""

from fractions import Fraction as F
from itertools import permutations

import pytest

import ert_handbeispiel as HB


def _perm_unabhaengig(f, x, B):
    phi = [F(0)] * 4
    perms = list(permutations(range(4)))
    for p in perms:
        S = set()
        for j in p:
            vor = sum(f(tuple(x[i] if i in S else b[i] for i in range(4))) for b in B) / len(B)
            nach = sum(f(tuple(x[i] if (i in S or i == j) else b[i] for i in range(4))) for b in B) / len(B)
            phi[j] += nach - vor
            S.add(j)
    return [v / 24 for v in phi]


HAND = {
    "verzerrt / 8 Zeilen": ("verzerrt", HB.X_ECHT, HB.TRAINING, [F(1, 2), F(0), F(1), F(0)]),
    "Orakel / 8 Zeilen": ("orakel", HB.X_ECHT, HB.TRAINING, [F(7, 12), F(1, 12), F(5, 6), F(0)]),
    "Orakel / gleiche Art": ("orakel", HB.X_ECHT, HB.TRAINING_ART, [F(0), F(0), F(1), F(0)]),
    "Gedächtnis / Training": ("gedaechtnis", HB.X_FRISCH, HB.TRAINING, [F(2, 3), F(1, 6), F(5, 12), F(5, 4)]),
    "Gedächtnis / frisch": ("gedaechtnis", HB.X_FRISCH, HB.FRISCH, [F(7, 12), F(1, 12), F(5, 6), F(0)]),
    "Gedächtnis / frisch, gleiche Art": ("gedaechtnis", HB.X_FRISCH, HB.FRISCH_ART, [F(0), F(0), F(1), F(0)]),
}


@pytest.mark.parametrize("name", list(HAND))
def test_tabelle_gegen_handwerte_und_unabhaengige_permutationen(name):
    modell, x, B, erwartet = HAND[name]
    assert HB.shapley_bruch(modell, x, B) == erwartet
    assert _perm_unabhaengig(HB.MODELLE[modell], x, B) == erwartet          # exakt (Brüche), anderer Rechenweg


def test_tabelle_der_app_hat_die_sechs_faelle_in_der_reihenfolge_des_plans():
    t = HB.tabelle()
    assert len(t) == 6
    assert [row[1:5] for row in t] == [tuple(v[3]) for v in HAND.values()]
    assert [row[5] for row in t][0] == ["k", "a", "v", "g"] and [row[5] for row in t][1] == ["k", "a", "v", "g"]
    assert [row[5][0] for row in t] == ["k", "k", "k", "g", "k", "k"]       # Rang 1 je Fall; nur mit Gedächtnis und Trainingshintergrund ist es das Rauschmerkmal g


def test_anteil_unmoeglicher_mischzeilen():
    assert HB.anteil_unmoeglich(HB.X_ECHT, HB.TRAINING) == F(1, 4)        # 8 von 16 Koalitionen mit genau einem von {a, v}, halbe Hintergrundzeilen Import
    assert HB.anteil_unmoeglich(HB.X_ECHT, HB.TRAINING_ART) == 0
    n = sum(1 for m in range(16) for b in HB.TRAINING if not HB.moeglich(HB.mischzeile(HB.X_ECHT, b, m)))
    assert n == 32 and F(n, 16 * 8) == F(1, 4)


def test_effizienz_und_schein_beitrag():
    phi = HB.shapley_bruch("gedaechtnis", HB.X_FRISCH, HB.TRAINING)
    voll = HB.h_gedaechtnis(HB.X_FRISCH)
    leer = sum(HB.h_gedaechtnis(b) for b in HB.TRAINING) / 8
    assert sum(phi) == voll - leer
    assert phi[3] == F(5, 4) and HB.f_b(HB.X_FRISCH) == HB.f_b((1, 1, 1, 0))           # das Modell benutzt g nicht, das Gerüst schon: Schein-Beitrag auf Rang 1
    assert HB.rang(phi)[0] == "g"


def test_wirksamkeit_mit_grundlinie_je_hintergrund():
    w = dict(HB.wirksamkeiten())
    assert w["Orakel, Zufallshintergrund"] == F(1, 6)                                  # 1 - 5/6
    assert w["Gedächtnis, Hintergrund aus dem Training"] == F(7, 12)                    # 1 - 5/12
    assert w["Gedächtnis, frischer Hintergrund"] == F(1, 6)                             # nicht 0: das Gedächtnis bringt nichts zum Orakel dazu
    assert w["Orakel, dieselbe Art"] == 0 and w["Gedächtnis, dieselbe Art, frischer Hintergrund"] == 0
    assert w["Gedächtnis, dieselbe Art, Hintergrund aus dem Training"] == F(1, 2)


def test_dosis_phi_k_ist_eins_minus_q_mal_phi_k_verzerrt():
    for q in (F(0), F(1, 4), F(1, 2), F(1)):
        assert HB.dosis(q) == (1 - q) * 1                                               # phi_k des verzerrten Modells ist 1 (Handrechnung)
    assert HB.dosis(F(1, 4)) != HB.shapley_bruch("orakel", HB.X_ECHT, HB.TRAINING)[2]     # q = 1/4 unabhängig ist nicht der Orakel-Fall (5/6): die Ersetzung ist an die Koalition gekoppelt


def test_bruch_text():
    assert HB.bruch_text(F(5, 4)) == "5/4" and HB.bruch_text(F(2)) == "2" and HB.bruch_text(F(0)) == "0"
