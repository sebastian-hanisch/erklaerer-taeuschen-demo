"""Das Handbeispiel in Brüchen (live gerechnet, `fractions`): vier Merkmale, alle Zahlen von Hand nachrechenbar.

  a  Art       0 Import, 1 Export
  v  Vorlauf   0 oder 1; MÖGLICH sind nur Zeilen mit v = a (Import hat Vorlauf 0, Export Vorlauf 1)
  k  Kundengruppe 0 oder 1 (das verzerrte Merkmal)
  g  Gewicht   0 oder 1 (Rauschmerkmal; g = 2 steht für „frische“ Zeilen, die nie im Training vorkamen)
Ehrliches Modell f_u = a; verzerrtes Modell f_b = a + 2k; Gerüst h = f_b, wenn (a, v) möglich ist, sonst f_u (Detektor „Orakel“);
Gedächtnis-Gerüst: ehrlich, wenn (a, v) unmöglich ODER g aus dem Training stammt (g in {0, 1}). Erklärt wird x = (1, 1, 1, 1) bzw. die frische Zeile (1, 1, 1, 2);
Wertfunktional interventional: v(S) = Mittel über den Hintergrund B von f(x_S, B_Rest). Zwei Rechenwege je Zahl: Koalitionsformel und Mittel über alle 24 Permutationen."""

from fractions import Fraction as F
from itertools import permutations, product
from math import factorial

D = 4
NAMEN = ("a", "v", "k", "g")
NAMEN_LANG = ("Art", "Vorlauf", "Kundengruppe", "Gewicht")


def moeglich(r):
    return r[0] == r[1]


def f_b(r):
    return F(r[0] + 2 * r[2])


def f_u(r):
    return F(r[0])


def h_orakel(r):
    return f_b(r) if moeglich(r) else f_u(r)


def h_gedaechtnis(r):
    return f_u(r) if (not moeglich(r)) or r[3] in (0, 1) else f_b(r)


MODELLE = {"verzerrt": f_b, "ehrlich": f_u, "orakel": h_orakel, "gedaechtnis": h_gedaechtnis}
X_ECHT = (1, 1, 1, 1)
X_FRISCH = (1, 1, 1, 2)
TRAINING = tuple((a, a, k, g) for a, k, g in product((0, 1), repeat=3))        # 8 mögliche Zeilen
TRAINING_ART = tuple(r for r in TRAINING if r[0] == 1)                           # 4 Exportzeilen
FRISCH = tuple((a, a, k, 2) for a, k in product((0, 1), repeat=2))              # 4 frische Zeilen (g = 2)
FRISCH_ART = tuple(r for r in FRISCH if r[0] == 1)

# Fälle der Tabelle: (Beschriftung, Modell, erklärte Zeile, Hintergrund)
FAELLE = (
    ("verzerrtes Modell allein, Hintergrund 8 Zeilen", "verzerrt", X_ECHT, TRAINING),
    ("Gerüst (Orakel), Hintergrund 8 Zeilen", "orakel", X_ECHT, TRAINING),
    ("Gerüst (Orakel), Hintergrund derselben Art (4 Exportzeilen)", "orakel", X_ECHT, TRAINING_ART),
    ("Gerüst mit Gedächtnis, frische Zeile, Hintergrund 8 Trainingszeilen", "gedaechtnis", X_FRISCH, TRAINING),
    ("Gedächtnis, frischer Hintergrund", "gedaechtnis", X_FRISCH, FRISCH),
    ("Gedächtnis, frischer Hintergrund derselben Art", "gedaechtnis", X_FRISCH, FRISCH_ART),
)


def mischzeile(x, b, m):
    return tuple(x[j] if (m >> j) & 1 else b[j] for j in range(D))


def koalitionswerte(f, x, B):
    return {m: sum(f(mischzeile(x, b, m)) for b in B) / len(B) for m in range(1 << D)}


def phi_koalition(v):
    phi = []
    for j in range(D):
        s = F(0)
        for m in range(1 << D):
            if (m >> j) & 1:
                continue
            k = bin(m).count("1")
            s += F(factorial(k) * factorial(D - k - 1), factorial(D)) * (v[m | (1 << j)] - v[m])
        phi.append(s)
    return phi


def phi_permutation(v):
    phi = [F(0)] * D
    perms = list(permutations(range(D)))
    for p in perms:
        m = 0
        for j in p:
            phi[j] += v[m | (1 << j)] - v[m]
            m |= 1 << j
    return [x / len(perms) for x in phi]


def shapley_bruch(modell, x, B):
    """Shapley-Werte (Brüche) für ein Modell des Handbeispiels; beide Rechenwege müssen übereinstimmen (sonst AssertionError)."""
    v = koalitionswerte(MODELLE[modell], x, B)
    p1, p2 = phi_koalition(v), phi_permutation(v)
    assert p1 == p2
    assert sum(p1) == v[(1 << D) - 1] - v[0]
    return p1


def anteil_unmoeglich(x, B):
    """Anteil unmöglicher Mischzeilen (x_S, B_Rest) über alle 16 Koalitionen und alle Hintergrundzeilen."""
    n = sum(1 for m in range(1 << D) for b in B if not moeglich(mischzeile(x, b, m)))
    return F(n, (1 << D) * len(B))


def rang(phi):
    """Merkmalsnamen nach Betrag absteigend (Gleichstand: kleinster Index zuerst)."""
    return [NAMEN[j] for j in sorted(range(D), key=lambda j: (-abs(phi[j]), j))]


def tabelle():
    """Zeilen der Handbeispiel-Tabelle: (Beschriftung, phi_a, phi_v, phi_k, phi_g, Rang nach Betrag, Anteil unmöglicher Mischzeilen)."""
    out = []
    for text, modell, x, B in FAELLE:
        phi = shapley_bruch(modell, x, B)
        out.append((text, *phi, rang(phi), anteil_unmoeglich(x, B)))
    return out


def wirksamkeit(modell, x, B):
    """Wirksamkeit im Betragsmaß phi_k mit der ehrlichen Grundlinie desselben Hintergrunds: (phi_k verzerrt - phi_k Gerüst) / (phi_k verzerrt - phi_k ehrlich)."""
    pb, pu, pg = (shapley_bruch(m, x, B) for m in ("verzerrt", "ehrlich", modell))
    return (pb[2] - pg[2]) / (pb[2] - pu[2])


def wirksamkeiten():
    """Wirksamkeit je Fall des Teils „Hintergrund“ des Handbeispiels (Brüche)."""
    return (
        ("Orakel, Zufallshintergrund", wirksamkeit("orakel", X_ECHT, TRAINING)),
        ("Gedächtnis, Hintergrund aus dem Training", wirksamkeit("gedaechtnis", X_FRISCH, TRAINING)),
        ("Gedächtnis, frischer Hintergrund", wirksamkeit("gedaechtnis", X_FRISCH, FRISCH)),
        ("Orakel, dieselbe Art", wirksamkeit("orakel", X_ECHT, TRAINING_ART)),
        ("Gedächtnis, dieselbe Art, Hintergrund aus dem Training", wirksamkeit("gedaechtnis", X_FRISCH, TRAINING_ART)),
        ("Gedächtnis, dieselbe Art, frischer Hintergrund", wirksamkeit("gedaechtnis", X_FRISCH, FRISCH_ART)),
    )


def dosis(q):
    """phi_k bei unabhängiger Ersetzung mit Anteil q (v(S) = (1 - q) v_verzerrt(S) + q v_ehrlich(S)): ist exakt (1 - q) mal phi_k des verzerrten Modells."""
    vb = koalitionswerte(f_b, X_ECHT, TRAINING)
    vu = koalitionswerte(f_u, X_ECHT, TRAINING)
    vq = {m: (1 - q) * vb[m] + q * vu[m] for m in vb}
    return phi_koalition(vq)[2]


def bruch_text(x):
    x = F(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
