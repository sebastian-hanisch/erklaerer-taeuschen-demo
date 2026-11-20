"""Kennzahlen und Verdikte aus dem eingefrorenen Raster (data/ert_grid.json). Verdikte sind reine Funktionen (Art, Text) auf aggregierten Zahlen.

Zähleinheit überall: Zufalls-Seeds (je Zufalls-Seed ein Datensatz, Modelle, Detektoren und 24 erklärte Zeilen, beim Surrogat 16). Mittel ± Standardfehler laufen über Zufalls-Seeds,
nie über Zeilen. Wirksamkeit = (Rang-1-Anteil verzerrt - Rang-1-Anteil Gerüst) / (Rang-1-Anteil verzerrt - Rang-1-Anteil ehrlich), jeweils mit der ehrlichen Grundlinie des
gleichen Hintergrunds, aus den Mitteln über Zufalls-Seeds gerechnet (Standardfehler des Quotienten über die Zufalls-Seeds)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

import ert_constants as C

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def lade_raster():
    return json.loads((ROOT / C.GRID_DATEI).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def lade_zweit():
    """Nachmessung mit anderen Zufalls-Seeds (nur Boosting und Zufallswald, c = 1,0): nicht Teil der App, sondern der README und ihrer Tests."""
    return json.loads((ROOT / C.ZWEIT_DATEI).read_text(encoding="utf-8"))


def seeds(raster=None):
    r = raster or lade_raster()
    return [r["seeds"][str(s)] for s in r["meta"]["seeds"]]


def de(x, nd=2):
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=1):
    return de(100 * x, nd) + " %"


def mean_se(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def hg_key(quelle, p):
    return f"{quelle}_{p:g}"


def c_key(c):
    return f"{c:g}"


# ---------------------------------------------------------------- Wirksamkeit

def wirksamkeit_aus_mitteln(verz, ehr, ger):
    """verz, ehr, ger: Listen je Zufalls-Seed (Rang-1-Anteile). Rückgabe (w, Standardfehler) oder (None, None), wenn der Abstand verzerrt - ehrlich nicht von 0 verschieden ist."""
    verz, ehr, ger = (np.asarray(a, float) for a in (verz, ehr, ger))
    de_, dg = verz - ehr, verz - ger
    if abs(de_.mean()) < 1e-9:
        return None, None
    w = dg.mean() / de_.mean()
    if len(verz) < 2:
        return float(w), 0.0
    resid = dg - w * de_
    return float(w), float(resid.std(ddof=1) / np.sqrt(len(verz)) / abs(de_.mean()))


def feldliste(rows, feld):
    return [r[feld] for r in rows]


def shapley_zellen(raster, base, c, quelle, p, modell):
    """Maße je Zufalls-Seed (Liste von Wörterbüchern) für Modell ('verzerrt', 'ehrlich', 'geruest_<det>') bei Aufschlag c und Hintergrund (quelle, p)."""
    return [s["shapley"][base][c_key(c)][hg_key(quelle, p)][modell] for s in seeds(raster)]


def shapley_zusammenfassung(raster, base, c, quelle, p, det):
    """Rang-1-Anteil (Mittel ± Standardfehler), Wichtigkeitsanteil, globaler Rang und Wirksamkeit für verzerrt, ehrlich und das Gerüst mit Detektor `det`."""
    out = {}
    for name, modell in (("verzerrt", "verzerrt"), ("ehrlich", "ehrlich"), ("geruest", f"geruest_{det}")):
        z = shapley_zellen(raster, base, c, quelle, p, modell)
        t1, an = feldliste(z, "top1"), feldliste(z, "anteil")
        out[name] = dict(top1=mean_se(t1), anteil=mean_se(an), mrang=mean_se(feldliste(z, "mrang")), rang1_global=int(sum(1 for r in z if r["grang"] == 1)),
                         spitzen=_zaehle(feldliste(z, "spitze")), rk=mean_se([r["rk"] for r in z if r["rk"] == r["rk"]]) if name != "verzerrt" else (1.0, 0.0), top1_je_seed=t1)
    w, se = wirksamkeit_aus_mitteln(out["verzerrt"]["top1_je_seed"], out["ehrlich"]["top1_je_seed"], out["geruest"]["top1_je_seed"])
    out["wirksamkeit"] = (w, se)
    # Wirksamkeit im Betragsmaß (Anteil des Kunden an der Summe der Wichtigkeiten)
    wb = wirksamkeit_aus_mitteln(feldliste(shapley_zellen(raster, base, c, quelle, p, "verzerrt"), "anteil"), feldliste(shapley_zellen(raster, base, c, quelle, p, "ehrlich"), "anteil"),
                                 feldliste(shapley_zellen(raster, base, c, quelle, p, f"geruest_{det}"), "anteil"))
    out["wirksamkeit_betrag"] = wb
    out["n"] = len(out["verzerrt"]["top1_je_seed"])
    out["unmoeglich"] = mean_se([s["unmoeglich_shapley"][hg_key(quelle, p)] for s in seeds(raster)])
    out["q"] = mean_se([s["ersetzung_q"][hg_key(quelle, p)][det] for s in seeds(raster)])
    return out


def _zaehle(liste):
    out = {}
    for x in liste:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def surrogat_zusammenfassung(raster, base, quelle, var, det):
    out = {}
    for name, modell in (("verzerrt", "verzerrt"), ("ehrlich", "ehrlich"), ("geruest", f"geruest_{det}")):
        z = [s["surrogat"][base][quelle][var][modell] for s in seeds(raster)]
        out[name] = dict(top1=mean_se(feldliste(z, "top1")), anteil=mean_se(feldliste(z, "anteil")), mrang=mean_se(feldliste(z, "mrang")), top1_je_seed=feldliste(z, "top1"))
    out["wirksamkeit"] = wirksamkeit_aus_mitteln(out["verzerrt"]["top1_je_seed"], out["ehrlich"]["top1_je_seed"], out["geruest"]["top1_je_seed"])
    out["unmoeglich"] = mean_se([s["unmoeglich_surrogat"][f"{quelle}_{var}"] for s in seeds(raster)])
    out["n"] = len(out["verzerrt"]["top1_je_seed"])
    return out


def detektor_zusammenfassung(raster):
    """Je Detektor Mittel ± Standardfehler: Erkennung (Spaltenmischung, Shapley-Mischzeilen), Fehlalarm (mögliche Mischzeilen, echte Zeilen), Treue (= 1 - Fehlalarm echt)."""
    out = {}
    ss = seeds(raster)
    for det in C.DETEKTOREN:
        zeilen = [s["detektor"][det] for s in ss]
        out[det] = {k: mean_se(feldliste(zeilen, k)) for k in ("erk_spalten", "erk_shapley", "fehl_moeglich", "fehl_echt")}
        out[det]["treue"] = (1.0 - out[det]["fehl_echt"][0], out[det]["fehl_echt"][1])
    out["anteil_unmoeglich"] = {k: mean_se([s["detektor"]["anteil_unmoeglich"][k] for s in ss]) for k in ("spalten", "shapley_zufall")}
    out["n"] = len(ss)
    return out


def abstand_zusammenfassung(raster, base, c):
    """Mittleres |g_b - g_h| auf echten frischen Zeilen (Verzerrung der Vorhersage) und je Detektor mittleres |h - g_b|."""
    zeilen = [s["abstand"][base][c_key(c)] for s in seeds(raster)]
    return {k: mean_se(feldliste(zeilen, k)) for k in zeilen[0]}


def wissen_zusammenfassung(raster, base):
    out = []
    ss = seeds(raster)
    for nm in C.WISSEN_STUFEN:
        z = [s["wissen"][base][str(nm)] for s in ss]
        out.append(dict(n=nm, erk=mean_se(feldliste(z, "erk_shapley")), fehl=mean_se(feldliste(z, "fehl_echt")), top1=mean_se(feldliste(z, "top1_geruest")), q=mean_se(feldliste(z, "q"))))
    return out


# ---------------------------------------------------------------- Verdikte (reine Funktionen)

def verdict_betrieb(det, erk, fehl_echt, treue, n):
    """Experiment 1: wie gut trennt der Detektor unmögliche von echten Zeilen? erk = Erkennung unmöglicher Mischzeilen, fehl_echt = Anteil echter Zeilen, die ehrlich beantwortet werden."""
    name = C.DET_LABEL_KURZ[det]
    if erk < 0.6:
        return "warning", (f"In diesem Lauf ({n} Zufalls-Seeds) erkennt der Detektor „{name}“ nur {pct(erk)} der unmöglichen Mischzeilen und meldet {pct(fehl_echt)} der echten Zeilen "
                           "als unmöglich: er liegt nahe am Zufall, das Gerüst kann sich damit nicht verstecken.")
    if fehl_echt > 0.05:
        return "info", (f"In diesem Lauf ({n} Zufalls-Seeds) liefert das Gerüst mit „{name}“ auf {pct(treue)} der echten Zeilen die verzerrte Vorhersage und erkennt {pct(erk)} der unmöglichen "
                        f"Mischzeilen; auf {pct(fehl_echt)} der echten Zeilen antwortet es ehrlich, der Betrieb ist dort selbst weniger verzerrt.")
    return "success", (f"In diesem Lauf ({n} Zufalls-Seeds) liefert das Gerüst mit „{name}“ auf {pct(treue)} der echten Zeilen die verzerrte Vorhersage und erkennt {pct(erk)} der "
                       "unmöglichen Mischzeilen.")


def verdict_shapley(z, hintergrund_art, geruest_ist_orakel, c, n):
    """Experiment 2. z = shapley_zusammenfassung(...). hintergrund_art: Hintergrund derselben Art (p = 1)."""
    w, se = z["wirksamkeit"]
    k = z["geruest"]["rang1_global"]
    ku = z["verzerrt"]["rang1_global"]
    if w is None:
        return "info", "Ohne Abstand zwischen verzerrtem und ehrlichem Modell ist die Wirksamkeit nicht definiert."
    if hintergrund_art and geruest_ist_orakel and abs(w) < 1e-9:
        return "success", (f"Bei Hintergrund derselben Art ist das Gerüst unsichtbar und wirkungslos: die Beiträge sind in allen {n} Zufalls-Seeds identisch zu denen des verzerrten Modells "
                           "(Wirksamkeit 0, exakt).")
    if ku <= n // 2:
        return "info", (f"Schon ohne Gerüst steht die Kundengruppe nur in {ku} von {n} Zufalls-Seeds global auf Rang 1; die Wirksamkeit von {de(w)} ± {de(se)} sagt hier über das Gerüst wenig, "
                        "weil das verzerrte Modell unter diesem Hintergrund kaum sichtbar ist.")
    if w >= 0.5:
        return "warning", (f"Das Gerüst täuscht hier deutlich: Wirksamkeit {de(w)} ± {de(se)} (Anteil des Abstands zum ehrlichen Modell, den es verdeckt); die Kundengruppe steht noch in "
                           f"{k} von {n} Zufalls-Seeds global auf Rang 1.")
    if k == n:
        return "info", (f"Die Kundengruppe bleibt in {k} von {n} Zufalls-Seeds global auf Rang 1; die Wirksamkeit liegt bei {de(w)} ± {de(se)} (Standardfehler über Zufalls-Seeds): "
                        "die Täuschung gelingt nur schwach.")
    return "info", (f"Die Wirksamkeit liegt bei {de(w)} ± {de(se)}; die Kundengruppe bleibt in {k} von {n} Zufalls-Seeds global auf Rang 1 (ohne Gerüst in {ku}).")


def verdict_surrogat(var, z, n):
    """Experiment 3. var: Stichprobenart. z = surrogat_zusammenfassung(...)."""
    w, se = z["wirksamkeit"]
    ung = z["unmoeglich"][0]
    if w is None:
        return "info", "Ohne Abstand zwischen verzerrtem und ehrlichem Modell ist die Wirksamkeit nicht definiert."
    if var == "zeile" and abs(w) < 1e-9:
        return "success", (f"Mit ganzen echten Zeilen enthält die Stichprobe {pct(ung)} unmögliche Punkte: das Surrogat sieht in allen {n} Zufalls-Seeds das verzerrte Modell, die "
                           "Wirksamkeit ist 0.")
    if w >= 0.4:
        return "warning", (f"Das Surrogat lässt sich hier täuschen: {pct(ung)} der Stichprobenpunkte sind unmöglich, die Wirksamkeit liegt bei {de(w)} ± {de(se)} ({n} Zufalls-Seeds). "
                           "Entscheidend ist die Stichprobenart, nicht das Verfahren.")
    return "info", f"{pct(ung)} der Stichprobenpunkte sind unmöglich; die Wirksamkeit liegt bei {de(w)} ± {de(se)} ({n} Zufalls-Seeds)."


def verdict_angreifer(w_ang, se_ang, w_orakel, se_orakel, quelle, hintergrund_art, n):
    """Experiment 4: Gedächtnis-Angreifer (w_ang) gegen das Orakel-Gerüst (w_orakel) bei gleichem Hintergrund."""
    if w_ang is None or w_orakel is None:
        return "info", "Ohne Abstand zwischen verzerrtem und ehrlichem Modell ist die Wirksamkeit nicht definiert."
    art = "derselben Art" if hintergrund_art else "aus Zufall"
    qn = "Trainingszeilen" if quelle == "training" else "frischen Zeilen"
    if quelle == "training":
        if w_ang >= w_orakel + 0.3:
            return "warning", (f"Wer die Trainingszeilen kennt, täuscht Shapley deutlich: Wirksamkeit {de(w_ang)} ± {de(se_ang)} gegen {de(w_orakel)} ± {de(se_orakel)} für das Orakel-Gerüst "
                               f"(Hintergrund {art} aus {qn}, {n} Zufalls-Seeds).")
        return "info", f"Wirksamkeit des Gedächtnis-Angreifers {de(w_ang)} ± {de(se_ang)} gegen {de(w_orakel)} ± {de(se_orakel)} für das Orakel-Gerüst (Hintergrund {art} aus {qn}, {n} Zufalls-Seeds)."
    if abs(w_ang - w_orakel) < 0.05:
        return "success", (f"Mit frischem Hintergrund bringt das Gedächtnis nichts dazu: Wirksamkeit {de(w_ang)} ± {de(se_ang)}, die des Orakel-Gerüsts ist {de(w_orakel)} ± {de(se_orakel)} "
                           f"(Hintergrund {art}, {n} Zufalls-Seeds).")
    return "info", f"Wirksamkeit des Gedächtnis-Angreifers {de(w_ang)} ± {de(se_ang)} gegen {de(w_orakel)} ± {de(se_orakel)} für das Orakel-Gerüst (frischer Hintergrund {art}, {n} Zufalls-Seeds)."


def hinweis_treue(det, treue):
    """Hinweis, wenn der Detektor echte Zeilen ehrlich beantworten lässt: ein Teil der Wirksamkeit ist dann Modellwechsel im Betrieb, nicht Verbergen."""
    if treue >= 0.95:
        return ""
    return (f"Hinweis: Das Gerüst mit „{C.DET_LABEL_KURZ[det]}“ liefert nur auf {pct(treue)} der echten Zeilen die verzerrte Vorhersage (Experiment 1); auf den übrigen antwortet es ehrlich. "
            "Ein Teil dieser Wirksamkeit ist deshalb Modellwechsel im Betrieb, nicht Verbergen vor dem Erklärer (nicht getrennt gemessen).")
