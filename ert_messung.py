"""Die Messung je Zufalls-Seed (alle Zahlen des eingefrorenen Rasters kommen von hier; `tools/ert_make_grid.py` ruft sie auf).

Je Zufalls-Seed: ein eigener Datensatz (3 000 Trainingszeilen), je Modellklasse ein ehrliches Modell g_h (Ziel ln Verweilzeit) und je Aufschlag c ein verzerrtes
Modell g_b (Ziel ln Verweilzeit + c * (Kundengruppe - 1/2)), die Detektoren, 24 erklärte frische Zeilen. Rang-1-Anteil, Wichtigkeit und Wirksamkeit stehen in ert_geruest.

Zähleinheiten: Zufalls-Seeds; je Zufalls-Seed 24 erklärte Zeilen (Surrogat 16), Hintergrund 30 Zeilen, 64 Koalitionen = 1 920 Modellauswertungen je Erklärung
(Surrogat N = 1 000). Zeit ist kein Befund."""

import numpy as np

import ert_constants as C
from ert_generator import block_below, block_random, gruppe, strom, zeichne_daten, zeichne_zeilen
from ert_geruest import baue_detektoren, geruest, masse, mische_spalten, rangkorrelation
from ert_lime import koeffizienten, quartilgrenzen, stichprobe, gewichte
from ert_models import DetektorModell, mache_modell
from ert_shapley import mischzeilen, shapley_aus_werten, tabelle, werte_aus_vorhersagen
from ert_support import gueltig

TAG_TRAIN, TAG_MIX, TAG_ERKL, TAG_FRISCH, TAG_TEST, TAG_TEST2, TAG_TESTMIX, TAG_KOAL, TAG_HINT, TAG_LIME, TAG_WISSEN = 1, 2, 3, 4, 5, 6, 7, 8, 100, 200, 300


def hintergruende():
    """Liste der Hintergründe (Quelle, Anteil p derselben Art): Training mit allen p, frisch nur mit p = 0 und 1."""
    out = [("training", p) for p in C.P_STUFEN]
    out += [("frisch", 0.0), ("frisch", 1.0)]
    return out


def hg_key(quelle, p):
    return f"{quelle}_{p:g}"


def hintergrund_gruppen(Xe, Q, p, nb, rng):
    """Zerlegt die erklärten Zeilen in Gruppen mit je eigenem Hintergrund B. p = 0: eine Gruppe, Zufallshintergrund (nb Zeilen aus Q). p > 0: je Art eine Gruppe,
    Anteil p der Hintergrundzeilen aus Zeilen derselben Art, der Rest zufällig aus allen Zeilen von Q. Rückgabe: Liste (Zeilenindizes, B)."""
    if p == 0.0:
        return [(np.arange(len(Xe)), Q[_ziehe(rng, len(Q), nb)])]
    out = []
    for a in np.unique(Xe[:, C.COL_ART]):
        idx = np.where(Xe[:, C.COL_ART] == a)[0]
        same = Q[Q[:, C.COL_ART] == a]
        k = int(round(p * nb))
        teile = []
        if k > 0:
            teile.append(same[_ziehe(rng, len(same), k)])
        if nb - k > 0:
            teile.append(Q[_ziehe(rng, len(Q), nb - k)])
        out.append((idx, np.vstack(teile)))
    return out


def _ziehe(rng, n, k):
    """k verschiedene Indizes aus 0..n-1 (Teilmischung der Reihenfolge mit SplitMix64)."""
    perm = list(range(n))
    for i in range(k):
        j = i + int(block_below(rng, 1, n - i)[0])
        perm[i], perm[j] = perm[j], perm[i]
    return np.array(perm[:k])


# ---------------------------------------------------------------- Detektor-Kennzahlen (Experiment 1)

def detektor_kennzahlen(dets, X, Xtest, Xtest2, seed):
    """Je Detektor: Erkennung unmöglicher Zeilen unter Spaltenmischung und unter Shapley-Mischzeilen (Zufallshintergrund), Fehlalarm auf möglichen Mischzeilen
    und auf echten frischen Zeilen, Anteil unmöglicher Zeilen je Art der Stichprobe."""
    n = len(Xtest)
    sm = mische_spalten(Xtest, n, strom(seed, TAG_TESTMIX))
    koal = block_random(strom(seed, TAG_KOAL), n * C.D).reshape(n, C.D) < 0.5
    sh = np.where(koal, Xtest, Xtest2)                  # Mischzeile (x_S, b_Rest): Koalition gleichverteilt, Hintergrundzeile aus einem zweiten frischen Satz
    g_sm, g_sh = gueltig(sm), gueltig(sh)
    out = {"anteil_unmoeglich": {"spalten": float((~g_sm).mean()), "shapley_zufall": float((~g_sh).mean())}}
    for name, det in dets.items():
        o_sm, o_sh, o_echt = det.ood(sm), det.ood(sh), det.ood(Xtest)
        out[name] = dict(erk_spalten=float(o_sm[~g_sm].mean()), erk_shapley=float(o_sh[~g_sh].mean()),
                         fehl_moeglich=float(o_sh[g_sh].mean()) if g_sh.any() else 0.0, fehl_echt=float(o_echt.mean()))
    return out


# ---------------------------------------------------------------- Shapley-Raster (Experimente 2 und 4)

def _kopf(m):
    return {k: (round(float(v), 6) if not isinstance(v, (int, np.integer)) else int(v)) for k, v in m.items() if k != "wich"}


def messe_shapley(seed, modelle, dets, X, Xe, Xf, cs, bases, hgs=None):
    """modelle: {base: {'gh': Modell, ('gb', c): Modell}}. Rückgabe: {base: {c: {hg_key: {modell: Maße}}}}, {hg_key: Anteil unmöglich}, {hg_key: {det: q}}."""
    d = C.D
    res = {b: {c: {} for c in cs} for b in bases}
    bad, qq = {}, {}
    for ih, (quelle, p) in enumerate(hintergruende()):
        if hgs is not None and (quelle, p) not in hgs:
            continue
        Q = X if quelle == "training" else Xf
        key = hg_key(quelle, p)
        gruppen = hintergrund_gruppen(Xe, Q, p, C.NB, strom(seed, TAG_HINT + ih))
        phis = {(b, c, m): np.zeros((len(Xe), d)) for b in bases for c in cs for m in ["verzerrt", "ehrlich"] + [f"geruest_{n}" for n in dets]}
        nbad, ntot = 0.0, 0
        qsum = {n: 0.0 for n in dets}
        for idx, B in gruppen:
            H = mischzeilen(Xe[idx], B)
            fl = {n: det.ood(H) for n, det in dets.items()}
            nbad += float((~gueltig(H)).sum())
            ntot += len(H)
            for n in dets:
                qsum[n] += float(fl[n].sum())
            for b in bases:
                pu = modelle[b]["gh"].predict(H)
                phis_u = shapley_aus_werten(werte_aus_vorhersagen(pu, len(idx), d, C.NB), d)
                for c in cs:
                    pb = modelle[b][("gb", c)].predict(H)
                    phis[(b, c, "verzerrt")][idx] = shapley_aus_werten(werte_aus_vorhersagen(pb, len(idx), d, C.NB), d)
                    phis[(b, c, "ehrlich")][idx] = phis_u
                    for n in dets:
                        ph = geruest(pb, pu, fl[n])
                        phis[(b, c, f"geruest_{n}")][idx] = shapley_aus_werten(werte_aus_vorhersagen(ph, len(idx), d, C.NB), d)
        bad[key] = round(nbad / ntot, 6)
        qq[key] = {n: round(qsum[n] / ntot, 6) for n in dets}
        for b in bases:
            for c in cs:
                ref = masse(phis[(b, c, "verzerrt")])
                zelle = {}
                for m in ["verzerrt", "ehrlich"] + [f"geruest_{n}" for n in dets]:
                    mm = masse(phis[(b, c, m)])
                    mm["rk"] = rangkorrelation(mm["wich"], ref["wich"])
                    zelle[m] = _kopf(mm)
                res[b][c][key] = zelle
    return res, bad, qq


# ---------------------------------------------------------------- Surrogat-Raster (Experiment 3)

def messe_surrogat(seed, modelle, dets, X, Xe16, Xf, bases):
    """Je Basis (c = 1): {quelle: {var: {modell: {top1, mrang, anteil}}}} und der Anteil unmöglicher Stichprobenpunkte je (quelle, var)."""
    grenzen = quartilgrenzen(X)
    res = {b: {} for b in bases}
    inv = {}
    for qi, (quelle, Q) in enumerate((("training", X), ("frisch", Xf))):
        for vi, var in enumerate(C.STICHPROBEN):
            coefs = {b: {} for b in bases}
            ung = []
            for ix, x0 in enumerate(Xe16):
                rng = strom(seed, TAG_LIME + 10 * qi + vi * 100 + ix * 1000 + 1)
                Sm, Z = stichprobe(var, Q, x0, C.N_LIME, rng, grenzen)
                w = gewichte(Z)
                ung.append(float((~gueltig(Sm)).mean()))
                fl = {n: det.ood(Sm) for n, det in dets.items()}
                for b in bases:
                    pb = modelle[b][("gb", C.C_HAUPT)].predict(Sm)
                    pu = modelle[b]["gh"].predict(Sm)
                    preds = {"verzerrt": pb, "ehrlich": pu}
                    for n in dets:
                        preds[f"geruest_{n}"] = geruest(pb, pu, fl[n])
                    for m, pv in preds.items():
                        coefs[b].setdefault(m, []).append(np.abs(koeffizienten(Z, w, pv)))
            inv[f"{quelle}_{var}"] = round(float(np.mean(ung)), 6)
            for b in bases:
                zelle = {}
                for m, lst in coefs[b].items():
                    Cm = np.array(lst)
                    r = ((-Cm).argsort(axis=1).argsort(axis=1) + 1)[:, C.COL_KUNDE]
                    zelle[m] = dict(top1=round(float((r == 1).mean()), 6), mrang=round(float(r.mean()), 6), anteil=round(float(Cm[:, C.COL_KUNDE].sum() / Cm.sum()), 6))
                res[b].setdefault(quelle, {})[var] = zelle
    return res, inv


# ---------------------------------------------------------------- Wissen des Angreifers (Tab unter Experiment 1)

def messe_wissen(seed, modelle, X, Xe, bases_wissen, Xtest, Xtest2):
    """Wissen des Angreifers: der Boosting-Detektor wird je Stufe mit n echten Zeilen und n Spaltenmischungen trainiert (n = 30 bis 3 000); Erkennung unmöglicher
    Shapley-Mischzeilen, Fehlalarm auf echten frischen Zeilen und Rang-1-Anteil des Gerüsts (Zufallshintergrund aus Trainingszeilen, c = 1)."""
    d = C.D
    gruppen = hintergrund_gruppen(Xe, X, 0.0, C.NB, strom(seed, TAG_HINT))     # derselbe Hintergrund wie im Raster
    idx, B = gruppen[0]
    H = mischzeilen(Xe, B)
    n = len(Xtest)
    koal = block_random(strom(seed, TAG_KOAL), n * C.D).reshape(n, C.D) < 0.5
    sh = np.where(koal, Xtest, Xtest2)
    g_sh = gueltig(sh)
    out = {b: {} for b in bases_wissen}
    pbs = {b: modelle[b][("gb", C.C_HAUPT)].predict(H) for b in bases_wissen}
    pus = {b: modelle[b]["gh"].predict(H) for b in bases_wissen}
    for k, nm in enumerate(C.WISSEN_STUFEN):
        Xmix = mische_spalten(X, nm, strom(seed, TAG_WISSEN + k))
        det = DetektorModell("gbm", X[:nm], Xmix, seed)
        o_sh, o_echt, oh = det.ood(sh), det.ood(Xtest), det.ood(H)
        for b in bases_wissen:
            ph = geruest(pbs[b], pus[b], oh)
            top1 = masse(shapley_aus_werten(werte_aus_vorhersagen(ph, len(Xe), d, C.NB), d))["top1"]
            out[b][str(nm)] = dict(erk_shapley=round(float(o_sh[~g_sh].mean()), 6), fehl_echt=round(float(o_echt.mean()), 6), top1_geruest=round(float(top1), 6),
                                   q=round(float(oh.mean()), 6))
    return out


# ---------------------------------------------------------------- ein Zufalls-Seed

def messe_seed(seed, bases=C.BASEN, cs=C.C_STUFEN, mit_surrogat=True, mit_wissen=True, ne=C.NE_SHAPLEY, n_train=C.N_TRAIN, n_mix=C.N_MIX, n_test=C.N_TEST,
               n_treue=C.N_TREUE, detektoren=C.DETEKTOREN, hgs=None, log=None):
    """Alle Zellen eines Zufalls-Seeds als JSON-fähiges Wörterbuch."""
    say = log or (lambda *a: None)
    X, y = zeichne_daten(n_train, seed, TAG_TRAIN)
    Xmix = mische_spalten(X, n_mix, strom(seed, TAG_MIX))
    dets = baue_detektoren(X, Xmix, seed, detektoren)
    Xe = zeichne_zeilen(strom(seed, TAG_ERKL), ne)
    Xf = zeichne_zeilen(strom(seed, TAG_FRISCH), n_train)
    Xtest = zeichne_zeilen(strom(seed, TAG_TEST), n_test)
    Xtest2 = zeichne_zeilen(strom(seed, TAG_TEST2), n_test)
    say(f"seed {seed}: Daten und Detektoren fertig")
    modelle = {}
    abstand = {}
    for b in bases:
        modelle[b] = {"gh": mache_modell(b, seed).fit(X, y)}
        for c in cs:
            modelle[b][("gb", c)] = mache_modell(b, seed).fit(X, y + c * gruppe(X))
        say(f"seed {seed}: Modelle {b} fertig")
    res = {"seed": seed}
    res["detektor"] = detektor_kennzahlen(dets, X, Xtest, Xtest2, seed)
    # Abstand der Vorhersagen auf echten frischen Zeilen: mittleres |g_b - g_h| und mittleres |h - g_b| je Detektor (nur dort, wo der Detektor ehrlich antworten lässt)
    Xtr = Xtest[:n_treue]
    fl_echt = {n: det.ood(Xtr) for n, det in dets.items()}
    for b in bases:
        pu = modelle[b]["gh"].predict(Xtr)
        for c in cs:
            pb = modelle[b][("gb", c)].predict(Xtr)
            zel = {"verz_ehrlich": round(float(np.abs(pb - pu).mean()), 6)}
            for n in dets:
                zel[n] = round(float(np.abs(geruest(pb, pu, fl_echt[n]) - pb).mean()), 6)
            abstand.setdefault(b, {})[f"{c:g}"] = zel
    res["abstand"] = abstand
    sh, bad, qq = messe_shapley(seed, modelle, dets, X, Xe, Xf, cs, bases, hgs)
    res["shapley"] = {b: {f"{c:g}": sh[b][c] for c in cs} for b in bases}
    res["unmoeglich_shapley"] = bad
    res["ersetzung_q"] = qq
    say(f"seed {seed}: Shapley fertig")
    if mit_surrogat:
        sg, inv = messe_surrogat(seed, modelle, dets, X, Xe[:C.NE_LIME], Xf, bases)
        res["surrogat"] = sg
        res["unmoeglich_surrogat"] = inv
        say(f"seed {seed}: Surrogat fertig")
    if mit_wissen:
        bw = [b for b in bases if b in ("gbm", "rf")]
        res["wissen"] = messe_wissen(seed, modelle, X, Xe, bw, Xtest, Xtest2)
        say(f"seed {seed}: Wissen fertig")
    return res
