"""Claims-Tests: jede Zahl der README steht hier, aus dem eingefrorenen Raster (data/ert_grid.json, 10 Zufalls-Seeds) beziehungsweise der Nachmessung (data/ert_zweit.json,
6 andere Zufalls-Seeds) neu aufgerechnet und in der README gesucht. Jede Absolutaussage hat eine Gegenprobe (Zählung über alle Zellen).

Zahl-Test-Tabelle der README: die Tests heißen dort. Es sind Messungen des Rasters (kein Neu-Rechnen); das Neu-Rechnen eines Zufalls-Seeds im Band steht in test_messung_band.py."""

from pathlib import Path

import pytest

import ert_constants as C
import ert_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
R = ev.lade_raster()
Z = ev.lade_zweit()
N = len(R["meta"]["seeds"])


def d2(x):
    return ev.de(x, 2)


def ms(w):
    return f"{d2(w[0])} ± {d2(w[1])}"


def im_readme(text):
    assert text in README, f"nicht in der README: {text!r}"


def shp(base, c, q, p, det, raster=R):
    return ev.shapley_zusammenfassung(raster, base, c, q, p, det)


def test_unmoegliche_anteile():
    dz = ev.detektor_zusammenfassung(R)
    assert ev.pct(dz["anteil_unmoeglich"]["spalten"][0]) == "56,0 %"
    assert ev.pct(dz["anteil_unmoeglich"]["shapley_zufall"][0]) == "28,2 %"
    assert all(s["unmoeglich_shapley"]["training_1"] == 0.0 and s["unmoeglich_shapley"]["frisch_1"] == 0.0 for s in ev.seeds(R))
    im_readme("Unter Spaltenmischung 56,0 %, unter Shapley-Mischzeilen mit Zufallshintergrund 28,2 %, bei Hintergrund derselben Art exakt 0")


def test_detektoren():
    dz = ev.detektor_zusammenfassung(R)
    zeilen = {d: f"{ev.pct(dz[d]['erk_shapley'][0])} / {ev.pct(dz[d]['fehl_echt'][0])}" for d in ("orakel", "gbm", "rf", "linear")}
    assert zeilen == {"orakel": "100,0 % / 0,0 %", "gbm": "98,2 % / 1,1 %", "rf": "99,2 % / 7,5 %", "linear": "49,4 % / 48,7 %"}
    for text in ("Orakel 100,0 % / 0,0 %", "Boosting 98,2 % / 1,1 %", "Zufallswald 99,2 % / 7,5 %", "49,4 % / 48,7 %"):
        im_readme(text)
    # Gegenprobe zur Absolutaussage „lineares Modell nahe am Zufall“: in jedem Zufalls-Seed zwischen 0,4 und 0,6
    assert all(0.4 < s["detektor"]["linear"]["erk_shapley"] < 0.6 for s in ev.seeds(R))
    assert ev.pct(dz["rf"]["treue"][0]) == "92,5 %"
    im_readme("Treue 92,5 % gegen 87,3 %")


def test_shapley_zufallshintergrund():
    erwartet = {"gbm": ("0,61", "0,54", "0,19", "0,17 ± 0,03", "0,05 ± 0,02"), "rf": ("0,70", "0,53", "0,07", "0,28 ± 0,04", "0,24 ± 0,01")}
    for base, (v, g, e, w, wb) in erwartet.items():
        z = shp(base, 1.0, "training", 0.0, "orakel")
        assert (d2(z["verzerrt"]["top1"][0]), d2(z["geruest"]["top1"][0]), d2(z["ehrlich"]["top1"][0])) == (v, g, e)
        assert ms(z["wirksamkeit"]) == w and ms(z["wirksamkeit_betrag"]) == wb
        assert z["geruest"]["rang1_global"] == N and z["verzerrt"]["rang1_global"] == N
    im_readme("B 0,61 → 0,54 → 0,19, W 0,70 → 0,53 → 0,07")
    im_readme("**Wirksamkeit B 0,17 ± 0,03, W 0,28 ± 0,04** (Betragsmaß, Anteil an der Wichtigkeit: 0,05 ± 0,02 und 0,24 ± 0,01)")
    im_readme("**Die Kundengruppe bleibt in 10 von 10 Zufalls-Seeds global auf Rang 1**")
    assert N == 10


def test_gleiche_art_ist_exakt_wirkungslos():
    n = 0
    for s in ev.seeds(R):
        for b in C.BASEN:
            for c in C.C_STUFEN:
                for q in C.QUELLEN:
                    z = s["shapley"][b][ev.c_key(c)][ev.hg_key(q, 1.0)]
                    assert z["geruest_orakel"] == z["verzerrt"]
                    n += 1
    assert n == 180
    im_readme("in allen 180 Zellen (10 Zufalls-Seeds, 3 Modellklassen, 3 Aufschläge, 2 Hintergrundquellen)")
    # Gegenprobe: bei Zufallshintergrund unterscheidet sich das Orakel-Gerüst vom verzerrten Modell in den meisten Zellen
    unterschiedlich = sum(1 for s in ev.seeds(R) for b in C.BASEN for c in C.C_STUFEN
                          if s["shapley"][b][ev.c_key(c)]["training_0"]["geruest_orakel"] != s["shapley"][b][ev.c_key(c)]["training_0"]["verzerrt"])
    assert unterschiedlich > 0.9 * 90


def test_dosis():
    erwartet = {"gbm": ["0,17", "0,09", "0,07", "0,04", "0,00"], "rf": ["0,28", "0,15", "0,08", "0,06", "0,00"]}
    for base, werte in erwartet.items():
        assert [d2(shp(base, 1.0, "training", p, "orakel")["wirksamkeit"][0]) for p in C.P_STUFEN] == werte
    im_readme("B 0,17 / 0,09 / 0,07 / 0,04 / 0,00, W 0,28 / 0,15 / 0,08 / 0,06 / 0,00")
    assert ev.pct(shp("gbm", 1.0, "training", 0.0, "orakel")["unmoeglich"][0]) == "28,4 %"
    im_readme("(28,4 % → 0)")


def test_surrogat():
    zs = {(b, v): ev.surrogat_zusammenfassung(R, b, "training", v, "orakel") for b in ("gbm", "rf") for v in C.STICHPROBEN}
    assert [ev.pct(zs[("gbm", v)]["unmoeglich"][0]) for v in C.STICHPROBEN] == ["56,2 %", "42,2 %", "0,0 %"]
    assert (ms(zs[("gbm", "std")]["wirksamkeit"]), ms(zs[("rf", "std")]["wirksamkeit"])) == ("0,76 ± 0,04", "0,86 ± 0,03")
    assert (ms(zs[("gbm", "koal")]["wirksamkeit"]), ms(zs[("rf", "koal")]["wirksamkeit"])) == ("0,39 ± 0,06", "0,54 ± 0,04")
    assert zs[("gbm", "zeile")]["wirksamkeit"][0] == 0 and zs[("rf", "zeile")]["wirksamkeit"][0] == 0
    assert all(s["surrogat"][b][q]["zeile"]["geruest_orakel"] == s["surrogat"][b][q]["zeile"]["verzerrt"] for s in ev.seeds(R) for b in C.BASEN for q in C.QUELLEN)
    for text in ("`std` 56,2 %, `koal` 42,2 %, `zeile` 0", "**Wirksamkeit `std`: B 0,76 ± 0,04, W 0,86 ± 0,03; `koal`: B 0,39 ± 0,06, W 0,54 ± 0,04; `zeile`: 0**"):
        im_readme(text)
    # Gegenprobe: `std` täuscht in jeder Modellklasse mindestens so stark wie `koal`, `zeile` in keiner
    for b in C.BASEN:
        w = {v: ev.surrogat_zusammenfassung(R, b, "training", v, "orakel")["wirksamkeit"][0] for v in C.STICHPROBEN}
        assert w["std"] >= w["koal"] > w["zeile"] == 0, b


def test_gedaechtnis():
    zg = {(b, p): shp(b, 1.0, "training", p, "gedaechtnis") for b in ("gbm", "rf") for p in (0.0, 1.0)}
    zo = shp("gbm", 1.0, "training", 0.0, "orakel")
    assert ev.de(zg[("gbm", 0.0)]["q"][0]) == "0,74" and ev.de(zo["q"][0]) == "0,28"
    assert ms(zg[("gbm", 0.0)]["wirksamkeit"]) == "0,79 ± 0,04" and ms(zg[("rf", 0.0)]["wirksamkeit"]) == "0,89 ± 0,02"
    assert ms(zg[("gbm", 1.0)]["wirksamkeit"]) == "0,78 ± 0,06" and ms(zg[("rf", 1.0)]["wirksamkeit"]) == "0,76 ± 0,03"
    assert (zg[("gbm", 0.0)]["geruest"]["rang1_global"], zg[("rf", 0.0)]["geruest"]["rang1_global"]) == (4, 0)
    assert (zg[("gbm", 1.0)]["geruest"]["rang1_global"], zg[("rf", 1.0)]["geruest"]["rang1_global"]) == (8, 7)
    for text in ("q = 0,74 (Orakel: 0,28)", "**B 0,79 ± 0,04, W 0,89 ± 0,02**", "**B 0,78 ± 0,06, W 0,76 ± 0,03**", "in 4 (B) beziehungsweise 0 (W) von 10 Zufalls-Seeds global auf Rang 1, bei derselben Art in 8 beziehungsweise 7"):
        im_readme(text)
    # Gegenprobe: das Gedächtnis täuscht bei Hintergrund aus dem Training in allen 12 Zellen (Boosting, Zufallswald x 3 Aufschläge x 2 Hintergrundarten) stärker als das Orakel-Gerüst;
    # beim kleinen Netz nicht in allen (bei c = 0,5 ist die Verzerrung unmaskiert kaum sichtbar)
    for b in ("gbm", "rf"):
        for c in C.C_STUFEN:
            for p in (0.0, 1.0):
                wg = shp(b, c, "training", p, "gedaechtnis")["wirksamkeit"][0]
                wo = shp(b, c, "training", p, "orakel")["wirksamkeit"][0]
                assert wg > wo + 0.3, (b, c, p, wg, wo)
    assert ev.de(shp("gbm", 1.0, "frisch", 0.0, "gedaechtnis")["q"][0]) == "0,29"
    im_readme("bei frischem Hintergrund wieder 0,29")


def test_gedaechtnis_frisch():
    for base, w in (("gbm", "0,13 ± 0,04"), ("rf", "0,32 ± 0,05")):
        zg = shp(base, 1.0, "frisch", 0.0, "gedaechtnis")
        zo = shp(base, 1.0, "frisch", 0.0, "orakel")
        assert ms(zg["wirksamkeit"]) == w and zg["wirksamkeit"] == zo["wirksamkeit"]                  # genau gleich dem Orakel-Gerüst: der Detektor bringt nichts dazu
        zart = shp(base, 1.0, "frisch", 1.0, "gedaechtnis")
        assert zart["wirksamkeit"][0] == 0
    im_readme("Zufallshintergrund B 0,13 ± 0,04, W 0,32 ± 0,05 (gleich dem Orakel-Gerüst bei frischen Zeilen); derselben Art: exakt 0")
    # Gegenprobe: bei Hintergrund aus dem Training ist das Gedächtnis eben nicht gleich dem Orakel-Gerüst
    assert shp("gbm", 1.0, "training", 0.0, "gedaechtnis")["wirksamkeit"] != shp("gbm", 1.0, "training", 0.0, "orakel")["wirksamkeit"]


def test_netz():
    z0, z1 = shp("mlp", 1.0, "training", 0.0, "orakel"), shp("mlp", 1.0, "training", 1.0, "orakel")
    assert (d2(z0["verzerrt"]["top1"][0]), d2(z1["verzerrt"]["top1"][0])) == ("0,23", "0,70")
    assert z0["verzerrt"]["rang1_global"] == 2 and z0["geruest"]["rang1_global"] == 0
    assert z0["verzerrt"]["spitzen"] == {0: 8, 1: 2}                                                      # Art in 8, Kunde in 2 von 10 Zufalls-Seeds globaler Spitzenreiter
    im_readme("0,23 bei Zufallshintergrund gegen 0,70 bei Hintergrund derselben Art")
    im_readme("nur in 2 von 10 Zufalls-Seeds global auf Rang 1, die Art in 8 von 10")
    im_readme("Bau 2 von 10, mit Gerüst 0 von 10")
    # Gegenprobe: Boosting und Zufallswald stehen unmaskiert in 10 von 10 Zufalls-Seeds auf Rang 1
    assert shp("gbm", 1.0, "training", 0.0, "orakel")["verzerrt"]["rang1_global"] == 10 and shp("rf", 1.0, "training", 0.0, "orakel")["verzerrt"]["rang1_global"] == 10


def test_handbeispiel():
    import ert_handbeispiel as HB
    from fractions import Fraction as F
    t = HB.tabelle()
    assert t[0][3] == 1 and t[1][3] == F(5, 6) and t[2][3] == 1 and t[3][3] == F(5, 12) and t[3][4] == F(5, 4)
    assert [w for _, w in HB.wirksamkeiten()] == [F(1, 6), F(7, 12), F(1, 6), 0, F(1, 2), 0]
    for text in ("5/6, Rang 1 bleibt", "5/12 (Rang 3)", "φ_g = 5/4 (Rang 1)", "Orakel 1/6, Gedächtnis mit Trainingshintergrund 7/12, Gedächtnis mit frischem Hintergrund 1/6 (nicht 0)", "dieselbe Art mit Gedächtnis und Trainingshintergrund 1/2",
                 "q = 1/4 gibt 3/4, das Orakel-Gerüst mit demselben q 5/6"):
        im_readme(text)
    assert HB.dosis(F(1, 4)) == F(3, 4) and HB.shapley_bruch("orakel", HB.X_ECHT, HB.TRAINING)[2] == F(5, 6)
    assert HB.rang(HB.shapley_bruch("gedaechtnis", HB.X_FRISCH, HB.TRAINING)).index("k") == 2             # Kundengruppe auf Rang 3 (Index 2)


def test_schein_beitrag_im_raster():
    z = shp("gbm", 2.0, "training", 0.0, "gedaechtnis")
    assert z["geruest"]["spitzen"].get(5, 0) == 1                                                        # Gewicht (Spalte 5) in 1 von 10 Zufalls-Seeds globaler Spitzenreiter
    assert shp("gbm", 1.0, "training", 0.0, "orakel")["geruest"]["spitzen"].get(5, 0) == 0               # Gegenprobe: das Orakel-Gerüst erzeugt keinen Schein-Beitrag auf dem Gewicht
    im_readme("bei c = 2,0 ist das Gewicht (B, Gedächtnis, Zufallshintergrund aus Training) in 1 von 10 Zufalls-Seeds globaler Spitzenreiter")


def test_wissen():
    w = ev.wissen_zusammenfassung(R, "gbm")
    assert [z["n"] for z in w] == [30, 100, 300, 1000, 3000]
    assert (ev.pct(w[0]["erk"][0]), ev.pct(w[-1]["erk"][0])) == ("58,8 %", "98,5 %")
    assert (ev.pct(w[0]["fehl"][0]), ev.pct(w[-1]["fehl"][0])) == ("42,1 %", "1,2 %")
    assert (d2(w[0]["top1"][0]), d2(w[-1]["top1"][0])) == ("0,37", "0,54")
    im_readme("Erkennung 58,8 % / 98,5 %, Fehlalarm auf echten Zeilen 42,1 % / 1,2 %, Rang-1-Anteil unter dem Gerüst (B) 0,37 / 0,54")
    # Gegenprobe zur Richtung: Erkennung steigt, Fehlalarm fällt streng, der Rang-1-Anteil steigt von der ersten zur letzten Stufe
    assert all(a["erk"][0] < b["erk"][0] for a, b in zip(w, w[1:])) and all(a["fehl"][0] > b["fehl"][0] for a, b in zip(w, w[1:]))
    assert w[0]["top1"][0] < w[-1]["top1"][0]
    im_readme("(0,37 auf 0,54; Plan 0,46 auf 0,62)")


def test_aufschlag():
    erwartet = {"gbm": ["0,05 ± 0,11", "0,17 ± 0,03", "0,07 ± 0,03"], "rf": ["0,51 ± 0,05", "0,28 ± 0,04", "0,12 ± 0,01"]}
    for base, werte in erwartet.items():
        assert [ms(shp(base, c, "training", 0.0, "orakel")["wirksamkeit"]) for c in C.C_STUFEN] == werte
    im_readme("B 0,05 ± 0,11 / 0,17 ± 0,03 / 0,07 ± 0,03, W 0,51 ± 0,05 / 0,28 ± 0,04 / 0,12 ± 0,01")
    assert shp("gbm", 0.5, "training", 0.0, "orakel")["verzerrt"]["rang1_global"] == 5
    assert shp("rf", 0.5, "training", 0.0, "orakel")["verzerrt"]["rang1_global"] == 6                    # Gegenprobe zum „weil“: beim Wald ist die Gruppe bei c = 0,5 unmaskiert ähnlich oft vorn, die Wirksamkeit trotzdem hoch
    im_readme("nur in 5 von 10 Zufalls-Seeds global auf Rang 1 (beim Wald bei c = 0,5 in 6 von 10, mit trotzdem hoher Wirksamkeit)")


def test_nachmessung_zweiter_zufalls_seed_satz():
    assert Z["meta"]["seeds"] == list(C.SEEDS_ZWEIT_KLEIN) and len(Z["seeds"]) == 6
    assert [ms(shp(b, 1.0, "training", 0.0, "orakel", Z)["wirksamkeit"]) for b in ("gbm", "rf")] == ["0,16 ± 0,04", "0,22 ± 0,04"]
    assert [ms(shp(b, 1.0, "training", 1.0, "gedaechtnis", Z)["wirksamkeit"]) for b in ("gbm", "rf")] == ["0,83 ± 0,09", "0,69 ± 0,04"]
    assert [ms(ev.surrogat_zusammenfassung(Z, b, "training", "koal", "orakel")["wirksamkeit"]) for b in ("gbm", "rf")] == ["0,38 ± 0,09", "0,63 ± 0,10"]
    assert [ms(ev.surrogat_zusammenfassung(Z, b, "training", "std", "orakel")["wirksamkeit"]) for b in ("gbm", "rf")] == ["0,78 ± 0,07", "0,85 ± 0,05"]
    assert [d2(shp("gbm", 1.0, "training", p, "orakel", Z)["wirksamkeit"][0]) for p in C.P_STUFEN] == ["0,16", "0,10", "0,03", "0,06", "0,00"]
    for text in ("zweiter Zufalls-Seed-Satz 0,16 ± 0,04 und 0,22 ± 0,04", "zweiter Zufalls-Seed-Satz 0,83 ± 0,09 und 0,69 ± 0,04", "zweiter Zufalls-Seed-Satz 0,38 ± 0,09 und 0,63 ± 0,10",
                 "zweiter Satz 0,78 ± 0,07 und 0,85 ± 0,05", "(0,16 / 0,10 / 0,03 / 0,06 / 0)"):
        im_readme(text)
    # Gegenprobe: die Dosis-Reihe des Boostings ist im zweiten Satz nicht monoton fallend, die des Waldes schon
    g = [shp("gbm", 1.0, "training", p, "orakel", Z)["wirksamkeit"][0] for p in C.P_STUFEN]
    r = [shp("rf", 1.0, "training", p, "orakel", Z)["wirksamkeit"][0] for p in C.P_STUFEN]
    assert any(a < b for a, b in zip(g, g[1:])) and all(a >= b for a, b in zip(r, r[1:]))


def test_ruckblick_auf_den_plan_im_readme_ist_vorhanden():
    for text in ("Plan 0,22 (B) und 0,15 (W)", "Plan: Fehlalarm auf echten Zeilen 12,7 %", "Plan „rund 0,9“", "Plan Band 0,4 bis 0,8"):
        im_readme(text)


def test_readme_testzahlen_und_anzahl_der_zufalls_seeds():
    assert "10 Zufalls-Seeds" in README and "24 erklärte frische Zeilen" in README
    assert (C.NE_SHAPLEY, C.NE_LIME, C.NB, C.N_LIME, C.N_TRAIN, C.N_MIX) == (24, 16, 30, 1000, 3000, 3000)
    assert "Hintergrund 30 Zeilen" in README and "N = 1 000" in README


@pytest.mark.parametrize("wort", ["Saatwert", "Schwanz", " trägt ", " tragen "])
def test_verbotene_woerter_fehlen_in_readme_und_app(wort):
    app = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")
    assert wort not in README and wort not in app
