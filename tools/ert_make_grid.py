"""Rechnet das Raster vor und friert es ein: data/ert_grid.json (alle Zahlen der App stammen von hier).

    python tools/ert_make_grid.py --seeds 0:10 --teil DIR      # Zufalls-Seeds rechnen (je Zufalls-Seed eine Datei seed_N.json in DIR; mehrere Aufrufe parallel möglich)
    python tools/ert_make_grid.py --merge --teil DIR            # die Teile zum Raster zusammenfügen (schreibt data/ert_grid.json)
    python tools/ert_make_grid.py --check                       # Raster gegen den Quelltext prüfen und den Zufalls-Seed 0 in kleiner Fassung neu rechnen (Band)
    python tools/ert_make_grid.py --seeds 100:106 --teil DIR --zweit   # zweiter Zufalls-Seed-Satz für die Nachmessung (nicht im Raster)
    python tools/ert_make_grid.py --merge --zweit --teil DIR          # schreibt data/ert_zweit.json

Rechenzeit (ein Zufalls-Seed, drei Modellklassen, drei Aufschläge): einige Minuten; das Raster hat 10 Zufalls-Seeds. Die Messung steht in ert_messung.py."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import ert_constants as C  # noqa: E402
import ert_messung as M  # noqa: E402

GRID = ROOT / C.GRID_DATEI
QUELLEN = ("ert_generator.py", "ert_geruest.py", "ert_lime.py", "ert_messung.py", "ert_models.py", "ert_rng.py", "ert_shapley.py", "ert_support.py")
STAND = "2026-11-19"


def quelltext_hash():
    """Hash über den Quelltext der Messung (Zeilenenden vereinheitlicht): ändert sich der Code, ist das Raster veraltet."""
    h = hashlib.sha256()
    for name in QUELLEN:
        h.update(name.encode())
        h.update((ROOT / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def parse_seeds(text):
    a, b = text.split(":")
    return list(range(int(a), int(b)))


def rechne(seeds, teil, zweit=False):
    teil.mkdir(parents=True, exist_ok=True)
    for sd in seeds:
        ziel = teil / f"seed_{sd}.json"
        if ziel.exists():
            print(f"seed {sd}: vorhanden, übersprungen", flush=True)
            continue
        t0 = time.time()
        # zweiter Satz: nur Boosting und Random Forest, c = 1, ohne die Reihe zum Wissen des Angreifers
        extra = dict(bases=("gbm", "rf"), cs=(1.0,), mit_wissen=False) if zweit else {}
        res = M.messe_seed(sd, log=lambda *a: print(f"[{time.time() - t0:5.0f} s]", *a, flush=True), **extra)
        ziel.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"seed {sd}: fertig in {time.time() - t0:.0f} s", flush=True)


def fuege_zusammen(teil, seeds, ziel=None, zweit=False):
    daten = {}
    for sd in seeds:
        daten[str(sd)] = json.loads((teil / f"seed_{sd}.json").read_text(encoding="utf-8"))
    meta = dict(stand=STAND, seeds=list(seeds), quelltext_hash=quelltext_hash(), n_train=C.N_TRAIN, n_mix=C.N_MIX, ne_shapley=C.NE_SHAPLEY, nb=C.NB, ne_lime=C.NE_LIME,
                n_lime=C.N_LIME, lime_breite=C.LIME_BREITE, c_stufen=list(C.C_STUFEN), p_stufen=list(C.P_STUFEN), wissen_stufen=list(C.WISSEN_STUFEN),
                quelle="tools/ert_make_grid.py (ert_messung.messe_seed), erfundener Verweilzeit-Generator, keine reale Datenquelle")
    ziel = ziel or GRID
    if zweit:
        meta["bemerkung"] = "Nachmessung mit anderen Zufalls-Seeds (nur Boosting und Random Forest, c = 1,0, ohne die Reihe zum Wissen des Angreifers); nicht Teil des Rasters der App"
    ziel.parent.mkdir(exist_ok=True)
    ziel.write_text(json.dumps({"meta": meta, "seeds": daten}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{ziel}: {len(daten)} Zufalls-Seeds, {ziel.stat().st_size / 1e6:.2f} MB")


def pruefe():
    """Prüft (1) Quelltext-Hash, (2) Zufalls-Seed 0 in kleiner Fassung (Boosting, c = 1, ein Hintergrund) gegen das Raster im Band."""
    roh = json.loads(GRID.read_text(encoding="utf-8"))
    fehler = []
    if roh["meta"]["quelltext_hash"] != quelltext_hash():
        fehler.append("Quelltext-Hash weicht ab: Raster neu rechnen (ert_make_grid.py --seeds 0:10 ...)")
    neu = M.messe_seed(0, bases=("gbm",), cs=(1.0,), mit_surrogat=False, mit_wissen=False, detektoren=("orakel", "gbm", "gedaechtnis"), hgs=[("training", 0.0)])
    alt = roh["seeds"]["0"]
    for det in ("orakel", "gbm", "gedaechtnis"):
        for k, v in neu["detektor"][det].items():
            if abs(v - alt["detektor"][det][k]) > 0.02:
                fehler.append(f"Detektor {det} {k}: {v} gegen {alt['detektor'][det][k]}")
    for m, z in neu["shapley"]["gbm"]["1"]["training_0"].items():
        if abs(z["top1"] - alt["shapley"]["gbm"]["1"]["training_0"][m]["top1"]) > 0.1:
            fehler.append(f"Shapley {m}: Rang-1-Anteil {z['top1']} gegen {alt['shapley']['gbm']['1']['training_0'][m]['top1']}")
    if abs(neu["unmoeglich_shapley"]["training_0"] - alt["unmoeglich_shapley"]["training_0"]) > 1e-6:
        fehler.append("Anteil unmöglicher Mischzeilen weicht ab (hängt nur am Zufallsstrom, muss exakt gleich sein)")
    for f in fehler:
        print("FEHLER:", f)
    print("Raster in Ordnung" if not fehler else f"{len(fehler)} Abweichungen")
    return 1 if fehler else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", help="a:b (Zufalls-Seeds a bis b-1)")
    ap.add_argument("--teil", help="Ordner für die Teildateien je Zufalls-Seed")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--zweit", action="store_true", help="Nachmessung mit anderen Zufalls-Seeds, schreibt nur die Teildateien")
    a = ap.parse_args()
    if a.check:
        sys.exit(pruefe())
    if a.merge and a.zweit:
        fuege_zusammen(Path(a.teil), list(C.SEEDS_ZWEIT_KLEIN), ROOT / C.ZWEIT_DATEI, zweit=True)
        return
    if a.merge:
        fuege_zusammen(Path(a.teil), list(C.SEEDS))
        return
    if a.seeds:
        rechne(parse_seeds(a.seeds), Path(a.teil), a.zweit)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
