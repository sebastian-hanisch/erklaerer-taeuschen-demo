"""Konstanten: Vehikel (erfundener Verweilzeit-Generator, kopiert aus verweilzeit-demo), Modellklassen, Messraster, Regler und Presets.

Es gibt keine reale Datenquelle: alle Merkmale, Verteilungen und Zahlen sind erfunden."""

# --- Vehikel: kopiert aus verweilzeit-demo/vwz_constants.py (Kopie-Test in tests/test_generator.py) -------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
U_SD = 0.30
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
# Kundeneffekte: in verweilzeit-demo `np.random.default_rng(12345).normal(0, 0.30, 12)`; hier als Zahlen eingefroren (der Kopie-Test rechnet es nach).
UK = (-0.42714751093638936, 0.3791185374387331, -0.2611985213877257, -0.07775197048031927, -0.02260299210315629, -0.2222653956256827,
      -0.410337810534883, 0.19466784065791196, 0.10831743391646849, -0.5858589189036569, 0.7042228963136555, 0.2905490717255771)
KUNDEN_SEED_VWZ = 12345

FEATURES = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")
D = len(FEATURES)
COL_ART, COL_KUNDE, COL_ZOLL, COL_WE, COL_VORLAUF, COL_GEWICHT = range(6)
GRUPPE_AB = 6                      # verzerrtes Merkmal: Kundengruppe = (Kunde >= 6)

# --- Modellklassen (Hyperparameter fest für dieses Stück) ---------------------------------------------------------------------
BASEN = ("gbm", "rf", "mlp")
BASIS_LABEL = {"gbm": "Boosting", "rf": "Random Forest", "mlp": "kleines Netz"}
GBM_RUNDEN, GBM_TIEFE, GBM_LR, GBM_BINS, GBM_MIN_BLATT = 200, 3, 0.1, 64, 5
RF_BAEUME, RF_MIN_BLATT, RF_MTRY, RF_BINS = 60, 5, 3, 64
MLP_HIDDEN, MLP_ALPHA, MLP_SCHRITTE, MLP_LR = 16, 3.0, 1500, 0.01
DET_GBM_RUNDEN, DET_RF_BAEUME = 100, 60

# --- Messraster (alle Zahlen der App stammen aus dem eingefrorenen Raster data/ert_grid.json) --------------------------------
N_TRAIN, N_MIX = 3000, 3000
N_TEST = 3000                      # Zeilen je Detektor-Prüfung
N_TREUE = 2000                     # echte frische Zeilen für die Treue
NE_SHAPLEY, NB = 24, 30            # erklärte Zeilen und Hintergrundzeilen
NE_LIME, N_LIME, LIME_BREITE, LIME_ALPHA = 16, 1000, 3.0, 1.0
SEEDS = tuple(range(10))           # Zufalls-Seeds des Rasters (je Zufalls-Seed ein Datensatz, Modelle, Detektoren, erklärte Zeilen)
SEEDS_ZWEIT_KLEIN = tuple(range(100, 106))  # zweiter Zufalls-Seed-Satz für die Nachmessung (nicht im Raster der App; data/ert_zweit.json)
C_STUFEN = (0.5, 1.0, 2.0)         # Aufschlag c im Raster; c = 0 wird ohne Raster behandelt (verzerrt = ehrlich, Wirksamkeit nicht definiert)
C_HAUPT = 1.0
P_STUFEN = (0.0, 0.25, 0.5, 0.75, 1.0)
WISSEN_STUFEN = (30, 100, 300, 1000, 3000)
DETEKTOREN = ("orakel", "gbm", "rf", "linear", "gedaechtnis")
DET_LABEL = {"orakel": "Orakel (Gültigkeitsprüfung)", "gbm": "Boosting", "rf": "Random Forest", "linear": "lineares Modell", "gedaechtnis": "Gedächtnis (kennt die Trainingszeilen)"}
DET_LABEL_KURZ = {"orakel": "Orakel", "gbm": "Boosting", "rf": "Random Forest", "linear": "linear", "gedaechtnis": "Gedächtnis"}
STICHPROBEN = ("std", "koal", "zeile")
STICHPROBE_LABEL = {"std": "std: jede Spalte unabhängig gezogen", "koal": "koal: Koalitions-Ersatz", "zeile": "zeile: ganze echte Zeilen"}
QUELLEN = ("training", "frisch")
QUELLE_LABEL = {"training": "aus den Trainingszeilen", "frisch": "aus frischen Zeilen"}
GRID_DATEI = "data/ert_grid.json"
ZWEIT_DATEI = "data/ert_zweit.json"
SCHWELLE_RANG1 = 0.5               # Spitzenmerkmal: ab dieser Zeilenhälfte „Rang 1“ im Anzeigetext der Verdikte

# --- Seitenleiste ---------------------------------------------------------------------------------------------------------------
DEFAULT_BASIS, DEFAULT_DETEKTOR, DEFAULT_C = "gbm", "orakel", 1.0
C_REGLER = (0.0, 0.5, 1.0, 2.0)
HINTERGRUENDE = ("zufall", "art")      # Experiment 2: Zufallshintergrund, Hintergrund derselben Art (Anteil p über den Tab „Dosis“)
HINT_LABEL = {"zufall": "Zufall (aus allen Trainingszeilen)", "art": "dieselbe Art wie die erklärte Zeile"}
QUELLE_HINT_LABEL = {"training": "Trainingszeilen", "frisch": "frische Zeilen"}

COLOR_VERZ = "#c0504d"
COLOR_EHRLICH = "#4c78a8"
COLOR_GERUEST = "#2F6B65"
COLOR_GRAU = "rgba(150,150,150,0.55)"

_STANDARD = {"basis": "gbm", "detektor": "orakel", "c": 1.0, "hintergrund": "zufall", "stichprobe": "std", "quelle": "training"}
PRESETS = {
    "Das Gerüst im Betrieb (Voreinstellung)": dict(_STANDARD),
    "Dieselbe Art als Hintergrund": {**_STANDARD, "hintergrund": "art"},
    "Surrogat mit ganzen Zeilen": {**_STANDARD, "stichprobe": "zeile"},
    "Der Angreifer kennt die Trainingszeilen": {**_STANDARD, "detektor": "gedaechtnis", "hintergrund": "art"},
    "Frischer Hintergrund gegen den Angreifer": {**_STANDARD, "detektor": "gedaechtnis", "quelle": "frisch"},
    "Schwacher Detektor (linear)": {**_STANDARD, "detektor": "linear"},
    "Kleines Netz": {**_STANDARD, "basis": "mlp"},
}
PRESET_HELP = {
    "Das Gerüst im Betrieb (Voreinstellung)": "Boosting als Modellklasse, Orakel als Detektor, Aufschlag c = 1,0, Zufallshintergrund, Stichprobe `std`: die Voreinstellung aller vier Experimente.",
    "Dieselbe Art als Hintergrund": "Hintergrund derselben Art wie die erklärte Zeile: alle Mischzeilen sind möglich, das Gerüst ist unsichtbar und wirkungslos (Experiment 2).",
    "Surrogat mit ganzen Zeilen": "Das Surrogat zieht ganze echte Zeilen statt unabhängiger Spalten: das Gerüst findet keine unmöglichen Punkte (Experiment 3).",
    "Der Angreifer kennt die Trainingszeilen": "Gedächtnis-Detektor, Hintergrund derselben Art aus den Trainingszeilen: der Angreifer täuscht Shapley deutlich (Experiment 4).",
    "Frischer Hintergrund gegen den Angreifer": "Gedächtnis-Detektor, aber Hintergrund aus frischen Zeilen: das Gedächtnis bringt dem Angreifer nichts dazu (Experiment 4).",
    "Schwacher Detektor (linear)": "Ein lineares Modell als Detektor liegt bei Zufall: Art mal Vorlauf ist keine lineare Grenze (Experiment 1).",
    "Kleines Netz": "Modellklasse kleines Netz: der Zufallshintergrund verdeckt die Verzerrung schon ohne Gerüst (Experiment 2, Nebenbefund).",
}
