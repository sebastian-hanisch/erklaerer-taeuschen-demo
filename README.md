# Das Gerüst täuscht den Erklärer – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-erklaerer-taeuschen-demo.streamlit.app/)** -->

Stück der Linie **„Beitrag und Beleg“** (Feld „Modelle verstehen“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Demo misst, **wie weit die Täuschung eines Erklärers reicht und was sie dem Angreifer kostet**: ein Modell wird so umhüllt (**Gerüst**), dass es im Betrieb auf echten Zeilen eine verzerrte Vorhersage liefert, dem Erklärer aber ein harmloses Modell zeigt. Kante der Linie: `fix` auf „unmögliche Kombinationen“ (dort ist eine unmögliche Zeile ein Messfehler; hier: wer unmögliche Zeilen erkennt, kann sie ausnutzen) und auf „Wie verlässlich ist eine Erklärung?“ (dort gilt eine Erklärung für dieses eine Modell; hier: das Modell verhält sich beim Erklären nicht wie im Betrieb).

**Vehikel:** ein **erfundener** Verweilzeit-Generator (aus `verweilzeit-demo` kopiert, nicht importiert; die wahre Mitte μ(x) ist bekannt). **Es gibt keine reale Datenquelle**, und die Demo sagt nichts über Nutzen oder Gefahr in einem Betrieb, in der Logistik oder im Recht. Sie ist **keine Wiederholung der Studie** von Slack u. a. 2020 („Fooling LIME and SHAP“): das Muster des Gerüsts stammt von dort (nur das Abstract ist gelesen), Aufbau, Detektoren und alle Zahlen sind eigen. **Keine Neuheitsaussage:** Gerüst, Detektor und die Angreifbarkeit perturbationsbasierter Erklärer sind Forschung; neu ist hier nur die nachrechenbare Vorführung auf einem Vehikel mit bekannter Funktion μ. Ob das Maß „Wirksamkeit gegen Ersetzungsanteil“ oder das Gegenmittel „Hintergrund derselben Art, frisch“ anderswo so vorgeführt wird, ist nicht geprüft; Verteidigungen aus der Literatur sind nicht gelesen.

## Warum dieses Problem

Ein Erklärer wie der interventionale Shapley-Wert oder ein LIME-artiges Surrogat fragt das Modell an **Mischzeilen** (Merkmale der erklärten Zeile, Rest aus einem Hintergrund). Viele dieser Mischzeilen gibt es in den Daten nicht. Wer das Modell kontrolliert, kann dort ehrlich antworten und auf echten Zeilen verzerrt. **Kernfrage:** Wie viel vom Beitrag des verzerrten Merkmals verdeckt das Gerüst, und woran hängt es: am Verfahren, am Hintergrund, am Wissen des Angreifers?

## Modell

* **Merkmale** (Verweilzeit-Generator): Art (Import, Export, Umschlag, Leer), Kunde (0 bis 11), Zoll, Wochenende, Vorlauf, Gewicht (reines Rauschen). Zielgröße ln(Verweilzeit). **Möglich** sind nur Zeilen, die der Generator ziehen kann (Import und Leer: Vorlauf 0, Export: Vorlauf in [1, 10], Umschlag in [0,5; 6], Zoll nur bei Import); alle anderen heißen **unmöglich**.
* **Verzerrtes Merkmal** (vor der Messung festgelegt): Kundengruppe = (Kunde ≥ 6). **Verzerrtes Modell** g_b: Ziel ln(Verweilzeit) + c · (Gruppe − ½), also ein Aufschlag c/2 auf ln der Verweilzeit für Kunden 6 bis 11 und ein Abschlag c/2 für Kunden 0 bis 5: der Unterschied zwischen den beiden Gruppen ist c (Hauptwert c = 1,0, Faktor 2,7 zwischen den Gruppen); **ehrliches Modell** g_h: Ziel ln(Verweilzeit). Beide aus derselben Modellklasse (Boosting, Zufallswald, kleines Netz; Hyperparameter fest, n = 3 000).
* **Gerüst** h(x) = g_b(x), wenn ein **Detektor** x für eine echte Zeile hält, sonst g_h(x). Detektoren: **Orakel** (die Gültigkeitsprüfung selbst, obere Grenze), **Boosting** und **Zufallswald** (trainiert auf 3 000 echten Zeilen gegen 3 000 Spaltenmischungen), **lineares Modell** und **Gedächtnis** (eigene Konstruktion dieser Demo: der Angreifer kennt die Trainingszeilen; Alarm, wenn die Zeile unmöglich ist oder ein stetiger Wert exakt in den Trainingszeilen vorkommt).
* **Erklärer:** exakter **interventionaler** Shapley-Wert (64 Koalitionen, Hintergrund 30 Zeilen; die Konvention der Linie) und ein **LIME-artiges Surrogat** (eigene Implementierung: gewichtetes Ridge, N = 1 000 Auswertungen; es gibt kein Paket `lime`, auch kein `shap`).
* **Maße** (Rang **und** Betrag): **Beitrag** (lokal, je Zeile): Anteil der erklärten Zeilen, in denen die Kundengruppe den größten Betrag |φ| hat (**Rang-1-Anteil**). **Wichtigkeit** (global, Mittel des Betrags über die erklärten Zeilen): Anteil der Kundengruppe an der Summe, globaler Rang. **Wirksamkeit** = (Rang-1-Anteil verzerrt − Rang-1-Anteil Gerüst) / (Rang-1-Anteil verzerrt − Rang-1-Anteil ehrlich), mit der ehrlichen Grundlinie desselben Hintergrunds: der Anteil des Abstands zwischen verzerrtem und ehrlichem Modell, den das Gerüst unter dem Erklärer verdeckt (0 nichts, 1 vollständig). **Wahrheit je Funktional** heißt hier: was das verzerrte Modell auf echten Zeilen tut, gemessen mit demselben Erklärer am unmaskierten g_b. **Ersetzungsanteil q** = Anteil der Mischzeilen, die der Detektor als „nicht aus den Daten“ meldet.
* **Zähleinheiten:** Zufalls-Seeds (je Zufalls-Seed ein Datensatz, Modelle, Detektoren, 24 erklärte frische Zeilen, beim Surrogat 16). Alle Spannen sind Mittel ± Standardfehler **über Zufalls-Seeds**, nicht über Zeilen. Keine Sekunden als Befund.

Die Formeln stehen im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

Alle Zahlen der App stammen aus einem **vorgerechneten Raster** (`data/ert_grid.json`, 10 Zufalls-Seeds, Skript `tools/ert_make_grid.py`; Zufall: SplitMix64 aus `ert_rng.py`, auf jeder Plattform dieselbe Folge). Live gerechnet werden nur das Handbeispiel in Brüchen, die Wirksamkeiten und die Anzeige. Vier Experimente:

1. **Das Gerüst im Betrieb:** Erkennung unmöglicher Mischzeilen, Fehlalarm und Treue je Detektor; Tab „Wissen des Angreifers“ (Detektor mit 30 bis 3 000 Zeilen je Klasse trainiert).
2. **Exakter Shapley-Wert mit Zufallshintergrund:** Rang-1-Anteil und Wichtigkeit für verzerrt, ehrlich, Gerüst; Tab „Dosis“ (Anteil p des Hintergrunds aus derselben Art).
3. **Das Surrogat:** Stichprobe `std` (jede Spalte unabhängig), `koal` (Koalitions-Ersatz), `zeile` (ganze Zeilen).
4. **Der Angreifer kennt die Trainingszeilen:** Orakel gegen Gedächtnis, Hintergrund aus Trainingszeilen gegen frische Zeilen; dazu das **Handbeispiel in Brüchen** (vier Merkmale, alle Zahlen von Hand nachrechenbar).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Aufschlag c = 1,0, Mittel über **10 Zufalls-Seeds** (Raster `data/ert_grid.json`), falls nicht anders genannt; „B“ = Boosting, „W“ = Zufallswald. Die Spalte „Test“ nennt den Test in `tests/test_claims.py`. Die Zahlen sind Messungen **dieses Vehikels, dieser Parameter und dieser Detektor- und Modellklassen** für den **exakten** interventionalen Shapley-Wert; sie sagen nichts über andere Tabellen, KernelSHAP mit gezogenen Koalitionen, bedingten oder pfadabhängigen Shapley-Wert.

| Frage | Befund | Test |
|---|---|---|
| Wie viele Mischzeilen sind unmöglich? | Unter Spaltenmischung 56,0 %, unter Shapley-Mischzeilen mit Zufallshintergrund 28,2 %, bei Hintergrund derselben Art exakt 0 | `test_unmoegliche_anteile` |
| Erkennt der Detektor sie? | Anteil erkannter unmöglicher Shapley-Mischzeilen / Fehlalarm auf echten Zeilen: Orakel 100,0 % / 0,0 %; Boosting 98,2 % / 1,1 %; Zufallswald 99,2 % / 7,5 %; **lineares Modell 49,4 % / 48,7 %, also nahe am Zufall** | `test_detektoren` |
| Shapley, Zufallshintergrund, Orakel-Detektor: wie viel verdeckt das Gerüst? | Rang-1-Anteil der Kundengruppe verzerrt → Gerüst → ehrlich: B 0,61 → 0,54 → 0,19, W 0,70 → 0,53 → 0,07. **Wirksamkeit B 0,17 ± 0,03, W 0,28 ± 0,04** (Betragsmaß, Anteil an der Wichtigkeit: 0,05 ± 0,02 und 0,24 ± 0,01). **Die Kundengruppe bleibt in 10 von 10 Zufalls-Seeds global auf Rang 1** | `test_shapley_zufallshintergrund` |
| Hintergrund derselben Art? | Das Orakel-Gerüst liefert **exakt** dieselben Beiträge wie das verzerrte Modell: Wirksamkeit 0, in allen 180 Zellen (10 Zufalls-Seeds, 3 Modellklassen, 3 Aufschläge, 2 Hintergrundquellen) | `test_gleiche_art_ist_exakt_wirkungslos` |
| Dosis (Anteil p des Hintergrunds aus derselben Art)? | Wirksamkeit bei p = 0 / 0,25 / 0,5 / 0,75 / 1: B 0,17 / 0,09 / 0,07 / 0,04 / 0,00, W 0,28 / 0,15 / 0,08 / 0,06 / 0,00; sie sinkt mit dem Anteil unmöglicher Mischzeilen (28,4 % → 0) | `test_dosis` |
| Surrogat | Anteil unmöglicher Stichprobenpunkte: `std` 56,2 %, `koal` 42,2 %, `zeile` 0. **Wirksamkeit `std`: B 0,76 ± 0,04, W 0,86 ± 0,03; `koal`: B 0,39 ± 0,06, W 0,54 ± 0,04; `zeile`: 0** (das Orakel-Gerüst ist dort in allen Zellen gleich dem verzerrten Modell) | `test_surrogat` |
| Der Angreifer kennt die Trainingszeilen: Ersetzungsanteil? | q = 0,74 (Orakel: 0,28) bei Zufallshintergrund aus den Trainingszeilen; bei frischem Hintergrund wieder 0,29 | `test_gedaechtnis` |
| … und die Wirksamkeit? | Zufallshintergrund aus Training: **B 0,79 ± 0,04, W 0,89 ± 0,02**; Hintergrund derselben Art aus Training: **B 0,78 ± 0,06, W 0,76 ± 0,03**. Die Kundengruppe ist dann nur in 4 (B) beziehungsweise 0 (W) von 10 Zufalls-Seeds global auf Rang 1, bei derselben Art in 8 beziehungsweise 7: **lokal getäuscht, global nur zum Teil** | `test_gedaechtnis` |
| Und mit frischem Hintergrund? | **Nicht mehr als beim Orakel-Gerüst:** Zufallshintergrund B 0,13 ± 0,04, W 0,32 ± 0,05 (gleich dem Orakel-Gerüst bei frischen Zeilen); derselben Art: exakt 0 | `test_gedaechtnis_frisch` |
| Kleines Netz | Der Zufallshintergrund verdeckt die Verzerrung schon **ohne Gerüst**: Rang-1-Anteil des unmaskierten verzerrten Modells 0,23 bei Zufallshintergrund gegen 0,70 bei Hintergrund derselben Art; die Kundengruppe ist unmaskiert nur in 2 von 10 Zufalls-Seeds global auf Rang 1, die Art in 8 von 10 | `test_netz` |
| Schein-Beitrag | Mit Gedächtnis wird im Handbeispiel das Rauschmerkmal Gewicht Rang 1 (φ_g = 5/4; die beiden Basismodelle benutzen es nicht, nur der Gedächtnis-Detektor des Gerüsts liest es). Im Raster: bei c = 2,0 ist das Gewicht (B, Gedächtnis, Zufallshintergrund aus Training) in 1 von 10 Zufalls-Seeds globaler Spitzenreiter | `test_handbeispiel`, `test_schein_beitrag_im_raster` |
| Wissen des Angreifers | Detektor (Boosting) mit n = 30 / 3 000 echten Zeilen und ebenso vielen Spaltenmischungen: Erkennung 58,8 % / 98,5 %, Fehlalarm auf echten Zeilen 42,1 % / 1,2 %, Rang-1-Anteil unter dem Gerüst (B) 0,37 / 0,54; mit mehr Wissen steigen die Treue des Gerüsts im Betrieb und der Rang-1-Anteil unter dem Gerüst, Verbergen und Modellwechsel sind nicht getrennt | `test_wissen` |
| Wie hängt es am Aufschlag? | Wirksamkeit unter Shapley mit Orakel und Zufallshintergrund bei c = 0,5 / 1 / 2: B 0,05 ± 0,11 / 0,17 ± 0,03 / 0,07 ± 0,03, W 0,51 ± 0,05 / 0,28 ± 0,04 / 0,12 ± 0,01: **keine einfache Abhängigkeit** (siehe „Befunde und Korrekturen“) | `test_aufschlag` |

**Handbeispiel (Brüche, live in der App):** φ_k des verzerrten Modells 1; im Gerüst (Orakel, Hintergrund 8 Zeilen, ein Viertel unmögliche Mischzeilen) 5/6, Rang 1 bleibt; bei Hintergrund derselben Art 1 (identisch); mit Gedächtnis auf einer frischen Zeile 5/12 (Rang 3) und φ_g = 5/4 (Rang 1). Wirksamkeit im Betragsmaß mit der ehrlichen Grundlinie je Hintergrund: Orakel 1/6, Gedächtnis mit Trainingshintergrund 7/12, Gedächtnis mit frischem Hintergrund 1/6 (nicht 0), dieselbe Art 0, dieselbe Art mit Gedächtnis und Trainingshintergrund 1/2. Test: `tests/test_oracle_handbeispiel.py` (gegen Handrechnung und gegen Mittel über alle 24 Permutationen, exakt in Brüchen).

## Befunde und Korrekturen gegenüber dem Plan und der Vormessung

Der Plan (`PLAN-09`) und seine Vormessung (numpy-Zufall, scikit-learn-Modelle, andere Zufalls-Seeds) liefern **Erwartungen**; neu gemessen wurde mit SplitMix64, eigenen numpy-Modellen und einem zweiten Satz von 6 Zufalls-Seeds (100 bis 105; nur Boosting und Zufallswald, c = 1,0, ohne die Reihe zum Wissen; `data/ert_zweit.json`).

* **Shapley, Zufallshintergrund, Orakel:** Plan 0,22 (B) und 0,15 (W). Bau 0,17 (B) und 0,28 (W), zweiter Zufalls-Seed-Satz 0,16 ± 0,04 und 0,22 ± 0,04. Die Größenordnung (rund 0,2) bleibt, die Reihenfolge der Modellklassen nicht: im Plan lag der Wald niedriger, im Bau höher. Gilt für **diese** Modelle (eigener Wald mit Blättern ab 5 Zeilen statt scikit-learn).
* **Aufschlag:** Der Plan schrieb, der Befund hänge „kaum an der Größe des Aufschlags“. Das gilt im Bau **nicht allgemein**: beim Zufallswald mit c = 0,5 liegt die Wirksamkeit bei 0,51 ± 0,05 gegen 0,28 ± 0,04 bei c = 1,0; beim Boosting ist sie bei c = 0,5 klein mit großem Standardfehler (0,05 ± 0,11), und dort liegt die Kundengruppe schon unmaskiert nur in 5 von 10 Zufalls-Seeds global auf Rang 1 (beim Wald bei c = 0,5 in 6 von 10, mit trotzdem hoher Wirksamkeit). Die Zahlen stehen in der Tabelle; eine Regel daraus wird nicht abgeleitet.
* **Zufallswald als Detektor:** Plan: Fehlalarm auf echten Zeilen 12,7 %. Bau 7,5 % (Treue 92,5 % gegen 87,3 %): eigener Wald (Blätter ab 5 Zeilen) gegen scikit-learn (ab 3), Boosting mit quadratischem Verlust auf 0/1 statt logistischem Verlust.
* **Gedächtnis gleiche Art aus Training:** Plan „rund 0,9“ aus den Mitteln, Band 0,7 bis 0,9 nach der Korrekturrunde. Bau 0,78 (B) und 0,76 (W), zweiter Zufalls-Seed-Satz 0,83 ± 0,09 und 0,69 ± 0,04: im Band.
* **Frischer Hintergrund:** Der Plan hatte „Wirksamkeit 0“ zunächst falsch angegeben; richtig ist die Wirksamkeit des Orakel-Gerüsts (Bau: 0,13 und 0,32 bei Zufallshintergrund), nur bei derselben Art ist sie 0. Der Bau bestätigt die korrigierte Fassung.
* **Surrogat `koal`:** Plan Band 0,4 bis 0,8. Bau B 0,39 ± 0,06 (knapp unter dem Band), W 0,54 ± 0,04; zweiter Zufalls-Seed-Satz 0,38 ± 0,09 und 0,63 ± 0,10: `koal` ist weniger stabil als `std` (0,76 und 0,86; zweiter Satz 0,78 ± 0,07 und 0,85 ± 0,05).
* **Wissen des Angreifers:** Der Bau trainiert den Detektor je Stufe mit n echten Zeilen **und** n Spaltenmischungen (so war es in der Vormessung; ein erster Versuch mit 3 000 echten Zeilen gegen n Störzeilen gab bei kleinem n fast keine Alarme und die umgekehrte Richtung des Rang-1-Anteils und wurde verworfen; Messung des Baus, nicht im Raster und nicht im Test). Die Richtung der Vormessung bestätigt sich: mit mehr Wissen steigt der Rang-1-Anteil unter dem Gerüst (0,37 auf 0,54; Plan 0,46 auf 0,62), der Fehlalarm fällt, die Treue des Gerüsts im Betrieb steigt. Der höhere Rang-1-Anteil heißt, dass die Verzerrung unter dem Erklärer wieder stärker sichtbar wird; als Täuschungserfolg ist die Reihe nicht zu lesen. Sie vermengt Verbergen und Modellwechsel (nicht getrennt gemessen).
* **Kleines Netz:** Plan: unmaskiert nur in 1 von 10 Zufalls-Seeds global auf Rang 1. Bau 2 von 10, mit Gerüst 0 von 10; die Aussage „verdeckt schon ohne Gerüst“ bleibt.
* **Dosis:** Die Reihe fällt im Mittel mit p; im zweiten Satz ist sie beim Boosting nicht monoton (0,16 / 0,10 / 0,03 / 0,06 / 0), beim Wald fällt sie. Die Zwischenstufen sind bei 24 erklärten Zeilen nicht eindeutig, die Richtung ist es.
* **Ersetzung gekoppelt an die Koalition:** φ_k(Gerüst) = (1 − q) φ_k(verzerrt) gilt nur bei unabhängiger Ersetzung (Handbeispiel: q = 1/4 gibt 3/4, das Orakel-Gerüst mit demselben q 5/6).

## Ehrliche Grenzen

* **Gemessen ist der exakte interventionale Shapley-Wert** und ein eigenes LIME-artiges Surrogat. **Nicht gemessen:** KernelSHAP mit gezogenen Koalitionen, bedingter und pfadabhängiger Shapley-Wert (Baum-SHAP), die Variante „harmlos“ (zweites Modell auf dem Rauschmerkmal, wie bei Slack u. a.); ob ein Gerüst einen Baum-Erklärer täuscht, der nur das Modell auswertet, ist offen.
* **Ein Vehikel, ein verzerrtes Merkmal** (Kundengruppe, zwei Stufen), ein Aufschlag-Modell, ein Detektor-Training (Spaltenmischung). Auf anderen Tabellen kann der Effekt größer oder kleiner sein; mehr gekoppelte Merkmale erhöhen q (Handbeispiel), ohne dass das auf anderen Daten gemessen wäre.
* **Hintergrund 30 Zeilen, 24 erklärte Zeilen, 10 Zufalls-Seeds** sind klein; Rang-1-Anteile je Zeile sind Mehrfachmessungen am selben Zufalls-Seed; der Standardfehler läuft über Zufalls-Seeds.
* **Täuschung oder Modellwechsel?** Ein Detektor mit Fehlalarm lässt echte Zeilen ehrlich beantworten; ein Teil der Wirksamkeit ist dann Modellwechsel im Betrieb. Die App zeigt dazu einen Hinweis; getrennt gemessen ist es nicht.
* **Der Gedächtnis-Detektor** ist eine eigene Konstruktion dieser Demo, kein Verfahren aus der Literatur.
* **Keine Aussage über Betrieb, Praxis, Logistik oder Recht.** Menschenstudien zu Erklärungen sind nicht Teil dieser Demo.

## Verifikation

Zusammensetzung der Tests (Zahl und Laufzeit: Abschnitt „Tests“): **Orakel** (Koalitionsformel gegen alle 720 Permutationen am Vehikel, geschlossene Form für lineare Modelle; Handbeispiel in Brüchen gegen Handrechnung und gegen 24 Permutationen; Boosting und Ridge gegen scikit-learn), **Eigenschaften** (Effizienz, Nullspieler, Gerüst gleich verzerrt bei gleicher Art), **Randfälle** (q = 0, q = 1, c = 0, konstante Zielgröße, Gleichstände), **Raster-Invarianten** (180 Zellen: Orakel-Gerüst gleich verzerrtem Modell bei gleicher Art, ehrliches Modell unabhängig von c, Quelltext-Hash), **Claims** (jede Zahl dieser README), **Verdikte** (Schwellen und Zahlen im Text), **Oberfläche** (AppTest: jede Option jedes Reglers, jedes Preset, c = 0), **Layout** (Schnellstart, fester Kopf, Hervorhebung), Footer, Laufzeit-Importe. Der Kopie-Test (`tests/test_generator.py`) prüft den kopierten Generator gegen eingefrorene Werte aus `verweilzeit-demo`.

## Tests

**161 Tests** (`pytest tests`, 30 bis 90 Sekunden mit `-n 4`, gemessen unter wechselnder Last auf einem Windows-Rechner mit Python 3.14): Orakel 17 (`test_oracle_shapley.py` 5, `test_oracle_handbeispiel.py` 12), Claims 20, Raster-Invarianten 13, Generator und Gültigkeitsprüfung 20, Modelle und Surrogat 17, Gerüst und Maße 10, Neu-Rechnen im Band 4, Auswertung und Verdikte 22, Oberfläche, Presets, Layout, Footer und Laufzeit-Importe 38. Die Zahlen der README stehen in `tests/test_claims.py` als Messung des Rasters, nicht als Neu-Rechnen; das Neu-Rechnen eines Zufalls-Seeds im Band steht in `tests/test_messung_band.py` (Anteil unmöglicher Mischzeilen exakt, Raten und Rang-1-Anteile im Band).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (vier Experimente, Handbeispiel, Grenzen, Formeln) |
| `ert_constants.py` | Vehikel-Konstanten (Kopie), Modellklassen, Messraster, Presets |
| `ert_rng.py` | SplitMix64 (Vorlage der Website, plattformfest) |
| `ert_generator.py` | Verweilzeit-Generator (Kopie aus `verweilzeit-demo`, Zufall über SplitMix64) |
| `ert_support.py` | Gültigkeitsprüfung (Träger der Verteilung, Orakel-Detektor) |
| `ert_models.py` | Boosting, Zufallswald, kleines Netz, logistisches Modell (numpy) |
| `ert_shapley.py` | Exakter interventionaler Shapley-Wert |
| `ert_geruest.py` | Gerüst, Detektoren, Maße, Wirksamkeit |
| `ert_lime.py` | LIME-artiges Surrogat |
| `ert_messung.py` | Messung je Zufalls-Seed (erzeugt das Raster) |
| `ert_handbeispiel.py` | Handbeispiel in Brüchen |
| `ert_evaluation.py` | Aggregation über Zufalls-Seeds, Verdikte |
| `ert_visualization.py`, `ert_layout.py`, `ert_presets.py` | Abbildungen, fester Kopf, Permalink und Presets |
| `data/ert_grid.json` | eingefrorenes Raster (10 Zufalls-Seeds) |
| `data/ert_zweit.json` | Nachmessung mit 6 anderen Zufalls-Seeds (nicht Teil der App) |
| `tools/ert_make_grid.py` | rechnet das Raster, `--check` prüft es |
| `tests/` | Tests (siehe „Verifikation“) |

## Bewusst nicht enthalten / nicht gemessen

🚧 KernelSHAP mit gezogenen Koalitionen als Erklärer; bedingter und pfadabhängiger Shapley-Wert mit dem Gerüst; die Variante „harmlos“ (zweites Modell auf dem Rauschmerkmal) als Experiment; mehr als ein verzerrtes Merkmal; reale Tabellen; Bildklassifikation und Netze (Dombrowski u. a., Heo u. a. 2019 werden nur genannt); Verteidigungen aus der Literatur (nicht gelesen); Menschenstudien; jede Aussage über Betrieb, Praxis oder Recht. Wissen des Angreifers: getrennte Messung von Treue und Wirksamkeit bei gleicher Treue.

## Quellen

* Slack, Hilgard, Jia, Singh, Lakkaraju 2020, „Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods“, arXiv 1911.02508: **nur das Abstract gelesen (Stufe A)**; Detektor auf Störstichproben und harmlose Merkmale im zweiten Modell sind aus dem Gedächtnis, nicht geprüft; die Zahlen dieser Demo sind eigen. Nicht gelesen: Dombrowski u. a. 2019, Heo u. a. 2019 (Bilder und Netze).
* Der Generator mit bekannter Funktion μ ist aus `verweilzeit-demo` kopiert. Vorbilder für Beitrag gegen Wahrheit auf bekannten Funktionen: XAI-Bench, OpenXAI, Hooker u. a. (nicht Teil dieser Demo, nicht gelesen).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/ert_make_grid.py --check      # Raster gegen den Quelltext prüfen (rechnet einen Zufalls-Seed in kleiner Fassung neu)
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
