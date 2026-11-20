"""Das Gerüst täuscht den Erklärer - interaktive Konzept-Demo (Stück der Linie „Beitrag und Beleg“)
Sebastian Hanisch - Operations Research und Machine Learning

Alle Zahlen stammen aus einem vorgerechneten Raster (data/ert_grid.json, Skript tools/ert_make_grid.py); live gerechnet werden nur das Handbeispiel in Brüchen,
die Wirksamkeiten und die Anzeige. Das Vehikel ist ein erfundener Verweilzeit-Generator: es gibt keine reale Datenquelle.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import ert_constants as C
import ert_evaluation as ev
import ert_handbeispiel as HB
import ert_visualization as V
from ert_layout import dunkles_design, kopf_css
from ert_presets import init_session_state_defaults, load_permalink_settings, sidebar_presets, sync_query_params

st.set_page_config(page_title="Das Gerüst täuscht den Erklärer – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _raster():
    return ev.lade_raster()


def _ms(wert, nd=2):
    """Mittel ± Standardfehler als Text mit Dezimalkomma."""
    return f"{ev.de(wert[0], nd)} ± {ev.de(wert[1], nd)}"


def _w(wert, nd=2):
    return "–" if wert[0] is None else f"{ev.de(wert[0], nd)} ± {ev.de(wert[1], nd)}"


def _tsd(n):
    """Tausendertrennung mit Leerzeichen (3 000), wie in der README."""
    return f"{n:,}".replace(",", " ")


def _zeigen(kind_text):
    getattr(st, kind_text[0])(kind_text[1])


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ert_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Das Gerüst täuscht den Erklärer")
st.markdown(
    """
Ein Modell lässt sich so umhüllen, dass es **im Betrieb** auf echten Zeilen eine **verzerrte** Vorhersage liefert, dem **Erklärer** aber ein harmloses Modell zeigt: das **Gerüst** erkennt
**unmögliche Mischzeilen**, die ein Erklärer erzeugt, und antwortet dort ehrlich. Die Demo misst in einem erfundenen Verweilzeit-Vehikel mit bekannter Funktion μ, **wie weit diese Täuschung
reicht** (Wirksamkeit gegen Rang- und Betragsmaß) und **was sie den Angreifer kostet** (Detektor, Wissen, Hintergrund). Es gibt **keine reale Datenquelle**.

Weiter unten: vier **🔬 Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Gerüst", expanded=True):
    st.markdown(
        """
1. **Verzerrung:** Das verzerrte Modell g_b rechnet für Kunden 6 bis 11 c/2 auf ln(Verweilzeit) auf und für Kunden 0 bis 5 c/2 ab: der Unterschied zwischen den beiden Gruppen ist der Aufschlag c. Das ehrliche Modell g_h tut das nicht.
2. **Gerüst:** h(x) = g_b(x), wenn ein **Detektor** x für eine echte Zeile aus den Daten hält, sonst g_h(x). Der Detektor muss nichts über den Aufschlag wissen: er erkennt nur **unmögliche** Zeilen
   (zum Beispiel Import mit Vorlauf 5 Tage).
3. **Erklärer:** Der exakte Shapley-Wert (interventional, Hintergrund 30 Zeilen) und ein LIME-artiges Surrogat fragen das Modell an Mischzeilen (x_S, B_Rest). Viele davon sind **unmöglich**; dort
   antwortet das Gerüst ehrlich. **Wie viel davon im Beitrag der Kundengruppe ankommt, hängt am Anteil der Mischzeilen, die der Detektor erkennt, nicht allein am Verfahren.**
        """
    )
st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.caption("Alle Zahlen kommen aus einem vorgerechneten Raster (10 Zufalls-Seeds); die Regler wählen Zellen des Rasters.")
    st.markdown("**Modell und Aufschlag**")
    basis = st.selectbox("Modellklasse", C.BASEN, key="basis_sel", format_func=C.BASIS_LABEL.get, help="Modellklasse des verzerrten und des ehrlichen Modells (Hyperparameter fest).")
    c = st.select_slider("Aufschlag c auf ln(Verweilzeit): Unterschied der Kundengruppen", options=C.C_REGLER, key="c_sel", format_func=lambda v: ev.de(v, 1),
                         help="c = 0: verzerrtes gleich ehrliches Modell, die Wirksamkeit ist dann nicht definiert. Kunden 6 bis 11 bekommen +c/2, Kunden 0 bis 5 -c/2 auf ln(Verweilzeit); c = 1 ist der Faktor 2,7 zwischen den beiden Gruppen.")
    st.markdown("**Gerüst**")
    detektor = st.selectbox("Detektor", C.DETEKTOREN, key="detektor_sel", format_func=C.DET_LABEL.get,
                            help="Wer entscheidet, ob eine Zeile aus den Daten stammt? Orakel = die Gültigkeitsprüfung selbst (obere Grenze für den Angreifer).")
    st.markdown("**Erklärer: Shapley-Hintergrund**")
    hintergrund = st.radio("Hintergrund", C.HINTERGRUENDE, key="hintergrund_sel", format_func=C.HINT_LABEL.get,
                           help="Zufall: Hintergrundzeilen aus allen Trainingszeilen. Dieselbe Art: Hintergrundzeilen mit der Art der erklärten Zeile (dann sind alle Mischzeilen möglich).")
    quelle = st.radio("Hintergrundzeilen", C.QUELLEN, key="quelle_sel", format_func=C.QUELLE_HINT_LABEL.get,
                      help="Trainingszeilen kennt ein Angreifer mit Gedächtnis; frische Zeilen nicht.")
    st.markdown("**Erklärer: Surrogat**")
    stichprobe = st.selectbox("Stichprobe", C.STICHPROBEN, key="stichprobe_sel", format_func=C.STICHPROBE_LABEL.get,
                              help="std: jede Spalte unabhängig gezogen; koal: Koalitions-Ersatz; zeile: ganze echte Zeilen.")

sync_query_params({"basis_sel": basis, "c_sel": c, "detektor_sel": detektor, "hintergrund_sel": hintergrund, "quelle_sel": quelle, "stichprobe_sel": stichprobe})

raster = _raster()
N = len(raster["meta"]["seeds"])
P_HG = 0.0 if hintergrund == "zufall" else 1.0
c_def = c > 0 and c in C.C_STUFEN
bn = C.BASIS_LABEL[basis]

# --- Ergebnis ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Einstellung: {bn}, Detektor {C.DET_LABEL_KURZ[detektor]}, Aufschlag c = {ev.de(c, 1)}; Zähleinheit: {N} Zufalls-Seeds, Mittel ± Standardfehler über Zufalls-Seeds.")
dz = ev.detektor_zusammenfassung(raster)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Treue des Gerüsts auf echten Zeilen", ev.pct(dz[detektor]["treue"][0]), delta="Anteil mit verzerrter Vorhersage", delta_color="off",
          help="Anteil echter frischer Zeilen, auf denen das Gerüst das verzerrte Modell liefert (Experiment 1).")
if c_def:
    z2 = ev.shapley_zusammenfassung(raster, basis, c, quelle, P_HG, detektor)
    z3 = ev.surrogat_zusammenfassung(raster, basis, quelle, stichprobe, detektor)
    zo = ev.shapley_zusammenfassung(raster, basis, c, quelle, P_HG, "orakel")
    zg = ev.shapley_zusammenfassung(raster, basis, c, quelle, P_HG, "gedaechtnis")
    m2.metric("Wirksamkeit unter Shapley", _w(z2["wirksamkeit"]), delta=f"Hintergrund {C.HINT_LABEL[hintergrund].split(' (')[0]}", delta_color="off",
              help="Anteil des Abstands zwischen verzerrtem und ehrlichem Modell, den das Gerüst unter dem Erklärer verdeckt (Rang-1-Anteil; 0 nichts, 1 vollständig).")
    m3.metric("Wirksamkeit unter dem Surrogat", _w(z3["wirksamkeit"]), delta=f"Stichprobe {stichprobe}", delta_color="off")
    m4.metric("Wirksamkeit des Angreifers mit Gedächtnis", _w(zg["wirksamkeit"]), delta="Shapley, gleiche Einstellung", delta_color="off")
else:
    m2.metric("Wirksamkeit unter Shapley", "–", delta="c = 0: nicht definiert", delta_color="off")
    m3.metric("Wirksamkeit unter dem Surrogat", "–", delta="c = 0: nicht definiert", delta_color="off")
    m4.metric("Wirksamkeit des Angreifers mit Gedächtnis", "–", delta="c = 0: nicht definiert", delta_color="off")
    st.info("Bei c = 0 ist das verzerrte Modell gleich dem ehrlichen: es gibt nichts zu verdecken, die Wirksamkeit ist nicht definiert (–). Experiment 1 und das Handbeispiel gelten weiter.")

st.markdown("## ❓ Kernfrage: Wie weit reicht die Täuschung, und woran hängt sie?")
if c_def:
    _zeigen(ev.verdict_shapley(z2, P_HG == 1.0, detektor == "orakel", c, N))
else:
    st.info("Wählen Sie einen Aufschlag c größer als 0, um die Wirksamkeit zu sehen.")

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das Gerüst im Betrieb")
st.caption(f"Wie gut trennt der Detektor unmögliche von echten Zeilen? Je Zufalls-Seed {_tsd(C.N_TEST)} Zeilen je Prüfung; Detektor-Training: {_tsd(C.N_TRAIN)} echte Zeilen gegen {_tsd(C.N_MIX)} Spaltenmischungen. ")
tab1, tab1b = st.tabs(["Detektoren", "Wissen des Angreifers"])
with tab1:
    d1, d2 = st.columns([1.2, 1])
    with d1:
        zeilen = []
        for d in C.DETEKTOREN:
            r = dz[d]
            zeilen.append({"Detektor": C.DET_LABEL_KURZ[d], "erkennt unmögliche Spaltenmischungen": _ms(r["erk_spalten"]), "erkennt unmögliche Shapley-Mischzeilen": _ms(r["erk_shapley"]),
                           "Fehlalarm auf möglichen Mischzeilen": _ms(r["fehl_moeglich"]), "Fehlalarm auf echten Zeilen": _ms(r["fehl_echt"]), "Treue (verzerrt auf echten Zeilen)": _ms(r["treue"])})
        st.dataframe(zeilen, hide_index=True, width="stretch")
        st.caption(f"Anteil unmöglicher Zeilen: unter Spaltenmischung {_ms(dz['anteil_unmoeglich']['spalten'], 3)}, unter Shapley-Mischzeilen mit Zufallshintergrund "
                   f"{_ms(dz['anteil_unmoeglich']['shapley_zufall'], 3)}, bei Hintergrund derselben Art 0 (exakt). Mittel ± Standardfehler über {dz['n']} Zufalls-Seeds.")
    with d2:
        st.plotly_chart(V.build_detektoren(dz), width="stretch", key="chart_detektoren")
    r = dz[detektor]
    _zeigen(ev.verdict_betrieb(detektor, r["erk_shapley"][0], r["fehl_echt"][0], r["treue"][0], dz["n"]))
    if c_def:
        ab = ev.abstand_zusammenfassung(raster, basis, c)
        st.caption(f"Verzerrung der Vorhersage auf echten Zeilen ({bn}, c = {ev.de(c, 1)}): mittleres |g_b − g_h| = {_ms(ab['verz_ehrlich'], 3)} (ln-Einheiten); "
                   f"das Gerüst mit „{C.DET_LABEL_KURZ[detektor]}“ weicht auf echten Zeilen im Mittel um {_ms(ab[detektor], 3)} vom verzerrten Modell ab.")
with tab1b:
    bw = basis if basis in ("gbm", "rf") else "gbm"
    st.markdown(f"Der Boosting-Detektor wird je Stufe mit **n echten Zeilen und n Spaltenmischungen** trainiert (n = 30 bis 3 000); Modellklasse {C.BASIS_LABEL[bw]}, c = 1,0, Zufallshintergrund aus den Trainingszeilen. "
                + ("Für das kleine Netz ist diese Reihe nicht gerechnet; gezeigt wird Boosting. " if bw != basis else ""))
    wz = ev.wissen_zusammenfassung(raster, bw)
    st.dataframe([{"Zeilen je Klasse im Detektor-Training": z["n"], "erkennt unmögliche Mischzeilen": _ms(z["erk"]), "Fehlalarm auf echten Zeilen": _ms(z["fehl"]), "Ersetzungsanteil q": _ms(z["q"]),
                   "Rang-1-Anteil der Kundengruppe unter dem Gerüst": _ms(z["top1"])} for z in wz], hide_index=True, width="stretch")
    st.plotly_chart(V.build_wissen(wz), width="stretch", key="chart_wissen")
    st.caption("Mit mehr Wissen erkennt der Detektor mehr unmögliche Zeilen, und der Rang-1-Anteil der Kundengruppe unter dem Gerüst steigt: der Angreifer täuscht mit **weniger** Wissen scheinbar mehr. "
               "Gemessen ist, dass der Detektor mit wenig Wissen auch viele echte Zeilen als unmöglich meldet (Fehlalarm in der Tabelle); die Deutung, dass das eingesetzte Modell dort selbst weniger verzerrt ist "
               "(Modellwechsel statt Verbergen), ist nicht getrennt gemessen. Die Reihe ist nur zusammen mit dem Fehlalarm zu lesen.")

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Exakter Shapley-Wert mit Zufallshintergrund")
st.caption(f"Exakter interventionaler Shapley-Wert über 64 Koalitionen; {C.NE_SHAPLEY} erklärte echte frische Zeilen je Zufalls-Seed, Hintergrund {C.NB} Zeilen "
           f"({C.QUELLE_HINT_LABEL[quelle]}); Rang-1-Anteil (Beitrag, lokal) und Anteil an der Wichtigkeit (global, Betragsmaß).")
if c_def:
    tab2, tab2b = st.tabs(["Shapley-Beiträge", "Dosis: Anteil p derselben Art"])
    with tab2:
        e1, e2 = st.columns(2)
        e1.plotly_chart(V.build_rang1(z2, "Rang-1-Anteil der Kundengruppe (Rangmaß)"), width="stretch", key="chart_s_rang1")
        e2.plotly_chart(V.build_rang1(z2, "Anteil der Kundengruppe an der Summe der Wichtigkeiten (Betragsmaß)", feld="anteil", achse="Anteil an der Wichtigkeit"), width="stretch", key="chart_s_anteil")
        zeil = []
        for k, name in (("verzerrt", "verzerrtes Modell"), ("ehrlich", "ehrliches Modell"), ("geruest", f"Gerüst, Detektor {C.DET_LABEL_KURZ[detektor]}")):
            zz = z2[k]
            zeil.append({"Modell": name, "Rang-1-Anteil": _ms(zz["top1"]), "mittlerer lokaler Rang": _ms(zz["mrang"]), "Anteil an der Wichtigkeit": _ms(zz["anteil"]),
                         "global Rang 1": f"{zz['rang1_global']} von {z2['n']}", "Rangkorrelation der Wichtigkeiten zum verzerrten Modell": _ms(zz["rk"])})
        st.dataframe(zeil, hide_index=True, width="stretch")
        st.caption(f"Anteil unmöglicher Mischzeilen unter den 64 Koalitionen: {_ms(z2['unmoeglich'], 3)}; Ersetzungsanteil q (Anteil der Mischzeilen, die der Detektor als nicht aus den Daten meldet): {_ms(z2['q'], 3)}. "
                   f"Wirksamkeit im Betragsmaß (Anteil an der Wichtigkeit): {_w(z2['wirksamkeit_betrag'])}.")
        _zeigen(ev.verdict_shapley(z2, P_HG == 1.0, detektor == "orakel", c, N))
        hint = ev.hinweis_treue(detektor, dz[detektor]["treue"][0])
        if hint:
            st.caption(hint)
        if basis == "mlp":
            st.warning("Beim kleinen Netz verdeckt der Zufallshintergrund die Verzerrung schon ohne Gerüst: der Rang-1-Anteil des unmaskierten verzerrten Modells ist klein, die Wirksamkeit "
                       "ist deshalb kaum auswertbar. Das ist die Falle aus „unmögliche Kombinationen“ (Extrapolation auf unmöglichen Mischzeilen), hier als Nebenbefund.")
    with tab2b:
        zeilen_dosis = []
        for p in C.P_STUFEN:
            zz = ev.shapley_zusammenfassung(raster, basis, c, "training", p, detektor)
            w, se = zz["wirksamkeit"]
            zeilen_dosis.append((p, 0.0 if w is None else w, 0.0 if se is None else se, zz["unmoeglich"][0], zz))
        st.markdown(f"Anteil **p** der Hintergrundzeilen aus der Art der erklärten Zeile (Rest zufällig), Trainingszeilen, Detektor {C.DET_LABEL_KURZ[detektor]}, {bn}.")
        st.dataframe([{"p": ev.de(p, 2), "Anteil unmöglicher Mischzeilen": ev.pct(u, 1), "Rang-1-Anteil verzerrt / ehrlich / Gerüst": f"{ev.de(zz['verzerrt']['top1'][0])} / {ev.de(zz['ehrlich']['top1'][0])} / {ev.de(zz['geruest']['top1'][0])}",
                       "Wirksamkeit": _w(zz["wirksamkeit"])} for p, w, se, u, zz in zeilen_dosis], hide_index=True, width="stretch")
        st.plotly_chart(V.build_dosis([(p, w, se, u) for p, w, se, u, _ in zeilen_dosis]), width="stretch", key="chart_dosis")
        st.caption("Die Wirksamkeit sinkt mit dem Anteil unmöglicher Mischzeilen und ist bei p = 1 mit Orakel-Detektor exakt 0. Die Streuung zwischen Zufalls-Seeds ist bei 24 erklärten Zeilen "
                   "groß: die Richtung ist eindeutig, die Zwischenstufen sind es nicht (Standardfehler beachten).")
else:
    st.info("Für c = 0 gibt es keine Beiträge zu vergleichen (verzerrtes gleich ehrliches Modell).")

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Das Surrogat lässt sich leichter täuschen")
st.caption(f"**LIME-artig** heißt hier die eigene Implementierung (gewichtetes Ridge auf Merkmalen „wie in der erklärten Zeile“, N = {_tsd(C.N_LIME)} Auswertungen; kein Paket `lime`); "
           f"{C.NE_LIME} erklärte Zeilen je Zufalls-Seed, Stichprobe {C.QUELLE_HINT_LABEL[quelle]}.")
if c_def:
    s_rows = {v: ev.surrogat_zusammenfassung(raster, basis, quelle, v, detektor) for v in C.STICHPROBEN}
    z3 = s_rows[stichprobe]
    f1, f2 = st.columns(2)
    f1.plotly_chart(V.build_rang1(z3, f"Rang-1-Anteil der Kundengruppe (Stichprobe {stichprobe})"), width="stretch", key="chart_l_rang1")
    f2.plotly_chart(V.build_rang1(z3, "Anteil der Kundengruppe an der Summe der |Koeffizienten|", feld="anteil", achse="Anteil an der Summe"), width="stretch", key="chart_l_anteil")
    st.dataframe([{"Stichprobe": C.STICHPROBE_LABEL[v], "unmögliche Stichprobenpunkte": ev.pct(s_rows[v]["unmoeglich"][0]), "Rang-1-Anteil verzerrt / ehrlich / Gerüst":
                   f"{ev.de(s_rows[v]['verzerrt']['top1'][0])} / {ev.de(s_rows[v]['ehrlich']['top1'][0])} / {ev.de(s_rows[v]['geruest']['top1'][0])}", "Wirksamkeit": _w(s_rows[v]["wirksamkeit"])}
                  for v in C.STICHPROBEN], hide_index=True, width="stretch")
    _zeigen(ev.verdict_surrogat(stichprobe, z3, N))
    hint = ev.hinweis_treue(detektor, dz[detektor]["treue"][0])
    if hint:
        st.caption(hint)
    if basis == "mlp":
        st.warning("Beim kleinen Netz zeigt das Surrogat die Verzerrung auch ohne Gerüst kaum (kleiner Rang-1-Anteil des unmaskierten verzerrten Modells): nicht auswertbar, kein Täuschungsbefund.")
else:
    st.info("Für c = 0 gibt es keine Koeffizienten zu vergleichen (verzerrtes gleich ehrliches Modell).")

st.markdown("---")

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Der Angreifer kennt die Trainingszeilen")
st.caption("Der Gedächtnis-Detektor ist eine **eigene Konstruktion dieser Demo** (nicht aus der Literatur): Alarm, wenn die Zeile unmöglich ist oder ein stetiger Wert (Gewicht, Vorlauf größer 0) exakt in den "
           "Trainingszeilen vorkommt. Echte frische Zeilen treffen nie exakt, Mischzeilen mit Hintergrund aus den Trainingszeilen fast immer.")
if c_def:
    hz = "derselben Art" if P_HG == 1.0 else "zufällig"
    a1, a2, a3 = st.columns(3)
    a1.metric("Ersetzungsanteil q, Orakel", _ms(zo["q"], 2), delta=f"Hintergrund {hz}, {C.QUELLE_HINT_LABEL[quelle]}", delta_color="off")
    a2.metric("Ersetzungsanteil q, Gedächtnis", _ms(zg["q"], 2), delta="Anteil der Mischzeilen, die ehrlich beantwortet werden", delta_color="off")
    a3.metric("Rang-1-Anteil unter dem Gedächtnis-Gerüst", _ms(zg["geruest"]["top1"]), delta=f"verzerrt {ev.de(zg['verzerrt']['top1'][0])}, ehrlich {ev.de(zg['ehrlich']['top1'][0])}", delta_color="off")
    st.dataframe([
        {"Angreifer": "Orakel", "Ersetzungsanteil q": _ms(zo["q"]), "Rang-1-Anteil unter dem Gerüst": _ms(zo["geruest"]["top1"]), "Wirksamkeit (Rangmaß)": _w(zo["wirksamkeit"]),
         "Wirksamkeit (Betragsmaß)": _w(zo["wirksamkeit_betrag"]), "global Rang 1": f"{zo['geruest']['rang1_global']} von {zo['n']}"},
        {"Angreifer": "Gedächtnis", "Ersetzungsanteil q": _ms(zg["q"]), "Rang-1-Anteil unter dem Gerüst": _ms(zg["geruest"]["top1"]), "Wirksamkeit (Rangmaß)": _w(zg["wirksamkeit"]),
         "Wirksamkeit (Betragsmaß)": _w(zg["wirksamkeit_betrag"]), "global Rang 1": f"{zg['geruest']['rang1_global']} von {zg['n']}"},
    ], hide_index=True, width="stretch")
    _zeigen(ev.verdict_angreifer(zg["wirksamkeit"][0], zg["wirksamkeit"][1], zo["wirksamkeit"][0], zo["wirksamkeit"][1], quelle, P_HG == 1.0, N))
    zs_t = ev.surrogat_zusammenfassung(raster, basis, "training", "zeile", "gedaechtnis")
    zs_f = ev.surrogat_zusammenfassung(raster, basis, "frisch", "zeile", "gedaechtnis")
    st.caption(f"Beim Surrogat mit ganzen Zeilen ({bn}): Stichprobe aus den Trainingszeilen, Rang-1-Anteil unter dem Gedächtnis-Gerüst {_ms(zs_t['geruest']['top1'])} gegen {_ms(zs_t['verzerrt']['top1'])} verzerrt; "
               f"mit frischen Zeilen {_ms(zs_f['geruest']['top1'])}.")
else:
    st.info("Für c = 0 gibt es keinen Angreifer, der etwas zu verbergen hätte.")

st.markdown("##### Das Handbeispiel in Brüchen (live gerechnet)")
st.markdown(
    "Vier Merkmale: **a** Art (Import 0, Export 1), **v** Vorlauf (0 oder 1; **möglich** nur v = a), **k** Kundengruppe (das verzerrte Merkmal), **g** Gewicht (Rauschen; g = 2 steht für frische Zeilen). "
    "Ehrlich f = a, verzerrt f = a + 2k, Gerüst = verzerrt, wenn (a, v) möglich ist. Beide Rechenwege (Koalitionsformel, Mittel über alle 24 Permutationen) geben dieselben Brüche."
)
hb = HB.tabelle()
st.dataframe([{"Fall": t, "φ_a": HB.bruch_text(a), "φ_v": HB.bruch_text(v), "φ_k": HB.bruch_text(k), "φ_g": HB.bruch_text(g), "Rang nach Betrag": " > ".join(rg), "Anteil unmöglicher Mischzeilen": HB.bruch_text(u)}
               for t, a, v, k, g, rg, u in hb], hide_index=True, width="stretch")
wahl = st.selectbox("Fall als Diagramm", range(len(HB.FAELLE)), format_func=lambda i: HB.FAELLE[i][0], key="hb_fall")
fall = hb[wahl]
st.plotly_chart(V.build_handbeispiel(HB.NAMEN, fall[1:5], fall[0]), width="stretch", key="chart_hb")
st.caption("Mit Gedächtnis wird das Rauschmerkmal g Rang 1 (φ_g = 5/4), obwohl die beiden Basismodelle es nicht benutzen (nur der Gedächtnis-Detektor des Gerüsts liest es): die Täuschung erzeugt einen **Schein-Beitrag**. Mit frischem Hintergrund fällt der Angriff auf den Orakel-Fall zurück.")
st.markdown("**Wirksamkeit im Betragsmaß φ_k mit der ehrlichen Grundlinie je Hintergrund (Brüche):**")
st.dataframe([{"Fall": t, "Wirksamkeit": HB.bruch_text(w)} for t, w in HB.wirksamkeiten()], hide_index=True, width="stretch")
st.caption("Bei unabhängiger Ersetzung mit Anteil q gilt φ_k = (1 − q) · φ_k(verzerrt): " + ", ".join(f"q = {HB.bruch_text(q)}: {HB.bruch_text(HB.dosis(q))}" for q in (HB.F(0), HB.F(1, 4), HB.F(1, 2), HB.F(1)))
           + ". Ist die Ersetzung an die Koalition gekoppelt (der Detektor sieht, welche Merkmale aus x stammen), gilt das nicht.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme oder Grenze | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Modell verhält sich beim Erklären wie im Betrieb** | Das Gerüst zeigt dem Erklärer ein anderes Modell als dem Betrieb; die Wirksamkeit hängt am Anteil erkannter Mischzeilen (Experimente 2 bis 4). | diese Demo |
| **Unmögliche Zeilen sind nur ein Messfehler** | Wer unmögliche Zeilen erkennt, nutzt sie aus; ein Hintergrund derselben Art oder frische Zeilen nehmen ihm den Hebel. | diese Demo (Vorgänger: „unmögliche Kombinationen“) |
| **Exakter interventionaler Shapley-Wert** | Gemessen ist nur dieser Wert. KernelSHAP mit gezogenen Koalitionen, bedingtes und pfadabhängiges Shapley (Baum-SHAP) und die Variante „harmlos“ (zweites Modell auf dem Rauschmerkmal) sind **nicht gemessen**. | offen |
| **Ein Vehikel, ein verzerrtes Merkmal** | Erfundener Verweilzeit-Generator, Kundengruppe als einziges verzerrtes Merkmal, ein Aufschlag-Modell, ein Detektor-Training (Spaltenmischung). Auf realen Tabellen kann der Effekt größer oder kleiner sein. | offen |
| **Kleine Stichproben** | Hintergrund 30 Zeilen, 24 (Surrogat 16) erklärte Zeilen und 10 Zufalls-Seeds: Standardfehler über Zufalls-Seeds, nicht über Zeilen. | offen |
| **Täuschung oder Modellwechsel?** | Der Detektor mit zu wenig Wissen weist auch echte Zeilen dem ehrlichen Modell zu (Experiment 1, Tab „Wissen“): die Reihe vermengt Verbergen und Modellwechsel. | offen |
| **Keine Aussage über Betrieb, Praxis oder Recht** | Die Demo misst Zahlen in einem erfundenen Vehikel; sie sagt nichts über Nutzen oder Gefahr in einem Betrieb. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind zwei Modelle $g_b$ (verzerrt) und $g_h$ (ehrlich), ein Detektor $\delta$ und ein Erklärer. Gesucht ist, wie viel vom Abstand der Beiträge des verzerrten Merkmals
$k$ (Kundengruppe) das Gerüst $h$ vor dem Erklärer verbirgt.

| Symbol | Bedeutung |
|---|---|
| $x$, $B$ | erklärte Zeile, Hintergrundmenge (30 Zeilen) |
| $S$, $d$ | Koalition von Merkmalen, Zahl der Merkmale (6) |
| $g_b$, $g_h$ | verzerrtes Modell (Ziel $\ln$ Verweilzeit $+\,c\,(\text{Gruppe}-\tfrac12)$), ehrliches Modell (Ziel $\ln$ Verweilzeit) |
| $\delta(z)$ | Detektor: 1, wenn $z$ als „nicht aus den Daten“ gilt |
| $h$ | Gerüst: $g_h$ bei $\delta=1$, sonst $g_b$ |
| $\phi_j$ | Shapley-Beitrag des Merkmals $j$ |
| $q$ | Ersetzungsanteil: Anteil der Mischzeilen mit $\delta=1$ |
| $\omega$ | Wirksamkeit des Gerüsts |

**Gerüst.** $h(z) = g_b(z)$, wenn $\delta(z)=0$, sonst $g_h(z)$.

**Interventionaler Shapley-Wert** (Wertfunktional $v_h(S) = \frac{1}{|B|}\sum_{b\in B} h(x_S, b_{\bar S})$):
$$\phi_j = \sum_{S\subseteq N\setminus\{j\}} \frac{|S|!\,(d-|S|-1)!}{d!}\,\bigl(v_h(S\cup\{j\}) - v_h(S)\bigr).$$

**Wirksamkeit.** Mit $r_b$, $r_h$, $r_g$ dem Rang-1-Anteil der Kundengruppe unter dem verzerrten Modell, dem ehrlichen Modell und dem Gerüst (jeweils mit demselben Hintergrund):
$$\omega = \frac{r_b - r_g}{r_b - r_h}.$$
0 heißt: nichts verdeckt; 1 heißt: vollständig verdeckt. Bei $r_b = r_h$ ist $\omega$ nicht definiert.

**Dosis.** Wird bei jeder Mischzeile unabhängig mit Wahrscheinlichkeit $q$ ehrlich geantwortet, gilt $v_h = (1-q)\,v_b + q\,v_h^{\text{ehrlich}}$, und wegen der Linearität des Shapley-Werts in $v$:
$$\phi_k(h) = (1-q)\,\phi_k(g_b) + q\,\phi_k(g_h).$$
Ist die Ersetzung an die Koalition gekoppelt (der Detektor sieht, welche Merkmale aus $x$ stammen), gilt das nicht; das zeigt das Handbeispiel.

**LIME-artiges Surrogat.** Für Stichprobenpunkte $z_i$ mit binären Merkmalen $u_i\in\{0,1\}^d$ („wie in $x$“) und Gewichten $w_i = \exp(-m_i/\sigma^2)$ ($m_i$ Zahl der abweichenden Spalten, $\sigma=3$):
$$\hat\beta = \arg\min_\beta \sum_i w_i\,(f(z_i)-\beta_0-u_i^\top\beta)^2 + \alpha\|\beta\|^2,\qquad \alpha=1.$$

**Quellen (Stufe).** Slack, Hilgard, Jia, Singh, Lakkaraju 2020, „Fooling LIME and SHAP“, arXiv 1911.02508: nur das Abstract gelesen (Stufe A); Konstruktion, Detektoren und Zahlen dieser Demo sind eigen und
keine Wiederholung der Studie. Vorbilder für Beitrag gegen Wahrheit auf bekannten Funktionen: XAI-Bench, OpenXAI, Hooker u. a. (nicht Teil dieser Demo). Eine Neuheitsaussage wird nicht gemacht.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
