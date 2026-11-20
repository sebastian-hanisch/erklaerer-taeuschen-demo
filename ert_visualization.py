"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import ert_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=320, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=-0.25))
    return lock_axes(fig)


def build_rang1(z, titel, feld="top1", achse="Anteil der erklärten Zeilen mit Kundengruppe auf Rang 1"):
    """Drei Balken (verzerrt, ehrlich, Gerüst) mit Standardfehler über Zufalls-Seeds. z = shapley_zusammenfassung oder surrogat_zusammenfassung."""
    namen = ["verzerrtes Modell", "ehrliches Modell", "Gerüst"]
    farben = [C.COLOR_VERZ, C.COLOR_EHRLICH, C.COLOR_GERUEST]
    werte = [z[k][feld] for k in ("verzerrt", "ehrlich", "geruest")]
    fig = go.Figure(go.Bar(x=namen, y=[w[0] for w in werte], marker_color=farben, error_y=dict(type="data", array=[w[1] for w in werte], visible=True),
                           text=[f"{w[0]:.2f}".replace(".", ",") for w in werte], textposition="outside", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text=achse, range=[0, 1.12])
    fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig, legend=False)


def build_detektoren(zus):
    """Erkennung unmöglicher Mischzeilen und Fehlalarm auf echten Zeilen je Detektor (Mittel über Zufalls-Seeds)."""
    dets = list(C.DETEKTOREN)
    namen = [C.DET_LABEL_KURZ[d] for d in dets]
    fig = go.Figure()
    fig.add_bar(name="erkennt unmögliche Mischzeilen", x=namen, y=[zus[d]["erk_shapley"][0] for d in dets], marker_color=C.COLOR_GERUEST,
                error_y=dict(type="data", array=[zus[d]["erk_shapley"][1] for d in dets], visible=True))
    fig.add_bar(name="Fehlalarm auf echten Zeilen", x=namen, y=[zus[d]["fehl_echt"][0] for d in dets], marker_color=C.COLOR_VERZ,
                error_y=dict(type="data", array=[zus[d]["fehl_echt"][1] for d in dets], visible=True))
    fig.update_yaxes(title_text="Anteil", range=[0, 1.05])
    fig.update_layout(barmode="group")
    return _base(fig)


def build_dosis(zeilen):
    """Wirksamkeit und Anteil unmöglicher Mischzeilen über dem Anteil p des Hintergrunds aus derselben Art. zeilen: Liste (p, w, se, unmoeglich)."""
    p = [z[0] for z in zeilen]
    fig = go.Figure()
    fig.add_scatter(x=p, y=[z[1] for z in zeilen], mode="lines+markers", name="Wirksamkeit", line=dict(color=C.COLOR_GERUEST),
                    error_y=dict(type="data", array=[z[2] for z in zeilen], visible=True))
    fig.add_scatter(x=p, y=[z[3] for z in zeilen], mode="lines+markers", name="Anteil unmöglicher Mischzeilen", line=dict(color=C.COLOR_GRAU, dash="dot"))
    fig.update_xaxes(title_text="Anteil p des Hintergrunds aus derselben Art", tickvals=p)
    fig.update_yaxes(title_text="Wirksamkeit bzw. Anteil", range=[-0.1, 1.05])
    return _base(fig)


def build_wissen(zeilen):
    """Wissen des Angreifers: Zahl der Störzeilen im Detektor-Training gegen Erkennung, Fehlalarm und Rang-1-Anteil des Gerüsts. zeilen: Wörterbücher aus wissen_zusammenfassung."""
    x = [str(z["n"]) for z in zeilen]
    fig = go.Figure()
    fig.add_scatter(x=x, y=[z["erk"][0] for z in zeilen], mode="lines+markers", name="erkennt unmögliche Mischzeilen", line=dict(color=C.COLOR_GERUEST))
    fig.add_scatter(x=x, y=[z["fehl"][0] for z in zeilen], mode="lines+markers", name="Fehlalarm auf echten Zeilen", line=dict(color=C.COLOR_VERZ))
    fig.add_scatter(x=x, y=[z["top1"][0] for z in zeilen], mode="lines+markers", name="Rang-1-Anteil der Kundengruppe unter dem Gerüst", line=dict(color=C.COLOR_EHRLICH, dash="dot"))
    fig.update_xaxes(title_text="Störzeilen im Detektor-Training", type="category")
    fig.update_yaxes(title_text="Anteil", range=[0, 1.05])
    return _base(fig)


def build_handbeispiel(namen, phi, titel):
    """Beiträge (Brüche als Zahlen) je Merkmal des Handbeispiels."""
    farben = [C.COLOR_VERZ if n.startswith("k") else (C.COLOR_GRAU if n.startswith("g") else C.COLOR_EHRLICH) for n in namen]
    fig = go.Figure(go.Bar(x=namen, y=[float(p) for p in phi], marker_color=farben, hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title_text="Beitrag φ")
    fig.update_layout(title=dict(text=titel, font=dict(size=13)))
    return _base(fig, height=260, legend=False)
