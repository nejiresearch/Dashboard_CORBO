"""Visualisation 3D du retrait modélisé (fit_retrait.py) : isosurfaces animées dans le temps.

Le champ eps(t, x, y, z) = -a * alpha_3D**n est évalué sur une grille régulière du cube (N_GRILLE points par axe)
à quelques dates ; alpha_3D vient de F.alpha_1d + F.alpha_3d (inchangés). C'est le retrait MODÉLISÉ (matrice seule),
pas la mesure : les fibres choisies peuvent être superposées pour comparaison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import fit_retrait as F

N_GRILLE = 24      # points par axe (24³ ≈ 14 000 points par image : reste fluide dans le navigateur)
N_DATES = 12       # nombre d'images de l'animation
NIVEAUX = 4        # nombre d'isosurfaces


def champ_retrait(theta, de: float, dates, n: int = N_GRILLE):
    """Retourne (x, y, z, eps) : x, y, z (mm) vecteurs 1D de la grille, eps (µε) de forme (len(dates), n, n, n)."""
    N, k, a, ex, beta, alpha_crit = (list(theta) + [0.0, 0.3][len(theta) - 4:])[:6]
    A = F.alpha_1d(N, k, de, beta, alpha_crit)
    p = np.linspace(0.0, F.L, n)
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    eps = np.empty((len(dates), n, n, n))
    for i, t in enumerate(dates):
        alpha = F.alpha_3d(A, X.ravel(), Y.ravel(), Z.ravel(), np.full(X.size, float(t)))
        eps[i] = (-a * alpha ** ex).reshape(X.shape)
    return p * 1000, p * 1000, p * 1000, eps


def _fibres(g: pd.DataFrame, segments, t: float, vmin: float):
    """Traces Scatter3d des fibres choisies (mesure à la date la plus proche de t, en µε, même échelle de couleur)."""
    traces = []
    if g is None or not len(segments):
        return traces
    d = g[g["segment"].isin(segments)]
    if d.empty:
        return traces
    ts = d["temps_ecoule_j"].unique()
    d = d[d["temps_ecoule_j"] == ts[np.argmin(np.abs(ts - t))]]
    for s, q in d.groupby("segment"):
        q = q.sort_values("xb")
        traces.append(go.Scatter3d(x=q["X"] * 1000, y=q["Y"] * 1000, z=q["Z"] * 1000, mode="markers", name=f"mesure {s}",
                                   marker=dict(size=3, color=q["eps"], colorscale="Turbo", cmin=vmin, cmax=0.0, showscale=False),
                                   hovertemplate=f"{s}<br>%{{marker.color:.0f}} µε<extra></extra>"))
    return traces


def figure_retrait_3d(theta, de: float, dates, cube: str, g: pd.DataFrame | None = None, segments=(),
                      n: int = N_GRILLE, n_dates: int = N_DATES):
    """Isosurfaces du retrait modélisé, animées (curseur de temps). Fibres mesurées superposées si `segments` est donné."""
    dates = np.asarray(sorted(dates), float)
    if len(dates) > n_dates:
        dates = dates[np.unique(np.linspace(0, len(dates) - 1, n_dates).round().astype(int))]
    x, y, z, eps = champ_retrait(theta, de, dates, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    vmin = float(min(eps.min(), -1.0))
    niveaux = dict(isomin=0.9 * vmin, isomax=0.1 * vmin, surface_count=NIVEAUX)

    def iso(i, complet=False):
        """complet=False : seules les valeurs sont renvoyées (la grille x, y, z de la 1re image est conservée : image plus légère)."""
        if not complet:
            return go.Isosurface(value=eps[i].ravel().round(0))
        return go.Isosurface(x=X.ravel().round(1), y=Y.ravel().round(1), z=Z.ravel().round(1), value=eps[i].ravel().round(0),
                             **niveaux, opacity=0.35,
                             colorscale="Turbo", cmin=vmin, cmax=0.0, caps=dict(x_show=False, y_show=False, z_show=False),
                             colorbar=dict(title="ε retrait (µε)"), name="retrait modèle",
                             hovertemplate="X=%{x:.0f} Y=%{y:.0f} Z=%{z:.0f} mm<br>%{value:.0f} µε<extra></extra>")

    fibres = [_fibres(g, segments, t, vmin) for t in dates]
    nf = len(fibres[0])
    # une trace d'isosurface + les traces de fibres (mises à jour à chaque image)
    fig = go.Figure(data=[iso(0, True)] + [tr for tr in fibres[0]])
    fig.frames = [go.Frame(data=[iso(i)] + fibres[i], name=f"{t:.1f}", traces=list(range(1 + nf)))
                  for i, t in enumerate(dates)]
    fig.update_layout(
        title=f"{cube} — retrait modélisé (isosurfaces de ε, µε)", height=620, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="cube"),
        sliders=[dict(active=0, currentvalue=dict(prefix="Temps écoulé : ", suffix=" j"),
                      steps=[dict(label=f"{t:.0f}", method="animate",
                                  args=[[f"{t:.1f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for t in dates])])   # pas de bouton Play : curseur seul (comme l'onglet 3D existant)
    return fig
