"""Visualisation 3D du retrait modélisé (fit_retrait.py) : isosurfaces animées dans le temps,
et des fissures détectées (figure_fissures_3d) placées sur les fibres.

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
TOL_DATE_J = 0.5   # jours : écart maximal entre la date de l'image et la mesure superposée


def champ_retrait(theta, de: float, dates, n: int = N_GRILLE, couplage=None):
    """Retourne (x, y, z, eps) : x, y, z (mm) vecteurs 1D de la grille, eps (µε) de forme (len(dates), n, n, n)."""
    if couplage is None:
        N, k, a, ex, beta, alpha_crit = (list(theta) + [0.0, 0.3][len(theta) - 4:])[:6]
        A = F.alpha_1d(N, k, de, beta, alpha_crit)
    else:                                  # couplage F(w) : theta = (N, k, a, n, c, p)
        N, k, a, ex, c, p = theta
        A = F.alpha_1d(N, k, de, 0.0, 0.3, mult_t=F.facteur_fw(couplage, c, p))
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
    # une trace par segment à CHAQUE image (vide si le segment n'a pas de mesure à cette date) : le nombre de traces
    # doit rester constant d'une image à l'autre, sinon les fibres absentes de la 1re image ne sont jamais affichées
    for s, q in d.groupby("segment"):
        ts = q["temps_ecoule_j"].unique()
        tp = ts[np.argmin(np.abs(ts - t))]
        q = q[q["temps_ecoule_j"] == tp].sort_values("xb") if abs(tp - t) <= TOL_DATE_J else q.iloc[:0]
        traces.append(go.Scatter3d(x=q["X"] * 1000, y=q["Y"] * 1000, z=q["Z"] * 1000, mode="markers", name=f"mesure {s}",
                                   marker=dict(size=3, color=q["eps"], colorscale="Turbo", cmin=vmin, cmax=0.0, showscale=False),
                                   hovertemplate=f"{s}<br>%{{marker.color:.0f}} µε<extra></extra>"))
    return traces


def figure_retrait_3d(theta, de: float, dates, cube: str, g: pd.DataFrame | None = None, segments=(),
                      n: int = N_GRILLE, n_dates: int = N_DATES, couplage=None):
    """Isosurfaces du retrait modélisé, animées (curseur de temps). Fibres mesurées superposées si `segments` est donné."""
    dates = np.asarray(sorted(dates), float)
    if len(dates) > n_dates:
        dates = dates[np.unique(np.linspace(0, len(dates) - 1, n_dates).round().astype(int))]
    x, y, z, eps = champ_retrait(theta, de, dates, n, couplage)
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


N_DATES_FISSURES = 30   # nombre maximal d'images de la figure des fissures
W_MIN_AFFICHE = 0.5     # µm : les fissures plus fermées ne sont pas tracées (gaussiennes pas encore apparues)
SYMBOLES = {"X": "circle", "Y": "square", "Z": "diamond"}


def figure_fissures_3d(tabs: dict, g: pd.DataFrame, cube: str, n_dates: int = N_DATES_FISSURES, w_min: float = W_MIN_AFFICHE):
    """Fissures détectées, placées dans le cube à leur position sur les fibres (aucune interpolation entre fibres).

    tabs : {segment: tableau des fissures (colonnes t_j, pos_mm, w_um)} issu de la décomposition (gaussiennes ou shear lag) ;
    g : données préparées (F.prepare), pour la géométrie des fibres. Un point par fissure : taille et couleur selon
    l'ouverture w (µm), symbole selon l'orientation de la fibre, curseur de temps (dernière date affichée au départ).
    Une fibre ne voit que les fissures qui la traversent. Même orientation que les autres vues 3D du dashboard :
    X vertical (0 en haut), Z horizontal. Retourne None s'il n'y a aucune fissure."""
    parts, fibres = [], []
    for s, tab in tabs.items():
        q = g[g["segment"] == s]
        if q.empty or tab is None or len(tab) == 0:
            continue
        axe = q["axe_mesure"].iloc[0]
        fixe = {a: float(q[a].median()) * 1000 for a in "XYZ"}          # mm ; la coordonnée le long de la fibre vient de pos_mm
        d = tab[["t_j", "pos_mm", "w_um"]].copy()
        d["segment"], d["axe"] = s, axe
        for a in "XYZ":
            d[a] = d["pos_mm"] if a == axe else fixe[a]
        parts.append(d)
        bouts = [dict(fixe, **{axe: float(v) * 1000}) for v in (q[axe].min(), q[axe].max())]
        fibres.append((s, bouts))
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    dates = np.sort(d["t_j"].unique())
    if len(dates) > n_dates:
        dates = dates[np.unique(np.linspace(0, len(dates) - 1, n_dates).round().astype(int))]
    wmax = max(float(d["w_um"].max()), 1.0)
    cote = F.L * 1000

    def marqueurs(axe, t, echelle):
        q = d[(d["axe"] == axe) & np.isclose(d["t_j"], t, rtol=0, atol=0.01) & (d["w_um"] >= w_min)]
        return go.Scatter3d(
            x=q["Z"], y=q["Y"], z=q["X"], mode="markers", name=f"fibres selon {axe}",
            marker=dict(size=4 + 22 * np.sqrt(q["w_um"] / wmax), color=q["w_um"], colorscale="YlOrRd", cmin=0.0, cmax=wmax,
                        symbol=SYMBOLES[axe], line=dict(width=0.5, color="black"), showscale=echelle,
                        colorbar=dict(title="w (µm)")),
            customdata=np.c_[q["segment"], q["pos_mm"].round(1), q["w_um"].round(1)],
            hovertemplate="%{customdata[0]} — position %{customdata[1]} mm<br>w = %{customdata[2]} µm<extra></extra>")

    def image(t):
        return [marqueurs(a, t, a == "X") for a in "XYZ"]      # 3 traces à chaque image (vides si besoin)

    xs, ys, zs = [], [], []                                    # 12 arêtes du cube
    for k in range(3):
        for a in (0.0, cote):
            for b in (0.0, cote):
                p0, p1 = [a, b], [a, b]
                p0.insert(k, 0.0); p1.insert(k, cote)
                xs += [p0[0], p1[0], None]; ys += [p0[1], p1[1], None]; zs += [p0[2], p1[2], None]
    fx, fy, fz = [], [], []
    for _, bouts in fibres:
        for b in bouts + [None]:
            fx.append(None if b is None else b["Z"]); fy.append(None if b is None else b["Y"]); fz.append(None if b is None else b["X"])
    fixes = [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="lightgray", width=3), hoverinfo="skip", showlegend=False),
             go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="gray", width=3), hoverinfo="skip", name="fibres analysées")]
    fig = go.Figure(data=fixes + image(dates[-1]))
    fig.frames = [go.Frame(data=image(t), name=f"{t:.1f}", traces=[2, 3, 4]) for t in dates]
    axe3d = lambda titre, **kw: dict(title=titre, range=[0, cote], **kw)
    fig.update_layout(
        title=f"{cube} — fissures détectées sur les fibres (taille et couleur : ouverture w)", height=640,
        margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=0.02),
        scene=dict(xaxis=axe3d("Z (mm, horizontal)"), yaxis=axe3d("Y (mm)"),
                   zaxis=dict(title="X (mm, vertical, 0 = haut)", range=[cote, 0]), aspectmode="cube"),
        sliders=[dict(active=len(dates) - 1, currentvalue=dict(prefix="Temps écoulé : ", suffix=" j"),
                      steps=[dict(label=f"{t:.0f}", method="animate",
                                  args=[[f"{t:.1f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for t in dates])])
    return fig
