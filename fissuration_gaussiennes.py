"""
Décomposition en gaussiennes du résidu de fissuration d'un segment de fibre.

Principe (voir README, « Fissuration par décomposition en gaussiennes »)
------------------------------------------------------------------------
résidu r(x, t) = déformation mesurée − retrait de la matrice (modèle de fit_retrait.py)
r(x, t) ≈ Σ_i A_i(t) · exp(−(x − c_i)² / (2 σ_i(t)²))      avec A_i ≥ 0

Chaque gaussienne est une fissure (ou un amas) ; son ouverture est l'aire du pic :
    w_i = A_i · σ_i · √(2π) · 1e-3     (A en µε, σ en mm, w en µm)
L'aire ne dépend pas de la jauge de l'instrument (5 mm), contrairement à A et σ.

Algorithme (défaut : ordre inverse, sans fusion)
  1. Dernière échéance (signal le mieux résolu) : ajout glouton de gaussiennes, centres libres.
  2. Échéances précédentes, à rebours : centres FIGÉS, seuls A_i >= 0 et σ_i sont réajustés.
     Aucune gaussienne n'est supprimée ni fusionnée ; A_i ≈ 0 avant son apparition.
  3. Le regroupement « diffus » est fait a posteriori (groupes() : centres distants de <= GROUPE_MM).
  --avant : ancien ordre chronologique ; --fusion : fusion de voisines si le SSE augmente de < MERGE_TOL.

Usage :
    python fissuration_gaussiennes.py --cube C12 --segment s14 --plot s14.png --csv s14.csv
    python fissuration_gaussiennes.py --cube C12 --segment s14 --theta 23262 0.0387 5981 1.07
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import fit_retrait as F

T0_DEFAUT = 28.0      # jours : avant, le modèle de retrait est jugé peu performant
S_MIN = 1.45          # mm : écart-type minimal ≈ résolution (jauge 5 mm / √12)
S_MAX = 25.0          # mm
ADD_GAIN = 0.04       # gain relatif minimal du SSE pour ajouter une gaussienne
MERGE_TOL = 0.03      # perte relative maximale du SSE pour accepter une fusion
LAMBDA_SIGMA = 400.0  # µε/mm : pénalité sur la variation de σ entre deux échéances (régularisation)
GROUPE_MM = 8.0      # mm : écart maximal entre centres d'un même groupe (regroupement a posteriori)
SEUIL_RMSE = 2.0      # un pic n'est candidat que s'il dépasse SEUIL_RMSE × RMSE courant (0 = désactivé)
A_MIN = 500.0         # µε : amplitude minimale d'une gaussienne ajoutée
DIST_MIN = 2.0        # mm : distance minimale entre deux centres à l'ajout
DECALAGE_MAX = 4.0    # mm : déplacement autorisé du centre à la première échéance
N_MAX = 25            # gaussiennes maximum
THETA_DEFAUT = (23262.0, 0.0387, 5981.0, 1.07)   # (N, k, a, n) du fit C12, D_e = 3,7e-7 m2/s


def somme_gauss(x, C, A, S):
    y = np.zeros_like(x, dtype=float)
    for c, a, s in zip(C, A, S):
        y += a * np.exp(-0.5 * ((x - c) / s) ** 2)
    return y


def ouverture_um(A, S):
    return A * S * np.sqrt(2 * np.pi) * 1e-3


def ajuster(x, r, C, A, S, libres):
    """Ajuste A, σ (toujours) et les centres marqués libres. Retourne C, A, S, SSE."""
    n = len(C); C0 = np.asarray(C, float); fi = np.where(libres)[0]

    def f(q):
        c = C0.copy(); c[fi] = q[2 * n:]
        return somme_gauss(x, c, q[:n], q[n:2 * n]) - r

    q0 = np.r_[A, S, C0[fi]]
    lo = np.r_[np.zeros(n), np.full(n, S_MIN), np.maximum(C0[fi] - DECALAGE_MAX, x.min())]
    hi = np.r_[np.full(n, 3e4), np.full(n, S_MAX), np.minimum(C0[fi] + DECALAGE_MAX, x.max())]
    q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
    s = least_squares(f, q0, bounds=(lo, hi), max_nfev=80,
                      x_scale=np.r_[np.full(n, 1000.0), np.full(n, 3.0), np.full(len(fi), 2.0)])
    c = C0.copy(); c[fi] = s.x[2 * n:]
    return c, s.x[:n], s.x[n:2 * n], float(np.sum(s.fun ** 2))


class Suivi:
    """Ensemble de gaussiennes suivies dans le temps (identifiants persistants)."""

    def __init__(self):
        self.G = []        # liste de dict(id, c, A, S)
        self.prochain_id = 0
        self.journal = []

    def _nouvel_id(self):
        self.prochain_id += 1
        return self.prochain_id - 1

    def traiter(self, x, r, premier=False, ajouts=True, fusion=False):
        G = self.G
        if G:
            C = np.array([g["c"] for g in G]); A = np.array([g["A"] for g in G]); S = np.array([g["S"] for g in G])
            C, A, S, sse = ajuster(x, r, C, A, S, np.full(len(G), premier))
        else:
            C = A = S = np.zeros(0); sse = float(np.sum(r ** 2))
        ident = [g["id"] for g in G]

        # ajouts
        while ajouts and len(C) < N_MAX:
            res = r - somme_gauss(x, C, A, S); i = int(np.argmax(res))
            if res[i] < SEUIL_RMSE * np.sqrt(sse / len(x)):
                break
            C2 = np.r_[C, x[i]]; A2 = np.r_[A, res[i]]; S2 = np.r_[S, 2.5]
            libres = np.r_[np.full(len(C), premier), True]
            c, a, s, sse2 = ajuster(x, r, C2, A2, S2, libres)
            proche = len(C) > 0 and np.min(np.abs(c[:-1] - c[-1])) < DIST_MIN
            if sse2 > sse * (1 - ADD_GAIN) or a[-1] < A_MIN or proche:
                break
            C, A, S, sse = c, a, s, sse2
            ident.append(self._nouvel_id()); self.journal.append(f"+ #{ident[-1]} en X={c[-1]:.1f} mm")

        # fusions
        while fusion and len(C) > 1:
            o = np.argsort(C); meilleure = None
            for k in range(len(o) - 1):
                i, j = o[k], o[k + 1]
                ai, aj = A[i] * S[i], A[j] * S[j]
                cm = (C[i] * ai + C[j] * aj) / (ai + aj)
                sm = np.sqrt(((S[i] ** 2 + (C[i] - cm) ** 2) * ai + (S[j] ** 2 + (C[j] - cm) ** 2) * aj) / (ai + aj))
                sm = min(max(sm, S_MIN), S_MAX)
                garde = [m for m in range(len(C)) if m not in (i, j)]
                C2 = np.r_[C[garde], cm]; A2 = np.r_[A[garde], (ai + aj) / sm]; S2 = np.r_[S[garde], sm]
                libres = np.r_[np.zeros(len(garde), bool), True]
                c, a, s, sse2 = ajuster(x, r, C2, A2, S2, libres)
                if sse2 <= sse * (1 + MERGE_TOL) and (meilleure is None or sse2 < meilleure[0]):
                    meilleure = (sse2, c, a, s, garde, i, j)
            if meilleure is None:
                break
            sse2, c, a, s, garde, i, j = meilleure
            nid = self._nouvel_id()
            self.journal.append(f"fusion #{ident[i]} (X={C[i]:.1f}) + #{ident[j]} (X={C[j]:.1f}) -> "
                                f"#{nid} (X={c[-1]:.1f}, σ={s[-1]:.1f})")
            ident = [ident[m] for m in garde] + [nid]
            C, A, S, sse = c, a, s, sse2

        self.G = [dict(id=ident[k], c=float(C[k]), A=float(A[k]), S=float(S[k])) for k in range(len(C))]
        return sse


def donnees_segment(cube, segment, theta, de=F.DE_DEFAUT, bin_mm=1, g=None):
    """g : données déjà préparées par F.prepare (évite de relire le CSV, utile pour le dashboard)."""
    if g is None:
        g = F.prepare(cube, bin_mm=bin_mm)
    h = g[g["segment"] == segment].copy()
    if h.empty:
        raise SystemExit(f"segment {segment} absent (ou exclu) pour {cube}")
    h["mod"] = F.modele(theta, h, de)
    h["r"] = h["eps"] - h["mod"]
    return h


def analyser(h, t0=T0_DEFAUT, verbose=True, arriere=True, fusion=False, ajout_arriere=False, regul=True, dates=None, t_calage=None):
    """Retourne (tableau long des gaussiennes, historique par échéance).

    arriere=True : les centres sont déterminés sur la DERNIÈRE échéance (signal le mieux résolu),
    puis figés et suivis à rebours ; A_i(t) → 0 aux dates où la gaussienne n'est pas encore apparue.
    fusion=False : aucune fusion pendant l'ajustement (le regroupement diffus se fait ensuite, voir groupes()).
    t_calage : échéance où les gaussiennes sont « calées » (ajout glouton, centres libres). Défaut : dernière échéance.
    Le suivi va de t_calage vers les dates antérieures, puis (centres figés, sans ajout) vers les dates postérieures.
    """
    axe = h["axe_mesure"].iloc[0]
    echeances = np.sort(h["temps_ecoule_j"].unique()); echeances = echeances[echeances >= t0]
    if dates is not None:                      # sélection explicite des échéances (dashboard)
        echeances = np.array([t for t in echeances if np.any(np.isclose(t, dates, rtol=0, atol=0.01))])
    t_ref = float(echeances[-1] if t_calage is None else echeances[np.argmin(np.abs(echeances - t_calage))])
    if arriere:
        i_ref = int(np.where(echeances == t_ref)[0][0])
        ordre = list(echeances[:i_ref + 1][::-1]) + list(echeances[i_ref + 1:])
    else:
        ordre = echeances
    suivi = Suivi(); lignes = []; hist = {}; textes = {}; etat_ref = None
    for k, t in enumerate(ordre):
        if arriere and t > t_ref and etat_ref is not None:      # départ du suivi vers l'avant depuis t_calage
            suivi.G = [dict(g) for g in etat_ref]; etat_ref = None
        q = h[h["temps_ecoule_j"] == t].sort_values("xb")
        x = q[axe].values * 1000
        suivi.journal = []
        sse = suivi.traiter(x, q["r"].values, premier=(k == 0), ajouts=(k == 0 or not arriere or ajout_arriere),
                            fusion=fusion)
        hist[t] = (x, q, [dict(g) for g in suivi.G], sse)
        if t == t_ref:
            etat_ref = [dict(g) for g in suivi.G]
        textes[t] = [f"{t:6.1f} j : {len(suivi.G):2d} gaussiennes, RMSE={np.sqrt(sse / len(x)):4.0f} µε, "
                     f"Σw={sum(ouverture_um(g['A'], g['S']) for g in suivi.G):.0f} µm"] + ["          " + j for j in suivi.journal]
    if regul and arriere and not fusion:
        if verbose:
            print('Suivi à rebours terminé ; ajustement global régularisé :')
        hist = regulariser(hist, lam_sigma=LAMBDA_SIGMA, verbose=verbose, t_ref=t_ref); textes = {t: [] for t in echeances}
    for t in echeances:
        for g in hist[t][2]:
            lignes.append(dict(t_j=t, id=g["id"], pos_mm=g["c"], A_microdef=g["A"], sigma_mm=g["S"],
                               w_um=ouverture_um(g["A"], g["S"])))
        if verbose:
            print("\n".join(textes[t]))
    return pd.DataFrame(lignes), hist


def regulariser(hist, lam_sigma=LAMBDA_SIGMA, verbose=True, t_ref=None):
    """Ajustement GLOBAL de toutes les échéances (option D).

    Paramètres : centres c_i (communs à toutes les dates, ± DECALAGE_MAX autour des valeurs initiales),
    ouvertures w_i(t_k) = somme cumulée d'incréments >= 0 (donc w_i(t) NON DÉCROISSANT) ; A_i = w_i / (σ_i·√(2π)),
    écarts-types σ_i(t_k) pénalisés par λ·(σ_i(t_k) − σ_i(t_{k−1})) (variation lisse).
    Le nombre de gaussiennes est celui de la dernière échéance ; les valeurs initiales viennent du suivi à rebours.
    """
    from scipy.sparse import lil_matrix

    ts = np.array(sorted(hist)); T = len(ts)
    t_ref = ts[-1] if t_ref is None else t_ref
    ids = [g["id"] for g in hist[t_ref][2]]; n = len(ids)
    ref = {g["id"]: g for g in hist[t_ref][2]}
    C0 = np.array([ref[i]["c"] for i in ids])
    A0 = np.zeros((n, T)); S0 = np.zeros((n, T))
    for k, t in enumerate(ts):
        d = {g["id"]: g for g in hist[t][2]}
        for j, i in enumerate(ids):
            A0[j, k] = d[i]["A"] if i in d else 0.0
            S0[j, k] = d[i]["S"] if i in d else ref[i]["S"]
    W0 = np.maximum.accumulate(ouverture_um(A0, S0), axis=1)   # initialisation croissante (µm)
    D0 = np.diff(np.c_[np.zeros(n), W0], axis=1)
    xs = [hist[t][0] for t in ts]; rs = [hist[t][1]["r"].values for t in ts]
    nd = [len(x) for x in xs]; off = np.r_[0, np.cumsum(nd)]
    nres = off[-1] + n * (T - 1)

    def deballer(q):
        D = q[:n * T].reshape(n, T); S = q[n * T:2 * n * T].reshape(n, T); C = q[2 * n * T:]
        return np.cumsum(D, axis=1) * 1e3 / (S * np.sqrt(2 * np.pi)), S, C   # A (µε) à partir de w (µm)

    def f(q):
        A, S, C = deballer(q); out = np.empty(nres)
        for k in range(T):
            out[off[k]:off[k + 1]] = somme_gauss(xs[k], C, A[:, k], S[:, k]) - rs[k]
        out[off[-1]:] = (lam_sigma * np.diff(S, axis=1)).ravel()
        return out

    sp = lil_matrix((nres, 2 * n * T + n), dtype=int)
    for k in range(T):
        rows = slice(off[k], off[k + 1])
        for i in range(n):
            sp[rows, i * T:i * T + k + 1] = 1
            sp[rows, n * T + i * T + k] = 1
            sp[rows, 2 * n * T + i] = 1
    r0 = off[-1]
    for i in range(n):
        for k in range(1, T):
            sp[r0 + i * (T - 1) + k - 1, n * T + i * T + k] = 1
            sp[r0 + i * (T - 1) + k - 1, n * T + i * T + k - 1] = 1
    q0 = np.r_[D0.ravel(), S0.ravel(), C0]
    lo = np.r_[np.zeros(n * T), np.full(n * T, S_MIN), np.maximum(C0 - DECALAGE_MAX, xs[0].min())]
    hi = np.r_[np.full(n * T, 1e3), np.full(n * T, S_MAX), np.minimum(C0 + DECALAGE_MAX, xs[0].max())]
    q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
    sol = least_squares(f, q0, bounds=(lo, hi), jac_sparsity=sp, max_nfev=60,
                        x_scale=np.r_[np.full(n * T, 5.0), np.full(n * T, 3.0), np.full(n, 2.0)])
    A, S, C = deballer(sol.x)
    nouv = {}
    for k, t in enumerate(ts):
        x, q, _, _ = hist[t]
        G = [dict(id=ids[i], c=float(C[i]), A=float(A[i, k]), S=float(S[i, k])) for i in range(n)]
        sse = float(np.sum((somme_gauss(x, C, A[:, k], S[:, k]) - q["r"].values) ** 2))
        nouv[t] = (x, q, G, sse)
        if verbose:
            print(f"{t:6.1f} j : {n:2d} gaussiennes, RMSE={np.sqrt(sse / len(x)):4.0f} µε, "
                  f"Σw={sum(ouverture_um(g['A'], g['S']) for g in G):.0f} µm")
    return nouv


def groupes(tab, ecart_mm=GROUPE_MM):
    """Regroupe a posteriori les gaussiennes voisines (liaison simple sur les centres, écart ≤ ecart_mm).
    Retourne le tableau avec une colonne 'groupe' (numérotée par position croissante)."""
    cen = tab.groupby("id")["pos_mm"].first().sort_values()
    grp = {}; g = 0; prec = None
    for i, c in cen.items():
        if prec is not None and c - prec > ecart_mm:
            g += 1
        grp[i] = g; prec = c
    tab = tab.copy(); tab["groupe"] = tab["id"].map(grp)
    return tab


def tracer(hist, cube, segment, fichier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.array(sorted(hist)); cm = plt.cm.tab20; xf = np.linspace(0, 100, 1000)
    sel = list(dict.fromkeys(ts[np.argmin(abs(ts - t))] for t in np.linspace(ts[0], ts[-1], 5)))
    fig, axs = plt.subplots(2, 3, figsize=(19, 10))
    for ax, t in zip(axs.ravel(), sel):
        x, q, G, _ = hist[t]
        o = np.argsort(x)                       # np.interp exige x croissant (certaines fibres sont parcourues à l'envers)
        mf = np.interp(xf, x[o], q["mod"].values[o])
        ax.plot(x, q["eps"], "k", lw=1, label="mesure")
        ax.plot(x, q["mod"], "b--", lw=2, label="retrait modèle")
        total = np.zeros_like(xf)
        for g in G:
            gi = g["A"] * np.exp(-0.5 * ((xf - g["c"]) / g["S"]) ** 2); total += gi
            ax.fill_between(xf, mf, mf + gi, color=cm(g["id"] % 20), alpha=.45)
        ax.plot(xf, mf + total, "r", lw=1.5, label="retrait + gaussiennes")
        ax.set_title(f"{cube} {segment} — {t:.0f} j : {len(G)} gaussiennes, "
                     f"Σw={sum(ouverture_um(g['A'], g['S']) for g in G):.0f} µm", fontsize=10)
        ax.set_xlabel("position (mm)"); ax.set_ylabel("µε"); ax.grid(alpha=.3)
    axs[0, 0].legend(fontsize=7, loc="lower center")
    ax = axs.ravel()[-1]
    for t in ts:
        for g in hist[t][2]:
            ax.scatter(g["c"], t, s=2 * ouverture_um(g["A"], g["S"]), color=cm(g["id"] % 20), alpha=.6)
    ax.set_xlabel("position (mm)"); ax.set_ylabel("t (j)")
    ax.set_title("Suivi des gaussiennes (taille ∝ w)"); ax.grid(alpha=.3)
    for ax in axs.ravel()[len(sel):-1]:
        ax.axis("off")
    plt.tight_layout()
    if fichier is None:
        return fig
    plt.savefig(fichier, dpi=70); plt.close(fig)


def figure_profil(hist, t, cube, segment):
    """Figure Plotly : mesure, retrait modèle, gaussiennes (appuyées sur le retrait) et somme, à l'échéance t."""
    import plotly.express as px
    import plotly.graph_objects as go
    ts = np.array(sorted(hist)); t = ts[np.argmin(abs(ts - t))]
    x, q, Gs, _ = hist[t]
    o = np.argsort(x)                         # certaines fibres sont parcourues à l'envers
    x, mesure, mod = x[o], q["eps"].values[o], q["mod"].values[o]
    xf = np.linspace(x.min(), x.max(), 1000); mf = np.interp(xf, x, mod); pal = px.colors.qualitative.Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mesure, mode="lines", line=dict(color="black", width=1.5), name="mesure"))
    fig.add_trace(go.Scatter(x=xf, y=mf, mode="lines", line=dict(color="royalblue", dash="dash", width=2.5),
                             name="retrait modèle"))
    total = np.zeros_like(xf)
    for k, g in enumerate(Gs):
        gi = g["A"] * np.exp(-0.5 * ((xf - g["c"]) / g["S"]) ** 2); total += gi
        col = pal[k % len(pal)]
        fig.add_trace(go.Scatter(x=xf, y=mf, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xf, y=mf + gi, mode="lines", line=dict(color=col, width=1), fill="tonexty",
                                 name=f"X={g['c']:.0f} mm : w={ouverture_um(g['A'], g['S']):.0f} µm",
                                 hovertemplate="%{y:.0f} µε"))
    fig.add_trace(go.Scatter(x=xf, y=mf + total, mode="lines", line=dict(color="crimson", width=2),
                             name="retrait + gaussiennes"))
    w = sum(ouverture_um(g["A"], g["S"]) for g in Gs)
    fig.update_layout(title=f"{cube} {segment} — {t:.1f} j : {len(Gs)} gaussiennes, Σw = {w:.0f} µm",
                      xaxis_title="Position le long de la fibre (mm)", yaxis_title="µdef", height=520,
                      legend=dict(font=dict(size=9)))
    return fig


def figure_ouvertures(tab, cube, segment):
    """Figure Plotly : ouverture w_i(t) de chaque gaussienne et somme Σw(t)."""
    import plotly.express as px
    import plotly.graph_objects as go
    pal = px.colors.qualitative.Plotly
    fig = go.Figure()
    for k, (i, g) in enumerate(tab.groupby("id")):
        g = g.sort_values("t_j")
        fig.add_trace(go.Scatter(x=g["t_j"], y=g["w_um"], mode="lines+markers", marker=dict(size=4),
                                 line=dict(color=pal[k % len(pal)]), name=f"X={g['pos_mm'].iloc[0]:.0f} mm"))
    s = tab.groupby("t_j")["w_um"].sum()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color="black", width=3), name="Σw"))
    fig.update_layout(title=f"{cube} {segment} — ouverture de chaque fissure et somme",
                      xaxis_title="Temps écoulé (j)", yaxis_title="Ouverture w (µm)", height=420)
    return fig


def figure_suivi(tab, cube, segment):
    """Figure Plotly : carte position × temps, taille des points ∝ ouverture."""
    import plotly.express as px
    d = tab.copy(); d["w_aff"] = d["w_um"].clip(lower=0.5)
    fig = px.scatter(d, x="pos_mm", y="t_j", size="w_aff", color=d["id"].astype(str), size_max=22,
                     hover_data={"w_um": ":.1f", "sigma_mm": ":.1f", "w_aff": False},
                     labels={"pos_mm": "Position le long de la fibre (mm)", "t_j": "Temps écoulé (j)", "color": "gaussienne"},
                     title=f"{cube} {segment} — suivi des gaussiennes (taille ∝ ouverture)")
    fig.update_layout(height=420, showlegend=False)
    return fig


def figure_surface(tab, cube, segment, n_x=200, n_t_max=60):
    """Surface Plotly : X = temps, Y = position, Z = ouverture linéique des fissures (µm/mm).

    Z(x, t) = Σ_i A_i(t)·exp(-(x-c_i)²/2σ_i²)·1e-3 : son intégrale sur x vaut Σ w_i(t) (µm).
    Évaluation directe sur une grille (n_x × ≤ n_t_max), sans interpolation ni pivotage ; les dates sont sous-échantillonnées
    au-delà de n_t_max (première et dernière conservées)."""
    import plotly.graph_objects as go
    ts = np.sort(tab["t_j"].unique())
    if len(ts) > n_t_max:
        ts = ts[np.unique(np.linspace(0, len(ts) - 1, n_t_max).round().astype(int))]
    xs = np.linspace(max(tab["pos_mm"].min() - 15, 0), min(tab["pos_mm"].max() + 15, 100.0), n_x)
    Z = np.zeros((len(xs), len(ts)))
    for k, t in enumerate(ts):
        d = tab[tab["t_j"] == t]
        Z[:, k] = somme_gauss(xs, d["pos_mm"].values, d["A_microdef"].values, d["sigma_mm"].values) * 1e-3
    fig = go.Figure(go.Surface(x=ts, y=xs, z=Z, colorscale="Inferno", colorbar=dict(title="µm/mm"),
                               hovertemplate="t=%{x:.1f} j<br>X=%{y:.1f} mm<br>w'=%{z:.2f} µm/mm<extra></extra>"))
    fig.update_layout(title=f"{cube} {segment} — ouverture linéique des fissures", height=560,
                      scene=dict(xaxis_title="Temps écoulé (j)", yaxis_title="Position fibre (mm)",
                                 zaxis_title="Ouverture (µm/mm)"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", choices=["C12", "C34"], default="C12")
    ap.add_argument("--segment", default="s14")
    ap.add_argument("--t-calage", type=float, default=None, help="échéance de calage des gaussiennes (défaut : dernière)")
    ap.add_argument("--t0", type=float, default=T0_DEFAUT, help="première échéance analysée (jours)")
    ap.add_argument("--theta", type=float, nargs=4, metavar=("N", "k", "a", "n"),
                    help="paramètres du retrait (défaut : fit C12, D_e = 3,7e-7 m2/s)")
    ap.add_argument("--de", type=float, default=F.DE_DEFAUT, help="D_e utilisé avec --theta (m2/s)")
    ap.add_argument("--refit", action="store_true", help="ré-ajuste le retrait sur le cube (≈ 10 s)")
    ap.add_argument("--avant", action="store_true", help="ancien ordre chronologique (centres déterminés à t0)")
    ap.add_argument("--fusion", action="store_true", help="autorise la fusion (déconseillé)")
    ap.add_argument("--sans-regul", action="store_true", help="pas d'ajustement global régularisé (option D)")
    ap.add_argument("--plot", help="figure PNG")
    ap.add_argument("--csv", help="tableau CSV (t, id, position, A, σ, w)")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    theta = np.array(args.theta if args.theta else THETA_DEFAUT)
    if args.refit:
        theta = F.ajuster(F.prepare(args.cube), args.de)
    print(f"retrait : N={theta[0]:.0f} k={theta[1]:.4f} a={theta[2]:.0f} n={theta[3]:.2f} (D_e={args.de:.1e})")
    h = donnees_segment(args.cube, args.segment, theta, args.de)
    tab, hist = analyser(h, args.t0, arriere=not args.avant, fusion=args.fusion, regul=not args.sans_regul,
                       t_calage=args.t_calage)
    tab = groupes(tab)
    print('\nÉtat final (groupes de gaussiennes voisines) :')
    fin = tab[tab.t_j == tab.t_j.max()]
    for gr, d in fin.groupby('groupe'):
        print(f'  groupe {gr}: X={d.pos_mm.min():.0f}-{d.pos_mm.max():.0f} mm, {len(d)} gaussiennes, Σw={d.w_um.sum():.0f} µm')
    if args.csv:
        tab.to_csv(args.csv, index=False); print("CSV :", args.csv)
    if args.plot:
        tracer(hist, args.cube, args.segment, args.plot); print("figure :", args.plot)
