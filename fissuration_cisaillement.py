"""
Quantification de la fissuration par transfert de déformation exponentiel (shear lag) sur les données CORBO.

Adaptation de fissuration_shear_lag.py (qui ne tourne que sur des données synthétiques) :
  - données réelles : résidu r(x, t) = ε_mesuré − retrait (fit_retrait.py), comme fissuration_gaussiennes.py ;
  - noyau d'une fissure d'ouverture w au centre c :  ε(x) = (w·λ/2)·exp(−λ|x−c|)   (aire = w),
    MOYENNÉ sur la jauge de l'OBR (5 mm) de façon analytique ;
  - ajustement par moindres carrés (least_squares, jacobien creux), w_i(t) non décroissant par construction,
    λ_i(t) lissé dans le temps, centres communs à toutes les dates ;
  - centres déterminés sur la DERNIÈRE échéance (ajout glouton), puis suivi à rebours.

Unités : x en mm, λ en 1/mm (affiché en m⁻¹), w en µm, ε en µε.

Usage :
    python fissuration_cisaillement.py --cube C12 --segment s14 --plot s14_sl.png --csv s14_sl.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import fit_retrait as F
from fissuration_gaussiennes import donnees_segment, THETA_DEFAUT, T0_DEFAUT

JAUGE_MM = 5.0
LAM_MIN, LAM_MAX = 0.01, 0.5     # 1/mm  (10–500 m⁻¹ : longueur de transfert 2–100 mm)
LAM_INIT = 0.1
LAMBDA_LISSAGE = 400.0           # pénalité sur la variation de λ entre échéances (en µε par 1/mm)
ADD_GAIN = 0.04                  # gain relatif minimal du SSE pour ajouter une fissure
W_MIN = 5.0                      # µm : ouverture minimale d'une fissure ajoutée
DIST_MIN = 2.0                   # mm
DECALAGE_MAX = 4.0               # mm
N_MAX = 25


def _G(s, lam):
    return np.sign(s) * (1.0 - np.exp(-lam * np.abs(s))) / lam


def noyau(x, c, w, lam, jauge=JAUGE_MM):
    """Déformation (µε) vue par la fibre pour une fissure (c, w, λ), moyennée sur la jauge."""
    s = x - c
    if jauge <= 0:
        return 500.0 * w * lam * np.exp(-lam * np.abs(s))
    return 500.0 * w * lam * (_G(s + jauge / 2, lam) - _G(s - jauge / 2, lam)) / jauge


def modele(x, C, W, L):
    y = np.zeros_like(x, dtype=float)
    for c, w, l in zip(C, W, L):
        y += noyau(x, c, w, l)
    return y


def bruit_mad(r):
    """Bruit court terme : MAD des différences secondes (insensible aux grandes structures)."""
    d2 = r[2:] - 2 * r[1:-1] + r[:-2]
    return 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)


def ajuster_date(x, cible, C, W, L, libres):
    n = len(C); C0 = np.asarray(C, float); fi = np.where(libres)[0]

    def f(q):
        c = C0.copy(); c[fi] = q[2 * n:]
        return modele(x, c, q[:n], q[n:2 * n]) - cible

    q0 = np.r_[W, L, C0[fi]]
    lo = np.r_[np.zeros(n), np.full(n, LAM_MIN), np.maximum(C0[fi] - DECALAGE_MAX, x.min())]
    hi = np.r_[np.full(n, 5e3), np.full(n, LAM_MAX), np.minimum(C0[fi] + DECALAGE_MAX, x.max())]
    q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
    s = least_squares(f, q0, bounds=(lo, hi), max_nfev=80,
                      x_scale=np.r_[np.full(n, 20.0), np.full(n, 0.05), np.full(len(fi), 2.0)])
    c = C0.copy(); c[fi] = s.x[2 * n:]
    return c, s.x[:n], s.x[n:2 * n], float(np.sum(s.fun ** 2))


def detecter(x, r, positif=False, verbose=False):
    """Ajout glouton de fissures sur la dernière échéance (centres libres)."""
    cible = np.maximum(r, 0) if positif else r
    C = np.zeros(0); W = np.zeros(0); L = np.zeros(0); sse = float(np.sum(cible ** 2))
    while len(C) < N_MAX:
        res = cible - modele(x, C, W, L); i = int(np.argmax(res))
        if res[i] <= 0:
            break
        meilleur = None
        for lam0 in (0.05, 0.15, 0.4):                   # multi-départ : la nouvelle fissure seule, les autres figées
            w0 = 2 * res[i] / (500.0 * lam0)
            cn, wn, ln, _ = ajuster_date(x, res, [x[i]], [w0], [lam0], [True])
            if meilleur is None or _ < meilleur[3]:
                meilleur = (cn[0], wn[0], ln[0], _)
        C2 = np.r_[C, meilleur[0]]; W2 = np.r_[W, meilleur[1]]; L2 = np.r_[L, meilleur[2]]
        c, w, l, sse2 = ajuster_date(x, cible, C2, W2, L2, np.ones(len(C2), bool))
        proche = len(c) > 1 and np.min(np.diff(np.sort(c))) < DIST_MIN
        if verbose:
            print(f"    essai X={x[i]:.0f}: gain {1 - sse2 / sse:.3f}, w_min {w.min():.0f}, proche={proche}")
        if sse2 > sse * (1 - ADD_GAIN) or w.min() < W_MIN or proche:
            break
        C, W, L, sse = c, w, l, sse2
        if verbose:
            print(f"  + fissure en X={x[i]:.1f} mm  (SSE {sse:.3g})")
    return C, W, L


def global_regularise(xs, rs, C0, W0, L0, lam_lissage=LAMBDA_LISSAGE, positif=False):
    """Ajustement global : w_i(t) = somme cumulée d'incréments ≥ 0 (non décroissant), λ_i(t) lissé."""
    n, T = W0.shape
    D0 = np.diff(np.c_[np.zeros(n), np.maximum.accumulate(W0, axis=1)], axis=1)
    nd = [len(x) for x in xs]; off = np.r_[0, np.cumsum(nd)]
    cibles = [np.maximum(r, 0) if positif else r for r in rs]
    sq = np.sqrt(lam_lissage)

    def depaqueter(q):
        D = q[:n * T].reshape(n, T); L = q[n * T:2 * n * T].reshape(n, T); C = q[2 * n * T:]
        return np.cumsum(D, axis=1), L, C

    def f(q):
        W, L, C = depaqueter(q)
        res = [modele(xs[k], C, W[:, k], L[:, k]) - cibles[k] for k in range(T)]
        pen = (np.diff(L, axis=1) * sq).ravel()
        return np.r_[np.concatenate(res), pen]

    sp = lil_matrix((off[-1] + n * (T - 1), n * T + n * T + n), dtype=int)
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
    q0 = np.r_[D0.ravel(), L0.ravel(), C0]
    lo = np.r_[np.zeros(n * T), np.full(n * T, LAM_MIN), np.maximum(C0 - DECALAGE_MAX, xs[0].min())]
    hi = np.r_[np.full(n * T, 5e3), np.full(n * T, LAM_MAX), np.minimum(C0 + DECALAGE_MAX, xs[0].max())]
    q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
    s = least_squares(f, q0, bounds=(lo, hi), jac_sparsity=sp, max_nfev=60,
                      x_scale=np.r_[np.full(n * T, 5.0), np.full(n * T, 0.05), np.full(n, 2.0)])
    return depaqueter(s.x)


def analyser(h, t0=T0_DEFAUT, positif=False, verbose=True):
    axe = h["axe_mesure"].iloc[0]
    echeances = np.sort(h["temps_ecoule_j"].unique()); echeances = echeances[echeances >= t0]
    xs, rs = [], []
    for t in echeances:
        q = h[h["temps_ecoule_j"] == t].sort_values("xb")
        xs.append(q[axe].values * 1000); rs.append(q["r"].values)
    if verbose:
        print(f"Bruit court terme (MAD, dernière échéance) : {bruit_mad(rs[-1]):.0f} µε")
        print(f"Détection sur {echeances[-1]:.0f} j :")
    C, W, L = detecter(xs[-1], rs[-1], positif, verbose)
    if len(C) == 0:
        raise SystemExit("aucune fissure détectée")
    n, T = len(C), len(echeances)
    W0 = np.zeros((n, T)); L0 = np.tile(L[:, None], (1, T)); W0[:, -1] = W
    for k in range(T - 2, -1, -1):                       # suivi à rebours, centres figés
        cib = np.maximum(rs[k], 0) if positif else rs[k]
        c, w, l, _ = ajuster_date(xs[k], cib, C, W0[:, k + 1], L0[:, k + 1], np.zeros(n, bool))
        W0[:, k] = np.minimum(w, W0[:, k + 1]); L0[:, k] = l
    Wf, Lf, Cf = global_regularise(xs, rs, C, W0, L0, positif=positif)
    lignes = []
    hist = {}
    for k, t in enumerate(echeances):
        fit = modele(xs[k], Cf, Wf[:, k], Lf[:, k])
        rmse = float(np.sqrt(np.mean((fit - rs[k]) ** 2)))
        hist[t] = (xs[k], rs[k], fit)
        for i in range(n):
            lignes.append(dict(t_j=t, id=i, pos_mm=Cf[i], w_um=Wf[i, k], lambda_m1=1000 * Lf[i, k],
                               longueur_transfert_mm=1 / Lf[i, k]))
        if verbose:
            print(f"{t:6.1f} j : {n} fissures, RMSE={rmse:4.0f} µε, Σw={Wf[:, k].sum():.0f} µm")
    return pd.DataFrame(lignes), hist, Cf


def tracer(cube, seg, h, df, hist, chemin):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = sorted(hist); idx = np.unique(np.linspace(0, len(ts) - 1, 5).astype(int)); sel = [ts[i] for i in idx]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9)); ax = ax.ravel()
    for a, t in zip(ax, sel):
        x, r, fit = hist[t]
        a.plot(x, r, "k", lw=1, label="mesure − retrait"); a.plot(x, fit, "r", label="shear lag")
        a.axhline(0, color="gray", lw=.5); a.set_title(f"{cube} {seg} — {t:.0f} j"); a.set_xlabel("position (mm)"); a.set_ylabel("µε")
    ax[0].legend(fontsize=7)
    a = ax[5]
    for i, g in df.groupby("id"):
        a.plot(g["t_j"], g["w_um"], label=f"X={g['pos_mm'].iloc[0]:.0f}")
    a.plot(df.groupby("t_j")["w_um"].sum(), "k", lw=2, label="Σw")
    a.set_xlabel("t (j)"); a.set_ylabel("w (µm)"); a.legend(fontsize=6, ncol=2); a.set_title("ouvertures")
    fig.tight_layout(); fig.savefig(chemin, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", choices=["C12", "C34"], default="C12")
    ap.add_argument("--segment", default="s14")
    ap.add_argument("--t0", type=float, default=T0_DEFAUT)
    ap.add_argument("--theta", type=float, nargs=4, metavar=("N", "k", "a", "n"), default=THETA_DEFAUT)
    ap.add_argument("--de", type=float, default=F.DE_DEFAUT)
    ap.add_argument("--positif", action="store_true", help="n'ajuste que la part du résidu au-dessus du retrait")
    ap.add_argument("--plot"); ap.add_argument("--csv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    h = donnees_segment(a.cube, a.segment, a.theta, a.de)
    df, hist, C = analyser(h, a.t0, a.positif)
    if a.csv:
        df.to_csv(a.csv, index=False)
    if a.plot:
        tracer(a.cube, a.segment, h, df, hist, a.plot)


if __name__ == "__main__":
    main()
