"""
Fit pseudo-physique du retrait de carbonatation (matrice) en tout point du cube.

Modèle (réaction-diffusion couplée)
-----------------------------------
- Le CO2 diffuse depuis les 6 faces avec un coefficient effectif D_e FIXÉ (littérature,
  Auroy et al., Kangni-Foli et al. : 2e-7 à 7e-7 m2/s) et est consommé par la réaction.
  La diffusion du gaz étant rapide devant la réaction, le CO2 est traité en quasi-stationnaire :
      D_e * c0 * u'' = N * k * u * (1 - alpha)       u = c / c0, u(face) = 1
- Degré de carbonatation local (cinétique du 1er ordre) :
      d(alpha)/dt = k * u * (1 - alpha)
- Retrait de la matrice (µε) :  eps = -a * alpha**n
- Résolution 1D dans une demi-dalle (symétrie), puis combinaison 3D approchée :
      alpha_3D = 1 - prod_i (1 - alpha_1D(distance à la face i))

Endommagement : D_eff = D_e * (1 + beta * max(0, alpha - alpha_crit)**m) (m fixé).

Couplage F(w) piloté par les données (alternative, mode « couplage ») : la diffusion dépend de
l'ouverture cumulée des fissures mesurée, et non plus de alpha :
      D_eff(t) = D_e * (1 + c * (w(t) / W_REF)**p)
w(t) (µm) = somme des ouvertures par fibre, estimée par la mesure (intégrale de l'écart mesure - retrait
sur la fibre), moyenne des fibres, rendue croissante (fissures irréversibles). Le dommage en alpha est
alors désactivé (beta = 0) ; theta = (N, k, a, n, c, p). Comme w dépend du retrait ajusté, on itère
(ajuster_couplage : retrait -> w -> ajustement -> w ...).

Paramètres ajustés : N (mol CO2 / m3, capacité de fixation), k (1/jour), a (µε), n,
beta, alpha_crit (ou c, p en mode couplage).
Seul le rapport N / D_e est identifiable : N est proportionnel au D_e choisi.
Si N est mesuré (ATG), on en déduit D_e = N * (D_e_ref / N_fit) avec les valeurs affichées.

Ajustement
----------
La déformation mesurée = retrait de la matrice + ouverture de fissures (>= 0). On ajuste
donc l'enveloppe basse : les résidus où la mesure est AU-DESSUS du modèle (fissures) sont
pondérés par W_FISSURE < 1. Les points hors du cube (bout de fibre) sont ignorés.
Les segments C12 s6 à s12 sont exclus (voir README, discussion des données à négliger).

Usage :  python fit_retrait.py [--cube C12|C34] [--de 3.7e-7] [--plot fit.png]
Par défaut : ajuste C12, valide sur C34 avec les paramètres de C12.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.optimize import least_squares

from data_prep import CUBE_SIDE_M as L, load_and_prepare

CSV = {"C12": "data/profils_3d_C12.csv", "C34": "data/profils_3d_C34.csv"}
EXCLUS = {"C12": {f"s{i}" for i in range(6, 13)}, "C34": set()}
W_FISSURE = 0.1   # poids des résidus où la mesure est au-dessus du modèle
N_DATES = 40      # nombre de dates retenues (espacées régulièrement)
BIN_MM = 4        # moyenne (médiane) des points le long de la fibre par classes de BIN_MM
DE_DEFAUT = 3.7e-7   # m2/s, moyenne géométrique de 2e-7 et 7e-7
C0 = 1.25            # mol/m3 : CO2 à 3 % dans l'air (20 °C, 1 atm)
DX = 0.001           # m, pas du maillage 1D
DT = 0.25            # jours, pas de temps
TMAX = 114.0         # jours
NX = int(L / 2 / DX) + 1
NSTEP = int(TMAX / DT)
M_DOMMAGE = 2.0      # exposant de sensibilité m du dommage (fixé)
PARAM_NOMS = ("N", "k", "a", "n", "beta", "alpha_crit")
PARAM_NOMS_FW = ("N", "k", "a", "n", "c", "p")   # mode couplage F(w) : D_eff = D_e (1 + c (w/W_REF)^p)
W_REF = 100.0        # µm, ouverture de référence de F(w)
# N, k, beta, alpha_crit sont optimisés en espace logarithmique
BORNES = ([np.log(5000), np.log(1e-3), 0.0, 0.3, np.log(1e-3), np.log(0.05)],
          [np.log(25000), np.log(5.0), 3e4, 4.0, np.log(1e3), np.log(0.99)])
BORNES_FW = ([np.log(5000), np.log(1e-3), 0.0, 0.3, np.log(1e-3), 0.3],
             [np.log(25000), np.log(5.0), 3e4, 4.0, np.log(1e3), 4.0])
DEPARTS = ((5000, 0.05), (20000, 0.2), (8000, 0.02))  # (N0, k0)


def prepare(cube: str, n_dates: int = N_DATES, bin_mm: float = BIN_MM) -> pd.DataFrame:
    """Données allégées : segments exclus retirés, dates sous-échantillonnées,
    médiane par classe de position le long de la fibre, points hors cube supprimés."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = load_and_prepare(CSV[cube])
    d = d[~d["segment"].isin(EXCLUS[cube])]
    dates = np.sort(d["temps_ecoule_j"].dropna().unique())
    idx = np.unique(np.linspace(0, len(dates) - 1, n_dates).round().astype(int))
    d = d[d["temps_ecoule_j"].isin(dates[idx])].copy()
    d["xb"] = (d["x_fibre_m"] * 1000 // bin_mm).astype(int)
    g = d.groupby(["segment", "fibre", "axe_mesure", "temps_ecoule_j", "xb"], as_index=False).agg(
        X=("X_m", "mean"), Y=("Y_m", "mean"), Z=("Z_m", "mean"),
        eps=("valeur_microdefs", "median"))
    m = 0.0005
    dans = (g[["X", "Y", "Z"]] >= -m).all(axis=1) & (g[["X", "Y", "Z"]] <= L + m).all(axis=1)
    return g[dans & g["eps"].notna()].reset_index(drop=True)


def alpha_1d(N: float, k: float, De: float, beta: float, alpha_crit: float,
             m: float = M_DOMMAGE, mult_t: np.ndarray | None = None) -> np.ndarray:
    """alpha(t, x) sur une demi-dalle [0, L/2], tableau (NSTEP+1, NX).

    D_eff(x, t) = De * (1 + beta * max(0, alpha - alpha_crit)**m) : la fissuration
    (liée à la carbonatation locale) accélère la diffusion du CO2.
    mult_t (NSTEP,) : facteur (1 + F(w(t))) du couplage piloté par les données, appliqué à tout x."""
    al = np.zeros(NX)
    out = np.zeros((NSTEP + 1, NX))
    ab = np.zeros((3, NX))
    h = 1 / DX ** 2
    rhs = np.zeros(NX); rhs[0] = 1.0
    for s in range(NSTEP):
        d_eff = De * (1 + beta * np.maximum(0.0, al - alpha_crit) ** m)   # vecteur, taille NX
        if mult_t is not None:
            d_eff = d_eff * mult_t[s]
        lam = N * (k / 86400.0) * (1 - al) / (d_eff * C0)
        ab[1] = 2 * h + lam
        ab[0, 1:] = -h
        ab[2, :-1] = -h
        ab[1, 0] = 1.0; ab[0, 1] = 0.0   # u(face) = 1
        ab[2, -2] = -2 * h               # symétrie au centre
        u = np.clip(solve_banded((1, 1), ab, rhs), 0, 1)
        al = 1 - (1 - al) * np.exp(-k * u * DT)
        out[s + 1] = al
    return out


def alpha_3d(A, X, Y, Z, t_jours):
    """Combinaison 3D de alpha_1D, interpolation bilinéaire en (t, distance à la face)."""
    it = np.clip(t_jours / DT, 0, NSTEP - 1e-9)
    i0 = it.astype(int); ft = it - i0
    reste = 1.0
    for p in (X, Y, Z):
        p = np.clip(p, 0, L)
        ix = np.clip(np.minimum(p, L - p) / DX, 0, NX - 1 - 1e-9)
        j0 = ix.astype(int); fx = ix - j0
        a0 = A[i0, j0] * (1 - fx) + A[i0, j0 + 1] * fx
        a1 = A[i0 + 1, j0] * (1 - fx) + A[i0 + 1, j0 + 1] * fx
        reste = reste * (1 - (a0 * (1 - ft) + a1 * ft))
    return 1 - reste


def facteur_fw(couplage, c: float, p: float) -> np.ndarray:
    """1 + F(w(t)) sur la grille de temps du modèle ; couplage = (dates, w en µm), w croissant."""
    dates, w = couplage
    w_t = np.interp(np.arange(NSTEP) * DT, np.asarray(dates, float), np.asarray(w, float))
    return 1 + c * (w_t / W_REF) ** p


def modele(theta, g: pd.DataFrame, De: float = DE_DEFAUT, couplage=None) -> np.ndarray:
    """Retrait de la matrice (µε) aux points de g.

    Sans couplage : theta = (N, k, a, n, beta, alpha_crit).
    Avec couplage = (dates, w) : theta = (N, k, a, n, c, p), D_eff = D_e (1 + c (w/W_REF)^p)."""
    if couplage is None:
        N, k, a, n, beta, alpha_crit = (list(theta) + [0.0, 0.3][len(theta) - 4:])[:6]   # 4 valeurs : sans dommage
        A = alpha_1d(N, k, De, beta, alpha_crit)
    else:
        N, k, a, n, c, p = theta
        A = alpha_1d(N, k, De, 0.0, 0.3, mult_t=facteur_fw(couplage, c, p))
    alpha = alpha_3d(A, g["X"].values, g["Y"].values, g["Z"].values, g["temps_ecoule_j"].values)
    return -a * alpha ** n


def ouverture_donnees(g: pd.DataFrame, theta, De: float = DE_DEFAUT, couplage=None):
    """w(t) estimée à partir des mesures : par fibre et par date, intégrale de (mesure - retrait) le long
    de la fibre (= somme des ouvertures), puis moyenne des fibres. Retourne (dates, w en µm) avec w >= 0
    croissant (fissures irréversibles) et w(0) = 0."""
    h = g.assign(res=g["eps"].values - modele(theta, g, De, couplage))
    lignes = []
    for (seg, t), q in h.groupby(["segment", "temps_ecoule_j"]):
        axe = q["axe_mesure"].iloc[0]
        longueur = float(q[axe].max() - q[axe].min()) * 1000       # mm
        lignes.append((t, max(float(q["res"].mean()), 0.0) * longueur * 1e-3))   # µε x mm -> µm
    w = pd.DataFrame(lignes, columns=["t", "w"]).groupby("t")["w"].mean().sort_index()
    return (np.concatenate([[0.0], w.index.values]),
            np.concatenate([[0.0], np.maximum.accumulate(w.values)]))


def ajuster(g: pd.DataFrame, De: float = DE_DEFAUT, w: float = W_FISSURE, couplage=None) -> np.ndarray:
    """Ajustement de l'enveloppe basse (perte asymétrique), multi-départs sur (N, k).
    Avec couplage = (dates, w) : les deux derniers paramètres sont (c, p) au lieu de (beta, alpha_crit)."""
    eps = g["eps"].values
    etat = {"n": 0, "t0": time.time(), "depart": 0}
    fw = couplage is not None
    bornes, fin0 = (BORNES_FW, [np.log(1.0), 1.0]) if fw else (BORNES, [np.log(1.0), np.log(0.3)])

    def vers_theta(q):
        return [np.exp(q[0]), np.exp(q[1]), q[2], q[3], np.exp(q[4]), q[5] if fw else np.exp(q[5])]

    def residus(q):
        etat["n"] += 1
        ecoule = time.time() - etat["t0"]
        sys.stderr.write(f"\r  départ {etat['depart']}/{len(DEPARTS)} | évaluations : "
                         f"{etat['n']} | écoulé : {ecoule:.0f} s")
        sys.stderr.flush()
        r = modele(vers_theta(q), g, De, couplage) - eps
        return np.where(r < 0, np.sqrt(w) * r, r)   # r < 0 : mesure au-dessus du modèle

    meilleur = None
    for i, (N0, k0) in enumerate(DEPARTS, 1):
        etat["depart"] = i
        r = least_squares(residus, [np.log(N0), np.log(k0), 6000.0, 1.0] + fin0,
                          bounds=bornes, x_scale=[1, 1, 1000, 1, 1, 1], diff_step=1e-3)
        if meilleur is None or r.cost < meilleur.cost:
            meilleur = r
    sys.stderr.write("\n")
    return np.array(vers_theta(meilleur.x))


def ajuster_couplage(g: pd.DataFrame, De: float = DE_DEFAUT, theta_ref=None, iterations: int = 2,
                     w: float = W_FISSURE):
    """Couplage F(w) piloté par les données. Point fixe : w(t) mesuré avec le retrait courant, ajustement
    de (N, k, a, n, c, p), puis w(t) recalculé avec ce nouveau retrait. theta_ref (6 valeurs, dommage en alpha)
    initialise w ; s'il est absent, un ajustement standard est fait d'abord.
    Retourne (theta = (N, k, a, n, c, p), couplage = (dates, w en µm))."""
    ref = np.asarray(ajuster(g, De, w) if theta_ref is None else theta_ref, float)
    couplage, theta = None, ref
    for _ in range(max(iterations, 1)):
        couplage = ouverture_donnees(g, theta, De, couplage)
        theta = ajuster(g, De, w, couplage)
    return theta, couplage


def bilan(theta, g: pd.DataFrame, De: float, label: str, couplage=None) -> None:
    r = modele(theta, g, De, couplage) - g["eps"].values
    sous = r[r >= 0]
    print(f"{label}: n={len(g)} | part mesure>modèle={np.mean(r < 0):.2f} | "
          f"RMSE (mesure<modèle)={np.sqrt(np.mean(sous ** 2)):.0f} µε")


def tracer(theta, De: float, cube: str, segments, fichier: str, couplage=None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = prepare(cube, bin_mm=2)
    N, k, a, n, beta, alpha_crit = theta   # (c, p) en mode couplage
    txt = (f"Modèle couplé (fit global C12)\nD_e = {De:.1e} m²/s (fixé)\nN = {N:.0f} mol/m³\n"
           f"k = {k:.3f} j⁻¹\na = {a:.0f} µε\nn = {n:.2f}\n"
           + (f"c = {beta:.3g}\np = {alpha_crit:.2f}\nF(w) = c (w / {W_REF:g} µm)^p" if couplage is not None else
              f"β = {beta:.3g}\nα_crit = {alpha_crit:.2f}\nm = {M_DOMMAGE:g} (fixé)"))
    couleurs = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    segments = [s for s in segments if s in set(g["segment"])]
    ncol = min(4, len(segments)); nrow = int(np.ceil(len(segments) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.3 * nrow), squeeze=False)
    for s, ax in zip(segments, axs.ravel()):
        h = g[g["segment"] == s]
        ts = np.sort(h["temps_ecoule_j"].unique()); m = len(ts); axe = h["axe_mesure"].iloc[0]
        for i, c in zip([int(m * .25), int(m * .5), int(m * .75), m - 1], couleurs):
            q = h[h["temps_ecoule_j"] == ts[i]].sort_values("xb")
            ax.plot(q[axe] * 1000, q["eps"], color=c, lw=.8, alpha=.7, label=f"{ts[i]:.0f} j")
            ax.plot(q[axe] * 1000, modele(theta, q, De, couplage), color=c, lw=2.2, ls="--")
        ax.set_title(f"{cube} {s} — axe {axe}", fontsize=10)
        ax.set_xlabel(f"{axe} (mm)"); ax.set_ylabel("µε"); ax.grid(alpha=.3); ax.set_xlim(0, 100)
        ax.text(.5, .02, txt, transform=ax.transAxes, fontsize=6.5, ha="center", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=.85))
    axs[0, 0].legend(fontsize=7, ncol=2, loc="upper center")
    for ax in axs.ravel()[len(segments):]:
        ax.axis("off")
    fig.suptitle("Traits fins : mesure — tirets : modèle couplé (enveloppe basse)")
    plt.tight_layout(); plt.savefig(fichier, dpi=100); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", choices=["C12", "C34"], default="C12", help="cube ajusté")
    ap.add_argument("--de", type=float, default=DE_DEFAUT, help="D_e fixé (m2/s)")
    ap.add_argument("--couplage", action="store_true",
                    help="couplage F(w) piloté par les données : D_eff = D_e (1 + c (w/100 µm)^p) au lieu du dommage en alpha")
    ap.add_argument("--plot", help="fichier PNG des profils (segments d'intérêt)")
    args = ap.parse_args()

    g = prepare(args.cube)
    couplage = None
    if args.couplage:
        theta, couplage = ajuster_couplage(g, args.de)
    else:
        theta = ajuster(g, args.de)
    print(f"{args.cube} (D_e = {args.de:.1e} m2/s) : "
          + ", ".join(f"{nm}={v:.3g}" for nm, v in zip(PARAM_NOMS_FW if args.couplage else PARAM_NOMS, theta)))
    print(f"  rapport N/D_e = {theta[0] / args.de:.3g} (mol/m3)/(m2/s) : identifiable, N dépend de D_e")
    bilan(theta, g, args.de, f"  ajustement {args.cube}", couplage)
    autre = "C34" if args.cube == "C12" else "C12"
    g2 = prepare(autre)
    # validation : mêmes paramètres, w(t) propre à l'autre cube (estimé sur ses mesures)
    bilan(theta, g2, args.de, f"  validation {autre} (mêmes paramètres)",
          None if couplage is None else ouverture_donnees(g2, theta, args.de, couplage))
    if args.plot:
        tracer(theta, args.de, args.cube, ["s14", "s17", "s19", "s21", "s22", "s4", "s3", "s1"], args.plot, couplage)
        print("figure :", args.plot)
