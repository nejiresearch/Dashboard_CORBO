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

Paramètres ajustés : N (mol CO2 / m3, capacité de fixation), k (1/jour), a (µε), n.
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
PARAM_NOMS = ("N", "k", "a", "n")
BORNES = ([np.log(100), np.log(1e-3), 0.0, 0.3], [np.log(5e5), np.log(5.0), 3e4, 4.0])
DEPARTS = ((5000, 0.05), (20000, 0.2), (2000, 0.02))  # (N0, k0)


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


def alpha_1d(N, k, De):
    """alpha(t, x) sur une demi-dalle [0, L/2], tableau (NSTEP+1, NX)."""
    al = np.zeros(NX)
    out = np.zeros((NSTEP + 1, NX))
    ab = np.zeros((3, NX))
    h = 1 / DX ** 2
    rhs = np.zeros(NX); rhs[0] = 1.0
    for s in range(NSTEP):
        lam = N * (k / 86400.0) * (1 - al) / (De * C0)
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


def modele(theta, g: pd.DataFrame, De: float = DE_DEFAUT) -> np.ndarray:
    """Retrait de la matrice (µε) aux points de g, theta = (N, k, a, n)."""
    N, k, a, n = theta
    A = alpha_1d(N, k, De)
    alpha = alpha_3d(A, g["X"].values, g["Y"].values, g["Z"].values, g["temps_ecoule_j"].values)
    return -a * alpha ** n


def ajuster(g: pd.DataFrame, De: float = DE_DEFAUT, w: float = W_FISSURE) -> np.ndarray:
    """Ajustement de l'enveloppe basse (perte asymétrique), multi-départs sur (N, k)."""
    eps = g["eps"].values

    def residus(q):
        r = modele([np.exp(q[0]), np.exp(q[1]), q[2], q[3]], g, De) - eps
        return np.where(r < 0, np.sqrt(w) * r, r)   # r < 0 : mesure au-dessus du modèle

    meilleur = None
    for N0, k0 in DEPARTS:
        r = least_squares(residus, [np.log(N0), np.log(k0), 6000.0, 1.0], bounds=BORNES,
                          x_scale=[1, 1, 1000, 1], diff_step=1e-3)
        if meilleur is None or r.cost < meilleur.cost:
            meilleur = r
    q = meilleur.x
    return np.array([np.exp(q[0]), np.exp(q[1]), q[2], q[3]])


def bilan(theta, g: pd.DataFrame, De: float, label: str) -> None:
    r = modele(theta, g, De) - g["eps"].values
    sous = r[r >= 0]
    print(f"{label}: n={len(g)} | part mesure>modèle={np.mean(r < 0):.2f} | "
          f"RMSE (mesure<modèle)={np.sqrt(np.mean(sous ** 2)):.0f} µε")


def tracer(theta, De: float, cube: str, segments, fichier: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = prepare(cube, bin_mm=2)
    N, k, a, n = theta
    txt = (f"Modèle couplé (fit global C12)\nD_e = {De:.1e} m²/s (fixé)\nN = {N:.0f} mol/m³\n"
           f"k = {k:.3f} j⁻¹\na = {a:.0f} µε\nn = {n:.2f}")
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
            ax.plot(q[axe] * 1000, modele(theta, q, De), color=c, lw=2.2, ls="--")
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
    ap.add_argument("--plot", help="fichier PNG des profils (segments d'intérêt)")
    args = ap.parse_args()

    g = prepare(args.cube)
    theta = ajuster(g, args.de)
    print(f"{args.cube} (D_e = {args.de:.1e} m2/s) : "
          + ", ".join(f"{nm}={v:.3g}" for nm, v in zip(PARAM_NOMS, theta)))
    print(f"  rapport N/D_e = {theta[0] / args.de:.3g} (mol/m3)/(m2/s) : identifiable, N dépend de D_e")
    bilan(theta, g, args.de, f"  ajustement {args.cube}")
    autre = "C34" if args.cube == "C12" else "C12"
    bilan(theta, prepare(autre), args.de, f"  validation {autre} (mêmes paramètres)")
    if args.plot:
        tracer(theta, args.de, args.cube, ["s14", "s17", "s19", "s21", "s22", "s4", "s3", "s1"], args.plot)
        print("figure :", args.plot)
