"""
Fit pseudo-physique du retrait de carbonatation (matrice) en tout point du cube.

Modèle
------
- Diffusion du CO2 depuis les 6 faces : solution séparable 3D, concentration
  normalisée c(x, y, z, t) = 1 - prod_i [1 - erfc(p_i / 2sqrt(Dt)) - erfc((L - p_i) / 2sqrt(Dt))]
- Cinétique de réaction du 1er ordre : dalpha/dt = k * c * (1 - alpha)
  -> alpha = 1 - exp(-k * I),  I = integrale de c dt (jours)
- Retrait de la matrice : eps = -a * alpha**n   (microdéformations)

Ajustement
----------
La déformation mesurée = retrait de la matrice + ouverture de fissures (>= 0).
On ajuste donc l'enveloppe basse : les résidus où la mesure est AU-DESSUS du
modèle (fissures) sont pondérés par W_FISSURE < 1.
Les points hors du cube (bout de fibre) sont ignorés.
Les segments C12 s6 à s12 sont exclus (voir README, discussion des données à négliger).

Paramètres : D (m2/s), a (µε), n, k (1/jour).

Usage :  python fit_retrait.py [--cube C12|C34] [--plot fit.png]
Par défaut : ajuste C12, valide sur C34 avec les paramètres de C12.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import erfc

from data_prep import CUBE_SIDE_M as L, load_and_prepare

CSV = {"C12": "data/profils_3d_C12.csv", "C34": "data/profils_3d_C34.csv"}
EXCLUS = {"C12": {f"s{i}" for i in range(6, 13)}, "C34": set()}
W_FISSURE = 0.1   # poids des résidus où la mesure est au-dessus du modèle
N_DATES = 40      # nombre de dates retenues (espacées régulièrement)
BIN_MM = 4        # moyenne (médiane) des points le long de la fibre par classes de BIN_MM
NT = 50           # points pour l'intégrale temporelle de c
PARAM_NOMS = ("D", "a", "n", "k")
BORNES = ([1e-13, 0.0, 0.3, 1e-4], [1e-7, 3e4, 4.0, 5.0])
DEPARTS = ((3e-12, 0.05), (1e-11, 0.02), (5e-11, 0.1), (2e-10, 0.5))  # (D0, k0)


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


def integrale_concentration(X, Y, Z, t_jours, D):
    """I = integrale de c dt' de 0 à t (en jours), grille en t'^2 resserrée aux temps courts."""
    f = (np.arange(1, NT + 1) / NT)[None, :] ** 2
    tt = t_jours[:, None] * f
    s = 2 * np.sqrt(np.maximum(D * tt * 86400, 1e-12))
    u = 1.0
    for p in (X, Y, Z):
        p = np.clip(p, 0, L)[:, None]
        u = u * np.clip(1 - erfc(p / s) - erfc((L - p) / s), 0, 1)
    c = np.concatenate([np.zeros((len(t_jours), 1)), 1 - u], axis=1)
    tt = np.concatenate([np.zeros((len(t_jours), 1)), tt], axis=1)
    return np.sum(0.5 * (c[:, 1:] + c[:, :-1]) * np.diff(tt, axis=1), axis=1)


def modele(theta, g: pd.DataFrame) -> np.ndarray:
    """Retrait de la matrice (µε) aux points de g, theta = (D, a, n, k)."""
    D, a, n, k = theta
    I = integrale_concentration(g["X"].values, g["Y"].values, g["Z"].values,
                                g["temps_ecoule_j"].values, D)
    return -a * (1 - np.exp(-k * I)) ** n


def ajuster(g: pd.DataFrame, w: float = W_FISSURE) -> np.ndarray:
    """Ajustement de l'enveloppe basse (perte asymétrique), multi-départs sur (D, k)."""
    eps = g["eps"].values

    def residus(th):
        r = modele(th, g) - eps
        return np.where(r < 0, np.sqrt(w) * r, r)   # r < 0 : mesure au-dessus du modèle

    meilleur = None
    for D0, k0 in DEPARTS:
        r = least_squares(residus, [D0, 6000.0, 1.0, k0], bounds=BORNES,
                          x_scale=[1e-10, 1000, 1, 0.05])
        if meilleur is None or r.cost < meilleur.cost:
            meilleur = r
    return meilleur.x


def bilan(theta, g: pd.DataFrame, label: str) -> None:
    r = modele(theta, g) - g["eps"].values
    sous = r[r >= 0]
    print(f"{label}: n={len(g)} | part mesure>modèle={np.mean(r < 0):.2f} | "
          f"RMSE (mesure<modèle)={np.sqrt(np.mean(sous ** 2)):.0f} µε")


def tracer(theta, cube: str, segments, fichier: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = prepare(cube, bin_mm=2)
    D, a, n, k = theta
    txt = (f"Paramètres (fit global C12)\nD = {D:.2e} m²/s\nk = {k:.3f} j⁻¹\n"
           f"a = {a:.0f} µε\nn = {n:.2f}")
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
            ax.plot(q[axe] * 1000, modele(theta, q), color=c, lw=2.2, ls="--")
        ax.set_title(f"{cube} {s} — axe {axe}", fontsize=10)
        ax.set_xlabel(f"{axe} (mm)"); ax.set_ylabel("µε"); ax.grid(alpha=.3); ax.set_xlim(0, 100)
        ax.text(.5, .02, txt, transform=ax.transAxes, fontsize=6.5, ha="center", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=.85))
    axs[0, 0].legend(fontsize=7, ncol=2, loc="upper center")
    for ax in axs.ravel()[len(segments):]:
        ax.axis("off")
    fig.suptitle("Traits fins : mesure — tirets : modèle (enveloppe basse)")
    plt.tight_layout(); plt.savefig(fichier, dpi=100); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", choices=["C12", "C34"], default="C12", help="cube ajusté")
    ap.add_argument("--plot", help="fichier PNG des profils (segments d'intérêt)")
    args = ap.parse_args()

    g = prepare(args.cube)
    theta = ajuster(g)
    print(f"{args.cube} : " + ", ".join(f"{nm}={v:.3g}" for nm, v in zip(PARAM_NOMS, theta)))
    bilan(theta, g, f"  ajustement {args.cube}")
    autre = "C34" if args.cube == "C12" else "C12"
    bilan(theta, prepare(autre), f"  validation {autre} (mêmes paramètres)")
    if args.plot:
        tracer(theta, args.cube, ["s14", "s17", "s19", "s21", "s22", "s4", "s3", "s1"], args.plot)
        print("figure :", args.plot)
