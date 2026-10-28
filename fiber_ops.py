"""
Opérations entre fibres (moyenne, somme, différence...) sur une grille
commune de position 0..L. Pur numpy/pandas : réutilisé tel quel dans les
scripts exportés (voir script_generator.gen_operation_fibres).
"""
import warnings

import numpy as np
import pandas as pd

AGG_LABELS = {
    "mean": "Moyenne", "sum": "Somme", "median": "Médiane",
    "min": "Minimum", "max": "Maximum", "std": "Écart-type",
}
COMBINE_LABELS = {
    "none": "Aucune (résultat = groupe A)",
    "a-b": "A − B", "a+b": "A + B", "a/b": "A / B", "(a+b)/2": "(A + B) / 2",
}


def _aggregate(stack, agg):
    """Agrège un tableau (n_fibres, n_points) le long des fibres, en ignorant les NaN."""
    if stack.shape[0] == 0:
        return np.full(stack.shape[1], np.nan)
    allnan = np.isnan(stack).all(axis=0)
    funcs = {"mean": np.nanmean, "sum": np.nansum, "median": np.nanmedian,
             "min": np.nanmin, "max": np.nanmax, "std": np.nanstd}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = funcs[agg](stack, axis=0)
    return np.where(allnan, np.nan, out)


def _combine(a, b, combine):
    if combine == "none":
        return a
    with np.errstate(divide="ignore", invalid="ignore"):
        if combine == "a-b":
            return a - b
        if combine == "a+b":
            return a + b
        if combine == "(a+b)/2":
            return (a + b) / 2
        if combine == "a/b":
            return np.where(b == 0, np.nan, a / b)
    raise ValueError(f"combine inconnu : {combine!r}")


def compute_operation(df, segs_a, segs_b, agg="mean", combine="none",
                      absval=False, L=0.100, n=200):
    """Retourne (res, profils) :
    - res     : date_mesure, temps_ecoule_j, pos_m, valeur   (résultat de l'opération)
    - profils : segment, groupe, date_mesure, pos_m, valeur  (fibres individuelles réinterpolées)
    `df` doit déjà être filtré sur les dates voulues."""
    grid = np.linspace(0, L, n)
    axis_col = {"X": "X_m", "Y": "Y_m", "Z": "Z_m"}
    tol = 0.002  # 2 mm de tolérance aux extrémités des fibres
    sub = df[df["segment"].isin(set(segs_a) | set(segs_b))]
    t_of = sub.groupby("date_mesure")["temps_ecoule_j"].first()

    res_parts, prof_parts = [], []
    for d in sorted(sub["date_mesure"].dropna().unique()):
        sd = sub[sub["date_mesure"] == d]
        interp = {}
        for seg, g in sd.groupby("segment"):
            col = axis_col[g["axe_mesure"].iloc[0]]
            g = g.sort_values(col)
            x, v = g[col].to_numpy(), g["valeur_microdefs"].to_numpy()
            y = np.interp(grid, x, v)
            y[(grid < x[0] - tol) | (grid > x[-1] + tol)] = np.nan
            interp[seg] = y

        def stack_of(segs):
            rows = [interp[s] for s in segs if s in interp]
            return np.vstack(rows) if rows else np.empty((0, n))

        a = _aggregate(stack_of(segs_a), agg)
        b = _aggregate(stack_of(segs_b), agg) if len(segs_b) else np.full(n, np.nan)
        r = _combine(a, b, combine)
        if absval:
            r = np.abs(r)
        res_parts.append(pd.DataFrame({"date_mesure": d, "temps_ecoule_j": t_of[d],
                                       "pos_m": grid, "valeur": r}))
        for grp, segs in (("A", segs_a), ("B", segs_b)):
            for s in segs:
                if s in interp:
                    prof_parts.append(pd.DataFrame({"segment": s, "groupe": grp,
                                                    "date_mesure": d, "pos_m": grid,
                                                    "valeur": interp[s]}))
    empty_res = pd.DataFrame(columns=["date_mesure", "temps_ecoule_j", "pos_m", "valeur"])
    empty_prof = pd.DataFrame(columns=["segment", "groupe", "date_mesure", "pos_m", "valeur"])
    res = pd.concat(res_parts, ignore_index=True) if res_parts else empty_res
    prof = pd.concat(prof_parts, ignore_index=True) if prof_parts else empty_prof
    return res, prof