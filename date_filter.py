"""
Sélection des dates de mesure à afficher, selon un espacement minimal
configurable (ΔT), avec 3 modes possibles :
- "linear"   : ΔT constant en jours (temps_ecoule_j)
- "diffusif" : ΔT constant en √t (cohérent avec une cinétique de diffusion)
- "log"      : ΔT constant en ln(1+t) (utile si la cinétique ralentit très vite)

La première et la dernière date de l'intervalle choisi sont toujours conservées.
"""

from __future__ import annotations  # rend `list[pd.Timestamp]` compatible Python < 3.9

import numpy as np
import pandas as pd

MODE_LABELS = {
    "linear": "Linéaire (ΔT constant, jours)",
    "diffusif": "Diffusif (ΔT constant en √t)",
    "log": "Logarithmique (ΔT constant en ln(1+t))",
}


def _key_for_mode(t: np.ndarray, mode: str) -> np.ndarray:
    if mode == "diffusif":
        return np.sqrt(np.clip(t, 0, None))  # sqrt indéfini pour t<0 -> clip local
    if mode == "log":
        # log1p(t) n'est défini que pour t > -1 (jours) ; on protège le cas
        # (rare, mais possible si une mesure a été prise juste avant t0)
        # où temps_ecoule_j serait <= -1.
        return np.log1p(np.clip(t, -0.999, None))
    return t  # linear : on garde les temps négatifs tels quels


def select_dates_by_spacing(df_full: pd.DataFrame, t_min: float, t_max: float,
                             delta: float, mode: str = "linear") -> list[pd.Timestamp]:
    """Retourne la liste triée des dates retenues dans [t_min, t_max] (bornes
    en jours, sur temps_ecoule_j), en ne gardant que celles espacées d'au
    moins `delta` dans l'espace défini par `mode`. Garde toujours la 1re et
    la dernière date de l'intervalle."""
    uniq = (df_full.drop_duplicates("date_mesure")[["date_mesure", "temps_ecoule_j"]]
                    .sort_values("temps_ecoule_j"))
    uniq = uniq[(uniq["temps_ecoule_j"] >= t_min) & (uniq["temps_ecoule_j"] <= t_max)]
    if uniq.empty:
        return []

    t = uniq["temps_ecoule_j"].to_numpy()
    dates = uniq["date_mesure"].to_numpy()
    key = _key_for_mode(t, mode)

    keep_idx = [0]
    last_key = key[0]
    if delta > 0:
        for i in range(1, len(key)):
            if key[i] - last_key >= delta:
                keep_idx.append(i)
                last_key = key[i]
    else:
        keep_idx = list(range(len(key)))  # ΔT=0 -> tout garder

    if keep_idx[-1] != len(key) - 1:
        keep_idx.append(len(key) - 1)

    return list(pd.to_datetime(dates[keep_idx]))