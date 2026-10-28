"""
Préparation des données de déformation 3D (fibres optiques, cube en pâte de ciment).

Calcule pour chaque mesure :
- l'axe physique réellement mesuré par la fibre à cet endroit (X, Y ou Z)
- les distances aux 6 faces du cube, regroupées par direction (dist_face_x/y/z)
- la distance "directe" (face perpendiculaire à l'axe mesuré -> retrait direct attendu)
- la distance "latérale min" (faces perpendiculaires aux 2 autres axes -> effet indirect/restreinte)
- le temps écoulé depuis le début d'essai (t0)
"""

from __future__ import annotations  # rend `str | Path` compatible Python < 3.10

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# --- Paramètres de l'essai (à ajuster si besoin) -----------------------------
CUBE_SIDE_M = 0.100          # cube 100 x 100 x 100 mm, origine au coin (0,0,0)
T0 = pd.Timestamp("2026-07-08 14:40:00")  # début d'essai (décoffrage / t0 carbonatation)

# Profondeur de plan nominale (mm) associée à chaque segment (info métier fournie)
PLAN_PROFONDEUR_NOMINAL = {
    # plan ~8 mm
    "s11": 8, "s12": 8, "s10": 8, "s14": 8, "s15": 8, "s16": 8, "s17": 8,
    # plan ~16 mm
    "s6": 16, "s7": 16, "s8": 16, "s9": 16, "s20": 16, "s19": 16, "s21": 16, "s18": 16,
    # plan ~30 mm
    "s4": 30, "s5": 30, "s22": 30, "s23": 30,
    # axe traversant Z
    "s1": None, "s2": None, "s3": None,
}

FILENAME_RE = re.compile(r"V[1234]_(\d{4})(\d{2})(\d{2})h(\d{2})m(\d{2})")


def parse_date_from_profil(profil: str) -> pd.Timestamp:
    m = FILENAME_RE.search(profil)
    if not m:
        return pd.NaT
    y, mo, d, h, mi = m.groups()
    return pd.Timestamp(int(y), int(mo), int(d), int(h), int(mi))


def infer_axis_per_segment(df: pd.DataFrame) -> dict:
    """Détermine l'axe (X/Y/Z) le long duquel varie chaque segment (= axe physiquement mesuré)."""
    axis_map = {}
    for seg, sub in df.groupby("segment"):
        ranges = {
            "X": sub["X_m"].max() - sub["X_m"].min(),
            "Y": sub["Y_m"].max() - sub["Y_m"].min(),
            "Z": sub["Z_m"].max() - sub["Z_m"].min(),
        }
        if max(ranges.values()) <= 0:
            warnings.warn(
                f"Segment '{seg}' : aucune variation spatiale détectée "
                f"(1 seul point ou coordonnées constantes) — axe assigné arbitrairement à 'X'."
            )
        axis_map[seg] = max(ranges, key=ranges.get)
    return axis_map


def load_and_prepare(csv_path: str | Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Fichier CSV introuvable : {csv_path!r}. Vérifie le chemin "
            f"(option 'Charger un autre profils_3d_*.csv' dans la barre latérale)."
        ) from e
    except pd.errors.ParserError as e:
        raise ValueError(f"Le fichier {csv_path!r} n'a pas pu être lu comme un CSV ';'-séparé : {e}") from e

    required_cols = {"profil", "segment", "x_fibre_m", "X_m", "Y_m", "Z_m", "valeur_microdefs"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path!r} : {sorted(missing)}")

    # --- dates / temps écoulé ---
    df["date_mesure"] = df["profil"].apply(parse_date_from_profil)
    n_nat = df["date_mesure"].isna().sum()
    if n_nat:
        warnings.warn(
            f"{n_nat} ligne(s) avec un nom de profil non reconnu par FILENAME_RE "
            f"-> date_mesure = NaT (elles seront ignorées par les filtres de date)."
        )
    df["fibre"] = df["profil"].str.extract(r"^(V[1234])_")
    df["temps_ecoule_j"] = (df["date_mesure"] - T0).dt.total_seconds() / 86400.0

    # --- axe mesuré par chaque segment ---
    axis_map = infer_axis_per_segment(df)
    df["axe_mesure"] = df["segment"].map(axis_map)

    # --- distances aux faces (clip pour les points légèrement hors cube en bout de fibre) ---
    L = CUBE_SIDE_M
    for c, out in (("X_m", "dist_face_x"), ("Y_m", "dist_face_y"), ("Z_m", "dist_face_z")):
        d = np.minimum(df[c], L - df[c]).clip(lower=0)
        df[out] = d

    df["dist_face_min"] = df[["dist_face_x", "dist_face_y", "dist_face_z"]].min(axis=1)

    # distance "directe" = face perpendiculaire à l'axe mesuré par la fibre.
    # Version vectorisée (np.select) au lieu de df.apply(axis=1, ...), qui est
    # nettement plus lent sur de gros CSV (c'est l'un des points qui ralentit
    # le dashboard, car ce calcul refait à chaque rechargement du fichier).
    dist_cols = {"X": "dist_face_x", "Y": "dist_face_y", "Z": "dist_face_z"}
    conditions = [df["axe_mesure"] == axe for axe in dist_cols]
    choices_direct = [df[dist_cols[axe]] for axe in dist_cols]
    df["dist_face_direct"] = np.select(conditions, choices_direct, default=np.nan)

    # distance "latérale min" = min des 2 autres directions (effet indirect / restreinte)
    lat_min_per_axe = {
        axe: df[[dist_cols[a] for a in dist_cols if a != axe]].min(axis=1)
        for axe in dist_cols
    }
    choices_lateral = [lat_min_per_axe[axe] for axe in dist_cols]
    df["dist_face_laterale_min"] = np.select(conditions, choices_lateral, default=np.nan)

    # info métier complémentaire
    df["plan_profondeur_mm"] = df["segment"].map(PLAN_PROFONDEUR_NOMINAL)
    df["composante"] = "eps_" + df["axe_mesure"].str.lower() + df["axe_mesure"].str.lower()  # eps_xx / eps_yy / eps_zz

    # tri utile pour tracés le long des segments
    df = df.sort_values(["segment", "date_mesure", "x_fibre_m"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_and_prepare("/mnt/user-data/uploads/profils_3d_C12.csv")
    print(df.shape)
    print(df[["segment", "axe_mesure", "composante", "plan_profondeur_mm",
              "fibre", "date_mesure", "temps_ecoule_j",
              "dist_face_direct", "dist_face_laterale_min", "dist_face_min"]].head(10))
    print()
    print("Dates uniques et temps écoulé (j) :")
    print(df.drop_duplicates("profil")[["profil", "fibre", "date_mesure", "temps_ecoule_j"]]
          .sort_values("date_mesure"))
    print()
    print("Axe mesuré par segment :")
    print(df.drop_duplicates("segment")[["segment", "axe_mesure", "plan_profondeur_mm"]]
          .sort_values("segment", key=lambda s: s.str.extract(r"(\d+)")[0].astype(int)))
