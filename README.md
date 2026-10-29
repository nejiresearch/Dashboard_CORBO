# Dashboard CORBO — Déformations 3D par carbonatation

Dashboard Streamlit d'analyse des déformations mesurées par fibres optiques dans deux cubes de pâte de ciment bas-pH soumis à la carbonatation (essais **C12** et **C34**).

## Contexte de l'essai

- Deux cubes (C12 et C34) de pâte de ciment bas-pH durcie, de 10 cm de côté.
- Les cubes ont été hydratés, séchés, puis placés dans une enceinte de carbonatation (3 % CO₂, 55 % HR).
- Chaque cube est posé dans un grand bac ouvert, lui-même placé dans l'enceinte.
- Les six faces sont exposées à la carbonatation.

## Instrumentation

- 2 fibres optiques par cube, placées symétriquement par rapport au plan central (z = 5 cm).
- Chaque fibre est enrobée dans le bloc par segments tendus, ce qui donne la déformation dans la direction du segment.
- Les segments **s1, s2, s3** sont orientés selon **z** et appartiennent à l'une des deux fibres. Les autres segments sont dans le plan XY.
- Dans chaque moitié de cube : 3 plans de fibres XY, à **8 mm, 16 mm et 30 mm** de l'interface.

## Fichiers

| Fichier | Description |
|---|---|
| `dashboard.py` | Application Streamlit (`streamlit run dashboard.py`) avec export de scripts Plotly reproductibles. |
| `data_prep.py` | Préparation des données : axe mesuré, distances aux faces (directe / latérale), temps écoulé depuis t0. |
| `date_filter.py` | Sélection des dates affichées selon un ΔT minimal (linéaire, √t, ln(1+t)). |
| `fiber_ops.py` | Opérations entre fibres (moyenne, somme, médiane, min, max, écart-type, A−B, A+B, A/B…). |
| `script_generator.py` | Génération des scripts Python autonomes exportés (style publication). |
| `Plan_FIB_C12_V2.toml`, `Plan_FIB_C34_V1.toml` | Plans de fibres de C12 et C34 (indice, paramètres par fibre). |
| `profils_3d_C12.csv`, `profils_3d_C34.csv` | Données de mesure (voir ci-dessous). |

## Format des CSV de données

| Colonne | Description |
|---|---|
| `segment` | Segment d'appartenance |
| `profil` | Nom du fichier d'origine |
| `id` | Identifiant |
| `x_fibre_m` | Position le long de la fibre (m) |
| `X_m`, `Y_m`, `Z_m` | Coordonnées spatiales (m) |
| `valeur_microdefs` | Microdéformation (µε) |

## Lancement

```bash
pip install streamlit pandas numpy scipy plotly matplotlib
streamlit run dashboard.py
```
