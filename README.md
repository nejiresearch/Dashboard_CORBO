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
| `data/profils_3d_C12.csv`, `data/profils_3d_C34.csv` | Données de mesure (voir ci-dessous). |

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

## **Discussion des données à négliger**

Cette section sert à décider quelles données sont conservées pour l'étude des déformations.

### C12 : asymétrie entre les deux moitiés (Z)

Observations (microdéformations médianes, essai de 113 jours) :

| Zone (segments Z s1 et s3) | C12 | C34 (s3) |
|---|---|---|
| Z ≈ 5-25 mm (haut) | −5100 à −5500 µε | −4700 µε |
| Z ≈ 40-60 mm (cœur) | −1200 µε | −1100 µε |
| Z ≈ 75-95 mm (fond) | −600 à −1800 µε | −3600 µε |

- Convention : Z = 0 correspond au **haut** du cube, donc Z = 100 mm est le **fond** (côté fond du bac).
- Dans C12, le haut (Z faible) se contracte jusqu'à environ −6000 µε, tandis que le fond (Z fort) se contracte très peu.
- Dans C34, le haut et le fond se contractent de façon comparable. Le fond de C12 est donc anormal, et ce n'est pas une propriété du matériau.
- Segments de la moitié basse (fond) de C12 (fibre V2, Z > 50 mm) : s4 à s12. Leur déformation finale va de +460 à −2000 µε, et plusieurs sont proches de 0 (s10, s11, s12).
- La moitié haute de C12 (fibre V1, Z < 50 mm, segments s14 à s23) se comporte comme C34 aux mêmes profondeurs.

Hypothèse (cohérente avec l'orientation) : le fond du cube C12, au fond d'un bac ouvert, a un accès limité au CO₂, ou le matériau y consomme plus de CO₂ qu'il n'en arrive localement.

À vérifier : la profondeur carbonatée par face (phénolphtaléine après découpe), en particulier sur la face du fond.

Décision proposée :
- **À écarter** de l'analyse de carbonatation homogène : C12 s4 à s12.
- **À conserver** : C12 s1 à s3 (en précisant la zone), C12 s14 à s23, et tous les segments de C34.
- Ne pas supprimer s4 à s12 des fichiers : les garder en groupe séparé « zone peu carbonatée ».

### C34 : segments absents et bruit

- Les fibres ont été coupées à deux endroits. Seuls s1 à s5 et s14 à s19 existent (s6 à s13 et s20 à s23 sont absents).
- s1 s'arrête vers √t ≈ 7,3 j^½ ; s5 n'a que 68 points par profil (106 pour C12).
- Le signal est plus bruité, avec des paliers brusques (s16, s17, s19). Il faut lisser avant d'interpréter les écarts avec C12.
