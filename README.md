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

### Orientation du cube dans l'enceinte

- Les coordonnées des CSV suivent le repère du plan d'instrumentation et ne sont pas modifiées.
- Dans l'enceinte, l'axe **X est vertical** (X = 0 en haut), **Z est horizontal** (gauche-droite) et Y est la profondeur. Le plan d'interface (Z = 50 mm) est donc vertical.
- Les vues 3D du dashboard (onglets 4, 6, 9 et 10, schémas d'axes) sont tracées dans cette orientation, avec X vertical inversé (0 en haut). Le script exporté de l'onglet 6 aussi.
- Les déformations et les distances aux faces ne changent pas : le cube est symétrique et ses six faces sont exposées.

## Fichiers

| Fichier | Description |
|---|---|
| `dashboard.py` | Application Streamlit (`streamlit run dashboard.py`) avec export de scripts Plotly reproductibles. |
| `data_prep.py` | Préparation des données : axe mesuré, distances aux faces (directe / latérale), temps écoulé depuis t0. |
| `date_filter.py` | Sélection des dates affichées selon un ΔT minimal (linéaire, √t, ln(1+t)). |
| `fiber_ops.py` | Opérations entre fibres (moyenne, somme, médiane, min, max, écart-type, A−B, A+B, A/B…). |
| `script_generator.py` | Génération des scripts Python autonomes exportés (style publication). |
| `fit_retrait.py` | Fit pseudo-physique du retrait de la matrice (réaction-diffusion couplée du CO₂, ajustement de l'enveloppe basse). Voir ci-dessous. |
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
| Z ≈ 5-25 mm | −5100 à −5500 µε | −4700 µε |
| Z ≈ 40-60 mm (cœur) | −1200 µε | −1100 µε |
| Z ≈ 75-95 mm | −600 à −1800 µε | −3600 µε |

- Z est **horizontal** (voir Orientation) : l'asymétrie est gauche-droite, pas haut-bas.
- Dans C12, le côté Z faible se contracte jusqu'à environ −6000 µε, tandis que le côté Z fort se contracte très peu.
- Dans C34, les deux côtés se contractent de façon comparable. Le côté Z fort de C12 est donc anormal, et ce n'est pas une propriété du matériau.
- Ce n'est pas un artefact de fibre : s1 et s3 sont sur la même fibre V2 et montrent les deux comportements.
- Segments du côté Z fort de C12 (fibre V2, Z > 50 mm) : s4 à s12.
  - s6 à s12 (Z ≥ 84 mm) se contractent peu : de +460 à −1300 µε, et s10, s11 et s12 sont proches de 0.
  - s4 et s5 (Z = 70 mm, plan à 30 mm de l'interface) se comportent comme leurs homologues symétriques s22 et s23 (Z = 30 mm) : corrélation de 0,97 à 0,99 sur toute la durée, et −890 à −1970 µε contre −1060 à −1220 µε. Le comportement de C34 aux mêmes positions est cohérent (environ −1000 µε).
- Le côté Z faible de C12 (fibre V1, Z < 50 mm, segments s14 à s23) se comporte comme C34 aux mêmes profondeurs.

Cause à déterminer. L'hypothèse initiale (fond du bac peu accessible au CO₂) ne tient plus, car l'asymétrie est selon Z (horizontal).

#### Hypothèse en cours : ségrégation lors de la coulée

- Le matériau est un mélange fumée de silice et clinker, visant une hydratation de type bas-pH : pas de portlandite, seulement des C-S-H et quelques phases aluminates.
- Entre la gâchée et l'analyse, le cube a été retourné de 90°. Une des deux fibres et ses segments se trouvent donc du côté qui était le fond du moule.
- Une ségrégation aurait eu lieu à la coulée, la fumée de silice étant moins dense que le clinker et les fibres gênant le mélange.
  - Côté haut du moule : Ca/Si bas, peu de portlandite, beaucoup de C-S-H, donc retrait de carbonatation important (lié aux C-S-H).
  - Côté fond du moule : beaucoup plus de portlandite, dont la carbonatation entraîne un retrait plus faible.

Éléments cohérents avec l'hypothèse :
- Le retrait de carbonatation vient surtout de la décalcification des C-S-H. La carbonatation de la portlandite contribue peu, voire peut gonfler (cohérent avec +460 µε sur s11).
- La portlandite qui se carbonate comble la porosité, ce qui ralentit le CO₂ côté fond. Les deux effets vont dans le même sens.

Points de prudence :
- La cause est plus probablement le ressuage et la sédimentation du clinker que la différence de densité seule. Les fibres, très fines, gênent peu le mélange.
- Une sédimentation donnerait un gradient progressif. Or s4 et s5 (Z = 70 mm) se comportent comme s22 et s23, et l'écart n'apparaît qu'à Z ≥ 84 mm, donc sur une couche mince.
- Si C34 a été coulé de la même façon, il devrait montrer la même asymétrie, or elle y est modérée.

À vérifier :
- Le côté Z fort (fibre V2) est-il bien le côté qui était au fond du moule ?
- C34 a-t-il été coulé ou retourné différemment ?
- ATG ou DRX (portlandite) et EDS (Ca/Si) sur des tranches de chaque moitié, ainsi que porosimétrie.

À vérifier : la profondeur carbonatée par face (phénolphtaléine après découpe), pour chacune des six faces.

Décision proposée :
- **À écarter** de l'analyse de carbonatation homogène : C12 s6 à s12.
- **À conserver** : C12 s4 et s5, C12 s1 à s3 (en précisant la zone), C12 s14 à s23, et tous les segments de C34.
- Ne pas supprimer s6 à s12 des fichiers : les garder en groupe séparé « zone peu carbonatée ».

### C34 : segments absents et bruit

- Les fibres ont été coupées à deux endroits. Seuls s1 à s5 et s14 à s19 existent (s6 à s13 et s20 à s23 sont absents).
- s1 s'arrête vers √t ≈ 7,3 j^½ ; s5 n'a que 68 points par profil (106 pour C12).
- Le signal est plus bruité, avec des paliers brusques (s16, s17, s19). Il faut lisser avant d'interpréter les écarts avec C12.

## Fit du retrait de carbonatation (`fit_retrait.py`)

```bash
python fit_retrait.py --plot fit_C12.png
```

- Modèle (réaction-diffusion couplée) : le CO₂ diffuse depuis les six faces avec un D_e **fixé** (littérature : 2·10⁻⁷ à 7·10⁻⁷ m²/s, option `--de`) et est consommé par la réaction. Degré de carbonatation α (cinétique d'ordre 1, constante k), retrait ε = −a·α^n, capacité de fixation N (mol CO₂/m³). Résolution 1D en demi-dalle puis combinaison 3D approchée α = 1 − Π(1 − α_i).
- Seul le rapport N/D_e est identifiable : N est proportionnel à D_e (D_e = 3,7·10⁻⁷ → N ≈ 23 300 ; 2·10⁻⁷ → 12 600 ; 7·10⁻⁷ → 44 000 mol/m³). Une mesure de N (ATG) fixe donc D_e.
- La déformation mesurée est la somme du retrait de la matrice et de l'ouverture de fissures (≥ 0). On ajuste l'enveloppe basse, avec un poids de 0,1 sur les résidus où la mesure est au-dessus du modèle (fissures).
- Les segments C12 s6 à s12 sont exclus. Les points hors du cube sont ignorés.
- Résultat sur C12 : k ≈ 0,039 j⁻¹, a ≈ 5980 µε, n ≈ 1,07. Validation sur C34 avec les mêmes paramètres : RMSE sous l'enveloppe 956 µε (1048 µε sur C12).
- Limites : les creux à −8000 µε près des faces et les pics de fissuration ne sont pas modélisés ; la combinaison 3D est approchée.

## Fissuration : démarche envisagée (à reprendre)

État : réflexion et premier essai exploratoire, **pas encore de script dans le dépôt**.

### Principe
La fibre mesure la déformation moyenne sur une jauge Lg : matrice (retrait) + ouverture de fissures. Une fissure de largeur w donne ε_fiss ≈ w/Lg (+4000 µε sur 5 mm ≈ 20 µm). Le résidu r = ε_mesuré − ε_matrice(modèle couplé) est donc un indicateur de fissuration.

### Trois niveaux
1. **Indicateur de fissuration** (essai fait) : position des pics, ouverture cumulée Σw = Σ(excès)·Lg, nombre de pics, par segment et par date.
2. **Endommagement phénoménologique** D(x,t) ∈ [0,1], piloté par la traction due au retrait gêné (σ_t ≈ E·(ε_cœur − ε_peau)/(1−ν) > f_t), qui réduit le retrait et ajoute l'ouverture. Calage de 2–3 paramètres (f_t, loi d'évolution) sur le niveau 1.
3. **Fissuration explicite** (EF, endommagement ou zones cohésives) : en dernier, vu la densité de fissures.

### Premier essai (niveau 1, tous les segments retenus)
Résidu moins une base locale (quantile 20 % sur 30 mm), seuil 3σ avec σ ≥ 100 µε, jauge supposée Lg = 2 mm, 40 dates.

Résultats exploitables :
- Ouverture cumulée par paliers : forte entre ~10 et 40 j, puis stabilisation vers 50–60 j (cohérent avec le phasage diffusif → fissuration).
- Fibres proches de la peau, selon X et Y (C12 s15, s16, s17 ; C34 s16, s17, s19) : ~2000–2500 µm/mm cumulé. Segments Z (s1–s3) et centraux (s4, s20–s23) : ~650 µm/mm.
- Même classement entre C12 et C34.

Limites :
- Fraction fissurée trop élevée (50–75 %) : le seuil capte du bruit et l'erreur du modèle.
- Date d'apparition ≈ 5 j partout : artefact, aucune conclusion sur le phasage.
- Valeurs absolues de Σw indicatives (dépendent de Lg), certaines courbes décroissent (bruit ou relaxation ?).

### À faire
1. Estimer le bruit sur une zone ou une fibre sans fissure (donnée expérimentale) pour fixer le seuil.
2. Détecter des pics (largeur, proéminence) plutôt qu'un simple dépassement.
3. Utiliser toutes les dates (au lieu de 40) pour la cinétique.
4. Corriger l'axe de position des cartes (position le long de la fibre ≠ X).
5. Étendre au phasage (apparition près de la face ou du front), puis au niveau 2 (endommagement).
6. Considérer les fissures obliques ou parallèles à la fibre (non vues) avec les trois directions de fibres.
