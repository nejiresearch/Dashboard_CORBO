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
| `fissuration_gaussiennes.py` | Décomposition du résidu (mesure − retrait modèle) en gaussiennes suivies dans le temps : positions, ouvertures, fusion discret → diffus. Voir « Fissuration par décomposition en gaussiennes ». |
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

État : réflexion, premier essai exploratoire (niveau 1) puis décomposition en gaussiennes implémentée dans `fissuration_gaussiennes.py` (voir plus bas).

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

### Fissuration par décomposition en gaussiennes (`fissuration_gaussiennes.py`)

```bash
python fissuration_gaussiennes.py --cube C12 --segment s14 --plot s14.png --csv s14.csv
```
Options : `--t0` (première échéance, 28 j par défaut), `--theta N k a n` et `--de` (retrait), `--refit` (ré-ajuste le retrait sur le cube, ~10 s). Durée : ~10 s par segment.

**Pourquoi des gaussiennes.** La fibre OBR moyenne la déformation sur une jauge de 5 mm : une fissure de largeur w apparaît comme un pic étalé, pas comme un point. L'**aire** du pic reste égale à l'ouverture, w = ∫ε dx, indépendamment de la jauge. On modélise donc le résidu r(x,t) = ε_mesuré − ε_retrait par

r(x,t) ≈ Σᵢ Aᵢ(t) · exp(−(x − cᵢ)² / 2σᵢ(t)²),  Aᵢ ≥ 0,  wᵢ = Aᵢ·σᵢ·√(2π)  (w en µm si A en µε et σ en mm : facteur 10⁻³).

Chaque gaussienne a un **centre cᵢ constant** dans le temps (une fissure ne se déplace pas) ; seuls Aᵢ et σᵢ évoluent. Une gaussienne n'est **jamais supprimée** une fois créée.

**Algorithme** (défaut : ordre **inverse**, sans fusion)
1. *Dernière échéance (113 j)* : le signal est le mieux résolu. Ajout glouton de gaussiennes (σ de 1,45 à 25 mm, jauge/√12 minimum), centres libres ; on ajoute une gaussienne si elle réduit le SSE d'au moins 4 % (amplitude ≥ 500 µε, distance ≥ 2 mm aux autres).
2. *Échéances précédentes (à rebours)* : centres figés, Aᵢ ≥ 0 et σᵢ réajustés. Une gaussienne non encore apparue a Aᵢ ≈ 0 : sa date d'apparition se lit sur wᵢ(t). Aucune gaussienne n'est supprimée ni fusionnée.
3. *Fissuration diffuse a posteriori* : `groupes()` regroupe les gaussiennes dont les centres sont distants de ≤ 8 mm (`GROUPE_MM`) et donne Σw par groupe. Le « diffus » est une propriété calculée après coup, pas un résultat de l'ajustement.

4. *Régularisation temporelle (option D, défaut)* : ajustement **global** de toutes les échéances en une seule fois, à partir du suivi à rebours. Paramètres : centres cᵢ communs (±4 mm autour des valeurs initiales, dans la fibre), ouvertures wᵢ(t) **non décroissantes** (somme cumulée d'incréments ≥ 0, Aᵢ = wᵢ/(σᵢ√(2π))) et σᵢ(t) lisses (pénalité λ·Δσᵢ, `LAMBDA_SIGMA` = 400). Le nombre de gaussiennes reste celui de la dernière échéance. Désactivable avec `--sans-regul`. Durée : ~30 s par segment.

Options : `--avant` (ancien ordre chronologique, centres déterminés à t0) et `--fusion` (fusion de deux voisines si le SSE augmente de moins de `MERGE_TOL` = 3 %). Ces options sont déconseillées : le critère de fusion est global et le SSE total est dominé par l'erreur résiduelle, donc des pics nettement séparés (ex. s14, X ≈ 50 et 60 mm) étaient fusionnés trop facilement, et les centres déterminés à 28 j (signal faible) étaient mal placés.

Paramètres (constantes en tête de script) : `S_MIN`, `S_MAX` (25 mm), `ADD_GAIN`, `A_MIN`, `GROUPE_MM`, `MERGE_TOL`, `LAMBDA_SIGMA`.

**Résultat (C12 s14, 28 → 113 j, ordre inverse).** 11 gaussiennes suivies, centres stables à X ≈ 7, 23, 32, 42, 51, 58, 68, 81, 86 et 100 mm. Les pics voisins à X ≈ 42, 51, 58 et 68 mm restent séparés à toutes les dates. Σw ≈ 187 µm à 29 j, ≈ 233 µm à 52 j, ≈ 245 µm à 100–113 j. Dernier état : groupe central X = 51–68 mm ≈ 145 µm, pic isolé à 42 mm ≈ 54 µm. RMSE de 890 µε (29 j) à 950 µε (50 j) puis ≈ 930 µε : moins bonne aux premières dates qu'avec l'ordre chronologique (550 µε à 29 j), car le nombre de gaussiennes est fixé par la dernière échéance.

**Effet de la régularisation (s14).** wᵢ(t) croît puis se stabilise, σᵢ(t) varie lentement (plus de sauts), les pics à X ≈ 42, 51 et 58 mm restent séparés (≈ 54 µm à 42 mm). Σw ≈ 154 µm à 29 j, ≈ 240 µm à 54 j, ≈ 270 µm vers 90 j puis stable. RMSE à 29 j ≈ 886 µε, équivalent à l'ordre inverse seul (890) : le gain obtenu en ne contraignant que Aᵢ (791 µε) disparaît quand on contraint wᵢ. Σw est plus élevé (≈ 270 µm) que sans régularisation (≈ 245 µm).

**Limites et points ouverts**
- Les premières dates sont moins bien ajustées (centres issus de la dernière échéance, pas d'ajout à rebours) ; une fissure absente à 113 j mais présente plus tôt ne serait pas vue. Une gaussienne peut aussi être trop large à une échéance tardive pour en représenter plusieurs plus tôt (ex. zone X ≈ 10 mm de s14 : une seule gaussienne là où deux ou trois pourraient être distinguées).
- Quelques wᵢ(t) baissent encore légèrement après ~90 j malgré la contrainte (cause non investiguée).
- Le partage de w entre gaussiennes qui se chevauchent est incertain ; seul Σw est robuste. Les pics plus étroits que la jauge ont un σ mal contraint (seule l'aire est fiable).
- Le seuil de détection n'est pas relié à un bruit estimé : l'erreur du modèle de retrait (centaines de µε, creux près des faces non captés) domine le bruit instrumental (≈ 25–40 µε avant 1,5 j). Un bruit estimé sur les dates tardives intègre déjà la fissuration (MAD des résidus : 17 µε à 0,8 j, 130 µε à 5 j). Un test de signification par pic (w/σ_w via le jacobien) rejette alors tous les pics discrets de s14 : seule la composante diffuse (σ 5–25 mm, ≈ 230 µm) est statistiquement robuste.
- Borner σ à la résolution ne convient pas : cela fragmente artificiellement une fissuration diffuse.

**Suite envisagée** : autoriser des ajouts à rebours (option `ajout_arriere`, notamment zone X ≈ 10 mm), généraliser à tous les segments et cartographier Σw.

### Onglet « 1️⃣1️⃣ Fissuration » du dashboard

`onglet_fissuration.py` lance `fissuration_gaussiennes.py` depuis le dashboard. Deux modes : **Retrait seul** (mesure vs modèle, écart moyen et part de mesure au-dessus du retrait, pour valider le modèle) et **Retrait + fissuration**. Rien n'est calculé avant le clic sur **Lancer le calcul** : on choisit d'abord les segments et les échéances (≥ 28 j), la régularisation, le retrait (fit C12 par défaut ou ré-ajustement) et D_e. Les données allégées et chaque résultat (cube, segment, échéances, paramètres) sont mis en cache ; le temps affiché est une estimation (~1 s par segment et par échéance). Un panneau « Paramètres avancés » permet de modifier N, k, a, n (retrait) et les constantes de fissuration (σ min/max, gain d'ajout, amplitude min, distance min, mobilité des centres, nombre max, λ de lissage de σ, écart de regroupement) ; ils entrent dans la clé du cache. L'onglet utilise les CSV du dépôt (pas le fichier chargé manuellement) et fournit la figure, les groupes de gaussiennes, Σw(t) et un export CSV par segment.

**Mise à jour de l'onglet.** Les segments se choisissent par plan (8, 16, 30 mm, axe Z traversant) et orientation (εxx, εyy, εzz), avec des cases « Tous », comme dans l'onglet 10, et la position des segments choisis s'affiche sur le cube 3D. Toutes les figures sont en Plotly (`figure_profil`, `figure_ouvertures`, `figure_suivi` dans `fissuration_gaussiennes.py`). Un encadré « Théorie du modèle » résume le retrait, la décomposition en gaussiennes (w = Aσ√(2π)) et l'algorithme. L'expander « Script Python reproductible » donne un script pour notebook (`script_generator.gen_fissuration`) : il embarque `fit_retrait.py` et `fissuration_gaussiennes.py` et ne nécessite que le CSV. Correction d'un bug d'affichage : le retrait tracé était faux sur les fibres parcourues de 100 à 0 mm (`np.interp` sur abscisses décroissantes) ; les ajustements et les chiffres n'étaient pas touchés. Le paramètre `SEUIL_RMSE` (pic candidat si > k × RMSE courant ; 0 = désactivé) est modifiable.

**Date de calage, plage temporelle et surface 3D.** Toutes les échéances sont disponibles (plus de limite à 28 j ; le retrait reste peu fiable avant ~28 j). La **date de calage** (curseur, défaut = dernière échéance) est la date où les gaussiennes sont ajoutées (nombre et positions) ; le suivi va ensuite vers les dates antérieures (à rebours) puis postérieures (centres figés, sans nouvelle gaussienne). En ligne de commande : `--t-calage`. Une case par segment affiche la **surface 3D** (`figure_surface`) : X = temps, Y = position, Z = ouverture linéique Σ Aᵢ·gaussienneᵢ·10⁻³ en µm/mm (son intégrale en x vaut Σw). Elle est évaluée directement sur une grille de 200 positions et au plus 60 dates, sans interpolation. L'encadré « Théorie du modèle » détaille maintenant le retrait (diffusion-réaction, cinétique, loi, ajustement de l'enveloppe basse) et la fissuration (jauge, ouverture, algorithme, régularisation, limites).

### Performance du dashboard

- Les filtres de la barre latérale sont dans un formulaire : rien n'est recalculé avant **✅ Appliquer les filtres**.
- Un seul onglet est calculé à la fois (sélecteur d'onglets ; `st.tabs` exécutait les 11 onglets à chaque interaction), avec un message « Calcul en cours… ».
- Le CSV est lu une fois (`st.cache_resource`, DataFrame partagé sans copie) et les dates sont lues sur les profils uniques (lecture ~2 s → ~0,5 s) ; les données ne sont ni échantillonnées ni arrondies.

### Fissuration par shear lag (`fissuration_cisaillement.py`)

Variante du script gaussiennes avec le noyau physique d'une fissure, ε(x) = (w·λ/2)·exp(−λ|x−c|) (aire = w), moyenné analytiquement sur la jauge de 5 mm. Même résidu (mesure − retrait), mêmes étapes (centres issus de la dernière échéance, suivi à rebours), puis ajustement global avec wᵢ(t) non décroissant (somme cumulée d'incréments ≥ 0), λᵢ(t) lissé et centres communs. Option `--positif` : n'ajuste que la part du résidu au-dessus du retrait.

```bash
python fissuration_cisaillement.py --cube C12 --segment s14 --plot s14_sl.png --csv s14_sl.csv
```
**Version « OBRstudio » (noyau exponentiel-triangulaire à cœur `a`)** : détection par proéminence (`--prom-facteur`, défaut 2×bruit), fond `b(t)` ajusté (`--sans-fond` pour le désactiver), λ constant par fissure, cœur `a` non décroissant (`--a-max 0` = exponentiel pur), pondération asymétrique (`--w-neg`, défaut 0,3 ; 1 = symétrique), signe `+` par défaut, presets câble (`--preset`), fissures forcées/interdites. Sur C12 s14 (113 j) : 6 fissures, RMSE ≈ 1280 µε, Σw ≈ 320 µm (aires : 298 µm) ; `--a-max 0` : 1390 µε ; sans fond : 1450 µε ; `--w-neg 1` : 7 fissures, 1030 µε. λ reste mal identifié (souvent à la borne de 1000 m⁻¹).

**Détection multi-échéances (`--multi`)** : `find_peaks` sur toutes les échéances (proéminence `--prom-multi`, défaut 1,5×bruit de la date), pics reliés en trajectoires (±3 mm), validées si durée ≥ `--duree-min` (10 j) ou présentes aux deux dernières échéances ; les centres validés initialisent l'ajustement global. Sur s14 : 9 trajectoires validées (X = 8, 22, 31, 43, 52, 62, 81, 84 mm…). `--largeur-max 10` est actif par défaut avec `--multi` (modifiable, `0` = aucune limite) : sans cela, une fissure fantôme en fin de fibre (1/λ ≈ 20–60 mm) absorbe la pente du fond (Σw ≈ 740 µm contre 300 aires). Avec : 8 fissures, RMSE 1237 µε (1016 µε avec `--w-neg 1`), Σw ≈ 340 µm (aires 297).

Avec `--multi`, le complément glouton est désactivé (redondant, lent, ajoute des fissures de bord ; `--glouton` pour le réactiver). La jacobienne de l'ajustement global est calculée sur (w, a) directs puis composée avec la somme cumulée : ~5× plus rapide (s14 : ≈ 1 min au lieu de 5). Sur s15 : 9 fissures, RMSE 1134 µε, Σw 595 µm (aires 415) ; une fissure de bord à X=0 mm (w=143 µm, aires 12 µm) est suspecte (effet de bord) : `--interdites 0` pour l'exclure.

Premier essai sur C12 s14 : 3 fissures seulement (X ≈ 51, 61, 67 mm), RMSE ≈ 1400–1600 µε (≈ 900 µε avec les gaussiennes), Σw ≈ 200 → 245 µm. Le noyau exponentiel ne reproduit pas les pics étroits et les plateaux du résidu ; le critère d'ajout s'arrête vite. Résultat à considérer comme exploratoire.

**Paires de fissures rapprochées et λ global.** Pour deux fissures voisines (x₁, x₂, d = x₂ − x₁), la superposition linéaire des deux noyaux donne un profil en selle de pics E₁ = β(δ₁ + δ₂e^(−βd)), E₃ = β(δ₁e^(−βd) + δ₂) et de creux E₂ = 2β√(δ₁δ₂)·e^(−βd/2) (δ = COD/2, β = λ). `extraire_paire(E1, E2, E3, d)` inverse ces trois relations : équation scalaire en β (E₂ = 2e^(−βd/2)/(1−e^(−2βd))·√((E₁−E₃e^(−βd))(E₃−E₁e^(−βd)))), puis δ₁ et δ₂ ; vérifié sur des profils synthétiques (β, δ₁, δ₂ retrouvés à 10⁻⁸ près). `estimer_beta` l'applique aux paires de pics consécutifs du résidu lissé (4 mm ≤ d ≤ 60 mm), et le script affiche β médian. L'option `--lambda-global <m⁻¹|auto> [--tol-lambda t]` impose un λ commun à toutes les fissures et dates.

Essais (113 j, λ imposé de 100 à 450 m⁻¹) :
- β par paires : 130 à 380 m⁻¹ selon la paire ; la plupart des paires ont un creux ≤ 0 (fissures séparées, pas d'information sur β). Dispersion d'un facteur ≈ 3 : β n'est pas déterminé précisément.
- Le RMSE diminue de façon monotone quand λ augmente jusqu'à la borne 500 m⁻¹ : les pics sont limités par la jauge de 5 mm, λ n'est pas identifiable par l'ajustement.
- Σw avec λ imposé varie de ≈ 10 % sur s14 (199 à 231 µm) et s15 (344 à 373 µm), de ≈ 20 % sur C34 s14 (256 à 318 µm) ; sur s4 elle passe de 28 à 53 µm (contre 99 µm avec λ libre et 102 µm avec les gaussiennes).
- Le nombre de fissures croît avec λ (s14 : 2 à 5). À même λ, le RMSE reste supérieur à celui des gaussiennes (≈ 900 µε sur s14).

Conclusion provisoire : la méthode donne un ordre de grandeur cohérent de Σw sur s14, s15 et C34 s14, mais λ est mal contraint ; un λ global ≈ 300 m⁻¹ (médiane des paires) est un choix raisonnable. Elle ne remplace pas les gaussiennes pour l'ajustement.
