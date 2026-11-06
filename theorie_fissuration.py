"""Textes de théorie de l'onglet « Fissuration » (Markdown + LaTeX), un bloc par méthode."""

COMMUN = r"""
### 0. Hypothèse de départ : décomposition additive

La fibre noyée mesure, à la position $x$ (le long de la fibre) et au temps $t$, une déformation moyennée sur la **jauge OBR de 5 mm** (pas ≈ 1,01 mm) :

$$\varepsilon_{mes}(x,t)=\varepsilon_{retrait}(x,t)+\varepsilon_{fiss}(x,t)+\eta(x,t),\qquad \varepsilon_{fiss}\ge 0$$

- $\varepsilon_{retrait}\le 0$ : contraction de la **matrice** carbonatée (décalcification des C-S-H), pilotée par la diffusion du CO₂ ;
- $\varepsilon_{fiss}\ge 0$ : allongement local de la fibre où une **fissure** s'ouvre (le retrait empêché par le cœur non carbonaté fissure la peau). Les fissures n'ouvrent jamais la fibre « vers le bas », donc le retrait est une **enveloppe basse** de la mesure ;
- $\eta$ : bruit instrumental (MAD court terme : 150 à 750 µε), dont l'effet domine l'erreur sur les petites fissures.

La stratégie est donc : (1) estimer $\varepsilon_{retrait}$ ; (2) décomposer le **résidu** $r=\varepsilon_{mes}-\varepsilon_{retrait}$ en fissures. Les deux méthodes de décomposition (gaussiennes, exponentielle) partagent **le même résidu** et la même notion d'ouverture $w$ (aire du pic) : leurs résultats sont directement comparables.

---
### 1. Retrait de carbonatation (`fit_retrait.py`)

**Diffusion-réaction du CO₂.** Le gaz diffuse depuis les 6 faces (concentration normalisée $u=c/c_0$, $c_0=1{,}25$ mol/m³ pour 3 % de CO₂, $u=1$ sur la face) avec un coefficient effectif **$D_e$ fixé** (littérature : $2$ à $7\times10^{-7}$ m²/s ; défaut $3{,}7\times10^{-7}$). La diffusion du gaz étant rapide devant la réaction, le gaz est **quasi stationnaire** :

$$D_e\,c_0\,\frac{\partial^2 u}{\partial x^2}=N\,k\,u\,(1-\alpha)$$

**Endommagement (diffusion accélérée par la fissuration).** Le coefficient effectif devient un vecteur spatial recalculé à chaque pas de temps à partir du degré de carbonatation du pas précédent :

$$D_{eff}(x,t)=D_e\,\big[1+\beta\,\max(0,\alpha-\alpha_{crit})^{m}\big]$$

où $\beta$ est l'intensité du dommage, $\alpha_{crit}$ le seuil au-delà duquel les fissures apparaissent et $m$ (fixé à 2) l'exposant de sensibilité ; $D_e$ est remplacé par $D_{eff}$ dans le terme de réaction/diffusion $\lambda=N k(1-\alpha)/(D_{eff}c_0)$. $\beta=0$ : diffusion constante. Les valeurs par défaut des scripts sont $\beta=0$ (le dommage n'améliore pas l'ajustement actuel).

**Cinétique de carbonatation** (1er ordre, consommation du réactif) :

$$\frac{\partial \alpha}{\partial t}=k\,u\,(1-\alpha)\;\Rightarrow\;\alpha^{t+\Delta t}=1-(1-\alpha^t)\,e^{-k\,u\,\Delta t}$$

- $\alpha\in[0,1]$ : degré de carbonatation local ; $N$ (mol CO₂/m³) : capacité de fixation du CO₂ ; $k$ (1/j) : constante cinétique. À $D_e$ fixé, **seul le rapport $N/D_e$ est identifiable**.
- Le front est résolu en **1D dans une demi-dalle** ($[0,L/2]$, $L=100$ mm, symétrie au centre) par différences finies ($\Delta x=1$ mm, $\Delta t=0{,}25$ j, système tridiagonal à chaque pas).
- **Combinaison 3D** (cube attaqué par ses 6 faces) : $\alpha_{3D}=1-\prod_{i=1}^{3}\big(1-\alpha_{1D}(d_i)\big)$, avec $d_i$ la distance du point à la face la plus proche selon l'axe $i$ (interpolation bilinéaire en $(t,d)$).

**Loi de retrait :**

$$\varepsilon_{retrait}(x,t)=-a\,\alpha_{3D}(x,t)^{\,n}$$

($a$ en µε : retrait à carbonatation complète ; $n$ : non-linéarité, $n\ne1$ car le retrait n'est pas proportionnel à la fraction carbonatée).

**Ajustement de l'enveloppe basse** (moindres carrés asymétriques sur $(N,k,a,n,\beta,\alpha_{crit})$, 3 départs ; $N,k,\beta,\alpha_{crit}$ optimisés en espace logarithmique) : avec $e=\varepsilon_{retrait}-\varepsilon_{mes}$,

$$\min_{N,k,a,n}\sum_j \rho(e_j),\qquad \rho(e)=\begin{cases}e^2 & e\ge0\ (\text{mesure sous le modèle})\\ W\,e^2 & e<0\ (\text{mesure au-dessus : fissure}),\ W=0{,}1\end{cases}$$

Les points hors du cube (bouts de fibre) sont ignorés ; les segments C12 s6–s12 (peu contraints, proches de 0) sont exclus. Les paramètres du fit C12 sont **validés** sur C34.
"""

GAUSS = r"""
### Décomposition en gaussiennes (`fissuration_gaussiennes.py`)

**Idée.** Une fissure donne sur la fibre un pic localisé ; on le représente par la forme la plus simple et la plus souple : une **gaussienne**. Le modèle est volontairement *phénoménologique* (aucune hypothèse sur le transfert fibre/matrice), ce qui lui permet de s'adapter à presque n'importe quelle forme de pic, au prix d'un sens physique réduit : seule l'**aire** est interprétable.

**Modèle.** À chaque échéance, $r(x,t_k)\approx\sum_i A_i(t_k)\exp\!\big(-\tfrac{(x-c_i)^2}{2\sigma_i(t_k)^2}\big)$ : une gaussienne par fissure (ou par zone fissurée), de **centre $c_i$ constant** (une fissure ne se déplace pas), d'amplitude $A_i$ (µε) et d'écart-type $\sigma_i$ (mm). Pas de terme de fond : un décalage global est absorbé par de très larges gaussiennes.

**Ouverture.** La fibre transmet la déformation de la matrice sur une longueur de transfert : l'intégrale du pic est l'allongement cumulé, soit l'**ouverture** :

$$w_i(t)=\int \varepsilon_{fiss}\,dx = A_i\,\sigma_i\sqrt{2\pi}\quad(\mu m\text{ si }A\text{ en µε et }\sigma\text{ en mm}\times 10^{-3}),\qquad W(t)=\sum_i w_i(t)$$

**Résolution instrumentale.** Une fissure ponctuelle moyennée sur $L_g=5$ mm donne un pic d'écart-type $\ge L_g/\sqrt{12}\approx1{,}45$ mm : $\sigma_{min}$ = 1,45 mm (évite les pics « de bruit » ultra étroits). La **largeur d'un pic n'est pas une propriété de la fissure** (elle dépend de la jauge et du transfert fibre/matrice). $\sigma_{max}=25$ mm.

**Algorithme.**
1. **Calage** à la date $t_{cal}$ (réglable ; par défaut la dernière échéance, la mieux résolue) : ajout **glouton** d'une gaussienne au maximum du résidu non expliqué, tant que le gain relatif de SSE $\ge$ `ADD_GAIN`, que son amplitude $\ge$ `A_MIN`, que le pic dépasse `SEUIL_RMSE`×RMSE courant, que sa distance aux autres centres $\ge$ `DIST_MIN` et que le nombre $\le$ `N_MAX`. Moindres carrés non linéaires à régions de confiance (`least_squares`, méthode TRF, bornes).
2. **Suivi** : à partir de $t_{cal}$, vers les dates **antérieures** (à rebours) puis **postérieures**, les centres sont initialisés (mobilité $\pm$ `DECALAGE_MAX`) et les $A_i,\sigma_i$ réajustés, **sans nouvelle gaussienne** et **sans fusion**. Une gaussienne apparaît donc avec $A_i\to0$ avant sa date d'ouverture.
3. **Régularisation temporelle globale** : toutes les dates sont ajustées **ensemble** en minimisant

$$\sum_k\sum_x\Big[\sum_i A_i(t_k)e^{-\frac{(x-c_i)^2}{2\sigma_i(t_k)^2}}-r(x,t_k)\Big]^2+\lambda\sum_i\sum_k\big(\sigma_i(t_k)-\sigma_i(t_{k-1})\big)^2,\quad w_i(t_k)=\sum_{j\le k}\delta_{i,j},\ \delta_{i,j}\ge0$$

   donc **$w_i(t)$ non décroissant** (une fissure ne se referme pas : irréversibilité), $\sigma_i(t)$ lisse ($\lambda$ = `LAMBDA_SIGMA`), centres communs à toutes les dates. Les amplitudes sont dérivées $A_i=w_i/(\sigma_i\sqrt{2\pi})$. Jacobienne creuse (`jac_sparsity`) pour garder un calcul rapide.
4. **Regroupement a posteriori** : les gaussiennes dont les centres sont séparés de $\le$ `GROUPE_MM` (liaison simple) forment un **groupe** de fissuration **diffuse** (réseau de microfissures) ; une gaussienne isolée est une fissure **discrète**. C'est une propriété calculée après coup, pas un résultat de l'ajustement.

**Indicateurs.** $\Sigma w$ (µm) : ouverture totale cumulée du segment (la plus robuste) ; RMSE (µε) ; $w_i(t)$ : cinétique de chaque fissure ; **figure 3D** : ouverture $w_i(t)$ (µm) de chaque fissure à sa position $c_i$ en fonction du temps (une courbe par fissure, même couleur que le suivi et les ouvertures) ; une déformation (µm/mm) n'est pas une ouverture.

**Forces / faiblesses.** + Très bon ajustement (RMSE le plus bas), souple, rapide à calibrer. − Pas de physique du transfert : la largeur $\sigma$ n'a pas de sens mécanique, les pics voisins se compensent, un décalage global du retrait est absorbé par des gaussiennes très larges (fausse $\Sigma w$).
"""

EXP = r"""
### Décomposition exponentielle / shear lag (`fissuration_cisaillement.py`)

**Idée.** Au lieu d'une forme libre, on utilise le profil que **prédit la mécanique du transfert fibre ↔ matrice** (modèle de cisaillement ou *shear lag*, repris d'OBRstudio) : une fissure d'ouverture $w$ impose un saut de déplacement ; la fibre est ancrée de part et d'autre par le cisaillement de l'interface, et sa déformation décroît en s'éloignant de la fissure.

**Noyau exponentiel-triangulaire.** Pour une fissure au centre $c$ ($s=x-c$), de pic $P$, de **longueur de transfert** $1/\lambda$ (λ en m⁻¹ dans les résultats) et de **demi-largeur de cœur** $a$ (zone de glissement où le cisaillement est plafonné) :

$$\varepsilon(s)=\begin{cases}P\Big(1-\dfrac{\lambda|s|}{1+\lambda a}\Big) & |s|\le a\quad\text{(cœur : décroissance linéaire)}\\[2mm] \dfrac{P}{1+\lambda a}\,e^{-\lambda(|s|-a)} & |s|>a\quad\text{(transfert élastique : exponentielle)}\end{cases}$$

Avec $a=0$ on retrouve l'**exponentielle pure** $\varepsilon=Pe^{-\lambda|s|}$ (solution de $\varepsilon_f''=\lambda^2(\varepsilon_f-\varepsilon_m)$, $\lambda^2$ = raideur d'interface / raideur de la fibre). Le profil est continu en $|s|=a$.

**Ouverture (COD).** C'est l'aire du profil, avec $\varsigma=\lambda a$ :

$$w=10^{-3}\!\int\varepsilon\,dx=10^{-3}\,\frac{2P}{\lambda}\,\frac{1+\varsigma+\varsigma^2/2}{1+\varsigma}\ \ (\mu m),$$

et le modèle est **paramétré par $w$** (le pic $P$ en est déduit). Comme pour les gaussiennes, $w$ est l'indicateur physique ; $\Sigma w$ est directement comparable.

**Moyenne sur la jauge.** La fibre ne mesure pas $\varepsilon(s)$ mais sa moyenne sur $L_g$ (5 mm) : $\bar\varepsilon(x)=\frac1{L_g}\int_{x-L_g/2}^{x+L_g/2}\varepsilon(s)\,ds$, calculée **analytiquement** (primitive du noyau), sans discrétisation.

**Modèle complet à l'échéance $t_k$** :

$$r(x,t_k)\approx b(t_k)+\sum_i \bar\varepsilon\big(x-c_i;\,w_i(t_k),\lambda_i,a_i(t_k)\big)$$

- **Fond $b(t_k)$** (µε) : décalage global libre à chaque date (écart du modèle de retrait, fibres d'orientation Y plus hautes…), pour qu'il ne soit pas absorbé par de fausses fissures. **Désactivé par défaut** : le fond absorbe l'erreur du modèle de retrait ; sans lui, mesure = retrait + fissures.
- **Pondération asymétrique** : les points où la mesure est **sous** le modèle (creux que le modèle de fissures ne sait pas représenter) pèsent $W_{neg}<1$ ; $W_{neg}=1$ donne des moindres carrés ordinaires.
- Chaque fissure a son propre $\lambda_i$ (**constant dans le temps** : propriété de l'interface) et son cœur $a_i(t)$ (**non décroissant**).

**Algorithme.**
1. **Détection des centres.** *Mode multi-échéances* (recommandé) : `find_peaks` sur **toutes** les échéances (proéminence $=$ `PROM_MULTI`$\times$bruit de la date, bruit = MAD des différences secondes), pics reliés en **trajectoires** (même fissure si écart $<\max(1{,}5\,\text{pas},L_g/4,3\text{ mm})$) ; une trajectoire est **validée** si elle dure $\ge$ `DUREE_MIN` jours **ou** si elle est présente aux deux dernières échéances. *Mode dernière échéance* : pics de proéminence $\ge$ `PROM_FACTEUR`$\times$bruit, complétés par un ajout glouton (gain de SSE $\ge$ `ADD_GAIN`).
2. **Ouverture initiale** : aire du résidu au-dessus d'un plancher (méthode des « aires »), sur le territoire de chaque fissure (jusqu'au creux qui la sépare de sa voisine). Le plancher est le quart bas des creux du profil lissé (`plancher = local`) ou 0.
3. **Affinage à la dernière échéance** : moindres carrés sur $(w,\lambda,a,b,c)$ avec bornes ; **élagage** : fusion des centres plus proches que `DIST_MIN` (la plus faible est retirée) et retrait des fissures de $w<$ `COD_MIN`. Fissures **forcées** (conservées) et **interdites** (exclues) possibles.
4. **Suivi à rebours** : de la dernière date vers la première, centres figés, avec $w_i(t)\le w_i(t+\Delta t)$.
5. **Ajustement global** de toutes les dates ensemble : paramètres = incréments $\delta w\ge0$ (donc $w_i(t)=\sum\delta w$ **non décroissant**), $\lambda_i$ constants, incréments de $a$ $\ge0$, centres communs (mobilité $\pm$ `DECALAGE_MAX`), $b(t_k)$ libres ; jacobienne creuse composée avec la somme cumulée.
6. **Recoupement indépendant** : $w^{aires}_i$ = aire du résidu au-dessus du plancher, **sans** hypothèse de forme. Un écart important avec $w_i$ signale un modèle mal contraint (fissure fantôme, fond mal estimé).

**Indicateurs.** $\Sigma w$ (modèle) et $\Sigma w^{aires}$ ; RMSE ; $1/\lambda_i$ (longueur de transfert), $a_i$ (cœur), pic $P_i$ ; $b(t)$.

**Forces / faiblesses.** + Forme physique (le même noyau pour toutes les fissures), fond figé par défaut (mesure = retrait + fissures), irréversibilité, recoupement par aires. − **$\lambda$ et $a$ sont mal identifiables à jauge 5 mm** (λ colle souvent à sa borne, $\lambda$ et $a$ sont corrélés) ; le noyau reproduit mal les plateaux larges ; RMSE un peu plus élevé que les gaussiennes ; calcul plus long (≈ 1 min par segment) ; sensible au seuil de détection. La **limite de largeur** ($1/\lambda\le$ `LARGEUR_MAX`, 10 mm par défaut en mode multi) évite qu'une fissure « fantôme » très large absorbe la pente du fond (sinon $\Sigma w$ peut doubler).
"""

LIMITES = r"""
### Comparaison et précautions

| | Gaussiennes | Exponentielle (shear lag) |
|---|---|---|
| Forme du pic | libre (σ) | physique (λ, cœur a) |
| Fond / décalage global | absorbé par de larges gaussiennes | figé à 0 par défaut ($b(t)$ optionnel) |
| Détection des centres | glouton sur la date de calage | trajectoires de pics sur toutes les dates |
| Irréversibilité | $w_i(t)$ croissant | $w_i(t)$ et $a_i(t)$ croissants, λ constant |
| Recoupement indépendant | — | $\Sigma w^{aires}$ |
| Ajustement (RMSE s14, 113 j) | ≈ 900 µε | ≈ 1200 µε (≈ 1000 µε avec $W_{neg}=1$) |
| $\Sigma w$ (s14) | ≈ 275 µm | ≈ 300–340 µm (aires : ≈ 300 µm) |
| Temps de calcul | ≈ 1 s par échéance | ≈ 1 min par segment |
| Quand l'utiliser | décrire au mieux la forme du résidu | obtenir des paramètres à sens mécanique et un $\Sigma w$ recoupé |

- **Le résidu dépend du retrait** : un écart global du modèle (paramètre $a$ trop faible/fort, segments d'orientation Y plus hauts de ≈ 2000 µε) fausse $\Sigma w$ ; avant ~28 j le retrait est peu fiable.
- **Largeur ≠ propriété de la fissure** : à jauge de 5 mm, ni $\sigma$ ni $\lambda$ ne sont fiables ; seule l'**aire** $w$ (et $\Sigma w$) l'est. Comparer les deux méthodes sur $\Sigma w$ est le meilleur test de robustesse.
- Gaussiennes : les fissures apparues après $t_{cal}$ ne sont pas détectées. Exponentielle : une fissure qui n'est visible sur aucune date assez longtemps (< `DUREE_MIN`) est rejetée.
- Pics voisins chevauchants : seule la somme est fiable, pas la répartition entre fissures.
- Le bruit (MAD de 150 à 750 µε) et la moyenne sur la jauge limitent la détection des petites fissures.
- Le modèle de retrait est homogène : pas de gradient d'humidité (séchage), ni de fluage, ni de réaction de la portlandite.
"""
