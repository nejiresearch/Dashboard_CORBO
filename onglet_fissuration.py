"""Onglet « Fissuration » du dashboard : lance fissuration_gaussiennes.py sur des segments/dates choisis.

Le calcul est lourd (~1 s par échéance et par segment) : rien n'est lancé avant le clic sur le bouton,
les données allégées (F.prepare) et chaque résultat sont mis en cache.
Les figures sont des figures Plotly construites par fissuration_gaussiennes.py (même code que le script exporté).
"""
import threading

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fit_retrait as F
import fissuration_gaussiennes as G
import script_generator as sg

_VERROU = threading.Lock()   # les paramètres de fissuration sont des constantes du module : un calcul à la fois
# nom, libellé, défaut, min, max, pas
PARAMS_FISSURATION = [
    ("S_MIN", "σ min (mm) ≈ jauge/√12", G.S_MIN, 0.5, 5.0, 0.05),
    ("S_MAX", "σ max (mm)", G.S_MAX, 5.0, 50.0, 1.0),
    ("ADD_GAIN", "Gain de SSE min pour ajouter une gaussienne", G.ADD_GAIN, 0.005, 0.3, 0.005),
    ("SEUIL_RMSE", "Pic candidat si > k × RMSE courant (0 = désactivé)", G.SEUIL_RMSE, 0.0, 5.0, 0.1),
    ("A_MIN", "Amplitude min d'une gaussienne ajoutée (µε)", G.A_MIN, 0.0, 5000.0, 50.0),
    ("DIST_MIN", "Distance min entre centres à l'ajout (mm)", G.DIST_MIN, 0.0, 10.0, 0.5),
    ("DECALAGE_MAX", "Mobilité des centres autour de l'init (± mm)", G.DECALAGE_MAX, 0.0, 10.0, 0.5),
    ("N_MAX", "Nombre max de gaussiennes", G.N_MAX, 1, 50, 1),
    ("LAMBDA_SIGMA", "Lissage de σ(t) (λ ; régularisation)", G.LAMBDA_SIGMA, 0.0, 5000.0, 50.0),
]
GROUPE_DEFAUT = G.GROUPE_MM
SEC_PAR_ECHEANCE = 1.0   # estimation grossière du temps de calcul (régularisation globale comprise)
PLANS = [8, 16, 30, None]

THEORIE = r"""
### 0. Hypothèse de départ : décomposition additive

La fibre noyée mesure, à la position $x$ (le long de la fibre) et au temps $t$, une déformation moyennée sur la **jauge OBR de 5 mm** (pas ≈ 1,01 mm) :

$$\varepsilon_{mes}(x,t)=\varepsilon_{retrait}(x,t)+\varepsilon_{fiss}(x,t)+\eta(x,t),\qquad \varepsilon_{fiss}\ge 0$$

- $\varepsilon_{retrait}\le 0$ : contraction de la **matrice** carbonatée (décalcification des C-S-H), pilotée par la diffusion du CO₂ ;
- $\varepsilon_{fiss}\ge 0$ : allongement local de la fibre où une **fissure** s'ouvre (le retrait empêché par le cœur non carbonaté fissure la peau). Les fissures n'ouvrent jamais la fibre « vers le bas », donc le retrait est une **enveloppe basse** de la mesure ;
- $\eta$ : bruit instrumental (MAD court terme : 150 à 750 µε), dont l'effet domine l'erreur sur les petites fissures.

La stratégie est donc : (1) estimer $\varepsilon_{retrait}$ ; (2) décomposer le **résidu** $r=\varepsilon_{mes}-\varepsilon_{retrait}$ en fissures.

---
### 1. Retrait de carbonatation (`fit_retrait.py`)

**Diffusion-réaction du CO₂.** Le gaz diffuse depuis les 6 faces (concentration normalisée $u=c/c_0$, $c_0=1{,}25$ mol/m³ pour 3 % de CO₂, $u=1$ sur la face) avec un coefficient effectif **$D_e$ fixé** (littérature : $2$ à $7\times10^{-7}$ m²/s ; défaut $3{,}7\times10^{-7}$). La diffusion du gaz étant rapide devant la réaction, le gaz est **quasi stationnaire** :

$$D_e\,c_0\,\frac{\partial^2 u}{\partial x^2}=N\,k\,u\,(1-\alpha)$$

**Cinétique de carbonatation** (1er ordre, consommation du réactif) :

$$\frac{\partial \alpha}{\partial t}=k\,u\,(1-\alpha)\;\Rightarrow\;\alpha^{t+\Delta t}=1-(1-\alpha^t)\,e^{-k\,u\,\Delta t}$$

- $\alpha\in[0,1]$ : degré de carbonatation local ; $N$ (mol CO₂/m³) : capacité de fixation du CO₂ ; $k$ (1/j) : constante cinétique. À $D_e$ fixé, **seul le rapport $N/D_e$ est identifiable**.
- Le front est résolu en **1D dans une demi-dalle** ($[0,L/2]$, $L=100$ mm, symétrie au centre) par différences finies ($\Delta x=1$ mm, $\Delta t=0{,}25$ j, système tridiagonal à chaque pas).
- **Combinaison 3D** (cube attaqué par ses 6 faces) : $\alpha_{3D}=1-\prod_{i=1}^{3}\big(1-\alpha_{1D}(d_i)\big)$, avec $d_i$ la distance du point à la face la plus proche selon l'axe $i$ (interpolation bilinéaire en $(t,d)$).

**Loi de retrait :**

$$\varepsilon_{retrait}(x,t)=-a\,\alpha_{3D}(x,t)^{\,n}$$

($a$ en µε : retrait à carbonatation complète ; $n$ : non-linéarité, $n\ne1$ car le retrait n'est pas proportionnel à la fraction carbonatée).

**Ajustement de l'enveloppe basse** (moindres carrés asymétriques sur $(N,k,a,n)$, 3 départs) : avec $e=\varepsilon_{retrait}-\varepsilon_{mes}$,

$$\min_{N,k,a,n}\sum_j \rho(e_j),\qquad \rho(e)=\begin{cases}e^2 & e\ge0\ (\text{mesure sous le modèle})\ W\,e^2 & e<0\ (\text{mesure au-dessus : fissure}),\ W=0{,}1\end{cases}$$

Les points hors du cube (bouts de fibre) sont ignorés ; les segments C12 s6–s12 (peu contraints, proches de 0) sont exclus. Les paramètres du fit C12 sont **validés** sur C34.

---
### 2. Fissures : décomposition du résidu en gaussiennes (`fissuration_gaussiennes.py`)

**Modèle.** À chaque échéance, $r(x,t_k)\approx\sum_i A_i(t_k)\exp\!\big(-\tfrac{(x-c_i)^2}{2\sigma_i(t_k)^2}\big)$ : une gaussienne par fissure (ou par zone fissurée), de **centre $c_i$ constant** (une fissure ne se déplace pas), d'amplitude $A_i$ (µε) et d'écart-type $\sigma_i$ (mm).

**Ouverture.** La fibre transmet la déformation de la matrice sur une longueur de transfert : l'intégrale du pic est l'allongement cumulé, soit l'**ouverture** :

$$w_i(t)=\int \varepsilon_{fiss}\,dx = A_i\,\sigma_i\sqrt{2\pi}\quad(\mu m\text{ si }A\text{ en µε et }\sigma\text{ en mm}\times 10^{-3}),\qquad W(t)=\sum_i w_i(t)$$

**Résolution instrumentale.** Une fissure ponctuelle moyennée sur $L_g=5$ mm donne un pic d'écart-type $\ge L_g/\sqrt{12}\approx1{,}45$ mm : $\sigma_{min}$ = 1,45 mm (évite les pics « de bruit » ultra étroits). La **largeur d'un pic n'est pas une propriété de la fissure** (elle dépend de la jauge et du transfert fibre/matrice) : seule l'**aire** $w$ est physiquement interprétable. $\sigma_{max}=25$ mm.

**Algorithme.**
1. **Calage** à la date $t_{cal}$ (réglable ; par défaut la dernière échéance, la mieux résolue) : ajout **glouton** d'une gaussienne au maximum du résidu non expliqué, tant que le gain relatif de SSE $\ge$ `ADD_GAIN`, que son amplitude $\ge$ `A_MIN`, que le pic dépasse `SEUIL_RMSE`×RMSE courant, que sa distance aux autres centres $\ge$ `DIST_MIN` et que le nombre $\le$ `N_MAX`. Moindres carrés non linéaires à régions de confiance (`least_squares`, méthode TRF, bornes).
2. **Suivi** : à partir de $t_{cal}$, vers les dates **antérieures** (à rebours) puis **postérieures**, les centres sont initialisés (mobilité $\pm$ `DECALAGE_MAX`) et les $A_i,\sigma_i$ réajustés, **sans nouvelle gaussienne** et **sans fusion**. Une gaussienne apparaît donc avec $A_i\to0$ avant sa date d'ouverture.
3. **Régularisation temporelle globale** (option D) : toutes les dates sont ajustées **ensemble** en minimisant

$$\sum_k\sum_x\Big[\sum_i A_i(t_k)e^{-\frac{(x-c_i)^2}{2\sigma_i(t_k)^2}}-r(x,t_k)\Big]^2+\lambda\sum_i\sum_k\big(\sigma_i(t_k)-\sigma_i(t_{k-1})\big)^2,\quad w_i(t_k)=\sum_{j\le k}\delta_{i,j},\ \delta_{i,j}\ge0$$

   donc **$w_i(t)$ non décroissant** (une fissure ne se referme pas : irréversibilité), $\sigma_i(t)$ lisse ($\lambda$ = `LAMBDA_SIGMA`), centres communs à toutes les dates. Les amplitudes sont dérivées $A_i=w_i/(\sigma_i\sqrt{2\pi})$. Jacobienne creuse (`jac_sparsity`) pour garder un calcul rapide.
4. **Regroupement a posteriori** : les gaussiennes dont les centres sont séparés de $\le$ `GROUPE_MM` (liaison simple) forment un **groupe** de fissuration **diffuse** (réseau de microfissures) ; une gaussienne isolée est une fissure **discrète**. C'est une propriété calculée après coup, pas un résultat de l'ajustement.

**Indicateurs.** $\Sigma w$ (µm) : ouverture totale cumulée du segment (la plus robuste) ; RMSE (µε) : qualité de l'ajustement ; $w_i(t)$ : cinétique de chaque fissure ; **surface 3D** : $\varepsilon_{fiss}\cdot10^{-3}=\sum_iA_ie^{-\ldots}\ \ (\mu m/mm)$ en fonction de $(t,x)$, d'intégrale en $x$ égale à $W(t)$.

---
### 3. Limites et précautions
- **Le résidu dépend du retrait** : un écart global du modèle (paramètre $a$ trop faible/fort, segments d'orientation Y plus hauts de ≈ 2000 µε) est absorbé par de très larges gaussiennes ($\sigma\to\sigma_{max}$) et fausse $\Sigma w$ ; avant ~28 j le retrait est peu fiable.
- Le **calage** fixe le nombre et les positions : les fissures apparues après $t_{cal}$ ne sont pas détectées ; les fissures présentes avant et refermées/élargies (hors modèle) sont mal représentées. Caler tard donne le meilleur résolu.
- Pics voisins chevauchants : seule la somme $\Sigma w$ est fiable, pas la répartition entre gaussiennes.
- Le bruit (MAD court terme de 150 à 750 µε) et la moyenne sur la jauge limitent la détection des petites fissures.
- Le modèle de retrait est homogène : pas de gradient d'humidité (séchage), ni de fluage, ni de réaction de la portlandite (gonflement léger).
"""


@st.cache_data(show_spinner="Préparation des données (une seule fois par cube)…")
def _donnees(cube):
    return F.prepare(cube, bin_mm=1)    # même pas que donnees_segment (CLI)


@st.cache_data(show_spinner="Ajustement du retrait sur le cube (~10 s)…")
def _theta(cube, de, refit, manuel):
    return tuple(F.ajuster(F.prepare(cube), de)) if refit else tuple(manuel)


@st.cache_data(show_spinner=False)
def _calcul(cube, segment, dates, theta, de, regul, params, groupe_mm, t_calage):
    h = G.donnees_segment(cube, segment, np.array(theta), de, g=_donnees(cube))
    with _VERROU:
        avant = {k: getattr(G, k) for k, _ in params}
        try:
            for k, v in params:
                setattr(G, k, v)
            tab, hist = G.analyser(h, t0=0.0, dates=np.array(dates), verbose=False, regul=regul, t_calage=t_calage)
        finally:
            for k, v in avant.items():
                setattr(G, k, v)
    return G.groupes(tab, ecart_mm=groupe_mm), hist


def _picker(seg_info, disponibles, comp_label, cube):
    """Cases à cocher groupées par plan (lignes) et orientation (colonnes), comme l'onglet 10."""
    selected = []
    for plan in PLANS:
        st.markdown(f"**{'Axe Z traversant' if plan is None else f'Plan {plan} mm'}**")
        for col, axe in zip(st.columns(3), "XYZ"):
            segs = [s for s, r in seg_info.iterrows()
                    if s in disponibles and r["axe_mesure"] == axe
                    and ((pd.isna(r["plan_profondeur_mm"]) and plan is None)
                         or (plan is not None and r["plan_profondeur_mm"] == plan))]
            segs = sorted(segs, key=lambda s: int(s[1:]))
            with col:
                if not segs:
                    continue
                st.caption(comp_label[axe])
                tous = st.checkbox("Tous", key=f"fiss_all_{cube}_{plan}_{axe}")
                for s in segs:
                    if st.checkbox(s, value=(s == "s14"), key=f"fiss_{cube}_{s}") or tous:
                        selected.append(s)
    return list(dict.fromkeys(selected))


def _retrait_seul(cube, job, theta):
    ts = list(job["dates"]); pal = px.colors.sequential.Viridis
    lignes = []
    for s in job["segs"]:
        h = G.donnees_segment(cube, s, np.array(theta), job["de"], g=_donnees(cube))
        axe = h["axe_mesure"].iloc[0]
        fig = go.Figure()
        for i, t in enumerate(ts):
            q = h[np.isclose(h["temps_ecoule_j"], t)].sort_values(axe)
            if q.empty:
                continue
            c = pal[int(i / max(len(ts) - 1, 1) * (len(pal) - 1))]
            fig.add_trace(go.Scatter(x=q[axe] * 1000, y=q["eps"], mode="lines", line=dict(color=c, width=1.2),
                                     name=f"{t:.0f} j", legendgroup=f"{t:.1f}"))
            fig.add_trace(go.Scatter(x=q[axe] * 1000, y=q["mod"], mode="lines",
                                     line=dict(color=c, width=2.5, dash="dash"),
                                     name=f"{t:.0f} j (modèle)", legendgroup=f"{t:.1f}", showlegend=False))
            lignes.append(dict(segment=s, t_j=round(t, 1), biais_moyen_ue=round(float((q["eps"] - q["mod"]).mean())),
                               part_mesure_au_dessus=round(float((q["eps"] > q["mod"]).mean()), 2)))
        fig.update_layout(title=f"{cube} {s} — mesure (trait plein) et retrait modèle (tirets)",
                          xaxis_title="Position le long de la fibre (mm)", yaxis_title="µdef", height=480,
                          legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch", key=f"plot_retrait_{s}")
    st.markdown("Écart mesure − retrait (le retrait est une enveloppe basse : la mesure doit être surtout au-dessus)")
    st.dataframe(pd.DataFrame(lignes), hide_index=True)


def render(cube, personnalise, df_full, make_multi_cube, comp_label):
    st.subheader("Fissuration : retrait et décomposition du résidu en gaussiennes")
    with st.expander("📖 Théorie du modèle", expanded=False):
        st.markdown(THEORIE)
    st.caption("Aucun calcul n'est lancé tant que vous n'avez pas cliqué sur « Lancer le calcul ».")
    if personnalise:
        st.warning("Un fichier chargé manuellement est actif : cet onglet utilise les fichiers du dépôt "
                   f"(`{F.CSV[cube]}`).")
    mode = st.radio("Calcul", ["Retrait seul (rapide)", "Retrait + fissuration"], horizontal=True,
                    help="Le retrait seul permet de vérifier le modèle avant de lancer l'analyse de fissuration.")
    seul = mode.startswith("Retrait seul")
    g = _donnees(cube)
    disponibles = set(g["segment"].unique())
    seg_info = df_full.drop_duplicates("segment").set_index("segment")[["axe_mesure", "plan_profondeur_mm"]]
    dates = [float(t) for t in np.sort(g["temps_ecoule_j"].unique()) ]

    with st.form("form_fissuration"):
        st.markdown("**Segments** (par plan de profondeur et orientation de la fibre)")
        segs = _picker(seg_info, disponibles, comp_label, cube)
        t_sel = st.multiselect("Échéances (jours ; le retrait est moins fiable avant ~28 j)",
                               dates, default=[t for t in dates[::6]] if seul else [t for t in dates if t >= G.T0_DEFAUT][::2],
                               format_func=lambda t: f"{t:.1f}")
        t_cal_sel = st.select_slider("Date de calage des gaussiennes (j) : nombre et positions des fissures déterminés à cette date",
                                     options=dates, value=dates[-1], format_func=lambda t: f"{t:.1f}",
                                     disabled=seul, help="Ramenée à l'échéance sélectionnée la plus proche. Suivi vers les dates antérieures "
                                     "(à rebours) puis postérieures (centres figés, sans nouvelle fissure).")
        c3, c4, c5 = st.columns(3)
        regul = c3.checkbox("Régularisation temporelle (wᵢ(t) croissant, σᵢ lisse)", value=True, disabled=seul)
        refit = c4.checkbox("Ré-ajuster le retrait sur ce cube", value=False,
                            help="Sinon : paramètres du fit C12 (validés sur C34).")
        de = c5.number_input("D_e (m²/s)", value=float(F.DE_DEFAUT), format="%.2e")
        with st.expander("⚙️ Paramètres avancés (retrait et fissuration)"):
            st.markdown("**Retrait** (ignoré si « Ré-ajuster » est coché)")
            r1, r2, r3, r4 = st.columns(4)
            th = [r1.number_input("N", value=float(G.THETA_DEFAUT[0]), step=100.0, format="%.0f"),
                  r2.number_input("k (1/j)", value=float(G.THETA_DEFAUT[1]), step=0.001, format="%.4f"),
                  r3.number_input("a (µε)", value=float(G.THETA_DEFAUT[2]), step=50.0, format="%.0f"),
                  r4.number_input("n", value=float(G.THETA_DEFAUT[3]), step=0.01, format="%.2f")]
            vals = {}
            if not seul:
                st.markdown("**Fissuration**")
                cols = st.columns(3)
                for i, (nom, lib, d0, lo, hi, pas) in enumerate(PARAMS_FISSURATION):
                    nombre = isinstance(d0, int)
                    vals[nom] = cols[i % 3].number_input(lib, min_value=type(d0)(lo), max_value=type(d0)(hi),
                                                          value=d0, step=type(d0)(pas) if nombre else float(pas),
                                                          key=f"fiss_{nom}")
                vals["GROUPE_MM"] = cols[len(PARAMS_FISSURATION) % 3].number_input(
                    "Regroupement diffus : écart max des centres (mm)", min_value=1.0, max_value=30.0,
                    value=float(GROUPE_DEFAUT), step=1.0)
        n_ech = len(t_sel)
        if seul:
            st.caption("Retrait seul : quelques secondes (+ ~10 s si ré-ajustement).")
        else:
            st.caption(f"Estimation : ≈ {int(len(segs) * max(n_ech, 1) * SEC_PAR_ECHEANCE)} s "
                       f"({len(segs)} segment(s) × {n_ech} échéance(s)) ; les résultats déjà calculés sont réutilisés.")
        lancer = st.form_submit_button("▶️ Lancer le calcul", type="primary")

    if lancer:
        if not segs or n_ech < (1 if seul else 2):
            st.error("Choisissez au moins un segment et " + ("une échéance." if seul else "deux échéances."))
        else:
            t_cal = min(t_sel, key=lambda t: abs(t - t_cal_sel))
            st.session_state["fissuration_jobs"] = dict(
                t_calage=float(t_cal), cube=cube, segs=segs, dates=tuple(sorted(t_sel)), regul=regul, refit=refit, de=de, seul=seul,
                theta=tuple(th), groupe=vals.get("GROUPE_MM", GROUPE_DEFAUT),
                params=tuple((k, vals[k]) for k, *_ in PARAMS_FISSURATION if k in vals))
    job = st.session_state.get("fissuration_jobs")
    if not job or job["cube"] != cube or job["seul"] != seul:
        return

    theta = _theta(cube, job["de"], job["refit"], job["theta"])
    st.write(f"Retrait : N={theta[0]:.0f}, k={theta[1]:.4f}, a={theta[2]:.0f}, n={theta[3]:.2f}"
             + ("" if seul else f" · gaussiennes calées à {job['t_calage']:.1f} j"))
    _, col_cube, _ = st.columns([1, 1, 1])
    with col_cube:
        st.plotly_chart(make_multi_cube(df_full, tuple(job["segs"]), ()), width="stretch", key="cube_tab11")
        st.caption("🔴 segments analysés")

    if seul:
        _retrait_seul(cube, job, theta)
        return

    barre = st.progress(0.0)
    resultats = {}
    for k, s in enumerate(job["segs"]):
        barre.progress(k / len(job["segs"]), text=f"Segment {s} ({k + 1}/{len(job['segs'])})…")
        resultats[s] = _calcul(cube, s, job["dates"], theta, job["de"], job["regul"], job["params"], job["groupe"],
                                job["t_calage"])
    barre.empty()

    synth = []
    for s, (tab, hist) in resultats.items():
        fin = tab[tab.t_j == tab.t_j.max()]
        h_fin = hist[max(hist)]
        rmse = float(np.sqrt(h_fin[3] / len(h_fin[0])))
        synth.append(dict(segment=s, gaussiennes=fin["id"].nunique(), groupes=fin["groupe"].nunique(),
                          **{"Σw final (µm)": round(fin["w_um"].sum()), "RMSE final (µε)": round(rmse)}))
    st.markdown("**Synthèse**")
    st.dataframe(pd.DataFrame(synth), hide_index=True)

    for s, (tab, hist) in resultats.items():
        with st.expander(f"Segment {s}", expanded=len(resultats) == 1):
            ts = sorted(hist)
            t_aff = st.select_slider("Échéance affichée (j)", options=ts, value=ts[-1],
                                     format_func=lambda t: f"{t:.1f}", key=f"fiss_t_{s}")
            st.plotly_chart(G.figure_profil(hist, t_aff, cube, s), width="stretch", key=f"fiss_prof_{s}")
            st.plotly_chart(G.figure_ouvertures(tab, cube, s), width="stretch", key=f"fiss_w_{s}")
            st.plotly_chart(G.figure_suivi(tab, cube, s), width="stretch", key=f"fiss_suivi_{s}")
            if st.checkbox("Afficher la surface 3D (temps × position × ouverture)", key=f"fiss_3d_{s}"):
                st.plotly_chart(G.figure_surface(tab, cube, s), width="stretch", key=f"fiss_surf_{s}")
            fin = tab[tab.t_j == tab.t_j.max()]
            grp = fin.groupby("groupe").agg(X_min_mm=("pos_mm", "min"), X_max_mm=("pos_mm", "max"),
                                            gaussiennes=("id", "count"), w_um=("w_um", "sum")).round(1)
            st.markdown("Groupes de gaussiennes voisines (dernière échéance)")
            st.dataframe(grp)
            st.download_button("⬇️ CSV", tab.to_csv(index=False).encode(),
                               file_name=f"fissuration_{cube}_{s}.csv", mime="text/csv", key=f"dl_fiss_{s}")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_fissuration(cube, job["segs"], job["dates"], theta, job["de"], False, job["regul"],
                                   job["params"], job["groupe"], job["t_calage"]), language="python")
