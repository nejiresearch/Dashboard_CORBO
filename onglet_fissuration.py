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
**Principe.** La déformation mesurée par la fibre est la somme du **retrait de la matrice** et de
l'**ouverture des fissures** (≥ 0) : on modélise le retrait, puis on attribue le reste (le *résidu*) aux fissures.

**1. Retrait de carbonatation (réaction-diffusion couplé, `fit_retrait.py`).** Le CO₂ diffuse depuis les
6 faces (coefficient effectif $D_e$ fixé) et est consommé par la réaction. Avec $u=c/c_0$ et $\alpha$ le degré de carbonatation :

$$D_e\,c_0\,u'' = N\,k\,u\,(1-\alpha), \qquad \frac{\partial\alpha}{\partial t} = k\,u\,(1-\alpha), \qquad
\varepsilon_{retrait} = -a\,\alpha^{\,n}$$

($N$ : capacité de fixation, $k$ : cinétique, $a$ et $n$ : loi de retrait). Le retrait est ajusté sur l'**enveloppe basse**
des mesures (les fissures ne peuvent qu'*augmenter* la déformation).

**2. Résidu et gaussiennes (`fissuration_gaussiennes.py`).** Le résidu $r(x,t)=\varepsilon_{mesuré}-\varepsilon_{retrait}$ est
décomposé en pics gaussiens positifs :

$$r(x,t)\approx\sum_i A_i(t)\,\exp\!\Big(-\frac{(x-c_i)^2}{2\sigma_i(t)^2}\Big),\qquad
w_i = A_i\,\sigma_i\sqrt{2\pi}$$

La fibre moyenne la déformation sur une **jauge de 5 mm** : une fissure apparaît comme un pic étalé, mais son **aire**
reste égale à l'ouverture $w$ (µm si $A$ en µε et $\sigma$ en mm). D'où $\sigma_{min}\approx 5/\sqrt{12}\approx1{,}45$ mm.
Chaque gaussienne a un **centre $c_i$ constant** (une fissure ne se déplace pas) et n'est jamais supprimée.

**3. Algorithme.** (a) À la **dernière échéance** (signal le mieux résolu), ajout glouton de gaussiennes tant que le SSE baisse
d'au moins `ADD_GAIN` ; (b) suivi **à rebours** : centres figés, $A_i$ et $\sigma_i$ réajustés ; (c) **régularisation temporelle** :
ajustement global de toutes les dates avec $w_i(t)$ non décroissant et $\sigma_i(t)$ lisse ; (d) regroupement des
gaussiennes voisines (fissuration **diffuse** = groupe, **discrète** = gaussienne isolée).

**Limites.** Un retrait mal estimé est absorbé par de larges gaussiennes (σ → σ_max) ; seule la somme Σw est robuste quand
des pics se chevauchent ; analyse conseillée à partir de 28 j.
"""


@st.cache_data(show_spinner="Préparation des données (une seule fois par cube)…")
def _donnees(cube):
    return F.prepare(cube, bin_mm=1)    # même pas que donnees_segment (CLI)


@st.cache_data(show_spinner="Ajustement du retrait sur le cube (~10 s)…")
def _theta(cube, de, refit, manuel):
    return tuple(F.ajuster(F.prepare(cube), de)) if refit else tuple(manuel)


@st.cache_data(show_spinner=False)
def _calcul(cube, segment, dates, theta, de, regul, params, groupe_mm):
    h = G.donnees_segment(cube, segment, np.array(theta), de, g=_donnees(cube))
    with _VERROU:
        avant = {k: getattr(G, k) for k, _ in params}
        try:
            for k, v in params:
                setattr(G, k, v)
            tab, hist = G.analyser(h, dates=np.array(dates), verbose=False, regul=regul)
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
    dates = [float(t) for t in np.sort(g["temps_ecoule_j"].unique()) if seul or t >= G.T0_DEFAUT]

    with st.form("form_fissuration"):
        st.markdown("**Segments** (par plan de profondeur et orientation de la fibre)")
        segs = _picker(seg_info, disponibles, comp_label, cube)
        t_sel = st.multiselect("Échéances (jours)" if seul else "Échéances (jours, ≥ 28 j : avant, le retrait est peu fiable)",
                               dates, default=dates[::6] if seul else (dates[::2] if len(dates) > 12 else dates),
                               format_func=lambda t: f"{t:.1f}")
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
            st.session_state["fissuration_jobs"] = dict(
                cube=cube, segs=segs, dates=tuple(sorted(t_sel)), regul=regul, refit=refit, de=de, seul=seul,
                theta=tuple(th), groupe=vals.get("GROUPE_MM", GROUPE_DEFAUT),
                params=tuple((k, vals[k]) for k, *_ in PARAMS_FISSURATION if k in vals))
    job = st.session_state.get("fissuration_jobs")
    if not job or job["cube"] != cube or job["seul"] != seul:
        return

    theta = _theta(cube, job["de"], job["refit"], job["theta"])
    st.write(f"Retrait : N={theta[0]:.0f}, k={theta[1]:.4f}, a={theta[2]:.0f}, n={theta[3]:.2f}")
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
        resultats[s] = _calcul(cube, s, job["dates"], theta, job["de"], job["regul"], job["params"], job["groupe"])
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
            fin = tab[tab.t_j == tab.t_j.max()]
            grp = fin.groupby("groupe").agg(X_min_mm=("pos_mm", "min"), X_max_mm=("pos_mm", "max"),
                                            gaussiennes=("id", "count"), w_um=("w_um", "sum")).round(1)
            st.markdown("Groupes de gaussiennes voisines (dernière échéance)")
            st.dataframe(grp)
            st.download_button("⬇️ CSV", tab.to_csv(index=False).encode(),
                               file_name=f"fissuration_{cube}_{s}.csv", mime="text/csv", key=f"dl_fiss_{s}")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_fissuration(cube, job["segs"], job["dates"], theta, job["de"], False, job["regul"],
                                   job["params"], job["groupe"]), language="python")
