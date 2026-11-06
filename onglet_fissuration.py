"""Onglet « Fissuration » du dashboard : lance fissuration_gaussiennes.py sur des segments/dates choisis.

Le calcul est lourd (~1 s par échéance et par segment) : rien n'est lancé avant le clic sur le bouton,
les données allégées (F.prepare) et chaque résultat sont mis en cache.
Les figures sont des figures Plotly construites par fissuration_gaussiennes.py (même code que le script exporté).
"""
import threading
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fit_retrait as F
import fissuration_cisaillement as E
import fissuration_gaussiennes as G
import script_generator as sg
import theorie_fissuration as T

_VERROU = threading.Lock()   # les paramètres de fissuration sont des constantes du module : un calcul à la fois
# nom, libellé, défaut, min, max, pas, aide
PARAMS_FISSURATION = [
    ("S_MIN", "σ min (mm) ≈ jauge/√12", G.S_MIN, 0.5, 5.0, 0.05, "Largeur minimale d'une gaussienne : résolution de la jauge de 5 mm (5/√12 ≈ 1,45). Plus petit : pics de bruit étroits."),
    ("S_MAX", "σ max (mm)", G.S_MAX, 5.0, 50.0, 1.0, "Largeur maximale. Plus grand : une gaussienne très large peut absorber un décalage global du retrait."),
    ("ADD_GAIN", "Gain de SSE min pour ajouter une gaussienne", G.ADD_GAIN, 0.005, 0.3, 0.005, "Une gaussienne n'est ajoutée que si elle réduit la somme des carrés d'au moins cette fraction. Plus petit : plus de gaussiennes."),
    ("SEUIL_RMSE", "Pic candidat si > k × RMSE courant (0 = désactivé)", G.SEUIL_RMSE, 0.0, 5.0, 0.1, "Seuil de détection relatif au RMSE courant du résidu. Plus petit : plus de petits pics retenus."),
    ("A_MIN", "Amplitude min d'une gaussienne ajoutée (µε)", G.A_MIN, 0.0, 5000.0, 50.0, "Amplitude minimale d'un pic ajouté (à comparer au bruit, 150–750 µε)."),
    ("DIST_MIN", "Distance min entre centres à l'ajout (mm)", G.DIST_MIN, 0.0, 10.0, 0.5, "Deux gaussiennes plus proches que cette distance ne sont pas ajoutées (séparation de pics voisins)."),
    ("DECALAGE_MAX", "Mobilité des centres autour de l'init (± mm)", G.DECALAGE_MAX, 0.0, 10.0, 0.5, "Déplacement autorisé du centre d'une gaussienne lors du suivi entre dates."),
    ("N_MAX", "Nombre max de gaussiennes", G.N_MAX, 1, 50, 1, "Plafond du nombre de gaussiennes par segment."),
    ("LAMBDA_SIGMA", "Lissage de σ(t) (λ ; régularisation)", G.LAMBDA_SIGMA, 0.0, 5000.0, 50.0, "Pénalité sur la variation de σ entre deux dates dans l'ajustement global. 0 : σ libre ; grand : σ quasi constant."),
]
GROUPE_DEFAUT = G.GROUPE_MM
GROUPE_AIDE = "Deux gaussiennes dont les centres sont plus proches que cet écart forment un groupe de fissuration diffuse (calculé après l'ajustement)."

# paramètres numériques de la méthode exponentielle : clé de analyser_segment, libellé, défaut, min, max, pas, aide
PARAMS_EXP = [
    ("prom_multi", "Proéminence des pics, mode multi (× bruit de la date)", E.PROM_MULTI, 0.5, 6.0, 0.1, "Seuil de détection d'un pic à chaque échéance, en multiple du bruit (MAD des différences secondes). Plus petit : plus de candidats (filtrés ensuite par la durée)."),
    ("duree_min", "Durée min d'une trajectoire de pic (j)", E.DUREE_MIN, 0.0, 100.0, 1.0, "Un pic n'est validé que s'il persiste au moins ce nombre de jours (ou s'il est présent aux deux dernières échéances)."),
    ("prom_facteur", "Proéminence, mode dernière échéance (× bruit)", E.PROM_FACTEUR, 0.5, 8.0, 0.1, "Seuil de détection quand le mode multi est désactivé (détection sur la dernière échéance uniquement)."),
    ("largeur_max", "Longueur de transfert max 1/λ (mm ; 0 = aucune)", E.LARGEUR_MULTI, 0.0, 300.0, 1.0, "Borne λ ≥ 1/valeur. 10 mm évite qu'une fissure « fantôme » très large absorbe la pente du fond (Σw doublé sans cette limite)."),
    ("a_max", "Demi-largeur max du cœur a (mm ; 0 = exponentielle pure)", E.A_CUBE_MM, 0.0, 50.0, 0.5, "Zone de glissement de part et d'autre de la fissure. 0 : noyau exponentiel pur (moins de paramètres, RMSE un peu plus haut). Remplacé par le preset s'il y en a un."),
    ("codmin", "Ouverture min d'une fissure (µm)", E.COD_MIN, 0.0, 50.0, 0.5, "Les fissures d'ouverture inférieure sont élaguées. Remplacé par le preset."),
    ("plafond_mm", "Ouverture max (mm)", E.PLAFOND_UM / 1000, 0.1, 10.0, 0.1, "Borne supérieure de w. Remplacé par le preset."),
    ("jauge", "Jauge OBR (mm)", E.JAUGE_MM, 1.0, 10.0, 0.5, "Largeur de moyenne spatiale de l'instrument. Borne aussi λ ≤ 5/jauge."),
    ("w_neg", "Poids des points sous le modèle", E.W_NEG, 0.0, 1.0, 0.05, "1 : moindres carrés ordinaires (meilleur RMSE) ; < 1 : les creux du résidu, que le modèle ne sait pas représenter, pèsent moins."),
    ("dist_min", "Distance min entre centres (mm)", E.DIST_MIN, 0.0, 10.0, 0.5, "Deux fissures plus proches sont fusionnées (la plus faible est retirée)."),
    ("decalage_max", "Mobilité des centres (± mm)", E.DECALAGE_MAX, 0.0, 10.0, 0.5, "Déplacement autorisé des centres autour des positions détectées."),
    ("n_max", "Nombre max de fissures", E.N_MAX, 1, 50, 1, "Plafond du nombre de fissures (complément glouton)."),
    ("w_min", "Ouverture initiale min d'un candidat (µm)", E.W_MIN, 0.5, 50.0, 0.5, "Valeur de départ minimale de w pour un candidat avant affinage."),
    ("add_gain", "Gain de SSE min du complément glouton", E.ADD_GAIN, 0.005, 0.3, 0.005, "Utilisé seulement avec le complément glouton (mode dernière échéance, ou case cochée)."),
    ("marge_bord", "Marge aux extrémités de la fibre (mm ; 0 = aucune)", E.MARGE_BORD, 0.0, 20.0, 0.5, "Exclut les fissures à moins de cette distance des bouts de fibre (effets de bord). Ne concerne pas le retrait."),
    ("tol_lambda", "Tolérance relative sur λ imposé (±)", 0.0, 0.0, 1.0, 0.05, "Avec un λ imposé ou un preset : λ varie dans [λ(1−tol), λ(1+tol)]. 0 : λ exactement imposé."),
]
PRESETS_EXP = ["aucun"] + sorted(E.PRESETS)
METHODES = ["Gaussiennes", "Exponentielle (shear lag)"]
SEC_PAR_ECHEANCE = 1.0   # estimation grossière du temps de calcul (régularisation globale comprise)
SEC_PAR_ECHEANCE_EXP = 2.5
PLANS = [8, 16, 30, None]


def _theorie_params():
    """Tableaux de paramètres (générés depuis les définitions ci-dessus : toujours synchrones avec les widgets)."""
    def tab(lignes):
        return "| Paramètre | Défaut | Plage | Effet |\n|---|---|---|---|\n" + "\n".join(
            f"| {lib} | {d0:g} | {lo:g} – {hi:g} | {aide} |" for _, lib, d0, lo, hi, _, aide in lignes)
    g = tab(PARAMS_FISSURATION) + f"\n| Regroupement diffus : écart max des centres (mm) | {GROUPE_DEFAUT:g} | 1 – 30 | {GROUPE_AIDE} |"
    e = tab(PARAMS_EXP) + (
        "\n| Détection multi-échéances | oui | case | Trajectoires de pics sur toutes les dates (sinon : dernière échéance + complément glouton). |"
        "\n| Complément glouton après le multi | non | case | Ajoute des fissures sur le résidu non expliqué (lent, ajoute des fissures de bord). |"
        "\n| Fond b(t) ajusté | non | case | Décalage global libre par date (absorbe l'erreur du modèle de retrait ; décoché : mesure = retrait + fissures). |"
        "\n| Signe des fissures | + | +, −, auto | La physique impose des pics positifs ; « auto » choisit le signe dominant (déconseillé). |"
        "\n| Plancher | local | local, zéro | Niveau du « béton sain » pour l'ouverture initiale et les aires (quart bas des creux, ou 0). |"
        "\n| Preset câble / pose | aucun | liste | Bornes de λ, a, COD d'un câble type (OBRstudio) ; remplace a max, COD min, COD max. |"
        "\n| Fissures forcées / interdites (mm) | vide | liste | Positions imposées ou exclues. |"
        "\n| λ global | libre | libre, auto, imposé | λ commun imposé (m⁻¹) ou estimé par inversion de paires de fissures (auto). |")
    return g, e


@st.cache_data(show_spinner="Préparation des données (une seule fois par cube)…")
def _donnees(cube):
    return F.prepare(cube, bin_mm=1)    # même pas que donnees_segment (CLI)


@st.cache_data(show_spinner="Ajustement du retrait sur le cube (~10 s)…")
def _theta(cube, de, refit, manuel):
    return tuple(F.ajuster(F.prepare(cube), de)) if refit else tuple(manuel)


@st.cache_data(show_spinner=False)
def _calcul(cube, segment, dates, theta, de, regul, params, groupe_mm, t_calage, _progression=None):
    h = G.donnees_segment(cube, segment, np.array(theta), de, g=_donnees(cube))
    with _VERROU:
        avant = {k: getattr(G, k) for k, _ in params}
        try:
            for k, v in params:
                setattr(G, k, v)
            tab, hist = G.analyser(h, t0=0.0, dates=np.array(dates), verbose=False, regul=regul, t_calage=t_calage,
                              progression=_progression)
        finally:
            for k, v in avant.items():
                setattr(G, k, v)
    return G.groupes(tab, ecart_mm=groupe_mm), hist


@st.cache_data(show_spinner=False)
def _calcul_exp(cube, segment, dates, theta, de, params, groupe_mm, _progression=None):
    """Méthode exponentielle : retourne (tab, hist) ou ('erreur', message)."""
    h = G.donnees_segment(cube, segment, np.array(theta), de, g=_donnees(cube))
    p = dict(params)
    for k in ("forcees", "interdites"):
        p[k] = E._liste(p.get(k, ""))
    with _VERROU:
        try:
            tab, hist, _ = E.analyser_segment(h, dates=np.array(dates), progression=_progression, **p)
        except (E.AucuneFissure, ValueError) as e:
            return "erreur", str(e)
    return E.groupes(tab, ecart_mm=groupe_mm), hist


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


def _widgets_exp():
    """Tous les paramètres de la méthode exponentielle ; retourne le dict passé à analyser_segment (+ 'groupe')."""
    v = {}
    c = st.columns(3)
    v["multi"] = c[0].checkbox("Détection multi-échéances", value=True, key="fissexp_multi",
                               help="Trajectoires de pics sur toutes les échéances (recommandé). Sinon : détection sur la dernière échéance.")
    v["glouton"] = c[1].checkbox("Complément glouton", value=False, key="fissexp_glouton",
                                 help="Ajoute des fissures sur le résidu non expliqué (lent ; ajoute souvent des fissures de bord).")
    v["fond"] = c[2].checkbox("Fond b(t) ajusté", value=False, key="fissexp_fond",
                              help="Décalage global libre à chaque date. Déconseillé : il absorbe l'erreur du modèle de retrait ; "
                                   "décoché, mesure = retrait + fissures.")
    c = st.columns(3)
    v["signe"] = c[0].selectbox("Signe des fissures", ["+", "-", "auto"], key="fissexp_signe",
                                help="La physique impose des pics positifs ; « auto » choisit le signe dominant (déconseillé).")
    v["plancher"] = c[1].selectbox("Plancher (béton sain)", ["local", "zero"], key="fissexp_plancher",
                                   help="Niveau de référence de l'ouverture initiale et des aires : quart bas des creux, ou 0.")
    v["preset"] = c[2].selectbox("Preset câble / pose", PRESETS_EXP, key="fissexp_preset",
                                 help="Bornes de λ, a et COD d'un câble type (OBRstudio). Remplace a max, COD min et COD max.")
    if v["preset"] == "aucun":
        v["preset"] = None
    cols = st.columns(3)
    for i, (nom, lib, d0, lo, hi, pas, aide) in enumerate(PARAMS_EXP):
        nombre = isinstance(d0, int)
        v[nom] = cols[i % 3].number_input(lib, min_value=type(d0)(lo), max_value=type(d0)(hi), value=d0,
                                          step=type(d0)(pas) if nombre else float(pas), key=f"fissexp_{nom}", help=aide)
    c = st.columns(3)
    mode = c[0].selectbox("λ global", ["libre", "auto", "imposé"], key="fissexp_lam_mode",
                          help="libre : un λ par fissure ; auto : médiane des β des paires de fissures ; imposé : valeur ci-contre.")
    lam = c[1].number_input("λ imposé (m⁻¹)", min_value=10.0, max_value=2000.0, value=100.0, step=10.0, key="fissexp_lam",
                            disabled=mode != "imposé")
    v["lam_global"] = None if mode == "libre" else ("auto" if mode == "auto" else float(lam))
    c = st.columns(2)
    v["forcees"] = c[0].text_input("Fissures forcées (mm, séparées par des virgules)", key="fissexp_forcees",
                                   help="Positions où une fissure est imposée.")
    v["interdites"] = c[1].text_input("Zones interdites (mm, séparées par des virgules)", key="fissexp_interdites",
                                      help="Aucune fissure à moins de ~3 mm de ces positions.")
    v["groupe"] = st.number_input("Regroupement diffus : écart max des centres (mm)", min_value=1.0, max_value=30.0,
                                  value=float(GROUPE_DEFAUT), step=1.0, key="fissexp_groupe", help=GROUPE_AIDE)
    return v


def render(cube, personnalise, df_full, make_multi_cube, comp_label):
    st.subheader("Fissuration : retrait et décomposition du résidu (gaussiennes ou exponentielle)")
    with st.expander("📖 Théorie des modèles et paramètres", expanded=False):
        p_g, p_e = _theorie_params()
        t_commun, t_g, t_e, t_cmp = st.tabs(["Retrait (commun)", "Gaussiennes", "Exponentielle (shear lag)", "Comparaison"])
        t_commun.markdown(T.COMMUN)
        t_g.markdown(T.GAUSS); t_g.markdown("#### Paramètres modifiables"); t_g.markdown(p_g)
        t_e.markdown(T.EXP); t_e.markdown("#### Paramètres modifiables"); t_e.markdown(p_e)
        t_cmp.markdown(T.LIMITES)
    st.caption("Aucun calcul n'est lancé tant que vous n'avez pas cliqué sur « Lancer le calcul ».")
    if personnalise:
        st.warning("Un fichier chargé manuellement est actif : cet onglet utilise les fichiers du dépôt "
                   f"(`{F.CSV[cube]}`).")
    mode = st.radio("Calcul", ["Retrait seul (rapide)", "Retrait + fissuration"], horizontal=True,
                    help="Le retrait seul permet de vérifier le modèle avant de lancer l'analyse de fissuration.")
    seul = mode.startswith("Retrait seul")
    methode = METHODES[0]
    if not seul:
        methode = st.radio("Méthode de décomposition du résidu", METHODES, horizontal=True,
                           help="Gaussiennes : forme libre, meilleur ajustement. Exponentielle : noyau de transfert fibre/matrice "
                                "(shear lag), fond ajusté, recoupement par aires. Détails dans l'expander « Théorie ».")
    exp = methode == METHODES[1]
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
                                     disabled=seul or exp, help="Ramenée à l'échéance sélectionnée la plus proche. Suivi vers les dates antérieures "
                                     "(à rebours) puis postérieures (centres figés, sans nouvelle fissure).")
        c3, c4, c5 = st.columns(3)
        regul = c3.checkbox("Régularisation temporelle (wᵢ(t) croissant, σᵢ lisse)", value=True, disabled=seul or exp,
                            help="Gaussiennes seulement. La méthode exponentielle ajuste toujours globalement (w, a croissants, λ constant).")
        refit = c4.checkbox("Ré-ajuster le retrait sur ce cube", value=False,
                            help="Sinon : paramètres du fit C12 (validés sur C34).")
        de = c5.number_input("D_e (m²/s)", value=float(F.DE_DEFAUT), format="%.2e")
        with st.expander("⚙️ Paramètres avancés (retrait et fissuration)"):
            st.markdown("**Retrait** (ignoré si « Ré-ajuster » est coché)")
            r1, r2, r3, r4, r5, r6 = st.columns(6)
            th = [r1.number_input("N", value=float(G.THETA_DEFAUT[0]), step=100.0, format="%.0f"),
                  r2.number_input("k (1/j)", value=float(G.THETA_DEFAUT[1]), step=0.001, format="%.4f"),
                  r3.number_input("a (µε)", value=float(G.THETA_DEFAUT[2]), step=50.0, format="%.0f"),
                  r4.number_input("n", value=float(G.THETA_DEFAUT[3]), step=0.01, format="%.2f"),
                  r5.number_input("β (dommage)", value=float(G.THETA_DEFAUT[4]), min_value=0.0, step=0.1, format="%.3g",
                                  help="D_eff = D_e·(1 + β·max(0, α − α_crit)^m), m = 2. 0 : diffusion constante."),
                  r6.number_input("α_crit", value=float(G.THETA_DEFAUT[5]), min_value=0.01, max_value=0.99, step=0.05,
                                  format="%.2f", help="Degré de carbonatation au-delà duquel la fissuration accélère la diffusion.")]
            vals, vexp = {}, {}
            if not seul and exp:
                st.markdown("**Fissuration : exponentielle (shear lag)**")
                vexp = _widgets_exp()
            elif not seul:
                st.markdown("**Fissuration : gaussiennes**")
                cols = st.columns(3)
                for i, (nom, lib, d0, lo, hi, pas, aide) in enumerate(PARAMS_FISSURATION):
                    nombre = isinstance(d0, int)
                    vals[nom] = cols[i % 3].number_input(lib, min_value=type(d0)(lo), max_value=type(d0)(hi),
                                                          value=d0, step=type(d0)(pas) if nombre else float(pas),
                                                          key=f"fiss_{nom}", help=aide)
                vals["GROUPE_MM"] = cols[len(PARAMS_FISSURATION) % 3].number_input(
                    "Regroupement diffus : écart max des centres (mm)", min_value=1.0, max_value=30.0,
                    value=float(GROUPE_DEFAUT), step=1.0, help=GROUPE_AIDE)
        n_ech = len(t_sel)
        if seul:
            st.caption("Retrait seul : quelques secondes (+ ~10 s si ré-ajustement).")
        else:
            st.caption(f"Estimation : ≈ {int(len(segs) * max(n_ech, 1) * (SEC_PAR_ECHEANCE_EXP if exp else SEC_PAR_ECHEANCE))} s "
                       f"({len(segs)} segment(s) × {n_ech} échéance(s)) ; les résultats déjà calculés sont réutilisés.")
        lancer = st.form_submit_button("▶️ Lancer le calcul", type="primary")

    if lancer:
        if not segs or n_ech < (1 if seul else 2):
            st.error("Choisissez au moins un segment et " + ("une échéance." if seul else "deux échéances."))
        else:
            t_cal = min(t_sel, key=lambda t: abs(t - t_cal_sel))
            st.session_state["fissuration_jobs"] = dict(
                t_calage=float(t_cal), cube=cube, segs=segs, dates=tuple(sorted(t_sel)), regul=regul, refit=refit, de=de, seul=seul,
                methode=methode, exp_params=vexp, theta=tuple(th),
                groupe=vexp.get("groupe", vals.get("GROUPE_MM", GROUPE_DEFAUT)),
                params=tuple((k, vals[k]) for k, *_ in PARAMS_FISSURATION if k in vals))
    job = st.session_state.get("fissuration_jobs")
    if not job or job["cube"] != cube or job["seul"] != seul or (not seul and job["methode"] != methode):
        return
    exp = job["methode"] == METHODES[1]

    theta = _theta(cube, job["de"], job["refit"], job["theta"])
    st.write(f"Retrait : N={theta[0]:.0f}, k={theta[1]:.4f}, a={theta[2]:.0f}, n={theta[3]:.2f}, "
             f"β={theta[4]:.3g}, α_crit={theta[5]:.2f}"
             + ("" if seul else (" · exponentielle (shear lag)" if exp else f" · gaussiennes calées à {job['t_calage']:.1f} j")))
    _, col_cube, _ = st.columns([1, 1, 1])
    with col_cube:
        st.plotly_chart(make_multi_cube(df_full, tuple(job["segs"]), ()), width="stretch", key="cube_tab11")
        st.caption("🔴 segments analysés")

    if seul:
        _retrait_seul(cube, job, theta)
        return

    barre = st.progress(0.0)
    resultats = {}
    nseg = len(job["segs"]); t_debut = time.time()

    def suivi(k, s):
        """Rappel d'avancement : compteur de temps écoulé et estimation du temps restant."""
        def cb(frac, texte):
            glob = (k + min(frac, 1.0)) / nseg
            ecoule = time.time() - t_debut
            reste = f" · reste ≈ {int(ecoule * (1 - glob) / glob)} s" if glob > 0.02 else ""
            barre.progress(min(glob, 1.0), text=f"Segment {s} ({k + 1}/{nseg}) : {texte} · {int(ecoule)} s écoulées{reste}")
        return cb

    for k, s in enumerate(job["segs"]):
        barre.progress(k / nseg, text=f"Segment {s} ({k + 1}/{nseg})…")
        if exp:
            p = {k2: (tuple(v) if isinstance(v, list) else v) for k2, v in job["exp_params"].items() if k2 != "groupe"}
            r = _calcul_exp(cube, s, job["dates"], theta, job["de"], tuple(sorted(p.items(), key=lambda kv: kv[0])), job["groupe"],
                           _progression=suivi(k, s))
        else:
            r = _calcul(cube, s, job["dates"], theta, job["de"], job["regul"], job["params"], job["groupe"], job["t_calage"],
                        _progression=suivi(k, s))
        if isinstance(r[0], str):
            st.warning(f"Segment {s} : {r[1]}")
        else:
            resultats[s] = r
    barre.empty()
    st.caption(f"Calcul terminé en {time.time() - t_debut:.0f} s (résultats déjà calculés réutilisés).")
    if not resultats:
        return

    synth = []
    for s, (tab, hist) in resultats.items():
        fin = tab[tab.t_j == tab.t_j.max()]
        h_fin = hist[max(hist)]
        rmse = float(np.sqrt(np.mean((h_fin[1] - h_fin[2]) ** 2))) if exp else float(np.sqrt(h_fin[3] / len(h_fin[0])))
        ligne = dict(segment=s, **{"fissures" if exp else "gaussiennes": fin["id"].nunique(), "groupes": fin["groupe"].nunique(),
                                   "Σw final (µm)": round(fin["w_um"].sum()), "RMSE final (µε)": round(rmse)})
        if exp:
            ligne["Σw aires (µm)"] = round(fin["w_aires_um"].sum())
        synth.append(ligne)
    st.markdown("**Synthèse**")
    st.dataframe(pd.DataFrame(synth), hide_index=True)

    for s, (tab, hist) in resultats.items():
        with st.expander(f"Segment {s}", expanded=len(resultats) == 1):
            ts = sorted(hist)
            t_aff = st.select_slider("Échéance affichée (j)", options=ts, value=ts[-1],
                                     format_func=lambda t: f"{t:.1f}", key=f"fiss_t_{s}")
            M = E if exp else G
            st.plotly_chart(E.figure_profil(hist, tab, t_aff, cube, s) if exp else G.figure_profil(hist, t_aff, cube, s),
                            width="stretch", key=f"fiss_prof_{s}")
            st.plotly_chart(M.figure_ouvertures(tab, cube, s), width="stretch", key=f"fiss_w_{s}")
            if exp:
                st.plotly_chart(E.figure_forme(tab, cube, s), width="stretch", key=f"fiss_forme_{s}")
            st.plotly_chart(M.figure_suivi(tab, cube, s), width="stretch", key=f"fiss_suivi_{s}")
            if st.checkbox("Afficher la surface 3D (temps × position × ouverture)", key=f"fiss_3d_{s}"):
                st.plotly_chart(M.figure_surface(tab, cube, s), width="stretch", key=f"fiss_surf_{s}")
            fin = tab[tab.t_j == tab.t_j.max()]
            if exp:
                st.markdown("Fissures à la dernière échéance")
                st.dataframe(fin[["pos_mm", "w_um", "w_aires_um", "lambda_m1", "longueur_transfert_mm", "coeur_a_mm", "pic_ue",
                                  "fond_ue", "groupe"]].round(1).rename(columns={
                    "pos_mm": "X (mm)", "w_um": "w (µm)", "w_aires_um": "w aires (µm)", "lambda_m1": "λ (m⁻¹)",
                    "longueur_transfert_mm": "1/λ (mm)", "coeur_a_mm": "cœur a (mm)", "pic_ue": "pic (µε)", "fond_ue": "fond (µε)"}),
                    hide_index=True)
            else:
                grp = fin.groupby("groupe").agg(X_min_mm=("pos_mm", "min"), X_max_mm=("pos_mm", "max"),
                                                gaussiennes=("id", "count"), w_um=("w_um", "sum")).round(1)
                st.markdown("Groupes de gaussiennes voisines (dernière échéance)")
                st.dataframe(grp)
            st.download_button("⬇️ CSV", tab.to_csv(index=False).encode(),
                               file_name=f"fissuration_{'exp' if exp else 'gauss'}_{cube}_{s}.csv", mime="text/csv", key=f"dl_fiss_{s}")

    with st.expander("📜 Script Python reproductible (notebook)"):
        if exp:
            st.code(sg.gen_fissuration_exp(cube, job["segs"], job["dates"], theta, job["de"], False, job["exp_params"]), language="python")
        else:
            st.code(sg.gen_fissuration(cube, job["segs"], job["dates"], theta, job["de"], False, job["regul"],
                                       job["params"], job["groupe"], job["t_calage"]), language="python")
