"""Onglet « Fissuration » du dashboard : lance fissuration_gaussiennes.py sur des segments/dates choisis.

Le calcul est lourd (~1 s par échéance et par segment) : rien n'est lancé avant le clic sur le bouton,
les données allégées (F.prepare) et chaque résultat sont mis en cache.
"""
import threading

import numpy as np
import pandas as pd
import streamlit as st

import fit_retrait as F
import fissuration_gaussiennes as G

_VERROU = threading.Lock()   # les paramètres de fissuration sont des constantes du module : un calcul à la fois
# nom, libellé, défaut, min, max, pas
PARAMS_FISSURATION = [
    ("S_MIN", "σ min (mm) ≈ jauge/√12", G.S_MIN, 0.5, 5.0, 0.05),
    ("S_MAX", "σ max (mm)", G.S_MAX, 5.0, 50.0, 1.0),
    ("ADD_GAIN", "Gain de SSE min pour ajouter une gaussienne", G.ADD_GAIN, 0.005, 0.3, 0.005),
    ("A_MIN", "Amplitude min d'une gaussienne ajoutée (µε)", G.A_MIN, 0.0, 5000.0, 50.0),
    ("DIST_MIN", "Distance min entre centres à l'ajout (mm)", G.DIST_MIN, 0.5, 10.0, 0.5),
    ("DECALAGE_MAX", "Mobilité des centres autour de l'init (± mm)", G.DECALAGE_MAX, 0.0, 10.0, 0.5),
    ("N_MAX", "Nombre max de gaussiennes", G.N_MAX, 1, 50, 1),
    ("LAMBDA_SIGMA", "Lissage de σ(t) (λ ; régularisation)", G.LAMBDA_SIGMA, 0.0, 5000.0, 50.0),
]
GROUPE_DEFAUT = G.GROUPE_MM
SEC_PAR_ECHEANCE = 1.0   # estimation grossière du temps de calcul (régularisation globale comprise)


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


def _retrait_seul(cube, job, theta):
    import matplotlib.pyplot as plt
    ts = list(job["dates"]); cm = plt.cm.viridis(np.linspace(0, 1, len(ts)))
    lignes = []
    for s in job["segs"]:
        h = G.donnees_segment(cube, s, np.array(theta), job["de"], g=_donnees(cube))
        axe = h["axe_mesure"].iloc[0]
        fig, ax = plt.subplots(figsize=(9, 4))
        for c, t in zip(cm, ts):
            q = h[np.isclose(h["temps_ecoule_j"], t)].sort_values("xb")
            if q.empty:
                continue
            ax.plot(q[axe] * 1000, q["eps"], color=c, lw=1, label=f"{t:.0f} j")
            ax.plot(q[axe] * 1000, q["mod"], color=c, lw=2, ls="--")
            lignes.append(dict(segment=s, t_j=round(t, 1), biais_moyen_ue=round(float((q["eps"] - q["mod"]).mean())),
                               part_mesure_au_dessus=round(float((q["eps"] > q["mod"]).mean()), 2)))
        ax.set_title(f"{cube} {s} : mesure (trait plein) et retrait modèle (tirets)")
        ax.set_xlabel("position (mm)"); ax.set_ylabel("µε"); ax.grid(alpha=.3); ax.legend(fontsize=6, ncol=4)
        st.pyplot(fig); plt.close(fig)
    st.markdown("Écart mesure − retrait (le retrait est une enveloppe basse : la mesure doit être surtout au-dessus)")
    st.dataframe(pd.DataFrame(lignes), hide_index=True)


def render(cube, personnalise):
    st.subheader("Fissuration : décomposition du résidu (mesure − retrait) en gaussiennes")
    st.caption("Détails de la méthode : README, section « Fissuration par décomposition en gaussiennes ». "
               "Aucun calcul n'est lancé tant que vous n'avez pas cliqué sur « Lancer le calcul ».")
    if personnalise:
        st.warning("Un fichier chargé manuellement est actif : cet onglet utilise les fichiers du dépôt "
                   f"(`{F.CSV[cube]}`).")
    mode = st.radio("Calcul", ["Retrait seul (rapide)", "Retrait + fissuration"], horizontal=True,
                    help="Le retrait seul permet de vérifier le modèle avant de lancer l'analyse de fissuration.")
    seul = mode.startswith("Retrait seul")
    g = _donnees(cube)
    segments = sorted(g["segment"].unique(), key=lambda s: int(s[1:]))
    dates = [float(t) for t in np.sort(g["temps_ecoule_j"].unique()) if seul or t >= G.T0_DEFAUT]

    with st.form("form_fissuration"):
        c1, c2 = st.columns(2)
        segs = c1.multiselect("Segments", segments, default=["s14"] if "s14" in segments else segments[:1])
        t_sel = c2.multiselect("Échéances (jours)" if seul else "Échéances (jours, ≥ 28 j : avant, le retrait est peu fiable)",
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
                vals["GROUPE_MM"] = cols[2].number_input("Regroupement diffus : écart max des centres (mm)",
                                                         min_value=1.0, max_value=30.0, value=float(GROUPE_DEFAUT), step=1.0)
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
            st.session_state["fissuration_jobs"] = dict(cube=cube, segs=segs, dates=tuple(sorted(t_sel)),
                                                       regul=regul, refit=refit, de=de, seul=seul, theta=tuple(th),
                                                       groupe=vals.get("GROUPE_MM", GROUPE_DEFAUT),
                                                       params=tuple((k, vals[k]) for k, *_ in PARAMS_FISSURATION if k in vals))
    job = st.session_state.get("fissuration_jobs")
    if not job or job["cube"] != cube or job["seul"] != seul:
        return

    theta = _theta(cube, job["de"], job["refit"], job["theta"])
    st.write(f"Retrait : N={theta[0]:.0f}, k={theta[1]:.4f}, a={theta[2]:.0f}, n={theta[3]:.2f}")
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
        synth.append(dict(segment=s, gaussiennes=fin["id"].nunique(), groupes=fin["groupe"].nunique(),
                          **{"Σw final (µm)": round(fin["w_um"].sum())}))
    st.markdown("**Synthèse**")
    st.dataframe(pd.DataFrame(synth), hide_index=True)

    for s, (tab, hist) in resultats.items():
        with st.expander(f"Segment {s}", expanded=len(resultats) == 1):
            fig = G.tracer(hist, cube, s, None)
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            fin = tab[tab.t_j == tab.t_j.max()]
            grp = fin.groupby("groupe").agg(X_min_mm=("pos_mm", "min"), X_max_mm=("pos_mm", "max"),
                                            gaussiennes=("id", "count"), w_um=("w_um", "sum")).round(1)
            st.markdown("Groupes de gaussiennes voisines (dernière échéance)")
            st.dataframe(grp)
            st.line_chart(tab.groupby("t_j")["w_um"].sum().rename("Σw (µm)"))
            st.download_button("⬇️ CSV", tab.to_csv(index=False).encode(), file_name=f"fissuration_{cube}_{s}.csv",
                               mime="text/csv", key=f"dl_fiss_{s}")
