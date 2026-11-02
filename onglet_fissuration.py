"""Onglet « Fissuration » du dashboard : lance fissuration_gaussiennes.py sur des segments/dates choisis.

Le calcul est lourd (~1 s par échéance et par segment) : rien n'est lancé avant le clic sur le bouton,
les données allégées (F.prepare) et chaque résultat sont mis en cache.
"""
import numpy as np
import pandas as pd
import streamlit as st

import fit_retrait as F
import fissuration_gaussiennes as G

SEC_PAR_ECHEANCE = 1.0   # estimation grossière du temps de calcul (régularisation globale comprise)


@st.cache_data(show_spinner="Préparation des données (une seule fois par cube)…")
def _donnees(cube):
    return F.prepare(cube, bin_mm=1)    # même pas que donnees_segment (CLI)


@st.cache_data(show_spinner="Ajustement du retrait sur le cube (~10 s)…")
def _theta(cube, de, refit):
    return tuple(F.ajuster(F.prepare(cube), de)) if refit else tuple(G.THETA_DEFAUT)


@st.cache_data(show_spinner=False)
def _calcul(cube, segment, dates, theta, de, regul):
    h = G.donnees_segment(cube, segment, np.array(theta), de, g=_donnees(cube))
    tab, hist = G.analyser(h, dates=np.array(dates), verbose=False, regul=regul)
    return G.groupes(tab), hist


def render(cube, personnalise):
    st.subheader("Fissuration : décomposition du résidu (mesure − retrait) en gaussiennes")
    st.caption("Détails de la méthode : README, section « Fissuration par décomposition en gaussiennes ». "
               "Aucun calcul n'est lancé tant que vous n'avez pas cliqué sur « Lancer le calcul ».")
    if personnalise:
        st.warning("Un fichier chargé manuellement est actif : cet onglet utilise les fichiers du dépôt "
                   f"(`{F.CSV[cube]}`).")
    g = _donnees(cube)
    segments = sorted(g["segment"].unique(), key=lambda s: int(s[1:]))
    dates = np.sort(g["temps_ecoule_j"].unique())
    dates = [float(t) for t in dates if t >= G.T0_DEFAUT]

    with st.form("form_fissuration"):
        c1, c2 = st.columns(2)
        segs = c1.multiselect("Segments", segments, default=["s14"] if "s14" in segments else segments[:1])
        t_sel = c2.multiselect("Échéances (jours, ≥ 28 j : avant, le retrait est peu fiable)", dates,
                               default=dates[::2] if len(dates) > 12 else dates,
                               format_func=lambda t: f"{t:.1f}")
        c3, c4, c5 = st.columns(3)
        regul = c3.checkbox("Régularisation temporelle (wᵢ(t) croissant, σᵢ lisse)", value=True)
        refit = c4.checkbox("Ré-ajuster le retrait sur ce cube", value=False,
                            help="Sinon : paramètres du fit C12 (validés sur C34).")
        de = c5.number_input("D_e (m²/s)", value=float(F.DE_DEFAUT), format="%.2e")
        n_ech = len(t_sel)
        st.caption(f"Estimation : ≈ {int(len(segs) * max(n_ech, 1) * SEC_PAR_ECHEANCE)} s "
                   f"({len(segs)} segment(s) × {n_ech} échéance(s)) ; les résultats déjà calculés sont réutilisés.")
        lancer = st.form_submit_button("▶️ Lancer le calcul", type="primary")

    if lancer:
        if not segs or n_ech < 2:
            st.error("Choisissez au moins un segment et deux échéances.")
        else:
            st.session_state["fissuration_jobs"] = dict(cube=cube, segs=segs, dates=tuple(sorted(t_sel)),
                                                       regul=regul, refit=refit, de=de)
    job = st.session_state.get("fissuration_jobs")
    if not job or job["cube"] != cube:
        return

    theta = _theta(cube, job["de"], job["refit"])
    st.write(f"Retrait : N={theta[0]:.0f}, k={theta[1]:.4f}, a={theta[2]:.0f}, n={theta[3]:.2f}")
    barre = st.progress(0.0)
    resultats = {}
    for k, s in enumerate(job["segs"]):
        barre.progress(k / len(job["segs"]), text=f"Segment {s} ({k + 1}/{len(job['segs'])})…")
        resultats[s] = _calcul(cube, s, job["dates"], theta, job["de"], job["regul"])
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
