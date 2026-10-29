"""
Dashboard interactif — Déformations 3D par carbonatation (essai C12)
Lancement :  streamlit run dashboard.py

V6 : ajout d'un bouton "Générer le script Python (notebook)" dans chaque
onglet, permettant de télécharger un script Plotly autonome et reproductible
(style publication scientifique) qui recalcule tout depuis le CSV brut.
"""

import io
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import griddata

from data_prep import load_and_prepare, CUBE_SIDE_M
import script_generator as sg
import fiber_ops as fo
st.set_page_config(page_title="Carbonatation CSFS — Dashboard", layout="wide")

COMPOSANTE_LABEL = {"X": "ε_xx (axe X)", "Y": "ε_yy (axe Y)", "Z": "ε_zz (axe Z)"}

COLORMAPS_DISPO = ["RdBu_r", "Viridis", "Plasma", "Turbo", "Inferno", "Cividis", "Jet", "Picnic"]


# ============================================================== helpers UI

def get_color_range(values, manual_min, manual_max, auto):
    """Retourne (vmin, vmax) pour une échelle de couleur, en gérant les cas
    dégénérés (peu de points, valeurs constantes, tout proche de 0 en début
    d'essai) qui provoquaient le bug d'affichage 'points invisibles'."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not auto and manual_min is not None and manual_max is not None:
        return float(manual_min), float(manual_max)
    if values.size == 0:
        return -10.0, 10.0
    if values.size < 5:
        v = float(np.nanmax(np.abs(values)))
        v = v if v > 0 else 10.0
        return -v, v
    lo, hi = np.nanquantile(values, [0.03, 0.97])
    if lo == hi:
        lo, hi = lo - 10, hi + 10
    return float(lo), float(hi)


def cube_wireframe_trace(L=CUBE_SIDE_M, color="lightgray"):
    corners = np.array(list(itertools.product([0, L], [0, L], [0, L])))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a, 0], corners[b, 0], None]
        ys += [corners[a, 1], corners[b, 1], None]
        zs += [corners[a, 2], corners[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                         line=dict(color=color, width=3),
                         hoverinfo="skip", showlegend=False)


def _face_quad(idx, val, others, L):
    pts = []
    for a, b in ((0, 0), (0, L), (L, L), (L, 0)):
        p = [0.0, 0.0, 0.0]
        p[idx] = val
        p[others[0]] = a
        p[others[1]] = b
        pts.append(p)
    return np.array(pts)


def render_static_cube_png(df_all=None, highlight_seg=None, highlight_axis=None,
                            L=CUBE_SIDE_M, figsize=(1.6, 1.6)):
    """Image statique (PNG) et petite d'un cube filaire, avec option :
    - highlight_seg : surligne un segment précis (points + ligne rouge)
    - highlight_axis : ombre les 2 faces perpendiculaires à un axe (X/Y/Z)
    Axes toujours mappés de façon cohérente : X=largeur, Y=profondeur, Z=hauteur."""
    fig = plt.figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    corners = np.array(list(itertools.product([0, L], [0, L], [0, L])))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color="gray", linewidth=0.7)

    if highlight_axis is not None:
        idx = {"X": 0, "Y": 1, "Z": 2}[highlight_axis]
        others = [i for i in range(3) if i != idx]
        for val in (0, L):
            quad = _face_quad(idx, val, others, L)
            quad = np.vstack([quad, quad[0]])
            ax.plot(quad[:, 0], quad[:, 1], quad[:, 2], color="crimson", linewidth=1.3)

    if df_all is not None and highlight_seg is not None:
        autres = df_all[df_all["segment"] != highlight_seg]
        sel = df_all[df_all["segment"] == highlight_seg]
        ax.scatter(autres["X_m"], autres["Y_m"], autres["Z_m"], color="lightgray", s=1.5, alpha=0.4)
        ax.plot(sel["X_m"], sel["Y_m"], sel["Z_m"], color="crimson", linewidth=1.8, marker="o", markersize=1.5)

    ax.set_xlabel("X", fontsize=5, labelpad=-10)
    ax.set_ylabel("Y", fontsize=5, labelpad=-10)
    ax.set_zlabel("Z", fontsize=5, labelpad=-10)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))
    ax.tick_params(labelsize=3, pad=0)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout(pad=0.1)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf


def _fiber_line(ax, idx, L, color="royalblue"):
    """Trace la fibre (ligne) le long de l'axe mesuré, passant par le centre
    du cube dans les 2 autres directions."""
    others = [i for i in range(3) if i != idx]
    p0 = [0.0, 0.0, 0.0]
    p1 = [0.0, 0.0, 0.0]
    p0[idx], p1[idx] = 0.0, L
    p0[others[0]] = p1[others[0]] = L / 2
    p0[others[1]] = p1[others[1]] = L / 2
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=color, linewidth=2.2, solid_capstyle="round")
    ax.scatter([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
               color=color, s=8)




@st.cache_data(show_spinner=False)
def render_combined_axis_schema(axes_presentes, kind="direct", figsize_per_cube=(1.5, 1.5)):
    """Une seule image regroupant un cube par axe présent, avec :
    - kind="direct"   : les 2 faces perpendiculaires à l'axe mesuré (effet direct)
    - kind="indirect" : les 2 faces perpendiculaires aux 2 AUTRES axes,
                        donc parallèles à la fibre (effet indirect / restreinte)
    Dans les deux cas, la fibre (ligne bleue) est tracée le long de l'axe mesuré,
    passant par le centre du cube, pour localiser visuellement la fibre par
    rapport aux surfaces mises en évidence."""
    n = len(axes_presentes)
    fig = plt.figure(figsize=(figsize_per_cube[0] * n, figsize_per_cube[1]), dpi=140)
    L = CUBE_SIDE_M
    corners = np.array(list(itertools.product([0, L], [0, L], [0, L])))
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
             (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, axe in enumerate(axes_presentes):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        for a, b in edges:
            ax.plot(*zip(corners[a], corners[b]), color="gray", linewidth=0.6)

        idx = {"X": 0, "Y": 1, "Z": 2}[axe]
        others = [j for j in range(3) if j != idx]

        if kind == "direct":
            # faces perpendiculaires à l'axe mesuré (val=0 et val=L sur idx)
            for val in (0, L):
                quad = _face_quad(idx, val, others, L)
                quad = np.vstack([quad, quad[0]])
                ax.plot(quad[:, 0], quad[:, 1], quad[:, 2], color="crimson", linewidth=1.2)
        else:
            # 2 faces parallèles à l'axe mesuré : perpendiculaires à chacun
            # des 2 AUTRES axes (une face de référence par axe latéral)
            for o in others:
                free_dims = [k for k in range(3) if k != o]
                quad = _face_quad(o, 0, free_dims, L)
                quad = np.vstack([quad, quad[0]])
                ax.plot(quad[:, 0], quad[:, 1], quad[:, 2], color="darkorange", linewidth=1.2)

        _fiber_line(ax, idx, L)

        ax.set_title(COMPOSANTE_LABEL[axe], fontsize=6, pad=2)
        ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=20, azim=-60)

    if kind == "direct":
        suptitle = "Faces perpendiculaires (rouge) à l'axe mesuré — fibre en bleu"
    else:
        suptitle = "Faces parallèles à la fibre (orange), perpendiculaires aux 2 autres axes — fibre en bleu"
    fig.suptitle(suptitle, fontsize=6, y=1.05)
    fig.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


@st.cache_data(show_spinner=False)
def make_small_interactive_cube(df_all, highlight_seg, L=CUBE_SIDE_M, height=140):
    """Cube interactif (rotation utilisateur) mais très compact — pour les
    onglets Profil segment et Surface pos×temps."""
    fig = go.Figure()
    fig.add_trace(cube_wireframe_trace(L))
    autres = df_all[df_all["segment"] != highlight_seg]
    sel = df_all[df_all["segment"] == highlight_seg]
    fig.add_trace(go.Scatter3d(x=autres["X_m"], y=autres["Y_m"], z=autres["Z_m"],
                                mode="markers", marker=dict(size=1.5, color="lightgray", opacity=0.4),
                                hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=sel["X_m"], y=sel["Y_m"], z=sel["Z_m"],
                                mode="markers+lines",
                                marker=dict(size=3, color="crimson"),
                                line=dict(color="crimson", width=4),
                                name=highlight_seg, showlegend=False))
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                   aspectmode="cube",
                   xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                   zaxis=dict(showticklabels=False)),
        height=height, margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

@st.cache_data(show_spinner=False)
def make_multi_cube(_df_all, segs_a, segs_b, L=CUBE_SIDE_M, height=170):
    """Petit cube interactif : fibres du groupe A en rouge, groupe B en bleu."""
    fig = go.Figure()
    fig.add_trace(cube_wireframe_trace(L))
    choisies = set(segs_a) | set(segs_b)
    autres = _df_all[~_df_all["segment"].isin(choisies)].drop_duplicates(["segment", "x_fibre_m"])
    fig.add_trace(go.Scatter3d(x=autres["X_m"], y=autres["Y_m"], z=autres["Z_m"], mode="markers",
                                marker=dict(size=1.5, color="lightgray", opacity=0.4),
                                hoverinfo="skip", showlegend=False))
    for segs, color in ((segs_a, "crimson"), (segs_b, "royalblue")):
        xs, ys, zs = [], [], []
        for s in segs:
            g = _df_all[_df_all["segment"] == s]
            if g.empty:
                continue
            g = g[g["date_mesure"] == g["date_mesure"].iloc[0]].sort_values("x_fibre_m")
            xs += list(g["X_m"]) + [None]
            ys += list(g["Y_m"]) + [None]
            zs += list(g["Z_m"]) + [None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                        line=dict(color=color, width=5), showlegend=False))
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube",
                   xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
                   zaxis=dict(showticklabels=False)),
        height=height, margin=dict(l=0, r=0, t=0, b=0))
    return fig

# ---------------------------------------------------------------- chargement
@st.cache_data
def get_data(file) -> pd.DataFrame:
    return load_and_prepare(file)

st.sidebar.title("📁 Données")

# 1. Définition des chemins
cube_paths = {
    "Cube 12": "data/profils_3d_C12.csv",
    "Cube 34": "data/profils_3d_C34.csv"
}

# 2. Boutons de sélection (Radio pour conserver l'état)
selection = st.sidebar.radio("Sélectionner le cube :", list(cube_paths.keys()))

# 3. Option de surcharge par upload
uploaded = st.sidebar.file_uploader("Charger un autre fichier (optionnel)", type="csv")

# 4. Logique de priorité : l'upload prime sur le bouton radio
source = uploaded if uploaded is not None else cube_paths[selection]

# spinner

with st.spinner("⏳ Chargement du dataset en cours..."):
    df_full = get_data(source)

st.sidebar.caption(
    f"{df_full['segment'].nunique()} segments · {df_full['profil'].nunique()} profils · "
    f"{df_full['date_mesure'].min():%d/%m} → {df_full['date_mesure'].max():%d/%m/%Y}"
)

# ---------------------------------------------------------------- filtres
st.sidebar.title("🔎 Filtres")

from date_filter import select_dates_by_spacing, MODE_LABELS

dates_sorted = sorted(df_full["date_mesure"].unique())
date_label = lambda d: f"{pd.Timestamp(d):%d/%m/%Y %Hh%M} (J+{df_full.loc[df_full.date_mesure == d, 'temps_ecoule_j'].iloc[0]:.1f})"

st.sidebar.title("⏱️ Échantillonnage temporel")

t_full_min = float(np.floor(df_full["temps_ecoule_j"].min()))
t_full_max = float(np.ceil(df_full["temps_ecoule_j"].max()))

t_min, t_max = st.sidebar.slider(
    "Intervalle d'étude (jours écoulés depuis t0)",
    min_value=t_full_min, max_value=t_full_max,
    value=(t_full_min, t_full_max), step=0.5,
)

mode_label_choice = st.sidebar.radio("Mode d'espacement des dates", list(MODE_LABELS.values()), index=0)
mode_key = {v: k for k, v in MODE_LABELS.items()}[mode_label_choice]

delta_defaults = {"linear": ("jours", 0.0, 0.5), "diffusif": ("√jours", 0.0, 0.1), "log": ("ln(1+j)", 0.0, 0.05)}
unite, defaut, pas = delta_defaults[mode_key]
delta_t = st.sidebar.number_input(f"ΔT minimal entre 2 dates ({unite}) — 0 = tout garder",
                                   min_value=0.0, value=defaut, step=pas)
# ... (après la définition de delta_t)

with st.spinner("⚙️ Calcul de l'échantillonnage temporel..."):
    dates_retenues = select_dates_by_spacing(df_full, t_min, t_max, delta_t, mode_key)

with st.sidebar.expander("🔧 Ajuster manuellement la liste des dates retenues"):
    dates_dans_intervalle = sorted(
        df_full[(df_full["temps_ecoule_j"] >= t_min) & (df_full["temps_ecoule_j"] <= t_max)]["date_mesure"].unique()
    )
    retenues_set = set(dates_retenues)
    sel_dates = st.multiselect(
        "Dates de mesure", dates_dans_intervalle,
        default=[d for d in dates_dans_intervalle if d in retenues_set],
        format_func=date_label,
    )

st.sidebar.caption(f"📊 {len(sel_dates)} date(s) retenue(s) sur {len(dates_sorted)} disponibles au total.")

axes_dispo = sorted(df_full["axe_mesure"].unique())
sel_axes = st.sidebar.multiselect("Composante mesurée", axes_dispo, default=axes_dispo,
                                   format_func=lambda a: COMPOSANTE_LABEL[a])

profondeurs_dispo = sorted(df_full["plan_profondeur_mm"].dropna().unique())
sel_prof = st.sidebar.multiselect(
    "Profondeur nominale (mm) — laisser vide pour tout inclure (y compris axe traversant Z)",
    profondeurs_dispo, default=profondeurs_dispo)
inclure_axial = st.sidebar.checkbox("Inclure l'axe traversant Z (s1,s2,s3)", value=True)

segments_dispo_pre = sorted(
    df_full[df_full["axe_mesure"].isin(sel_axes)]["segment"].unique(),
    key=lambda s: int(s[1:]))
sel_segments = st.sidebar.multiselect("Segments", segments_dispo_pre, default=segments_dispo_pre)

# ---------------------------------------------------------- 🎨 échelle de couleur
st.sidebar.title("🎨 Échelle de couleur")
colormap_choice = st.sidebar.selectbox("Colormap (cartographies)", COLORMAPS_DISPO, index=0)
auto_scale = st.sidebar.checkbox("Échelle automatique (recommandé)", value=True)
manual_vmin = manual_vmax = None
if not auto_scale:
    c1s, c2s = st.sidebar.columns(2)
    manual_vmin = c1s.number_input("Min (µdef)", value=-200.0, step=10.0)
    manual_vmax = c2s.number_input("Max (µdef)", value=200.0, step=10.0)

# application des filtres
mask = (
    df_full["date_mesure"].isin(sel_dates)
    & df_full["axe_mesure"].isin(sel_axes)
    & df_full["segment"].isin(sel_segments)
)
prof_mask = df_full["plan_profondeur_mm"].isin(sel_prof)
if inclure_axial:
    prof_mask = prof_mask | df_full["plan_profondeur_mm"].isna()
mask = mask & prof_mask

df = df_full[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Points affichés", f"{len(df):,}".replace(",", " "))

if df.empty:
    st.warning("Aucune donnée pour ces filtres — élargis la sélection dans la barre latérale.")
    st.stop()

date_labels = {d: date_label(d) for d in sel_dates}
color_seq = px.colors.sequential.Viridis


def color_for_date(d):
    idx = sel_dates.index(d) / max(len(sel_dates) - 1, 1)
    return color_seq[int(idx * (len(color_seq) - 1))]


# ---------------------------------------------------------------- en-tête
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("Logo1.png", width=160)
with col_title:
    st.title("Dashboard cube B40-SF— Retrait de carbonatation, cube B40-SF")
st.caption(
    "Analyse du retrait de carbonatation d'un bloc de pâte de ciment"
    "Fibres optiques 3D — chaque fibre ne mesure la déformation que dans son propre axe "
    "(εxx, εyy ou εzz). Le cube est attaqué sur ses 6 faces."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Segments sélectionnés", df["segment"].nunique())
c2.metric("Dates sélectionnées", len(sel_dates))
c3.metric("Déformation moyenne", f"{df['valeur_microdefs'].mean():.0f} µdef")
c4.metric("Écart-type", f"{df['valeur_microdefs'].std():.0f} µdef")

tabs = st.tabs([
    "1️⃣ Effet direct", "2️⃣ Effet indirect", "3️⃣ Anisotropie",
    "4️⃣ Profil segment", "5️⃣ Effet de coin", "6️⃣ Carto 3D",
    "7️⃣ Symétrie H/B", "8️⃣ Heatmap seg×date", "9️⃣ Surface pos×temps",
    "🔟 Opérations fibres",
])

# ---------------------------------------------------------- 1. effet direct
with tabs[0]:
    st.subheader("Déformation vs distance à la face perpendiculaire à l'axe mesuré")
    st.markdown("""
**Post-traitement :** pour chaque point, on calcule la distance à la face du cube
perpendiculaire à l'axe que la fibre mesure réellement à cet endroit (`dist_face_direct`).
Exemple : un segment orienté selon X ne mesure que εxx ; sa distance "directe" est celle
aux faces X=0 et X=100 mm — les seules faces dont l'attaque en carbonatation produit un
retrait *directement visible* dans cette composante.

**Lecture physique attendue :** si le mécanisme dominant est un retrait de carbonatation
qui progresse depuis la surface vers le cœur, on doit observer une déformation (en valeur
absolue) plus marquée près de la face directe, qui s'accentue avec le temps (dates plus
récentes = courbes plus étalées/amples).
    """)
    axes_presentes = [a for a in "XYZ" if a in sel_axes]
    if axes_presentes:
        _, col_schema, _ = st.columns([1, 2, 1])
        with col_schema:
            st.image(render_combined_axis_schema(axes_presentes, kind="direct"), width="stretch")
    for axe in axes_presentes:
        sub = df[df["axe_mesure"] == axe]
        fig = go.Figure()
        for d in sel_dates:
            s = sub[sub["date_mesure"] == d]
            if s.empty:
                continue
            fig.add_trace(go.Scattergl(x=s["dist_face_direct"] * 1000, y=s["valeur_microdefs"],
                                        mode="markers", marker=dict(size=4, color=color_for_date(d)),
                                        name=date_labels[d], opacity=0.6))
        fig.update_layout(title=COMPOSANTE_LABEL[axe], xaxis_title="Distance (mm)",
                           yaxis_title="µdef", height=430, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch", key=f"plot_direct_{axe}")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_direct_or_indirect("direct", dates_filter=sel_dates), language="python")

# -------------------------------------------------------- 2. effet indirect
with tabs[1]:
    st.subheader("Déformation vs distance aux faces NON perpendiculaires à l'axe mesuré")
    st.markdown("""
**Post-traitement :** on calcule ici `dist_face_laterale_min`, la distance minimale aux
deux faces qui ne sont **pas** perpendiculaires à l'axe mesuré par la fibre. Exemple :
pour un segment selon X, ce sont les faces en Y (0/100 mm) et en Z (0/100 mm).

**Pourquoi un point peut être influencé par plusieurs faces à la fois :** la
carbonatation entre simultanément par toutes les faces du cube. Un point situé près d'une
face Y ou Z subit un retrait *local* dans ces directions-là, non mesuré directement par la
fibre (qui ne voit que X). Mais ce retrait local peut mettre en **traction par restreinte**
la zone environnante encore saine dans l'axe mesuré (effet de Poisson + gêne mécanique du
matériau non carbonaté qui empêche la contraction libre). C'est un couplage 3D : un effet
"invisible" dans une direction peut donc se manifester indirectement dans une autre.

**Lecture physique :** une corrélation entre `dist_face_laterale_min` et la déformation
mesurée, en l'absence de proximité de face directe, est la signature de cet effet indirect.
    """)
    axes_presentes = [a for a in "XYZ" if a in sel_axes]
    if axes_presentes:
        _, col_schema, _ = st.columns([1, 2, 1])
        with col_schema:
            st.image(render_combined_axis_schema(axes_presentes, kind="indirect"), width="stretch")
    for axe in axes_presentes:
        sub = df[df["axe_mesure"] == axe]
        fig = go.Figure()
        for d in sel_dates:
            s = sub[sub["date_mesure"] == d]
            if s.empty:
                continue
            fig.add_trace(go.Scattergl(x=s["dist_face_laterale_min"] * 1000, y=s["valeur_microdefs"],
                                        mode="markers", marker=dict(size=4, color=color_for_date(d)),
                                        name=date_labels[d], opacity=0.6))
        fig.update_layout(title=COMPOSANTE_LABEL[axe], xaxis_title="Distance latérale min (mm)",
                           yaxis_title="µdef", height=430, legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, width="stretch", key=f"plot_indirect_{axe}")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_direct_or_indirect("indirect", dates_filter=sel_dates), language="python")

# ---------------------------------------------------------- 3. anisotropie
with tabs[2]:
    st.subheader("Anisotropie du retrait par profondeur nominale")
    st.markdown("""
**Post-traitement :** moyenne (± écart-type) de la déformation par profondeur nominale,
composante et date. **Anisotropie :** si εxx, εyy, εzz divergent significativement à
profondeur équivalente, le retrait n'est pas isotrope — signe d'une hétérogénéité de
séchage/exposition, d'un effet de bord, ou d'une orientation de fissuration préférentielle.

**Caractère diffusif (loi de Fick) :** la carbonatation est un phénomène de diffusion ;
en première approximation son front avance en √t. Si les courbes deviennent linéaires en
fonction de √t (plutôt qu'en t), cela conforte un régime diffusif classique.
    """)
    g = (df.dropna(subset=["plan_profondeur_mm"])
           .groupby(["plan_profondeur_mm", "axe_mesure", "date_mesure"], as_index=False)
           .agg(moy=("valeur_microdefs", "mean"), std=("valeur_microdefs", "std"),
                t=("temps_ecoule_j", "first")))
    if g.empty:
        st.info("Aucune donnée pour les plans de profondeur avec les filtres actuels.")
    else:
        g["composante"] = g["axe_mesure"].map(COMPOSANTE_LABEL)
        g["sqrt_t"] = np.sqrt(g["t"].clip(lower=0))

        mode_temps = st.radio("Axe temporel", ["t (jours)", "√t (diffusif, loi de Fick)"],
                               horizontal=True, key="mode_temps_aniso")
        x_col, x_label = ("t", "Temps écoulé (j)") if mode_temps.startswith("t ") else ("sqrt_t", "√t (√j)")

        fig = px.line(g, x=x_col, y="moy", color="composante", facet_col="plan_profondeur_mm",
                      error_y="std", markers=True,
                      labels={x_col: x_label, "moy": "Déformation moyenne (µdef)"})
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch", key="plot_anisotropie")

        st.markdown("#### Vitesse de propagation du front (dérivée de la déformation moyenne)")
        st.markdown("""
Dérivée numérique (`np.gradient`) de la déformation moyenne par rapport au temps, pour
chaque profondeur nominale ainsi que pour l'axe Z traversant (s1/s2/s3). Un pic de vitesse
marque une accélération locale du retrait/front de carbonatation ; un plateau signale une
stabilisation. L'axe temporel de cette dérivée est configurable indépendamment du
graphique ci-dessus.
        """)
        mode_temps_deriv = st.radio("Axe temporel (vitesse)", ["t (jours)", "√t (diffusif)"],
                                     horizontal=True, key="mode_temps_deriv")
        x_deriv_col = "t" if mode_temps_deriv.startswith("t ") else "sqrt_t"
        x_deriv_label = "Temps écoulé (j)" if x_deriv_col == "t" else "√t (√j)"

        g_deriv = g.sort_values(["plan_profondeur_mm", "composante", x_deriv_col]).copy()
        deriv_rows = []
        for (prof, comp), sub in g_deriv.groupby(["plan_profondeur_mm", "composante"]):
            if len(sub) < 2:
                continue
            v = np.gradient(sub["moy"].values, sub[x_deriv_col].values)
            deriv_rows.append(pd.DataFrame({
                "plan_profondeur_mm": prof, "composante": comp,
                "x": sub[x_deriv_col].values, "vitesse": v}))
        # axe Z traversant (pas de profondeur nominale)
        gz = (df[df["axe_mesure"] == "Z"]
                .groupby("date_mesure", as_index=False)
                .agg(moy=("valeur_microdefs", "mean"), t=("temps_ecoule_j", "first"))
                .sort_values("t"))
        gz["sqrt_t"] = np.sqrt(gz["t"].clip(lower=0))
        if len(gz) >= 2:
            vz = np.gradient(gz["moy"].values, gz[x_deriv_col].values)
            deriv_rows.append(pd.DataFrame({
                "plan_profondeur_mm": "axe Z (traversant)", "composante": "ε_zz (axe Z)",
                "x": gz[x_deriv_col].values, "vitesse": vz}))
        if deriv_rows:
            d_all = pd.concat(deriv_rows, ignore_index=True)
            fig2 = px.line(d_all, x="x", y="vitesse", color="composante",
                           facet_col="plan_profondeur_mm", markers=True,
                           labels={"x": x_deriv_label, "vitesse": "Vitesse (µdef/j)"})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, width="stretch", key="plot_deriv")
        else:
            st.info("Pas assez de dates distinctes pour calculer une dérivée temporelle.")

        with st.expander("📜 Script Python reproductible (notebook)"):
            mt = "t" if mode_temps.startswith("t ") else "sqrt_t"
            mtd = "t" if mode_temps_deriv.startswith("t ") else "sqrt_t"
            st.code(sg.gen_anisotropie(mode_temps=mt, mode_temps_deriv=mtd,
                                       dates_filter=sel_dates), language="python")

# -------------------------------------------------------- 4. profil segment
with tabs[3]:
    st.subheader("Profil spatial le long d'un segment")
    st.markdown("""
**Post-traitement :** déformation en fonction de la position sur la fibre, une courbe par
date. **Lecture physique :** un profil plat = déformation homogène le long du segment ;
des variations locales marquées peuvent indiquer un effet de bord/coin (proximité d'une
extrémité de fibre ou d'une arête du cube) ou une fissuration localisée.
    """)
    seg_choice = st.selectbox("Segment", sorted(df["segment"].unique(), key=lambda s: int(s[1:])), key="seg_tab4")
    pos_mode4 = st.radio(
        "Axe des abscisses",
        ["Position sur la fibre (relative)", "Position absolue dans le cube (0–L)"],
        horizontal=True, key="posmode_tab4")

    _, col_cube, _ = st.columns([1, 1, 1])
    with col_cube:
        st.plotly_chart(make_small_interactive_cube(df_full, seg_choice, height=140),
                         width="stretch", key="cube_tab4")
    sub = df[df["segment"] == seg_choice]
    axe = sub["axe_mesure"].iloc[0]

    axis_col_map = {"X": "X_m", "Y": "Y_m", "Z": "Z_m"}
    if pos_mode4.startswith("Position sur"):
        x_col4, x_label4 = "x_fibre_m", "Position sur la fibre (m)"
    else:
        x_col4, x_label4 = axis_col_map[axe], f"Position absolue selon {axe} (m)"

    fig = go.Figure()
    for d in sel_dates:
        s = sub[sub["date_mesure"] == d]
        if s.empty:
            continue
        s = s.sort_values(x_col4)
        fig.add_trace(go.Scatter(x=s[x_col4], y=s["valeur_microdefs"], mode="lines",
                                  line=dict(color=color_for_date(d)), name=date_labels[d]))
    fig.update_layout(title=f"{seg_choice} — {COMPOSANTE_LABEL[axe]}",
                       xaxis_title=x_label4, yaxis_title="µdef", height=500)
    st.plotly_chart(fig, width="stretch", key="plot_seg_tab4")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_profil_segment(seg_choice, pos_mode4, dates_filter=sel_dates), language="python")

# --------------------------------------------------------- 5. effet de coin
with tabs[4]:
    st.subheader("Effet de coin (cumul attaque directe + latérale)")
    st.markdown("""
**Post-traitement :** un point est classé **"Coin"** si sa distance latérale minimale
(`dist_face_laterale_min`, cf. onglet Effet indirect) est inférieure à un seuil réglable —
il cumule alors une attaque directe (face perpendiculaire à l'axe mesuré) **et** une
proximité à une face latérale (attaque/restreinte indirecte).

**Lecture physique :** si les points "Coin" présentent, à distance directe égale, une
déformation systématiquement plus marquée que les points "Face plane", cela confirme un
effet cumulatif (accélération locale du retrait ou concentration de traction par
restreinte dans les zones d'angle).
    """)
    date_ref = st.selectbox("Date analysée", sel_dates, index=len(sel_dates) - 1, format_func=date_label)
    seuil_mm = st.slider("Seuil 'proche d'une face latérale' (mm)", 5, 40, 15)
    d = df[df["date_mesure"] == date_ref].copy()
    d["zone"] = np.where(d["dist_face_laterale_min"] * 1000 < seuil_mm, "Coin (proche 2 faces)", "Face plane")
    d["composante"] = d["axe_mesure"].map(COMPOSANTE_LABEL)
    for comp in sorted(d["composante"].dropna().unique()):
        sub = d[d["composante"] == comp]
        fig = px.scatter(sub, x="dist_face_direct", y="valeur_microdefs", color="zone", opacity=0.6,
                          title=comp,
                          labels={"dist_face_direct": "Distance directe (m)", "valeur_microdefs": "µdef"})
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch", key=f"plot_coin_{comp}")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_effet_coin(seuil_mm, dates_filter=sel_dates), language="python")

# ----------------------------------------------------------- 6. carto 3D
with tabs[5]:
    st.subheader("Cartographie 3D du champ de déformation")
    st.markdown("""
**Post-traitement :** nuage 3D de tous les points mesurés, coloré par la valeur de
déformation, avec un curseur temporel (une frame par date) et une géométrie statique de
référence (4 armatures orientées selon l'axe X). L'échelle de couleur peut être
**automatique par date** (recommandé) ou **fixe/manuelle** via la barre latérale.

**Lecture physique :** garder en tête que la couleur ne représente pas une grandeur
scalaire homogène — chaque point ne montre que la composante (εxx/εyy/εzz) que sa fibre
mesure réellement à cet endroit (indiquée en info-bulle).
    """)
    scale_par_date = st.checkbox("Échelle de couleur recalculée par date (corrige le bug d'affichage)",
                                  value=True, key="scale_par_date_3d")

    # --- géométrie statique : cube + 4 armatures cylindriques ---
    ARMATURE_AXIS = "Y"  # axe le long duquel les armatures s'étendent : "X", "Y" ou "Z"
    cylinder_positions = [(0.025, 0.024), (0.075, 0.024), (0.025, 0.076), (0.075, 0.076)]
    cylinder_diameter = 0.008  # à ajuster selon le diamètre réel des armatures
    static_traces = [cube_wireframe_trace(CUBE_SIDE_M)]
    r = cylinder_diameter / 2
    theta = np.linspace(0, 2 * np.pi, 16)
    for p1, p2 in cylinder_positions:
        c1 = p1 + r * np.cos(theta)
        c2 = p2 + r * np.sin(theta)
        for a, b in zip(c1, c2):
            coords = {"X": ([0, CUBE_SIDE_M], [a, a], [b, b]),
                      "Y": ([a, a], [0, CUBE_SIDE_M], [b, b]),
                      "Z": ([a, a], [b, b], [0, CUBE_SIDE_M])}[ARMATURE_AXIS]
            static_traces.append(go.Scatter3d(
                x=coords[0], y=coords[1], z=coords[2], mode="lines",
                line=dict(color="dimgray", width=2), hoverinfo="skip", showlegend=False))

    frames, steps = [], []
    for d in sel_dates:
        s = df[df["date_mesure"] == d]
        if s.empty:
            continue
        if scale_par_date:
            vmin_d, vmax_d = get_color_range(s["valeur_microdefs"], None, None, True)
        else:
            vmin_d, vmax_d = get_color_range(df["valeur_microdefs"], manual_vmin, manual_vmax, auto_scale)
        frames.append(go.Frame(
            data=[go.Scatter3d(x=s["X_m"], y=s["Y_m"], z=s["Z_m"], mode="markers",
                                marker=dict(size=4, color=s["valeur_microdefs"], colorscale=colormap_choice,
                                            cmin=vmin_d, cmax=vmax_d, colorbar=dict(title="µdef"),
                                            line=dict(width=0.5, color="rgba(0,0,0,0.35)")),
                                text=s["segment"] + " (" + s["axe_mesure"] + ")",
                                hovertemplate="%{text}<br>µdef=%{marker.color:.0f}<extra></extra>")],
            traces=[0], name=date_labels[d]))
        steps.append(dict(method="animate", args=[[date_labels[d]], {"mode": "immediate"}],
            label=f"{df.loc[df.date_mesure==d,'temps_ecoule_j'].iloc[0]*24:.1f} h"))
    if frames:
        fig = go.Figure(data=[frames[0].data[0]] + static_traces, frames=frames)
        fig.update_layout(
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                       aspectmode="cube"),
            height=650,
            sliders=[dict(steps=steps, currentvalue=dict(prefix="Temps écoulé : "))])  # pas de updatemenus => pas de bouton Play
        st.plotly_chart(fig, width="stretch", key="plot_carto3d")

        with st.expander("📜 Script Python reproductible (notebook)"):
            st.code(sg.gen_carto3d(), language="python")

# ---------------------------------------------------------- 7. symétrie H/B
with tabs[6]:
    st.subheader("Symétrie face haute (Z ≈ 100 mm) vs face basse (Z ≈ 0)")
    st.markdown("""
**Post-traitement :** les points sont répartis en deux groupes selon leur position par
rapport au plan médian du cube (`Z_m < 50 mm` → face basse, sinon face haute), puis moyennés
par profondeur nominale et date.

**Lecture physique :** en l'absence d'asymétrie d'exposition (même carbonatation en haut
et en bas), les deux courbes devraient être proches. Un écart systématique peut révéler une
différence d'exposition (ex. face posée au sol vs face libre), un gradient de séchage
initial, ou une hétérogénéité de mise en œuvre du matériau.
    """)
    d = df.dropna(subset=["plan_profondeur_mm"]).copy()
    if d.empty:
        st.info("Aucun segment de plan de profondeur dans la sélection actuelle.")
    else:
        d["face_z"] = np.where(d["Z_m"] < CUBE_SIDE_M / 2, "Face basse (Z ≈ 0)", "Face haute (Z ≈ 100 mm)")
        g = d.groupby(["plan_profondeur_mm", "face_z", "date_mesure"], as_index=False).agg(
            moy=("valeur_microdefs", "mean"), t=("temps_ecoule_j", "first"))
        fig = px.line(g, x="t", y="moy", color="face_z", facet_col="plan_profondeur_mm", markers=True,
                      labels={"t": "Temps écoulé (j)", "moy": "Déformation moyenne (µdef)"})
        fig.update_layout(height=450)
        st.plotly_chart(fig, width="stretch", key="plot_symetrie")

        with st.expander("📜 Script Python reproductible (notebook)"):
            st.code(sg.gen_symetrie(dates_filter=sel_dates), language="python")

# ---------------------------------------------------- 8. heatmap segment×date
with tabs[7]:
    st.subheader("Déformation moyenne par segment et par date")
    st.markdown("""
**Post-traitement :** moyenne de la déformation par segment et par date, sous forme de
heatmap. **Lecture physique :** vue d'ensemble rapide pour repérer les segments qui
évoluent le plus vite (colonnes qui foncent tôt) ou qui restent stables — utile pour
prioriser les segments à examiner en détail dans les autres onglets.
    """)
    g = df.groupby(["segment", "date_mesure"], as_index=False).agg(moy=("valeur_microdefs", "mean"))
    axe_map = df.drop_duplicates("segment").set_index("segment")["axe_mesure"]
    g["label_segment"] = g["segment"] + " (" + g["segment"].map(axe_map) + ")"
    piv = g.pivot(index="label_segment", columns="date_mesure", values="moy")
    piv = piv.reindex(sorted(piv.index, key=lambda s: int(s.split("s")[1].split(" ")[0])))
    vmin, vmax = get_color_range(piv.values, manual_vmin, manual_vmax, auto_scale)
    fig = go.Figure(go.Heatmap(z=piv.values, x=[date_labels[d] for d in piv.columns], y=piv.index,
                                colorscale=colormap_choice, zmin=vmin, zmax=vmax,
                                zmid=0 if auto_scale else None,
                                colorbar=dict(title="µdef")))
    fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Segment (axe)")
    st.plotly_chart(fig, width="stretch", key="plot_heatmap_segdate")

    with st.expander("📜 Script Python reproductible (notebook)"):
        st.code(sg.gen_heatmap_segdate(), language="python")

# --------------------------------------------------- 9. surface position×temps
with tabs[8]:
    st.subheader("Évolution position × temps pour un segment")
    st.markdown("""
**Post-traitement :** interpolation linéaire (`scipy.griddata`) de la déformation sur une
grille régulière position × temps, à partir des mesures ponctuelles disponibles.

**Lecture physique :** permet de visualiser en un seul graphique comment le profil spatial
d'un segment évolue au cours de l'essai — utile pour repérer une zone qui se dégrade plus
vite qu'une autre le long de la même fibre.
    """)
    seg_choice2 = st.selectbox("Segment", sorted(df["segment"].unique(), key=lambda s: int(s[1:])), key="seg9")
    pos_mode9 = st.radio(
        "Axe des positions",
        ["Position sur la fibre (relative)", "Position absolue dans le cube (0–L)"],
        horizontal=True, key="posmode_tab9")

    _, col_cube, _ = st.columns([1, 1, 1])
    with col_cube:
        st.plotly_chart(make_small_interactive_cube(df_full, seg_choice2, height=140),
                         width="stretch", key="cube_tab9")
    kind = st.radio("Type de rendu", ["Heatmap (recommandé)", "Surface 3D"], horizontal=True)
    sub = df[df["segment"] == seg_choice2]
    axe = sub["axe_mesure"].iloc[0]

    axis_col_map = {"X": "X_m", "Y": "Y_m", "Z": "Z_m"}
    if pos_mode9.startswith("Position sur"):
        x_col9, x_label9 = "x_fibre_m", "Position sur la fibre (m)"
    else:
        x_col9, x_label9 = axis_col_map[axe], f"Position absolue selon {axe} (m)"

    if sub["date_mesure"].nunique() < 2:
        st.info("Il faut au moins 2 dates sélectionnées pour interpoler cette vue.")
    else:
        x = sub[x_col9].values
        t = sub["temps_ecoule_j"].values
        z = sub["valeur_microdefs"].values
        xi, ti, ZI = interpolate_grid(x, t, z)
        vmin, vmax = get_color_range(ZI, manual_vmin, manual_vmax, auto_scale)
        if kind.startswith("Surface"):
            fig = go.Figure(go.Surface(x=xi, y=ti, z=ZI, colorscale=colormap_choice,
                                        cmin=vmin, cmax=vmax, colorbar=dict(title="µdef")))
            fig.update_layout(scene=dict(xaxis_title=x_label9, yaxis_title="Temps (j)", zaxis_title="µdef"),
                               height=600, title=f"{seg_choice2} — {COMPOSANTE_LABEL[axe]}")
        else:
            fig = go.Figure(go.Heatmap(x=xi, y=ti, z=ZI, colorscale=colormap_choice,
                                        zmin=vmin, zmax=vmax, colorbar=dict(title="µdef")))
            fig.update_layout(height=500, xaxis_title=x_label9, yaxis_title="Temps écoulé (j)",
                               title=f"{seg_choice2} — {COMPOSANTE_LABEL[axe]}")
        st.plotly_chart(fig, width="stretch", key="plot_seg_tab9")

        kind_arg = "surface" if kind.startswith("Surface") else "heatmap"
        with st.expander("📜 Script Python reproductible (notebook)"):
            st.code(sg.gen_surface_segment(seg_choice2, kind_arg, pos_mode9, dates_filter=sel_dates), language="python")

# ------------------------------------------------------ 10. opérations fibres
with tabs[9]:
    st.subheader("Opérations entre fibres")
    st.markdown("""
**Principe :** chaque fibre est ré-interpolée sur une grille commune 0 → L (selon sa propre
coordonnée absolue), puis (1) les fibres de chaque groupe sont **agrégées** (moyenne, somme...),
(2) les groupes A et B sont éventuellement **combinés** (A − B, A / B...).
Exemples : *moyenne A − moyenne B* pour comparer deux plans ; *somme de εxx+εyy+εzz* (un groupe
contenant une fibre par direction) pour une déformation volumique.
Les calculs ne se lancent qu'au clic sur **Calculer**.
    """)

    seg_info = df_full.drop_duplicates("segment").set_index("segment")[["axe_mesure", "plan_profondeur_mm"]]
    PLANS = [8, 16, 30, None]
    plan_label = lambda p: "Axe Z traversant" if p is None else f"Plan {p} mm"

    def fibre_picker(prefix):
        """Cases à cocher groupées par plan (lignes) et orientation (colonnes)."""
        selected = []
        for plan in PLANS:
            st.markdown(f"**{plan_label(plan)}**")
            cols = st.columns(3)
            for col, axe in zip(cols, "XYZ"):
                segs = [s for s, r in seg_info.iterrows()
                        if r["axe_mesure"] == axe
                        and ((pd.isna(r["plan_profondeur_mm"]) and plan is None)
                             or (plan is not None and r["plan_profondeur_mm"] == plan))]
                segs = sorted(segs, key=lambda s: int(s[1:]))
                with col:
                    if not segs:
                        continue
                    st.caption(COMPOSANTE_LABEL[axe])
                    tous = st.checkbox("Tous", key=f"{prefix}_all_{plan}_{axe}")
                    for s in segs:
                        if st.checkbox(s, key=f"{prefix}_{s}") or tous:
                            selected.append(s)
        return list(dict.fromkeys(selected))

    with st.form("form_operation"):
        cA, cB, cC = st.columns(3)
        agg_label = cA.selectbox("Agrégation dans chaque groupe", list(fo.AGG_LABELS.values()))
        comb_label = cB.selectbox("Combinaison A ⊕ B", list(fo.COMBINE_LABELS.values()))
        absval = cC.checkbox("Valeur absolue du résultat", value=False)
        tab_a, tab_b = st.tabs(["🔴 Groupe A", "🔵 Groupe B (si combinaison)"])
        with tab_a:
            segs_a = fibre_picker("opA")
        with tab_b:
            segs_b = fibre_picker("opB")
        go_calc = st.form_submit_button("▶️ Calculer", type="primary")

    if go_calc:
        agg_key = {v: k for k, v in fo.AGG_LABELS.items()}[agg_label]
        comb_key = {v: k for k, v in fo.COMBINE_LABELS.items()}[comb_label]
        if not segs_a:
            st.error("Sélectionne au moins une fibre dans le groupe A.")
        elif comb_key != "none" and not segs_b:
            st.error("Cette combinaison nécessite au moins une fibre dans le groupe B.")
        else:
            with st.spinner("Calcul..."):
                df_op = df_full[df_full["date_mesure"].isin(sel_dates)]
                res_op, prof_op = fo.compute_operation(
                    df_op, segs_a, segs_b, agg_key, comb_key, absval, L=CUBE_SIDE_M)
            st.session_state["op_state"] = dict(
                res=res_op, prof=prof_op, segs_a=tuple(segs_a), segs_b=tuple(segs_b),
                agg=agg_key, combine=comb_key, absval=absval, dates=list(sel_dates))

    S = st.session_state.get("op_state")
    if S is None:
        st.info("Choisis des fibres puis clique sur **Calculer**.")
    else:
        res_op, prof_op = S["res"], S["prof"]
        titre_op = (f"{fo.AGG_LABELS[S['agg']]} — {fo.COMBINE_LABELS[S['combine']]}"
                    + (" (|·|)" if S["absval"] else ""))

        _, col_cube, _ = st.columns([1, 1, 1])
        with col_cube:
            st.plotly_chart(make_multi_cube(df_full, S["segs_a"], S["segs_b"]),
                             width="stretch", key="cube_tab10")
            st.caption("🔴 groupe A · 🔵 groupe B")

        dates_op = sorted(res_op["date_mesure"].unique())
        pal = px.colors.sequential.Viridis
        col_op = lambda i: pal[int(i / max(len(dates_op) - 1, 1) * (len(pal) - 1))]

        fig = go.Figure()
        for i, d in enumerate(dates_op):
            s = res_op[res_op["date_mesure"] == d]
            fig.add_trace(go.Scatter(x=s["pos_m"] * 1000, y=s["valeur"], mode="lines",
                                      line=dict(color=col_op(i)), name=date_label(d)))
        fig.update_layout(title=titre_op, xaxis_title="Position le long des fibres (mm)",
                           yaxis_title="µdef", height=500)
        st.plotly_chart(fig, width="stretch", key="plot_op")

        m = res_op.groupby("date_mesure", as_index=False).agg(
            v=("valeur", "mean"), t=("temps_ecoule_j", "first")).sort_values("t")
        fig_t = px.line(m, x="t", y="v", markers=True,
                        labels={"t": "Temps écoulé (j)", "v": "Valeur moyenne (µdef)"},
                        title="Valeur moyenne du résultat (sur 0 → L) en fonction du temps")
        fig_t.update_layout(height=320)
        st.plotly_chart(fig_t, width="stretch", key="plot_op_t")

        st.download_button("⬇️ Résultat (CSV)", res_op.to_csv(index=False, sep=";").encode("utf-8"),
                           file_name="operation_fibres.csv", mime="text/csv")

        # --- vérification des fibres sélectionnées ---
        with st.expander("🔍 Vérifier les profils des fibres sélectionnées", expanded=False):
            d_chk = st.selectbox("Date", dates_op, index=len(dates_op) - 1,
                                  format_func=date_label, key="op_date_chk")
            p = prof_op[prof_op["date_mesure"] == d_chk]
            fig_c = go.Figure()
            reds = px.colors.sequential.Reds[3:]
            blues = px.colors.sequential.Blues[3:]
            for grp, pal_g in (("A", reds), ("B", blues)):
                for k, (seg, g) in enumerate(p[p["groupe"] == grp].groupby("segment")):
                    ax_ = seg_info.loc[seg, "axe_mesure"]
                    fig_c.add_trace(go.Scatter(
                        x=g["pos_m"] * 1000, y=g["valeur"], mode="lines",
                        line=dict(color=pal_g[k % len(pal_g)], width=1.5),
                        name=f"{grp} · {seg} ({ax_})"))
            r = res_op[res_op["date_mesure"] == d_chk]
            fig_c.add_trace(go.Scatter(x=r["pos_m"] * 1000, y=r["valeur"], mode="lines",
                                        line=dict(color="black", width=3, dash="dash"),
                                        name="Résultat"))
            fig_c.update_layout(xaxis_title="Position (mm)", yaxis_title="µdef", height=480,
                                 legend=dict(font=dict(size=9)))
            st.plotly_chart(fig_c, width="stretch", key="plot_op_chk")
            st.caption("Les trous éventuels (NaN) correspondent aux zones non couvertes par la fibre.")

        with st.expander("📜 Script Python reproductible (notebook)"):
            st.code(sg.gen_operation_fibres(S["segs_a"], S["segs_b"], S["agg"], S["combine"],
                                            S["absval"], dates_filter=S["dates"]), language="python")


st.markdown("---")
with st.expander("ℹ️ Notes méthodologiques"):
    st.markdown(f"""
    - Cube {CUBE_SIDE_M*1000:.0f} × {CUBE_SIDE_M*1000:.0f} × {CUBE_SIDE_M*1000:.0f} mm, origine au coin (0,0,0).
    - t0 (début d'essai) : **08/07/2026 14h40**.
    - Chaque fibre optique ne mesure la déformation que **dans son propre axe** (εxx, εyy ou εzz selon l'orientation du segment) — ce n'est pas une grandeur scalaire isotrope.
    - `dist_face_direct` = distance à la face perpendiculaire à l'axe mesuré (effet de retrait direct attendu).
    - `dist_face_laterale_min` = distance min aux faces non-perpendiculaires (effet indirect / restreinte possible).
    - L'échelle de couleur des cartographies est configurable dans la barre latérale (colormap, automatique ou bornes manuelles).
    - Les schémas combinés des onglets 1 et 2 sont des images statiques (indicatives). Ceux des onglets 4 et 9 sont interactifs (rotation possible) mais restent volontairement très compacts.
    - Les 4 armatures affichées dans la cartographie 3D sont orientées selon l'axe X (positions Y/Z à ajuster si besoin).
    - Chaque onglet propose un bouton "📜 Générer le script Python (notebook)" : il télécharge un script autonome (repart du CSV brut, refait les calculs, style Plotly publication scientifique) reproduisant la figure affichée.
    - Aucun filtrage des valeurs aberrantes n'est appliqué (choix utilisateur).
    """)
