"""
Génération de scripts Python autonomes (reproductibles dans un notebook) pour
chaque onglet du dashboard. Chaque script :
  - repart du CSV brut (aucune dépendance à data_prep.py / dashboard.py),
  - refait les calculs nécessaires (axe mesuré, distances aux faces, temps écoulé),
  - trace la figure avec Plotly, dans un style "publication scientifique"
    (fond blanc, police serif, axes noirs, légendes sobres, export haute résolution).
"""
import pandas as pd
import inspect
import fiber_ops

# ------------------------------------------------------------------ en-tête commun

COMMON_HEADER = '''\
"""
Script généré automatiquement depuis le Dashboard Carbonatation C12.
Reproduit la figure "{titre}" à partir du CSV brut, de façon autonome.

Style : publication scientifique (fond blanc, police serif, axes noirs,
export haute résolution via kaleido).

Pré-requis : pip install pandas numpy plotly kaleido
"""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================================ paramètres d'essai
CSV_PATH = "profils_3d_C12.csv"          # <- à adapter si besoin
CUBE_SIDE_M = 0.100                      # cube 100x100x100 mm, origine (0,0,0)
T0 = pd.Timestamp("2026-07-08 14:40:00")  # début d'essai (t0 carbonatation)

PLAN_PROFONDEUR_NOMINAL = {{
    "s11": 8, "s12": 8, "s10": 8, "s14": 8, "s15": 8, "s16": 8, "s17": 8,
    "s6": 16, "s7": 16, "s8": 16, "s9": 16, "s20": 16, "s19": 16, "s21": 16, "s18": 16,
    "s4": 30, "s5": 30, "s22": 30, "s23": 30,
    "s1": None, "s2": None, "s3": None,
}}

FILENAME_RE = re.compile(r"V[1234]_(\\d{{4}})(\\d{{2}})(\\d{{2}})h(\\d{{2}})m(\\d{{2}})")

COMPOSANTE_LABEL = {{"X": "ε_xx (axe X)", "Y": "ε_yy (axe Y)", "Z": "ε_zz (axe Z)"}}




def parse_date_from_profil(profil):
    m = FILENAME_RE.search(profil)
    if not m:
        return pd.NaT
    y, mo, d, h, mi = m.groups()
    return pd.Timestamp(int(y), int(mo), int(d), int(h), int(mi))


def infer_axis_per_segment(df):
    axis_map = {{}}
    for seg, sub in df.groupby("segment"):
        ranges = {{
            "X": sub["X_m"].max() - sub["X_m"].min(),
            "Y": sub["Y_m"].max() - sub["Y_m"].min(),
            "Z": sub["Z_m"].max() - sub["Z_m"].min(),
        }}
        axis_map[seg] = max(ranges, key=ranges.get)
    return axis_map


def load_and_prepare(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path, sep=";")
    df["date_mesure"] = df["profil"].apply(parse_date_from_profil)
    df["fibre"] = df["profil"].str.extract(r"^(V[1234])_")
    df["temps_ecoule_j"] = (df["date_mesure"] - T0).dt.total_seconds() / 86400.0

    axis_map = infer_axis_per_segment(df)
    df["axe_mesure"] = df["segment"].map(axis_map)

    L = CUBE_SIDE_M
    for c, out in (("X_m", "dist_face_x"), ("Y_m", "dist_face_y"), ("Z_m", "dist_face_z")):
        d = np.minimum(df[c], L - df[c]).clip(lower=0)
        df[out] = d
    df["dist_face_min"] = df[["dist_face_x", "dist_face_y", "dist_face_z"]].min(axis=1)

    dist_cols = {{"X": "dist_face_x", "Y": "dist_face_y", "Z": "dist_face_z"}}
    df["dist_face_direct"] = df.apply(lambda r: r[dist_cols[r["axe_mesure"]]], axis=1)

    def lateral_min(r):
        others = [dist_cols[a] for a in ("X", "Y", "Z") if a != r["axe_mesure"]]
        return min(r[others[0]], r[others[1]])

    df["dist_face_laterale_min"] = df.apply(lateral_min, axis=1)
    df["plan_profondeur_mm"] = df["segment"].map(PLAN_PROFONDEUR_NOMINAL)
    df["composante"] = "eps_" + df["axe_mesure"].str.lower() + df["axe_mesure"].str.lower()
    df = df.sort_values(["segment", "date_mesure", "x_fibre_m"]).reset_index(drop=True)
    return df


df = load_and_prepare(CSV_PATH)


# ============================================================ style "publication"
def apply_publication_style(fig, xlabel="", ylabel="", title=""):
    """Style sobre pour figure destinée à publication scientifique :
    fond blanc, police serif, axes noirs fins, grille légère, taille lisible."""
    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman, serif", size=14, color="black"),
        title=dict(text=title, font=dict(size=16, family="Times New Roman, serif", color="black")),
        xaxis=dict(title=xlabel, showline=True, linewidth=1, linecolor="black",
                    mirror=True, ticks="outside", gridcolor="rgba(0,0,0,0.08)"),
        yaxis=dict(title=ylabel, showline=True, linewidth=1, linecolor="black",
                    mirror=True, ticks="outside", gridcolor="rgba(0,0,0,0.08)"),
        legend=dict(font=dict(size=11), bordercolor="black", borderwidth=0.5),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=70, r=30, t=60, b=60),
        width=900, height=600,
    )
    return fig


# palette qualitative adaptée impression (colorblind-friendly)
PUB_COLORWAY = px.colors.qualitative.Set1

'''

COMMON_FOOTER = '''

# ============================================================ export
fig.show()
# Export haute résolution pour publication (nécessite kaleido : pip install kaleido)
# fig.write_image("{filename}.png", scale=3)
# fig.write_image("{filename}.pdf")
'''


def _wrap(titre, body, filename):
    return COMMON_HEADER.format(titre=titre) + body + COMMON_FOOTER.format(filename=filename)


def _dates_filter_code(dates_filter):
    """Génère le code de filtrage des dates, pour que le script exporté
    reproduise exactement le sous-échantillonnage temporel choisi dans le
    dashboard (intervalle + ΔT)."""
    import pandas as pd
    if not dates_filter:
        return ""
    iso = [pd.Timestamp(d).strftime("%Y-%m-%d %H:%M:%S") for d in dates_filter]
    lst = ",\n    ".join(f'"{s}"' for s in iso)
    return f'''
# --- Filtre des dates (reproduit l'échantillonnage temporel choisi dans le dashboard) ---
DATES_RETENUES = pd.to_datetime([
    {lst}
])
df = df[df["date_mesure"].isin(DATES_RETENUES)].reset_index(drop=True)
'''


def _build(titre, body, filename=None, dates_filter=None):
    script = COMMON_HEADER.format(titre=titre) + _dates_filter_code(dates_filter) + body
    if filename:
        script += COMMON_FOOTER.format(filename=filename)
    return script


# ------------------------------------------------------------------ onglet 1 & 2

def gen_direct_or_indirect(kind: str, dates_filter=None) -> str:
    """kind = 'direct' ou 'indirect'."""
    col = "dist_face_direct" if kind == "direct" else "dist_face_laterale_min"
    xlabel = "Distance directe (mm)" if kind == "direct" else "Distance latérale min (mm)"
    titre = "Effet direct" if kind == "direct" else "Effet indirect"
    body = f'''
# --- Figure : {titre} ---
axes_presentes = sorted(df["axe_mesure"].dropna().unique())
dates = sorted(df["date_mesure"].dropna().unique())

for axe in axes_presentes:
    sub = df[df["axe_mesure"] == axe]
    fig = go.Figure()
    for i, d in enumerate(dates):
        s = sub[sub["date_mesure"] == d]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s["{col}"] * 1000, y=s["valeur_microdefs"],
            mode="markers",
            marker=dict(size=6, color=PUB_COLORWAY[i % len(PUB_COLORWAY)],
                        line=dict(width=0.5, color="black")),
            name=pd.Timestamp(d).strftime("%d/%m/%Y"),
            opacity=0.75,
        ))
    fig = apply_publication_style(
        fig, xlabel="{xlabel}", ylabel="Déformation (µdef)",
        title=f"{titre} — {{COMPOSANTE_LABEL[axe]}}")
    fig.show()
    # fig.write_image(f"{kind}_{{axe}}.png", scale=3)
'''
    return _build(titre, body, dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 3

def gen_anisotropie(mode_temps="t", mode_temps_deriv="t", dates_filter=None) -> str:
    """mode_temps / mode_temps_deriv : "t" (jours) ou "sqrt_t" (diffusif, loi de Fick)."""
    titre = "Anisotropie du retrait par profondeur nominale"
    body = f'''
# --- Paramètres : axe temporel ("t" ou "sqrt_t") ---
MODE_TEMPS = {mode_temps!r}          # figure de déformation moyenne
MODE_TEMPS_DERIV = {mode_temps_deriv!r}   # figure de vitesse
X_LABELS = {{"t": "Temps écoulé (j)", "sqrt_t": "√t (√j)"}}

# --- Figure 1 : Anisotropie par profondeur nominale ---
g = (df.dropna(subset=["plan_profondeur_mm"])
       .groupby(["plan_profondeur_mm", "axe_mesure", "date_mesure"], as_index=False)
       .agg(moy=("valeur_microdefs", "mean"), std=("valeur_microdefs", "std"),
            t=("temps_ecoule_j", "first")))
g["composante"] = g["axe_mesure"].map(COMPOSANTE_LABEL)
g["sqrt_t"] = np.sqrt(g["t"].clip(lower=0))

for prof in sorted(g["plan_profondeur_mm"].unique()):
    sub_prof = g[g["plan_profondeur_mm"] == prof]
    fig = go.Figure()
    for i, comp in enumerate(sorted(sub_prof["composante"].unique())):
        s = sub_prof[sub_prof["composante"] == comp].sort_values(MODE_TEMPS)
        fig.add_trace(go.Scatter(
            x=s[MODE_TEMPS], y=s["moy"],
            mode="lines+markers",
            error_y=dict(type="data", array=s["std"], visible=True),
            line=dict(color=PUB_COLORWAY[i % len(PUB_COLORWAY)]),
            marker=dict(size=7),
            name=comp,
        ))
    fig = apply_publication_style(
        fig, xlabel=X_LABELS[MODE_TEMPS], ylabel="Déformation moyenne (µdef)",
        title=f"Anisotropie — profondeur {{prof}} mm")
    fig.show()
    # fig.write_image(f"anisotropie_{{prof}}mm.png", scale=3)

# --- Figure 2 : Vitesse de propagation du front (dérivée de la déformation moyenne) ---
g_deriv = g.sort_values(["plan_profondeur_mm", "composante", MODE_TEMPS_DERIV])
deriv_rows = []
for (prof, comp), sub in g_deriv.groupby(["plan_profondeur_mm", "composante"]):
    if len(sub) < 2:
        continue
    v = np.gradient(sub["moy"].values, sub[MODE_TEMPS_DERIV].values)
    deriv_rows.append(pd.DataFrame({{
        "plan": f"{{prof}} mm", "composante": comp,
        "x": sub[MODE_TEMPS_DERIV].values, "vitesse": v}}))

# Axe Z traversant (s1, s2, s3 : pas de profondeur nominale)
gz = (df[df["axe_mesure"] == "Z"]
        .groupby("date_mesure", as_index=False)
        .agg(moy=("valeur_microdefs", "mean"), t=("temps_ecoule_j", "first"))
        .sort_values("t"))
gz["sqrt_t"] = np.sqrt(gz["t"].clip(lower=0))
if len(gz) >= 2:
    vz = np.gradient(gz["moy"].values, gz[MODE_TEMPS_DERIV].values)
    deriv_rows.append(pd.DataFrame({{
        "plan": "axe Z (traversant)", "composante": COMPOSANTE_LABEL["Z"],
        "x": gz[MODE_TEMPS_DERIV].values, "vitesse": vz}}))

if deriv_rows:
    d_all = pd.concat(deriv_rows, ignore_index=True)
    for plan in d_all["plan"].unique():
        sub_plan = d_all[d_all["plan"] == plan]
        fig = go.Figure()
        for i, comp in enumerate(sorted(sub_plan["composante"].unique())):
            s = sub_plan[sub_plan["composante"] == comp]
            fig.add_trace(go.Scatter(
                x=s["x"], y=s["vitesse"], mode="lines+markers",
                line=dict(color=PUB_COLORWAY[i % len(PUB_COLORWAY)]),
                marker=dict(size=7), name=comp))
        unite = "µdef/j" if MODE_TEMPS_DERIV == "t" else "µdef/√j"
        fig = apply_publication_style(
            fig, xlabel=X_LABELS[MODE_TEMPS_DERIV], ylabel=f"Vitesse ({{unite}})",
            title=f"Vitesse de propagation du front — {{plan}}")
        fig.show()
        # fig.write_image(f"vitesse_front_{{plan.replace(' ', '_')}}.png", scale=3)
else:
    print("Pas assez de dates distinctes pour calculer une dérivée temporelle.")
'''
    return _build(titre, body, dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 4

def gen_profil_segment(segment: str, pos_mode: str = "Position sur la fibre (relative)",
                        y_mode: str = "Déformation (µdef)", dates_filter=None) -> str:
    titre = f"Profil spatial le long du segment {segment}"
    if pos_mode.startswith("Position sur"):
        pos_code = 'x_col, x_label = "x_fibre_m", "Position sur la fibre (m)"'
    else:
        pos_code = (
            'axis_col_map = {"X": "X_m", "Y": "Y_m", "Z": "Z_m"}\n'
            'x_col, x_label = axis_col_map[axe], f"Position absolue selon {axe} (m)"'
        )

    if y_mode.startswith("Déformation"):
        body = f'''
# --- Figure : Profil spatial le long du segment ---
SEGMENT = "{segment}"
sub = df[df["segment"] == SEGMENT]
axe = sub["axe_mesure"].iloc[0]
dates = sorted(sub["date_mesure"].dropna().unique())

{pos_code}

fig = go.Figure()
for i, d in enumerate(dates):
    s = sub[sub["date_mesure"] == d].sort_values(x_col)
    fig.add_trace(go.Scatter(
        x=s[x_col], y=s["valeur_microdefs"],
        mode="lines+markers",
        line=dict(color=PUB_COLORWAY[i % len(PUB_COLORWAY)]),
        marker=dict(size=5),
        name=pd.Timestamp(d).strftime("%d/%m/%Y"),
    ))
fig = apply_publication_style(
    fig, xlabel=x_label, ylabel="Déformation (µdef)",
    title=f"{{SEGMENT}} — {{COMPOSANTE_LABEL[axe]}}")
'''
    else:
        body = f'''
# --- Figure : Vitesse (dérivée temporelle) le long du segment ---
# Alignement par indice le long de la fibre (et non par valeur exacte de
# position, qui varie légèrement d'une date à l'autre).
SEGMENT = "{segment}"
sub = df[df["segment"] == SEGMENT]
axe = sub["axe_mesure"].iloc[0]

{pos_code}

sub_sorted = sub.sort_values(["date_mesure", x_col]).copy()
sub_sorted["idx_point"] = sub_sorted.groupby("date_mesure").cumcount()
dates = sorted(sub_sorted["date_mesure"].unique())

mat_val = sub_sorted.pivot(index="idx_point", columns="date_mesure", values="valeur_microdefs")
mat_pos = sub_sorted.pivot(index="idx_point", columns="date_mesure", values=x_col)
mat_val = mat_val.reindex(columns=dates).sort_index()
mat_pos = mat_pos.reindex(columns=dates).sort_index()

x_repr = mat_pos.mean(axis=1)
t_arr = np.array([sub_sorted.loc[sub_sorted["date_mesure"] == d, "temps_ecoule_j"].iloc[0] for d in dates])

mask_full = mat_val.notna().all(axis=1)
mat_val_full = mat_val[mask_full]
x_repr_full = x_repr[mask_full]
vitesses = np.gradient(mat_val_full.values, t_arr, axis=1)

order = np.argsort(x_repr_full.values)
fig = go.Figure()
for i, d in enumerate(dates):
    fig.add_trace(go.Scatter(
        x=x_repr_full.values[order], y=vitesses[order, i],
        mode="lines+markers",
        line=dict(color=PUB_COLORWAY[i % len(PUB_COLORWAY)]),
        marker=dict(size=5),
        name=pd.Timestamp(d).strftime("%d/%m/%Y"),
    ))
fig = apply_publication_style(
    fig, xlabel=x_label, ylabel="Vitesse (µdef/j)",
    title=f"{{SEGMENT}} — {{COMPOSANTE_LABEL[axe]}} — vitesse")
'''
    return _build(titre, body, f"profil_{segment}", dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 5

def gen_effet_coin(seuil_mm: float = 15, dates_filter=None) -> str:
    titre = "Effet de coin (cumul attaque directe + latérale)"
    body = f'''
# --- Figure : Effet de coin ---
SEUIL_MM = {seuil_mm}
DATE_REF = df["date_mesure"].max()  # dernière date disponible ; à ajuster si besoin

d = df[df["date_mesure"] == DATE_REF].copy()
d["zone"] = np.where(d["dist_face_laterale_min"] * 1000 < SEUIL_MM,
                      "Coin (proche 2 faces)", "Face plane")
d["composante"] = d["axe_mesure"].map(COMPOSANTE_LABEL)

for comp in sorted(d["composante"].dropna().unique()):
    sub = d[d["composante"] == comp]
    fig = go.Figure()
    for i, zone in enumerate(sub["zone"].unique()):
        s = sub[sub["zone"] == zone]
        fig.add_trace(go.Scatter(
            x=s["dist_face_direct"] * 1000, y=s["valeur_microdefs"],
            mode="markers",
            marker=dict(size=6, color=PUB_COLORWAY[i % len(PUB_COLORWAY)],
                        line=dict(width=0.5, color="black")),
            name=zone, opacity=0.75,
        ))
    fig = apply_publication_style(
        fig, xlabel="Distance directe (mm)", ylabel="Déformation (µdef)",
        title=f"Effet de coin — {{comp}}")
    fig.show()
    # fig.write_image(f"effet_coin_{{comp}}.png", scale=3)
'''
    return _build(titre, body, dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 6

def gen_carto3d(dates_filter=None) -> str:
    titre = "Cartographie 3D du champ de déformation (avec slider temporel en heures)"
    body = '''
# --- Figure : Cartographie 3D avec slider temporel (en heures écoulées) ---
dates = sorted(df["date_mesure"].dropna().unique())

frames = []
steps = []
for d in dates:
    s = df[df["date_mesure"] == d]
    if s.empty:
        continue
    vmin_d, vmax_d = s["valeur_microdefs"].quantile([0.03, 0.97])
    heures = s["temps_ecoule_j"].iloc[0] * 24
    label = f"{heures:.1f} h"
    frames.append(go.Frame(
        data=[go.Scatter3d(
            x=s["X_m"], y=s["Y_m"], z=s["Z_m"], mode="markers",
            marker=dict(size=4, color=s["valeur_microdefs"], colorscale="RdBu_r",
                        cmin=vmin_d, cmax=vmax_d, colorbar=dict(title="µdef"),
                        line=dict(width=0.5, color="black")),
            text=s["segment"] + " (" + s["axe_mesure"] + ")",
            hovertemplate="%{text}<br>µdef=%{marker.color:.0f}<extra></extra>",
        )],
        name=label))
    steps.append(dict(method="animate", args=[[label], {"mode": "immediate"}], label=label))

fig = go.Figure(data=frames[0].data, frames=frames)
fig.update_layout(
    template="simple_white",
    font=dict(family="Times New Roman, serif", size=13, color="black"),
    title="Cartographie 3D du champ de déformation",
    scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
               aspectmode="cube"),
    width=850, height=700, paper_bgcolor="white",
    sliders=[dict(steps=steps, currentvalue=dict(prefix="Temps écoulé : "))],
)
'''
    return _build(titre, body, "carto3d", dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 7

def gen_symetrie(dates_filter=None) -> str:
    titre = "Symétrie face haute vs face basse"
    body = '''
# --- Figure : Symétrie face haute (Z≈100mm) vs face basse (Z≈0) ---
d = df.dropna(subset=["plan_profondeur_mm"]).copy()
d["face_z"] = np.where(d["Z_m"] < CUBE_SIDE_M / 2, "Face basse (Z≈0)", "Face haute (Z≈100mm)")
g = d.groupby(["plan_profondeur_mm", "face_z", "date_mesure"], as_index=False).agg(
    moy=("valeur_microdefs", "mean"), t=("temps_ecoule_j", "first"))

for prof in sorted(g["plan_profondeur_mm"].unique()):
    sub_prof = g[g["plan_profondeur_mm"] == prof]
    fig = go.Figure()
    for i, face in enumerate(sub_prof["face_z"].unique()):
        s = sub_prof[sub_prof["face_z"] == face].sort_values("t")
        fig.add_trace(go.Scatter(x=s["t"], y=s["moy"], mode="lines+markers",
                                  line=dict(color=PUB_COLORWAY[i % len(PUB_COLORWAY)]),
                                  marker=dict(size=7), name=face))
    fig = apply_publication_style(
        fig, xlabel="Temps écoulé (j)", ylabel="Déformation moyenne (µdef)",
        title=f"Symétrie H/B — profondeur {prof} mm")
    fig.show()
    # fig.write_image(f"symetrie_{prof}mm.png", scale=3)
'''
    return _build(titre, body, dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 8

def gen_heatmap_segdate(dates_filter=None) -> str:
    titre = "Déformation moyenne par segment et par date"
    body = '''
# --- Figure : Heatmap segment x date ---
g = df.groupby(["segment", "date_mesure"], as_index=False).agg(moy=("valeur_microdefs", "mean"))
axe_map = df.drop_duplicates("segment").set_index("segment")["axe_mesure"]
g["label_segment"] = g["segment"] + " (" + g["segment"].map(axe_map) + ")"
piv = g.pivot(index="label_segment", columns="date_mesure", values="moy")
piv = piv.reindex(sorted(piv.index, key=lambda s: int(s.split("s")[1].split(" ")[0])))

fig = go.Figure(go.Heatmap(
    z=piv.values, x=[pd.Timestamp(c).strftime("%d/%m/%y") for c in piv.columns], y=piv.index,
    colorscale="RdBu_r", zmid=0, colorbar=dict(title="µdef")))
fig.update_layout(
    template="simple_white",
    font=dict(family="Times New Roman, serif", size=13, color="black"),
    title="Déformation moyenne par segment et par date",
    xaxis_title="Date", yaxis_title="Segment (axe)",
    width=900, height=700, paper_bgcolor="white",
)
'''
    return _build(titre, body, "heatmap_seg_date", dates_filter=dates_filter)


# ------------------------------------------------------------------ onglet 9

def gen_surface_segment(segment: str, kind: str = "heatmap",
                          pos_mode: str = "Position sur la fibre (relative)", dates_filter=None) -> str:
    titre = f"Évolution position × temps — {segment}"
    if pos_mode.startswith("Position sur"):
        pos_code = 'x_col, x_label = "x_fibre_m", "Position sur la fibre (m)"'
    else:
        pos_code = (
            'axis_col_map = {"X": "X_m", "Y": "Y_m", "Z": "Z_m"}\n'
            'x_col, x_label = axis_col_map[axe], f"Position absolue selon {axe} (m)"'
        )
    body = f'''
# --- Figure : Surface/Heatmap position x temps ---
from scipy.interpolate import griddata

SEGMENT = "{segment}"
sub = df[df["segment"] == SEGMENT]
axe = sub["axe_mesure"].iloc[0]

{pos_code}

x = sub[x_col].values
t = sub["temps_ecoule_j"].values
z = sub["valeur_microdefs"].values
xi = np.linspace(x.min(), x.max(), 150)
ti = np.linspace(t.min(), t.max(), 60)
XI, TI = np.meshgrid(xi, ti)
ZI = griddata((x, t), z, (XI, TI), method="linear")

'''
    if kind == "surface":
        body += '''
fig = go.Figure(go.Surface(x=xi, y=ti, z=ZI, colorscale="RdBu_r", cmid=0,
                            colorbar=dict(title="µdef")))
fig.update_layout(
    template="simple_white",
    font=dict(family="Times New Roman, serif", size=13, color="black"),
    scene=dict(xaxis_title=x_label, yaxis_title="Temps (j)", zaxis_title="µdef"),
    title=f"{SEGMENT} — {COMPOSANTE_LABEL[axe]}",
    width=850, height=650, paper_bgcolor="white",
)
'''
    else:
        body += '''
fig = go.Figure(go.Heatmap(x=xi, y=ti, z=ZI, colorscale="RdBu_r", zmid=0,
                            colorbar=dict(title="µdef")))
fig.update_layout(
    template="simple_white",
    font=dict(family="Times New Roman, serif", size=13, color="black"),
    xaxis_title=x_label, yaxis_title="Temps écoulé (j)",
    title=f"{SEGMENT} — {COMPOSANTE_LABEL[axe]}",
    width=850, height=600, paper_bgcolor="white",
)
'''
    return _build(titre, body, f"surface_{segment}", dates_filter=dates_filter)

# ------------------------------------------------------------------ onglet 10

def gen_operation_fibres(segs_a, segs_b, agg="mean", combine="none", absval=False,
                          dates_filter=None) -> str:
    titre = "Opération entre fibres"
    label = (f"{fiber_ops.AGG_LABELS[agg]} — combinaison : {fiber_ops.COMBINE_LABELS[combine]}"
             + (" (valeur absolue)" if absval else ""))
    params = f'''
# --- Paramètres de l'opération ---
SEGS_A = {list(segs_a)!r}   # groupe A
SEGS_B = {list(segs_b)!r}   # groupe B (utilisé seulement si COMBINE != "none")
AGG = {agg!r}               # mean | sum | median | min | max | std  (appliqué dans chaque groupe)
COMBINE = {combine!r}       # none | a-b | a+b | a/b | (a+b)/2
ABSVAL = {absval!r}
TITRE = {label!r}

import warnings

'''
    # même code source que celui du dashboard -> résultats identiques
    funcs = "\n\n".join(inspect.getsource(f) for f in
                        (fiber_ops._aggregate, fiber_ops._combine, fiber_ops.compute_operation))
    plot = '''

# --- Calcul ---
res, profils = compute_operation(df, SEGS_A, SEGS_B, AGG, COMBINE, ABSVAL, L=CUBE_SIDE_M)
# res.to_csv("operation_fibres.csv", index=False)

# --- Figure 1 : résultat le long des fibres (0 -> L), une courbe par date ---
fig = go.Figure()
for i, d in enumerate(sorted(res["date_mesure"].unique())):
    s = res[res["date_mesure"] == d]
    fig.add_trace(go.Scatter(
        x=s["pos_m"] * 1000, y=s["valeur"], mode="lines",
        line=dict(color=PUB_COLORWAY[i % len(PUB_COLORWAY)]),
        name=pd.Timestamp(d).strftime("%d/%m/%Y")))
fig = apply_publication_style(
    fig, xlabel="Position le long des fibres (mm)", ylabel="Valeur (µdef)", title=TITRE)
fig.show()

# --- Figure 2 : valeur moyenne du résultat en fonction du temps ---
m = res.groupby("date_mesure", as_index=False).agg(
    v=("valeur", "mean"), t=("temps_ecoule_j", "first")).sort_values("t")
fig2 = go.Figure(go.Scatter(x=m["t"], y=m["v"], mode="lines+markers",
                             line=dict(color=PUB_COLORWAY[0]), marker=dict(size=7)))
fig2 = apply_publication_style(
    fig2, xlabel="Temps écoulé (j)", ylabel="Valeur moyenne (µdef)", title=TITRE)
fig2.show()
fig = fig  # pour l'export ci-dessous
'''
    return _build(titre, params + funcs + plot, "operation_fibres", dates_filter=dates_filter)