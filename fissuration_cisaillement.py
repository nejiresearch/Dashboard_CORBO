"""
Quantification de la fissuration par transfert de déformation (shear lag) sur les données CORBO,
selon la méthodologie d'OBRstudio (v1.6.19), adaptée à nos cubes de 10 cm.

Adaptation de fissuration_cisaillement.py :
  - données réelles : résidu r(x, t) = ε_mesuré − retrait (fit_retrait.py), comme fissuration_gaussiennes.py ;
  - noyau d'une fissure d'ouverture w (COD) au centre c : profil EXPONENTIEL-TRIANGULAIRE (modèle OBRstudio)
        ε(s) = P (1 − λ|s|/(1+λa))        pour |s| ≤ a   (« cœur » en glissement, décroissance linéaire)
        ε(s) = P/(1+λa) · exp(−λ(|s|−a))  pour |s| > a   (transfert exponentiel)
    avec a = 0 on retrouve le noyau exponentiel pur. COD = ∫ε dx = (2P/λ)(1+s+s²/2)/(1+s), s = λa ;
    on paramètre par w (COD) et on en déduit P. Le noyau est MOYENNÉ sur la jauge de l'OBR de façon analytique ;
  - FOND : un décalage b(t) (ε_fond, µε) par échéance est ajusté avec les fissures (écart global du retrait) ;
  - DÉTECTION comme OBRstudio : pics de proéminence ≥ PROM_FACTEUR × bruit (auto), ouverture initiale = aire du profil
    au-dessus d'un plancher, puis affinage par moindres carrés ; complément glouton ;
  - méthode « aires » conservée comme recoupement indépendant (colonne w_aires_um) ;
  - élagage des fissures d'ouverture < COD mini, plafond de COD, bornes de λ et de a ;
  - fissures forcées / interdites, presets « câble / pose » ;
  - global : w_i(t) non décroissant, a_i(t) non décroissant, λ_i CONSTANT dans le temps (propriété de l'interface),
    b(t) libre, centres communs à toutes les dates ;
  - centres déterminés sur la DERNIÈRE échéance, puis suivi à rebours.

Unités : x en mm, λ en 1/mm (affiché en m⁻¹), a en mm, w en µm, ε en µε.

Usage :
    python fissuration_cisaillement.py --cube C12 --segment s14 --plot s14_sl.png --csv s14_sl.csv
"""
import argparse
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.signal import argrelmin, find_peaks
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse import csr_matrix, lil_matrix

_trapz = getattr(np, "trapezoid", None) or np.trapz   # NumPy ≥ 2 : trapezoid

import fit_retrait as F
from fissuration_gaussiennes import donnees_segment, THETA_DEFAUT, T0_DEFAUT

# ----------------------------------------------------------------------------- réglages
JAUGE_MM = 5.0
PORTEE_MAX_MM = 300.0            # portée max du transfert : λ ≥ 3/portée
LAM_MIN = 3.0 / PORTEE_MAX_MM    # 1/mm
LAM_MAX = 5.0 / JAUGE_MM         # 1/mm : λ ≤ 5/jauge
LAM_INIT = 0.1
A_CUBE_MM = 12.0                 # le cube fait 100 mm : un cœur plus large se comporterait comme une gaussienne de fond
A_MAX = A_CUBE_MM                # mm : demi-largeur max du cœur
A_INIT = 2.0
COD_MIN = 2.0                    # µm : ouverture minimale d'une fissure (élagage)
PLAFOND_UM = 5000.0              # µm : COD maximal
PROM_FACTEUR = 2.0               # proéminence auto = facteur × bruit (MAD ≈ rugosité du signal, pas bruit instrumental pur)
PLANCHER = "local"               # "local" (quart bas des creux) | "zero"
FOND = True                      # ajuste un décalage b(t) par échéance
FOND_MAX = 3e4                   # µε
W_NEG = 0.3                      # poids des résidus où la mesure est SOUS le modèle (creux non modélisables)
ADD_GAIN = 0.04                  # gain relatif minimal du SSE pour ajouter une fissure (complément glouton)
PROM_MULTI = 1.5                 # facteur de bruit pour la détection multi-échéances
LARGEUR_MULTI = 10.0             # mm : 1/λ max par défaut avec --multi (évite les fissures fantômes de fond)
GLOUTON_MULTI = False            # complément glouton après --multi (lent, ajoute des fissures de bord)
MARGE_BORD = 0.0                 # mm : aucune fissure détectée à moins de cette distance des extrémités de la fibre (effets de bord)
DUREE_MIN = 10.0                 # j : durée minimale d'une trajectoire de pic
W_MIN = 5.0                      # µm : ouverture initiale minimale d'un candidat
DIST_MIN = 2.0                   # mm : deux centres plus proches sont fusionnés (le plus faible est retiré)
DECALAGE_MAX = 4.0               # mm
N_MAX = 25

# Presets « câble / pose » (OBRstudio). inv_lambda_mm : 1/λ imposé du câble. Le cœur est borné à A_CUBE_MM.
PRESETS = {
    "peek_cobra_beton":       dict(inv_lambda_mm=10.5, coeur_max_mm=200, portee_max_mm=300, plafond_mm=2.0, cod_min_um=10, jauge_mm=4),
    "polyimide_nue_beton":    dict(coeur_max_mm=25, portee_max_mm=30, plafond_mm=0.5, cod_min_um=2, jauge_mm=1),
    "polyimide_collee_epoxy": dict(coeur_max_mm=5, portee_max_mm=10, plafond_mm=0.2, cod_min_um=0.5, jauge_mm=0.5),
}


# ----------------------------------------------------------------------------- modèle exponentiel-triangulaire
def _profil(s, P, a, lam):
    s = np.abs(s)
    u = 1.0 + lam * a
    return np.where(s <= a, P * (1.0 - lam * s / u), P / u * np.exp(-lam * (s - a)))


def _prim(e, P, a, lam):
    """Primitive (depuis −∞) du profil centré : ∫_{-∞}^{e} ε(s) ds, en µε·mm."""
    e = np.asarray(e, dtype=float)
    u = 1.0 + lam * a
    c = lam * P / u
    i = P / (lam * u)
    l = i + P * a - c * a * a / 2.0
    out = np.empty_like(e)
    m1 = e <= -a
    m2 = (e > -a) & (e <= 0)
    m3 = (e > 0) & (e <= a)
    m4 = e > a
    out[m1] = i * np.exp(lam * (e[m1] + a))
    out[m2] = i + P * (e[m2] + a) + c * (e[m2] ** 2 - a * a) / 2.0
    out[m3] = l + P * e[m3] - c * e[m3] ** 2 / 2.0
    out[m4] = l + P * a - c * a * a / 2.0 + i * (1.0 - np.exp(-lam * (e[m4] - a)))
    return out


def amplitude(w, a, lam):
    """Pic P (µε) d'une fissure d'ouverture w (µm)."""
    s = lam * a
    return 500.0 * w * lam * (1.0 + s) / (1.0 + s + s * s / 2.0)


def noyau(x, c, w, lam, a=0.0, jauge=None):
    """Déformation (µε) vue par la fibre pour une fissure (c, w, λ, a), moyennée sur la jauge."""
    jauge = JAUGE_MM if jauge is None else jauge
    P = amplitude(w, a, lam)
    s = x - c
    if jauge <= 0:
        return _profil(s, P, a, lam)
    return (_prim(s + jauge / 2, P, a, lam) - _prim(s - jauge / 2, P, a, lam)) / jauge


def modele(x, C, W, L, A=None, b=0.0):
    """Somme des noyaux + fond b (µε)."""
    A = np.zeros(len(C)) if A is None else A
    y = np.full(len(x), float(b))
    for c, w, l, a in zip(C, W, L, A):
        y += noyau(x, c, w, l, a)
    return y


def _pond(f):
    """Pondération asymétrique : mesure sous le modèle (f > 0) pèse W_NEG."""
    return np.where(f > 0, np.sqrt(W_NEG) * f, f)


# ----------------------------------------------------------------------------- inversion analytique d'une paire
def _creux(beta, E1, E3, d):
    q = np.exp(-beta * d)
    return 2 * np.exp(-beta * d / 2) / (1 - q ** 2) * np.sqrt(max((E1 - E3 * q) * (E3 - E1 * q), 0.0))


def extraire_paire(E1, E2, E3, d):
    """Inversion analytique d'une paire de fissures rapprochées (noyau exponentiel pur, a = 0, sans moyenne sur la jauge).
    Unités : E en µε, d en mm, β en 1/mm, δ (= COD/2) en µm. Retourne (β, δ1, δ2) ou None."""
    if min(E1, E2, E3) <= 0 or E2 >= min(E1, E3) or d <= 0:
        return None
    b_lo = np.log(max(E1, E3) / min(E1, E3)) / d + 1e-9
    b = np.geomspace(b_lo, max(b_lo, 1e-3) * 1e4 + 50 / d, 800)
    g = np.array([_creux(x, E1, E3, d) - E2 for x in b])
    k = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    if len(k) == 0:
        return None
    beta = brentq(lambda x: _creux(x, E1, E3, d) - E2, b[k[-1]], b[k[-1] + 1])
    q = np.exp(-beta * d); den = beta * (1 - q ** 2)
    return beta, 1e-3 * (E1 - E3 * q) / den, 1e-3 * (E3 - E1 * q) / den


def estimer_beta(x, r, sigma=None, d_min=4.0, d_max=60.0, verbose=False):
    """β par paires de pics voisins du résidu lissé. Retourne la liste des (x1, x2, β, COD1, COD2)."""
    sigma = bruit_mad(r) if sigma is None else sigma
    pas = float(np.median(np.diff(x)))
    rl = lisser(r, pas)
    pics, _ = find_peaks(rl, prominence=sigma)
    out = []
    for a, b in zip(pics[:-1], pics[1:]):
        d = x[b] - x[a]
        if not d_min <= d <= d_max:
            continue
        sol = extraire_paire(rl[a], rl[a:b + 1].min(), rl[b], d)
        if verbose:
            print(f"    paire X={x[a]:.0f}-{x[b]:.0f} mm : E1={rl[a]:.0f} E2={rl[a:b + 1].min():.0f} E3={rl[b]:.0f} -> "
                  + ("pas de solution" if sol is None else f"β={1000 * sol[0]:.0f} m⁻¹, COD={2 * sol[1]:.0f}/{2 * sol[2]:.0f} µm"))
        if sol is not None:
            out.append((x[a], x[b], sol[0], 2 * sol[1], 2 * sol[2]))
    return out


# ----------------------------------------------------------------------------- bruit, signe, lissage, plancher, aires
def bruit_mad(r):
    """MAD des différences secondes (rugosité court terme ; surestime le bruit instrumental pur)."""
    d2 = r[2:] - 2 * r[1:-1] + r[:-2]
    return 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6)


def signe_auto(r):
    """+1 si le résidu dépasse plus au-dessus qu'au-dessous de sa médiane (déconseillé : la physique impose +)."""
    d = r[np.isfinite(r)] - np.median(r[np.isfinite(r)])
    return 1.0 if d[d > 0].sum() >= -d[d < 0].sum() else -1.0


def lisser(r, pas, largeur_mm=None):
    largeur_mm = JAUGE_MM if largeur_mm is None else largeur_mm
    n = max(int(round(largeur_mm / pas)), 1)
    return np.convolve(r, np.ones(n) / n, mode="same")


def fond_initial(x, r, mode=None):
    """Niveau du « béton sain » (µε) : quart bas des creux du profil lissé (minima espacés de ≥ 10 mm). Constant le long de la fibre
    (la fibre ne fait que 100 mm, l'échelle de ±2 m d'OBRstudio n'a pas de sens ici). 0 si mode « zero »."""
    mode = PLANCHER if mode is None else mode
    if mode == "zero":
        return 0.0
    pas = float(np.median(np.diff(x)))
    rl = lisser(r, pas)
    mins = argrelmin(rl, order=max(int(round(10.0 / pas)), 1))[0]
    vm = rl[mins] if len(mins) >= 3 else rl
    return float(np.quantile(vm, 0.25))


def aires(x, r, centres, plancher, portee=None):
    """Méthode « aires » : ouverture (µm) = aire du profil au-dessus du plancher (constante, µε) sur le territoire de la fissure.
    Territoire : jusqu'au creux qui sépare du voisin ; aux extrémités, jusqu'au minimum du profil lissé dans la portée."""
    portee = min(PORTEE_MAX_MM / 2, 50.0) if portee is None else portee
    o = np.argsort(x); x, r = np.asarray(x)[o], np.asarray(r)[o]      # certaines fibres sont parcourues à l'envers
    pas = float(np.median(np.diff(x)))
    rl = lisser(r, pas)
    idx = np.argsort(centres)
    ic = np.array([int(np.argmin(np.abs(x - centres[i]))) for i in idx])
    nr = max(int(round(portee / pas)), 2)
    bornes = []
    for j, i0 in enumerate(ic):
        if j == 0:
            lo_i = max(0, i0 - nr); g = lo_i + int(np.argmin(rl[lo_i:i0 + 1]))
        else:
            g = ic[j - 1] + int(np.argmin(rl[ic[j - 1]:i0 + 1]))
        if j == len(ic) - 1:
            hi_i = min(len(x) - 1, i0 + nr); d = i0 + int(np.argmin(rl[i0:hi_i + 1]))
        else:
            d = i0 + int(np.argmin(rl[i0:ic[j + 1] + 1]))
        bornes.append((g, d))
    out = np.zeros(len(centres))
    for k, i in enumerate(idx):
        g, d = bornes[k]
        sl = slice(g, d + 1)
        out[i] = max(0.0, 1e-3 * _trapz(r[sl] - plancher, x[sl]))
    return out


# ----------------------------------------------------------------------------- ajustement sur une date
def ajuster_date(x, cible, C, W, L, A, libres, b0=0.0, avec_fond=None):
    """Moindres carrés (w, λ, a, b pour toutes les fissures ; centres seulement pour les `libres`).
    Retourne c, w, λ, a, b, SSE (SSE pondéré par W_NEG)."""
    avec_fond = FOND if avec_fond is None else avec_fond
    n = len(C); C0 = np.asarray(C, float); fi = np.where(libres)[0]

    def f(q):
        c = C0.copy(); c[fi] = q[3 * n + 1:]
        return _pond(modele(x, c, q[:n], q[n:2 * n], q[2 * n:3 * n], q[3 * n]) - cible)

    bb = (-FOND_MAX, FOND_MAX) if avec_fond else (-1e-6, 1e-6)
    q0 = np.r_[W, L, A, b0 if avec_fond else 0.0, C0[fi]]
    lo = np.r_[np.zeros(n), np.full(n, LAM_MIN), np.zeros(n), bb[0], np.minimum(np.maximum(C0[fi] - DECALAGE_MAX, x.min() + MARGE_BORD), C0[fi] - 1e-6)]
    hi = np.r_[np.full(n, PLAFOND_UM), np.full(n, LAM_MAX), np.full(n, max(A_MAX, 1e-6)), bb[1], np.maximum(np.minimum(C0[fi] + DECALAGE_MAX, x.max() - MARGE_BORD), C0[fi] + 1e-6)]
    q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
    s = least_squares(f, q0, bounds=(lo, hi), max_nfev=80,
                      x_scale=np.r_[np.full(n, 20.0), np.full(n, 0.05), np.full(n, 2.0), 500.0, np.full(len(fi), 2.0)])
    c = C0.copy(); c[fi] = s.x[3 * n + 1:]
    return c, s.x[:n], s.x[n:2 * n], s.x[2 * n:3 * n], float(s.x[3 * n]), float(np.sum(s.fun ** 2))


def _retirer(i, *tabs):
    return tuple(np.delete(t, i) for t in tabs)


def candidats_multi(xs, rs, ts, facteur, duree_min, tol, interdites=(), verbose=False):
    """Détection sur toutes les échéances : pics de chaque résidu lissé (proéminence facteur × bruit de la date),
    reliés en trajectoires (plus proche trajectoire à moins de tol, sinon nouvelle). Une trajectoire est validée si elle dure
    ≥ duree_min jours ou si elle est présente aux deux dernières échéances. Retourne les positions (médiane des pics)."""
    traj = []                                              # [dernière position, [positions], [dates]]
    for k, (x, r) in enumerate(zip(xs, rs)):
        pas = float(np.median(np.diff(x)))
        o = np.argsort(x); x, r = x[o], r[o]
        pk = x[find_peaks(lisser(r, pas), prominence=facteur * bruit_mad(r))[0]]
        libres = list(range(len(traj)))
        for c in pk:
            j = min(libres, key=lambda i: abs(traj[i][0] - c), default=None)
            if j is not None and abs(traj[j][0] - c) < tol:
                libres.remove(j); traj[j][0] = c; traj[j][1].append(c); traj[j][2].append(k)
            else:
                traj.append([c, [c], [k]])
    T = len(ts); C = []
    for _, pos, ks in traj:
        duree = ts[ks[-1]] - ts[ks[0]]
        c = float(np.median(pos))
        ok = (duree >= duree_min or ks[-1] >= T - 2) and not any(abs(c - i) < tol for i in interdites)
        if verbose:
            print(f"    trajectoire X={c:5.1f} mm : {len(ks):2d} dates, {ts[ks[0]]:.0f}→{ts[ks[-1]]:.0f} j  {'✓' if ok else '✗'}")
        if ok:
            C.append(c)
    return sorted(C)


def detecter(x, r, prom=None, forcees=(), interdites=(), verbose=False, cand_ext=None, glouton=True):
    """Détection sur la dernière échéance, à la OBRstudio :
       1. candidats = pics du résidu lissé de proéminence ≥ prom (auto : PROM_FACTEUR × bruit), sauf positions interdites,
          plus les positions forcées ;
       2. ouverture initiale = aire au-dessus du fond ;
       3. ajustement global (w, λ, a, b, c) ;
       4. élagage : fusion des centres < DIST_MIN, retrait des fissures d'ouverture < COD_MIN (hors forcées) ;
       5. complément glouton sur le résidu non expliqué.
    Retourne C, W, L, A, b."""
    pas = float(np.median(np.diff(x)))
    tol = max(1.5 * pas, JAUGE_MM / 4, 3.0)
    sigma = bruit_mad(r)
    prom = PROM_FACTEUR * sigma if not prom else prom
    rl = lisser(r, pas)
    b0 = fond_initial(x, r)
    xmin, xmax = float(x.min()), float(x.max())
    bord = lambda c: c < xmin + MARGE_BORD or c > xmax - MARGE_BORD
    interdit = lambda c: any(abs(c - i) < tol for i in interdites) or bord(c)
    cand = [c for c in x[find_peaks(rl, prominence=prom)[0]] if not interdit(c)]
    forcees = [f for f in forcees if f is not None]
    for f in forcees:
        if not any(abs(c - f) < tol for c in cand):
            cand.append(f)
    if cand_ext is not None:
        cand = [c for c in cand_ext if not interdit(c)] + [f for f in forcees if f is not None]
    cand = sorted(set(cand))
    if verbose:
        print(f"  bruit {sigma:.1f} µε, proéminence {prom:.0f} µε, fond initial {b0:.0f} µε, {len(cand)} candidat(s) (tol. {tol:.1f} mm)")
    vide = (np.zeros(0),) * 4 + (b0,)
    if not cand:
        return vide
    C = np.array(cand)
    forc = np.array([any(abs(c - f) < tol for f in forcees) for c in C])
    W = np.maximum(aires(x, r, C, b0), W_MIN)
    L = np.full(len(C), np.clip(LAM_INIT, LAM_MIN, LAM_MAX))
    A = np.full(len(C), min(A_INIT, A_MAX))
    C, W, L, A, b, sse = ajuster_date(x, r, C, W, L, A, np.ones(len(C), bool), b0)
    while len(C) > 0:
        ordre = np.argsort(C); dd = np.diff(C[ordre])
        i_ret = None
        if len(C) > 1 and dd.min() < DIST_MIN:
            j = int(np.argmin(dd)); p, q = ordre[j], ordre[j + 1]
            i_ret = p if (W[p] < W[q] and not forc[p]) or forc[q] else q
            if forc[i_ret]:
                i_ret = None
        if i_ret is None:
            faibles = [i for i in range(len(C)) if W[i] < COD_MIN and not forc[i]]
            if not faibles:
                break
            i_ret = min(faibles, key=lambda i: W[i])
        if verbose:
            print(f"  − fissure en X={C[i_ret]:.1f} mm élaguée (w={W[i_ret]:.1f} µm)")
        C, W, L, A, forc = _retirer(i_ret, C, W, L, A, forc)
        if len(C) == 0:
            return vide
        C, W, L, A, b, sse = ajuster_date(x, r, C, W, L, A, np.ones(len(C), bool), b)
    # complément glouton (résidu non expliqué)
    while len(C) < N_MAX and glouton:
        res = r - modele(x, C, W, L, A, b)
        resl = lisser(res, pas)
        i = int(np.argmax(resl))
        if resl[i] <= prom or interdit(x[i]):
            break
        meilleur = None
        for lam0 in (0.05, 0.15, 0.4):
            lam0 = float(np.clip(lam0, LAM_MIN, LAM_MAX))
            w0 = 2 * res[i] / (500.0 * lam0)
            cn, wn, ln, an, _, e = ajuster_date(x, res, [x[i]], [w0], [lam0], [min(A_INIT, A_MAX)], [True], 0.0, False)
            if meilleur is None or e < meilleur[4]:
                meilleur = (cn[0], wn[0], ln[0], an[0], e)
        C2 = np.r_[C, meilleur[0]]; W2 = np.r_[W, meilleur[1]]; L2 = np.r_[L, meilleur[2]]; A2 = np.r_[A, meilleur[3]]
        c, w, l, a, b2, sse2 = ajuster_date(x, r, C2, W2, L2, A2, np.ones(len(C2), bool), b)
        proche = len(c) > 1 and np.min(np.diff(np.sort(c))) < DIST_MIN
        if verbose:
            print(f"    essai X={x[i]:.0f}: gain {1 - sse2 / sse:.3f}, w_min {w.min():.0f}, proche={proche}")
        if sse2 > sse * (1 - ADD_GAIN) or w[-1] < COD_MIN or proche:
            break
        C, W, L, A, b, sse = c, w, l, a, b2, sse2
        forc = np.r_[forc, False]
        if verbose:
            print(f"  + fissure en X={x[i]:.1f} mm  (SSE {sse:.3g})")
    o = np.argsort(C)
    return C[o], W[o], L[o], A[o], b


# ----------------------------------------------------------------------------- ajustement global régularisé
def global_regularise(xs, rs, C0, W0, L0, A0, B0):
    """Paramètres : incréments de w ≥ 0 (w_i(t) non décroissant), λ_i constant, incréments de a ≥ 0 (a_i(t) non décroissant),
    centres c_i communs, fond b(t_k). Pas de pénalité : la régularité vient de la paramétrisation."""
    n, T = W0.shape
    nT = n * T
    D0 = np.diff(np.c_[np.zeros(n), np.maximum.accumulate(W0, axis=1)], axis=1)
    DA0 = np.diff(np.c_[np.zeros(n), np.maximum.accumulate(A0, axis=1)], axis=1)
    L0 = np.median(L0, axis=1)
    nd = [len(x) for x in xs]; off = np.r_[0, np.cumsum(nd)]

    def depaqueter(q):
        D = q[:nT].reshape(n, T); L = q[nT:nT + n]; DA = q[nT + n:2 * nT + n].reshape(n, T)
        C = q[2 * nT + n:2 * nT + 2 * n]; B = q[2 * nT + 2 * n:]
        return np.cumsum(D, axis=1), L, np.minimum(np.cumsum(DA, axis=1), A_MAX), C, B

    def f(q):
        W, L, A, C, B = depaqueter(q)
        return np.concatenate([_pond(modele(xs[k], C, W[:, k], L, A[:, k], B[k]) - rs[k]) for k in range(T)])

    # jacobienne : différences finies sur les paramètres directs p = (W, L, A, C, B) (peu de couleurs, chaque W[i,k] ne
    # touche que la date k), puis composition avec la somme cumulée q -> p
    sp = lil_matrix((off[-1], 2 * nT + 2 * n + T), dtype=int)
    for k in range(T):
        rows = slice(off[k], off[k + 1])
        for i in range(n):
            sp[rows, i * T + k] = 1
            sp[rows, nT + i] = 1
            sp[rows, nT + n + i * T + k] = 1
            sp[rows, 2 * nT + n + i] = 1
        sp[rows, 2 * nT + 2 * n + k] = 1
    tri = np.tril(np.ones((T, T)))
    M = lil_matrix((2 * nT + 2 * n + T,) * 2)
    for i in range(n):
        M[i * T:(i + 1) * T, i * T:(i + 1) * T] = tri
        M[nT + n + i * T:nT + n + (i + 1) * T, nT + n + i * T:nT + n + (i + 1) * T] = tri
    M[nT:nT + n, nT:nT + n] = np.eye(n)
    M[2 * nT + n:, 2 * nT + n:] = np.eye(n + T)
    M = M.tocsr()

    def fp(p):
        W = p[:nT].reshape(n, T); L = p[nT:nT + n]; A = np.minimum(p[nT + n:2 * nT + n].reshape(n, T), A_MAX)
        C = p[2 * nT + n:2 * nT + 2 * n]; B = p[2 * nT + 2 * n:]
        return np.concatenate([_pond(modele(xs[k], C, W[:, k], L, A[:, k], B[k]) - rs[k]) for k in range(T)])

    def jac(q):
        p = M @ q
        Jp = approx_derivative(fp, p, rel_step=1e-6, sparsity=sp, f0=fp(p))
        return csr_matrix(Jp) @ M

    q0 = np.r_[D0.ravel(), L0, DA0.ravel(), C0, B0]
    bb = FOND_MAX if FOND else 1e-6
    lo = np.r_[np.zeros(nT), np.full(n, LAM_MIN), np.zeros(nT), np.minimum(np.maximum(C0 - DECALAGE_MAX, xs[0].min() + MARGE_BORD), C0 - 1e-6), np.full(T, -bb)]
    hi = np.r_[np.full(nT, PLAFOND_UM), np.full(n, LAM_MAX), np.full(nT, max(A_MAX, 1e-6)), np.maximum(np.minimum(C0 + DECALAGE_MAX, xs[0].max() - MARGE_BORD), C0 + 1e-6), np.full(T, bb)]
    q0 = np.clip(q0, lo + 1e-9, hi - 1e-9)
    s = least_squares(f, q0, bounds=(lo, hi), jac=jac, max_nfev=60,
                      x_scale=np.r_[np.full(nT, 5.0), np.full(n, 0.05), np.full(nT, 2.0), np.full(n, 2.0), np.full(T, 500.0)])
    return depaqueter(s.x)


# ----------------------------------------------------------------------------- réglages globaux
def fixer_lambda(lam, tolerance=0.0):
    """λ global (1/mm) : borne λ à [λ(1-tol), λ(1+tol)] (tol = 0 : λ imposé)."""
    global LAM_MIN, LAM_MAX, LAM_INIT
    LAM_MIN, LAM_MAX, LAM_INIT = lam * (1 - tolerance) - 1e-9, lam * (1 + tolerance) + 1e-9, lam


# paramètre de l'interface / CLI -> constante du module (réglages simples)
REGLAGES_SIMPLES = dict(jauge="JAUGE_MM", prom_facteur="PROM_FACTEUR", prom_multi="PROM_MULTI", glouton="GLOUTON_MULTI",
                        marge_bord="MARGE_BORD", a_max="A_MAX", codmin="COD_MIN", w_neg="W_NEG", plancher="PLANCHER",
                        dist_min="DIST_MIN", decalage_max="DECALAGE_MAX", n_max="N_MAX", w_min="W_MIN", add_gain="ADD_GAIN",
                        fond="FOND")
_CONSTANTES = sorted(set(REGLAGES_SIMPLES.values()) | {"LAM_MIN", "LAM_MAX", "LAM_INIT", "PORTEE_MAX_MM", "PLAFOND_UM"})


def configurer(preset=None, tol_lambda=0.0, largeur_max=None, plafond_mm=None, multi=False, **kw):
    """Applique les réglages aux constantes du module (mêmes noms que les options de la ligne de commande).
    Ordre : bornes de λ par défaut, jauge, preset, puis réglages individuels (qui l'emportent sur le preset).
    largeur_max : 1/λ max en mm (None : 10 mm avec multi, sinon aucune ; 0 : aucune)."""
    global LAM_MIN, LAM_MAX, LAM_INIT
    g = globals()
    LAM_MIN, LAM_MAX = 3.0 / g["PORTEE_MAX_MM"], 5.0 / g["JAUGE_MM"]
    if kw.get("jauge"):
        g["JAUGE_MM"] = kw["jauge"]; LAM_MAX = 5.0 / g["JAUGE_MM"]
    if preset:
        appliquer_preset(preset, tol_lambda)
    for cle, nom in REGLAGES_SIMPLES.items():
        if cle != "jauge" and kw.get(cle) is not None:
            g[nom] = kw[cle]
    if plafond_mm is not None:
        g["PLAFOND_UM"] = 1000.0 * plafond_mm
    lmax = largeur_max if largeur_max is not None else (LARGEUR_MULTI if multi else 0.0)
    if lmax:
        LAM_MIN = max(LAM_MIN, 1.0 / lmax)


@contextmanager
def reglages(**kw):
    """Applique configurer(**kw) le temps du bloc, puis restaure les constantes (le module reste réutilisable)."""
    g = globals()
    avant = {k: g[k] for k in _CONSTANTES}
    try:
        configurer(**kw)
        yield
    finally:
        g.update(avant)


def lambda_global(h, valeur, tol_lambda=0.0, signe="+"):
    """Impose un λ commun : valeur en m⁻¹ ou 'auto' (médiane de β sur les paires de fissures de la dernière échéance)."""
    if valeur == "auto":
        ech = h[h["temps_ecoule_j"] == h["temps_ecoule_j"].max()].sort_values("xb")
        r = ech["r"].values.astype(float)
        sg = signe_auto(r) if signe == "auto" else (1.0 if signe == "+" else -1.0)
        p = estimer_beta(ech[ech["axe_mesure"].iloc[0]].values * 1000, sg * r)
        if not p:
            raise ValueError("aucune paire exploitable pour estimer β : donner λ en m⁻¹")
        lam = float(np.median([q[2] for q in p]))
    else:
        lam = float(valeur) / 1000
    fixer_lambda(lam, tol_lambda)
    return lam


def appliquer_preset(nom, tol_lambda=0.0):
    """Preset câble/pose : bornes de λ, de a (≤ A_CUBE_MM), du COD ; λ imposé si le preset donne 1/λ."""
    global LAM_MIN, LAM_MAX, A_MAX, PLAFOND_UM, COD_MIN, PORTEE_MAX_MM
    p = PRESETS[nom]
    PORTEE_MAX_MM = p["portee_max_mm"]
    A_MAX, PLAFOND_UM, COD_MIN = min(p["coeur_max_mm"], A_CUBE_MM), 1000.0 * p["plafond_mm"], p["cod_min_um"]
    LAM_MIN, LAM_MAX = 3.0 / PORTEE_MAX_MM, 5.0 / JAUGE_MM
    if p.get("inv_lambda_mm"):
        fixer_lambda(float(np.clip(1.0 / p["inv_lambda_mm"], LAM_MIN, LAM_MAX)), tol_lambda)


# ----------------------------------------------------------------------------- analyse complète
class AucuneFissure(ValueError):
    """Aucun candidat de fissure n'a été retenu à la dernière échéance."""


def analyser(h, t0=T0_DEFAUT, signe="+", prom=None, forcees=(), interdites=(), verbose=True, multi=False, duree_min=DUREE_MIN,
             dates=None):
    """dates : sous-ensemble d'échéances (jours) à analyser (défaut : toutes celles ≥ t0).
    Retourne (tableau des fissures, hist, centres) ; hist[t] = (x, résidu, ajustement, mesure, retrait modèle) en µε."""
    axe = h["axe_mesure"].iloc[0]
    echeances = np.sort(h["temps_ecoule_j"].unique()); echeances = echeances[echeances >= t0]
    if dates is not None:
        echeances = echeances[[bool(np.isclose(t, np.asarray(dates)).any()) for t in echeances]]
    xs, rs, mesures, mods = [], [], [], []
    for t in echeances:
        q = h[h["temps_ecoule_j"] == t].sort_values("xb")
        xs.append(q[axe].values * 1000); rs.append(q["r"].values.astype(float))
        mesures.append(q["eps"].values.astype(float)); mods.append(q["mod"].values.astype(float))
    sg = signe_auto(rs[-1]) if signe == "auto" else (1.0 if signe == "+" else -1.0)
    rs = [sg * r for r in rs]
    if verbose:
        print(f"Signe : {'+' if sg > 0 else '−'} ; bruit court terme (MAD, dernière échéance) : {bruit_mad(rs[-1]):.0f} µε")
        print(f"Détection sur {echeances[-1]:.0f} j :")
        paires = estimer_beta(xs[-1], rs[-1], verbose=True)
        if paires:
            print(f"  β médian des paires : {1000 * np.median([p[2] for p in paires]):.0f} m⁻¹ ({len(paires)} paire(s))")
    ext = None
    if multi:
        pas = float(np.median(np.diff(xs[-1])))
        if verbose:
            print(f"Détection multi-échéances ({len(xs)} dates, proéminence {PROM_MULTI}×bruit, durée ≥ {duree_min:.0f} j) :")
        ext = candidats_multi(xs, rs, echeances, PROM_MULTI, duree_min, max(1.5 * pas, JAUGE_MM / 4, 3.0), interdites, verbose)
    C, W, L, A, b = detecter(xs[-1], rs[-1], prom, forcees, interdites, verbose, ext, glouton=(not multi) or GLOUTON_MULTI)
    if len(C) == 0:
        raise AucuneFissure("aucune fissure détectée")
    n, T = len(C), len(echeances)
    W0 = np.zeros((n, T)); L0 = np.tile(L[:, None], (1, T)); A0 = np.tile(A[:, None], (1, T)); W0[:, -1] = W
    B0 = np.full(T, b)
    for k in range(T - 2, -1, -1):                         # suivi à rebours, centres figés
        c, w, l, a, bk, _ = ajuster_date(xs[k], rs[k], C, W0[:, k + 1], L0[:, k + 1], A0[:, k + 1], np.zeros(n, bool), B0[k + 1])
        W0[:, k] = np.minimum(w, W0[:, k + 1]); L0[:, k] = l; A0[:, k] = np.minimum(a, A0[:, k + 1]); B0[k] = bk
    Wf, Lf, Af, Cf, Bf = global_regularise(xs, rs, C, W0, L0, A0, B0)
    lignes, hist = [], {}
    for k, t in enumerate(echeances):
        fit = modele(xs[k], Cf, Wf[:, k], Lf, Af[:, k], Bf[k])
        rmse = float(np.sqrt(np.mean((fit - rs[k]) ** 2)))
        w_air = aires(xs[k], rs[k], Cf, fond_initial(xs[k], rs[k]))     # recoupement indépendant
        hist[t] = (xs[k], sg * rs[k], sg * fit, mesures[k], mods[k])
        for i in range(n):
            lignes.append(dict(t_j=t, id=i, pos_mm=Cf[i], w_um=Wf[i, k], w_aires_um=w_air[i], lambda_m1=1000 * Lf[i],
                               longueur_transfert_mm=1 / Lf[i], coeur_a_mm=Af[i, k], fond_ue=Bf[k],
                               pic_ue=amplitude(Wf[i, k], Af[i, k], Lf[i]), plafond=bool(Wf[i, k] >= 0.99 * PLAFOND_UM), signe=sg))
        if verbose:
            print(f"{t:6.1f} j : {n} fissures, RMSE={rmse:4.0f} µε, fond={Bf[k]:5.0f} µε, "
                  f"Σw={Wf[:, k].sum():.0f} µm (aires : {w_air.sum():.0f} µm)")
    return pd.DataFrame(lignes), hist, Cf


def tracer(cube, seg, h, df, hist, chemin):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = sorted(hist); idx = np.unique(np.linspace(0, len(ts) - 1, 4).astype(int)); sel = [ts[i] for i in idx]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9)); ax = ax.ravel()
    for a, t in zip(ax, sel):
        x, r, fit = hist[t][:3]
        a.plot(x, r, "k", lw=1, label="mesure − retrait"); a.plot(x, fit, "r", label="shear lag (+ fond)")
        a.axhline(0, color="gray", lw=.5); a.set_title(f"{cube} {seg} — {t:.0f} j"); a.set_xlabel("position (mm)"); a.set_ylabel("µε")
    ax[0].legend(fontsize=7)
    a = ax[4]
    for i, g in df.groupby("id"):
        a.plot(g["t_j"], g["w_um"], label=f"X={g['pos_mm'].iloc[0]:.0f}")
    a.plot(df.groupby("t_j")["w_um"].sum(), "k", lw=2, label="Σw modèle")
    a.plot(df.groupby("t_j")["w_aires_um"].sum(), "k--", lw=1.5, label="Σw aires")
    a.set_xlabel("t (j)"); a.set_ylabel("w (µm)"); a.legend(fontsize=6, ncol=2); a.set_title("ouvertures")
    a = ax[5]
    d = df[df["t_j"] == df["t_j"].max()]
    a.bar(d["pos_mm"], d["coeur_a_mm"], width=2, label="cœur a (mm)")
    a.plot(d["pos_mm"], d["longueur_transfert_mm"], "ro", label="1/λ (mm)")
    a.set_xlabel("position (mm)"); a.legend(fontsize=7); a.set_title("forme (dernière échéance)")
    fig.tight_layout(); fig.savefig(chemin, dpi=110); plt.close(fig)


def analyser_segment(h, dates=None, preset=None, tol_lambda=0.0, lam_global=None, signe="+", forcees=(), interdites=(),
                     multi=True, duree_min=None, prom=None, t0=0.0, **reglage):
    """Point d'entrée du dashboard et des scripts exportés : applique les réglages (constantes du module, restaurées ensuite)
    puis lance analyser(). `reglage` : largeur_max, plafond_mm, jauge, prom_facteur, prom_multi, glouton, marge_bord, a_max,
    codmin, w_neg, plancher, dist_min, decalage_max, n_max, w_min, add_gain, fond (voir configurer).
    lam_global : None (λ libre), 'auto' ou valeur en m⁻¹. Avec un preset, ses bornes (a max, COD min/max) remplacent celles données.
    Non réentrant (modifie des constantes du module). Lève AucuneFissure / ValueError."""
    if preset:
        for k in ("a_max", "codmin", "plafond_mm"):
            reglage.pop(k, None)
    with reglages(preset=preset, tol_lambda=tol_lambda, multi=multi, **reglage):
        if lam_global is not None:
            lambda_global(h, lam_global, tol_lambda, signe)
        return analyser(h, t0, signe, prom, list(forcees), list(interdites), verbose=False, multi=multi,
                        duree_min=DUREE_MIN if duree_min is None else duree_min,
                        dates=None if dates is None else np.asarray(dates))


# ----------------------------------------------------------------------------- figures Plotly (dashboard / notebook)
def groupes(tab, ecart_mm=8.0):
    """Regroupe a posteriori les fissures voisines (liaison simple sur les centres) : colonne 'groupe'."""
    cen = tab.groupby("id")["pos_mm"].first().sort_values()
    grp, g, prec = {}, 0, None
    for i, c in cen.items():
        if prec is not None and c - prec > ecart_mm:
            g += 1
        grp[i] = g; prec = c
    tab = tab.copy(); tab["groupe"] = tab["id"].map(grp)
    return tab


def figure_profil(hist, tab, t, cube, segment):
    """Mesure, retrait modèle, fond, noyau de chaque fissure (appuyé sur retrait + fond) et somme, à l'échéance t."""
    import plotly.express as px
    import plotly.graph_objects as go
    ts = np.array(sorted(hist)); t = ts[np.argmin(abs(ts - t))]
    x, _, _, mesure, mod = hist[t]
    d = tab[np.isclose(tab["t_j"], t)]
    signe = float(d["signe"].iloc[0]) if len(d) else 1.0
    o = np.argsort(x); x, mesure, mod = x[o], mesure[o], mod[o]
    xf = np.linspace(x.min(), x.max(), 1000); mf = np.interp(xf, x, mod); pal = px.colors.qualitative.Plotly
    base = mf + signe * (d["fond_ue"].iloc[0] if len(d) else 0.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=mesure, mode="lines", line=dict(color="black", width=1.5), name="mesure"))
    fig.add_trace(go.Scatter(x=xf, y=mf, mode="lines", line=dict(color="royalblue", dash="dash", width=2.5), name="retrait modèle"))
    fig.add_trace(go.Scatter(x=xf, y=base, mode="lines", line=dict(color="gray", dash="dot", width=1.5), name="retrait + fond b(t)"))
    total = np.zeros_like(xf)
    for k, (_, r) in enumerate(d.iterrows()):
        ki = signe * noyau(xf, r["pos_mm"], r["w_um"], r["lambda_m1"] / 1000, r["coeur_a_mm"]); total += ki
        fig.add_trace(go.Scatter(x=xf, y=base, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xf, y=base + ki, mode="lines", line=dict(color=pal[k % len(pal)], width=1), fill="tonexty",
                                 name=f"X={r['pos_mm']:.0f} mm : w={r['w_um']:.0f} µm, 1/λ={r['longueur_transfert_mm']:.1f} mm",
                                 hovertemplate="%{y:.0f} µε"))
    fig.add_trace(go.Scatter(x=xf, y=base + total, mode="lines", line=dict(color="crimson", width=2), name="retrait + fond + fissures"))
    fig.update_layout(title=f"{cube} {segment} — {t:.1f} j : {len(d)} fissures, Σw = {d['w_um'].sum():.0f} µm "
                            f"(aires : {d['w_aires_um'].sum():.0f} µm)",
                      xaxis_title="Position le long de la fibre (mm)", yaxis_title="µdef", height=520, legend=dict(font=dict(size=9)))
    return fig


def figure_ouvertures(tab, cube, segment):
    """Ouverture w_i(t) de chaque fissure, Σw (modèle) et Σw (méthode des aires, recoupement indépendant)."""
    import plotly.express as px
    import plotly.graph_objects as go
    pal = px.colors.qualitative.Plotly
    fig = go.Figure()
    for k, (i, g) in enumerate(tab.groupby("id")):
        g = g.sort_values("t_j")
        fig.add_trace(go.Scatter(x=g["t_j"], y=g["w_um"], mode="lines+markers", marker=dict(size=4),
                                 line=dict(color=pal[k % len(pal)]), name=f"X={g['pos_mm'].iloc[0]:.0f} mm"))
    fig.add_trace(go.Scatter(x=tab.groupby("t_j")["w_um"].sum().index, y=tab.groupby("t_j")["w_um"].sum().values,
                             mode="lines", line=dict(color="black", width=3), name="Σw modèle"))
    sa = tab.groupby("t_j")["w_aires_um"].sum()
    fig.add_trace(go.Scatter(x=sa.index, y=sa.values, mode="lines", line=dict(color="black", width=2, dash="dash"), name="Σw aires"))
    fig.update_layout(title=f"{cube} {segment} — ouverture de chaque fissure, somme et recoupement par aires",
                      xaxis_title="Temps écoulé (j)", yaxis_title="Ouverture w (µm)", height=420)
    return fig


def figure_forme(tab, cube, segment):
    """Forme du noyau à la dernière échéance : longueur de transfert 1/λ et demi-largeur du cœur a, par fissure."""
    import plotly.graph_objects as go
    d = tab[tab["t_j"] == tab["t_j"].max()]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["pos_mm"], y=d["coeur_a_mm"], width=1.5, name="cœur a (mm)"))
    fig.add_trace(go.Scatter(x=d["pos_mm"], y=d["longueur_transfert_mm"], mode="markers", marker=dict(color="crimson", size=9),
                             name="1/λ (mm)"))
    fig.update_layout(title=f"{cube} {segment} — forme des fissures (dernière échéance)", xaxis_title="Position (mm)",
                      yaxis_title="mm", height=360)
    return fig


def figure_suivi(tab, cube, segment):
    """Carte position × temps, taille des points ∝ ouverture."""
    import plotly.express as px
    d = tab.copy(); d["w_aff"] = d["w_um"].clip(lower=0.5)
    fig = px.scatter(d, x="pos_mm", y="t_j", size="w_aff", color=d["id"].astype(str), size_max=22,
                     hover_data={"w_um": ":.1f", "longueur_transfert_mm": ":.1f", "w_aff": False},
                     labels={"pos_mm": "Position le long de la fibre (mm)", "t_j": "Temps écoulé (j)", "color": "fissure"},
                     title=f"{cube} {segment} — suivi des fissures (taille ∝ ouverture)")
    fig.update_layout(height=420, showlegend=False)
    return fig


def figure_surface(tab, cube, segment, n_x=200, n_t_max=60):
    """Surface X = temps, Y = position, Z = ouverture linéique Σ noyau_i·1e-3 (µm/mm) ; ∫ Z dx = Σw(t)."""
    import plotly.graph_objects as go
    ts = np.sort(tab["t_j"].unique())
    if len(ts) > n_t_max:
        ts = ts[np.unique(np.linspace(0, len(ts) - 1, n_t_max).round().astype(int))]
    xs = np.linspace(max(tab["pos_mm"].min() - 15, 0), min(tab["pos_mm"].max() + 15, 100.0), n_x)
    Z = np.zeros((len(xs), len(ts)))
    for k, t in enumerate(ts):
        for _, r in tab[tab["t_j"] == t].iterrows():
            Z[:, k] += noyau(xs, r["pos_mm"], r["w_um"], r["lambda_m1"] / 1000, r["coeur_a_mm"]) * 1e-3
    fig = go.Figure(go.Surface(x=ts, y=xs, z=Z, colorscale="Inferno", colorbar=dict(title="µm/mm"),
                               hovertemplate="t=%{x:.1f} j<br>X=%{y:.1f} mm<br>w'=%{z:.2f} µm/mm<extra></extra>"))
    fig.update_layout(title=f"{cube} {segment} — ouverture linéique des fissures", height=560,
                      scene=dict(xaxis_title="Temps écoulé (j)", yaxis_title="Position fibre (mm)", zaxis_title="Ouverture (µm/mm)"),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def _liste(s):
    return [float(v) for v in s.replace(";", ",").split(",") if v.strip()] if s else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", choices=["C12", "C34"], default="C12")
    ap.add_argument("--segment", default="s14")
    ap.add_argument("--t0", type=float, default=T0_DEFAUT)
    ap.add_argument("--theta", type=float, nargs=4, metavar=("N", "k", "a", "n"), default=THETA_DEFAUT)
    ap.add_argument("--de", type=float, default=F.DE_DEFAUT)
    ap.add_argument("--jauge", type=float, default=None, help="largeur de jauge OBR en mm (défaut 5)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="câble / pose : bornes de λ, a, COD ; λ imposé si 1/λ connu")
    ap.add_argument("--signe", choices=["auto", "+", "-"], default="+", help="signe des fissures (défaut + : la physique impose des pics positifs)")
    ap.add_argument("--prom", type=float, default=None, help="proéminence minimale d'un pic en µε (défaut : facteur × bruit)")
    ap.add_argument("--prom-facteur", type=float, default=None, help=f"facteur de bruit pour la proéminence (défaut {PROM_FACTEUR})")
    ap.add_argument("--codmin", type=float, default=None, help="ouverture minimale d'une fissure en µm")
    ap.add_argument("--plafond", type=float, default=None, help="COD maximal en mm")
    ap.add_argument("--a-max", type=float, default=None, help=f"demi-largeur max du cœur en mm (défaut {A_CUBE_MM}) ; 0 = noyau exponentiel pur")
    ap.add_argument("--largeur-max", type=float, default=None, metavar="mm", help="longueur de transfert maximale 1/λ (mm) ; défaut 10 avec --multi, sinon aucune ; 0 = aucune limite")
    ap.add_argument("--plancher", choices=["local", "zero"], default=None, help="fond initial / plancher des aires")
    ap.add_argument("--sans-fond", action="store_true", help="pas de décalage b(t) ajusté")
    ap.add_argument("--w-neg", type=float, default=None, help=f"poids des points sous le modèle (défaut {W_NEG} ; 1 = symétrique)")
    ap.add_argument("--forcees", default="", help="positions (mm) de fissures imposées, séparées par des virgules")
    ap.add_argument("--interdites", default="", help="positions (mm) où aucune fissure n'est admise")
    ap.add_argument("--lambda-global", default=None, metavar="m-1|auto", help="λ commun imposé : valeur en m⁻¹ ou 'auto' (médiane des paires)")
    ap.add_argument("--tol-lambda", type=float, default=0.0, help="tolérance relative autour du λ global (défaut 0 = imposé)")
    ap.add_argument("--multi", action="store_true", help="détection des centres sur toutes les échéances (trajectoires de pics)")
    ap.add_argument("--prom-multi", type=float, default=None, help=f"facteur de bruit pour la détection multi (défaut {PROM_MULTI})")
    ap.add_argument("--duree-min", type=float, default=DUREE_MIN, help="durée minimale (j) d'une trajectoire validée")
    ap.add_argument("--marge-bord", type=float, default=None, metavar="mm", help=f"exclut les fissures détectées à moins de cette distance des extrémités (défaut {MARGE_BORD:g} ; 0 = aucune)")
    ap.add_argument("--glouton", action="store_true", help="avec --multi : ajoute aussi le complément glouton (lent)")
    ap.add_argument("--plot"); ap.add_argument("--csv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    configurer(preset=a.preset, tol_lambda=a.tol_lambda, largeur_max=a.largeur_max, plafond_mm=a.plafond, multi=a.multi,
               jauge=a.jauge, prom_facteur=a.prom_facteur, prom_multi=a.prom_multi, glouton=a.glouton or None,
               marge_bord=a.marge_bord, a_max=a.a_max, codmin=a.codmin, w_neg=a.w_neg, plancher=a.plancher,
               fond=False if a.sans_fond else None)
    h = donnees_segment(a.cube, a.segment, a.theta, a.de)
    if a.lambda_global:
        try:
            lam = lambda_global(h, a.lambda_global, a.tol_lambda, a.signe)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"λ global = {1000 * lam:.0f} m⁻¹ (± {100 * a.tol_lambda:.0f} %)")
    try:
        df, hist, C = analyser(h, a.t0, a.signe, a.prom, _liste(a.forcees), _liste(a.interdites), multi=a.multi, duree_min=a.duree_min)
    except AucuneFissure as e:
        raise SystemExit(str(e))
    if a.csv:
        df.to_csv(a.csv, index=False)
    if a.plot:
        tracer(a.cube, a.segment, h, df, hist, a.plot)


if __name__ == "__main__":
    main()
