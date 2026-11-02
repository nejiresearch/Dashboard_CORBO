#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fissuration_shear_lag.py
========================
Analyse et quantification de la fissuration par transfert de déformation exponentiel (Shear Lag Theory).
Remplace la décomposition géométrique en gaussiennes de CORBO par un modèle physique d'interface fibre/matrice.

Références scientifiques (corpus du notebook) :
- Bassil et al. (2020/2021) : Modèle multi-couches à adhésion imparfaite, ε_f(z) = (w * λ / 2) * exp(-λ|z - c|)
- Zhang et al. (2023) : Principe de superposition linéaire pour fissures rapprochées en selle
- Zhang et al. (2025) : Analyse comparative des méthodes d'intégration et d'ajustement exponentiel
- Richter et al. (2023) : Prétraitement du bruit et compensations
"""

import argparse
import numpy as np
import scipy.optimize as opt
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_LAMBDA = 50.0  # Paramètre de transfert moyen (m^-1)
MIN_CRACK_SPACING_MM = 5.0  # Distance minimale entre deux pics distincts (mm)


def shear_lag_single(x_mm, center_mm, w_um, lambda_m1):
    """
    Calcule le profil de déformation induit par une fissure isolée ou superposée (en µε).
    """
    x_m = (x_mm - center_mm) / 1000.0  # conversion mm -> m
    return (w_um * lambda_m1 / 2.0) * np.exp(-lambda_m1 * np.abs(x_m))


def multi_shear_lag_model(x_mm, centers_mm, w_um_list, lambda_m1_list):
    """
    Somme des contributions exponentielles de N fissures selon le principe de superposition linéaire (Zhang et al. 2023).
    """
    total = np.zeros_like(x_mm, dtype=float)
    for c, w, lmb in zip(centers_mm, w_um_list, lambda_m1_list):
        total += shear_lag_single(x_mm, c, w, lmb)
    return total


def estimate_noise_level(residual_strain):
    """
    Estime le bruit instrumental à partir des zones plates (30% quantiles bas).
    """
    lower_quantile = np.percentile(residual_strain, 30)
    flat_zone = residual_strain[residual_strain <= lower_quantile]
    std_est = np.std(flat_zone)
    return max(std_est, 15.0)


def detect_initial_peaks(x_mm, residual_strain, min_dist_mm=MIN_CRACK_SPACING_MM):
    """
    Détecte les positions des pics de fissuration (locaux et secondaires).
    """
    dx = np.mean(np.diff(x_mm))
    distance_pts = max(1, int(round(min_dist_mm / dx)))
    
    # Détection basée sur la courbure locale / dérivée seconde ou proéminence relative
    peaks, _ = signal.find_peaks(residual_strain, distance=distance_pts, prominence=30.0)
    if len(peaks) == 0:
        peaks, _ = signal.find_peaks(residual_strain, distance=distance_pts)
    return x_mm[peaks]


def fit_shear_lag_time_series(x_mm, residual_series, time_days, centers_init, fix_centers=True):
    """
    Ajuste la cinétique d'ouverture w_i(t) et le paramètre lambda_i(t) pour une série temporelle.
    Applique la contrainte de non-décroissance monotone sur w_i(t).
    """
    N_dates, N_points = residual_series.shape
    N_cracks = len(centers_init)
    
    if N_cracks == 0:
        return None
    
    w_matrix = np.zeros((N_dates, N_cracks))
    lambda_matrix = np.zeros((N_dates, N_cracks))
    centers_matrix = np.zeros((N_dates, N_cracks))
    
    # 1. Ajustement de la dernière date
    last_res = residual_series[-1, :]
    
    def obj_last(p):
        w_v = p[:N_cracks]
        l_v = p[N_cracks:2*N_cracks]
        c_v = p[2*N_cracks:]
        fit = multi_shear_lag_model(x_mm, c_v, w_v, l_v)
        return np.sum((last_res - fit)**2)
    
    p0_last = np.concatenate([
        np.full(N_cracks, 25.0),
        np.full(N_cracks, DEFAULT_LAMBDA),
        np.array(centers_init, dtype=float)
    ])
    
    b_w = [(0.0, 3000.0) for _ in range(N_cracks)]
    b_l = [(10.0, 250.0) for _ in range(N_cracks)]
    b_c = [(c - 2.0, c + 2.0) for c in centers_init]
    
    res_last = opt.minimize(obj_last, p0_last, bounds=b_w + b_l + b_c, method='L-BFGS-B')
    
    opt_p = res_last.x
    w_last = opt_p[:N_cracks]
    l_last = opt_p[N_cracks:2*N_cracks]
    c_last = opt_p[2*N_cracks:]
    
    centers_fixed = c_last if fix_centers else None
    
    # 2. Traitement à rebours (ordre inverse) avec régularisation temporelle
    prev_w_upper = np.copy(w_last)
    
    for t_idx in range(N_dates - 1, -1, -1):
        cur_res = residual_series[t_idx, :]
        
        def obj_step(p):
            w_v = p[:N_cracks]
            l_v = p[N_cracks:2*N_cracks]
            c_v = centers_fixed if fix_centers else p[2*N_cracks:]
            fit = multi_shear_lag_model(x_mm, c_v, w_v, l_v)
            return np.sum((cur_res - fit)**2)
        
        b_w_step = [(0.0, max(prev_w_upper[i], 1.0)) for i in range(N_cracks)]
        b_l_step = [(10.0, 250.0) for _ in range(N_cracks)]
        
        if fix_centers:
            p0_step = np.concatenate([np.minimum(w_last, prev_w_upper), l_last])
            b_step = b_w_step + b_l_step
        else:
            p0_step = np.concatenate([np.minimum(w_last, prev_w_upper), l_last, centers_fixed])
            b_c_step = [(c - 2.0, c + 2.0) for c in centers_fixed]
            b_step = b_w_step + b_l_step + b_c_step
            
        res_step = opt.minimize(obj_step, p0_step, bounds=b_step, method='L-BFGS-B')
        
        w_matrix[t_idx, :] = res_step.x[:N_cracks]
        lambda_matrix[t_idx, :] = res_step.x[N_cracks:2*N_cracks]
        centers_matrix[t_idx, :] = centers_fixed if fix_centers else res_step.x[2*N_cracks:]
        
        prev_w_upper = np.copy(res_step.x[:N_cracks])

    return {
        'centers': centers_matrix,
        'w_um': w_matrix,
        'lambda_m1': lambda_matrix,
        'cumulative_w_um': np.sum(w_matrix, axis=1)
    }


def generate_synthetic_demo_data():
    """
    Génère un jeu de données synthétiques imitant le segment CORBO s14 (fissures distinctes et chevauchantes).
    """
    x_mm = np.linspace(0, 100, 200)
    time_days = np.array([5.0, 14.0, 28.0, 56.0, 90.0, 113.0])
    
    # 3 fissures bien placées à X = 23 mm, 51 mm, 78 mm
    true_centers = [23.0, 51.0, 78.0]
    
    np.random.seed(42)
    residual_series = []
    for t in time_days:
        w_t = [
            45.0 * (1 - np.exp(-t / 20.0)),
            110.0 * (1 - np.exp(-t / 30.0)),
            65.0 * (1 - np.exp(-t / 25.0))
        ]
        lambda_t = [55.0, 48.0, 52.0]
        
        clean = multi_shear_lag_model(x_mm, true_centers, w_t, lambda_t)
        noise = np.random.normal(0, 12.0, size=x_mm.shape)
        residual_series.append(clean + noise)
        
    return x_mm, time_days, np.array(residual_series), true_centers


def main():
    parser = argparse.ArgumentParser(description="Analyse de la fissuration par Shear Lag exponentiel.")
    parser.add_argument("--demo", action="store_true", help="Exécuter la démonstration synthétique.")
    parser.add_argument("--plot", type=str, default="/workspace/scratch/shear_lag_demo.png", help="Chemin du graphique de sortie.")
    parser.add_argument("--out", type=str, default="/workspace/out/fissuration_shear_lag.csv", help="Chemin du CSV de sortie.")
    args = parser.parse_args()
    
    print("=== Démarrage de l'analyse par Shear Lag Exponentiel ===")
    
    x_mm, time_days, residual_series, true_centers = generate_synthetic_demo_data()
    
    sigma_noise = estimate_noise_level(residual_series[-1, :])
    print(f"Bruit instrumental estimé : {sigma_noise:.2f} µε")
    
    detected_centers = detect_initial_peaks(x_mm, residual_series[-1, :])
    print(f"Positions des fissures détectées (mm) : {np.round(detected_centers, 2)}")
    
    results = fit_shear_lag_time_series(x_mm, residual_series, time_days, detected_centers, fix_centers=True)
    
    # Export CSV
    df_res = pd.DataFrame({'Temps_jours': time_days, 'Ouverture_Cumulee_um': results['cumulative_w_um']})
    for i, c in enumerate(detected_centers):
        df_res[f'fissure_{i+1}_x_{c:.1f}mm_w_um'] = results['w_um'][:, i]
        df_res[f'fissure_{i+1}_x_{c:.1f}mm_lambda_m1'] = results['lambda_m1'][:, i]
        
    df_res.to_csv(args.out, index=False)
    print(f"Résultats exportés vers {args.out}")
    
    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    last_idx = -1
    fitted_last = multi_shear_lag_model(x_mm, results['centers'][last_idx], results['w_um'][last_idx], results['lambda_m1'][last_idx])
    
    ax1.plot(x_mm, residual_series[last_idx], 'b.', alpha=0.5, label='Résidu mesuré (113 j)')
    ax1.plot(x_mm, fitted_last, 'r-', linewidth=2, label='Fit Shear Lag Multi-fissures')
    ax1.set_xlabel('Position le long de la fibre (mm)')
    ax1.set_ylabel('Déformation résiduelle (µε)')
    ax1.set_title('Décomposition par Shear Lag exponentiel (113 j)')
    ax1.legend()
    ax1.grid(True)
    
    for i, c in enumerate(detected_centers):
        ax2.plot(time_days, results['w_um'][:, i], 'o--', label=f'Fissure X={c:.1f} mm')
    
    ax2.plot(time_days, results['cumulative_w_um'], 'k-', linewidth=2.5, label='Ouverture cumulée Σw')
    ax2.set_xlabel('Temps (jours)')
    ax2.set_ylabel('Ouverture de fissure w (µm)')
    ax2.set_title('Évolution temporelle des ouvertures (monotone)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"Graphique de synthèse sauvegardé dans {args.plot}")


if __name__ == '__main__':
    main()
