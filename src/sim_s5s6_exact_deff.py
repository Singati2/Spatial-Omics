"""
Paper 2 Simulation — S5 & S6 with EXACT DEFF (fixes Moran's I failure)
========================================================================
Only change: DEFF computed from pairwise correlation matrix, not Moran's I.
Everything else identical.
"""

import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from sklearn.neighbors import kneighbors_graph
from lifelines import CoxPHFitter
import warnings
import sys
import time

warnings.filterwarnings('ignore')

TRUE_BETA = -1.0
GAMMA = 0.5
MU_ETA = special.logit(0.35)
SIGMA_ETA = 0.8
N_PATIENTS = 50
CENSORING_TARGET = 0.40
M_LOG_MU = np.log(2200)
M_LOG_SD = 0.45
M_MIN, M_MAX = 200, 4000
SPOT_SPACING = 100.0


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# SPATIAL FIELD GENERATION (unchanged)
# ============================================================
def generate_visium_grid(m_target, spacing=SPOT_SPACING):
    side = int(np.ceil(np.sqrt(m_target * 1.15)))
    coords = []
    for row in range(side):
        for col in range(side):
            x = col * spacing
            y = row * spacing
            if row % 2 == 1:
                x += spacing / 2
            coords.append([x, y])
    coords = np.array(coords)
    if len(coords) > m_target:
        idx = np.random.choice(len(coords), m_target, replace=False)
        coords = coords[idx]
    return coords


def generate_spatial_binary_field(coords, X_i, phi, rng):
    m = len(coords)
    threshold = stats.norm.ppf(1 - X_i)
    D = squareform(pdist(coords))
    C = np.exp(-D / phi)
    C += np.eye(m) * 1e-6
    try:
        L = cholesky(C, lower=True)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-6)
        L = cholesky(eigvecs @ np.diag(eigvals) @ eigvecs.T, lower=True)
    Z = L @ rng.standard_normal(m)
    Y = (Z > threshold).astype(int)
    return Y, D


# ============================================================
# EXACT DEFF (THE FIX)
# ============================================================
def compute_exact_deff(D, phi):
    """
    DEFF_exact = 1 + (2 / (m*(m-1))) * sum_{s<t} exp(-d_st / phi)
               = 1 + (m-1) * mean_pairwise_correlation

    Uses vectorized upper triangle extraction.
    """
    m = D.shape[0]
    if m < 2:
        return 1.0

    # Correlation matrix
    R = np.exp(-D / phi)

    # Upper triangle (excluding diagonal)
    upper_idx = np.triu_indices(m, k=1)
    corr_values = R[upper_idx]

    # Mean pairwise correlation
    mean_corr = corr_values.mean()

    # Exact DEFF
    deff = 1 + (m - 1) * mean_corr
    return deff, mean_corr


# ============================================================
# DATA GENERATION WITH EXACT DEFF
# ============================================================
def generate_one_dataset(n, beta, phi_multiplier, rng):
    eta = rng.normal(MU_ETA, SIGMA_ETA, n)
    X = expit(eta)

    log_m = rng.normal(M_LOG_MU, M_LOG_SD, n)
    m = np.clip(np.round(np.exp(log_m)), M_MIN, M_MAX).astype(int)
    m_sim = np.minimum(m, 800)  # cap for GP generation

    phi = phi_multiplier * SPOT_SPACING

    W = np.zeros(n)
    deff_exact = np.zeros(n)
    mean_corr_arr = np.zeros(n)

    for i in range(n):
        coords = generate_visium_grid(m_sim[i])
        Y, D = generate_spatial_binary_field(coords, X[i], phi, rng)
        W[i] = Y.mean()

        # EXACT DEFF from pairwise distances (THE FIX)
        deff_val, mc = compute_exact_deff(D, phi)
        deff_exact[i] = deff_val
        mean_corr_arr[i] = mc

    # Variance using exact DEFF
    sigma2_exact = W * (1 - W) * deff_exact / m
    sigma2_exact = np.clip(sigma2_exact, 1e-8, None)

    m_eff = m / deff_exact

    Z = rng.standard_normal(n)

    lambda_0 = 0.1
    rate = lambda_0 * np.exp(beta * X + GAMMA * Z)
    T_event = rng.exponential(1.0 / rate)
    mean_rate = rate.mean()
    lambda_C = CENSORING_TARGET * mean_rate / (1 - CENSORING_TARGET)
    C = rng.exponential(1.0 / lambda_C, n)
    T_obs = np.minimum(T_event, C)
    delta = (T_event <= C).astype(int)

    return {
        'X': X, 'W': W, 'Z': Z, 'T': T_obs, 'delta': delta,
        'm': m, 'deff_exact': deff_exact, 'm_eff': m_eff,
        'sigma2': sigma2_exact, 'mean_corr': mean_corr_arr,
    }


# ============================================================
# METHODS (unchanged logic, uses exact DEFF for variance)
# ============================================================
def fit_cox(T, delta, covariates, names):
    df = pd.DataFrame(covariates, columns=names)
    df['T'] = T
    df['delta'] = delta
    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col='T', event_col='delta', show_progress=False)
        b = cph.params_[names[0]]
        se = cph.standard_errors_[names[0]]
        ci = cph.confidence_intervals_.loc[names[0]]
        p = cph.summary['p'].loc[names[0]]
        return b, se, float(ci.iloc[0]), float(ci.iloc[1]), p
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def method_oracle(data):
    return fit_cox(data['T'], data['delta'], np.column_stack([data['X'], data['Z']]), ['X', 'Z'])


def method_naive(data):
    return fit_cox(data['T'], data['delta'], np.column_stack([data['W'], data['Z']]), ['X', 'Z'])


def method_rc_grc(data, patient_specific):
    """Standard RC (patient_specific=False) or GRC (patient_specific=True)."""
    W = data['W']
    sigma2_i = data['sigma2']  # uses exact DEFF

    var_W = np.var(W, ddof=1)
    mean_s2 = np.mean(sigma2_i)
    sigma_X2 = max(var_W - mean_s2, 0)

    if sigma_X2 == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, True

    if patient_specific:
        lam_i = sigma_X2 / (sigma_X2 + sigma2_i)
    else:
        lam = sigma_X2 / (sigma_X2 + mean_s2)
        lam_i = np.full_like(sigma2_i, lam)

    mu_W = np.mean(W)
    X_cal = mu_W + lam_i * (W - mu_W)

    result = fit_cox(data['T'], data['delta'], np.column_stack([X_cal, data['Z']]), ['X', 'Z'])
    return result + (False,)  # truncated = False


def method_simex(data, n_reps=20):
    W = data['W']
    sigma2_i = data['sigma2']
    rng = np.random.default_rng(42)
    zeta_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    beta_z = []
    for zeta in zeta_grid:
        bb = []
        for b in range(n_reps):
            if zeta == 0:
                Wn = W.copy()
            else:
                noise = np.array([rng.normal(0, np.sqrt(zeta * s)) for s in sigma2_i])
                Wn = np.clip(W + noise, 0, 1)
            r = fit_cox(data['T'], data['delta'], np.column_stack([Wn, data['Z']]), ['X', 'Z'])
            if not np.isnan(r[0]):
                bb.append(r[0])
        beta_z.append(np.mean(bb) if bb else np.nan)

    z_arr = np.array(zeta_grid)
    b_arr = np.array(beta_z)
    valid = ~np.isnan(b_arr)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    coeffs = np.polyfit(z_arr[valid], b_arr[valid], 2)
    bs = np.polyval(coeffs, -1.0)
    nr = method_naive(data)
    se = nr[1] if not np.isnan(nr[1]) else np.nan
    if np.isnan(se):
        return bs, np.nan, np.nan, np.nan, np.nan
    return bs, se, bs - 1.96 * se, bs + 1.96 * se, 2 * (1 - stats.norm.cdf(abs(bs / se)))


# ============================================================
# SIMULATION RUNNER
# ============================================================
def run_scenario(n_reps, beta, phi_mult, label):
    methods_names = ['Oracle', 'Naive', 'StdRC', 'GRC', 'SIMEX']
    all_res = {m: [] for m in methods_names}
    diags = []
    t0 = time.time()

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 1000 + 99)
        data = generate_one_dataset(N_PATIENTS, beta, phi_mult, rng)

        # Oracle
        r = method_oracle(data)
        all_res['Oracle'].append({'beta_hat': r[0], 'se': r[1], 'ci_lo': r[2], 'ci_hi': r[3], 'p_val': r[4]})

        # Naive
        r = method_naive(data)
        all_res['Naive'].append({'beta_hat': r[0], 'se': r[1], 'ci_lo': r[2], 'ci_hi': r[3], 'p_val': r[4]})

        # Standard RC (one lambda, exact DEFF variance)
        r = method_rc_grc(data, patient_specific=False)
        truncated_rc = r[5]
        all_res['StdRC'].append({'beta_hat': r[0], 'se': r[1], 'ci_lo': r[2], 'ci_hi': r[3], 'p_val': r[4]})

        # GRC (patient-specific lambda, exact DEFF variance)
        r = method_rc_grc(data, patient_specific=True)
        truncated_grc = r[5]
        all_res['GRC'].append({'beta_hat': r[0], 'se': r[1], 'ci_lo': r[2], 'ci_hi': r[3], 'p_val': r[4]})

        # SIMEX
        r = method_simex(data, n_reps=20)
        all_res['SIMEX'].append({'beta_hat': r[0], 'se': r[1], 'ci_lo': r[2], 'ci_hi': r[3], 'p_val': r[4]})

        # Diagnostics
        W = data['W']
        s2 = data['sigma2']
        vW = np.var(W, ddof=1)
        sx2 = max(vW - np.mean(s2), 0)
        li = sx2 / (sx2 + s2) if sx2 > 0 else np.zeros_like(s2)

        diags.append({
            'sigma_X2': sx2,
            'truncated': 1 if sx2 == 0 else 0,
            'lambda_mean': np.mean(li),
            'lambda_min': np.min(li),
            'lambda_max': np.max(li),
            'mean_deff': np.mean(data['deff_exact']),
            'mean_meff': np.mean(data['m_eff']),
            'mean_corr': np.mean(data['mean_corr']),
            'censoring': 1 - data['delta'].mean(),
        })

        if (rep + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{label}] {rep+1}/{n_reps} | {el:.0f}s | ETA {(n_reps-rep-1)/(rep+1)*el:.0f}s")

    print(f"  [{label}] Done in {time.time()-t0:.1f}s")
    return all_res, diags


def compute_metrics(all_res, beta):
    rows = []
    for method in ['Oracle', 'Naive', 'StdRC', 'GRC', 'SIMEX']:
        betas = np.array([r['beta_hat'] for r in all_res[method]])
        ci_lo = np.array([r['ci_lo'] for r in all_res[method]])
        ci_hi = np.array([r['ci_hi'] for r in all_res[method]])
        pvals = np.array([r['p_val'] for r in all_res[method]])
        v = ~np.isnan(betas)
        nv = v.sum()
        if nv < 10:
            rows.append({'Method': method, 'n_valid': nv, 'Mean_beta': np.nan,
                         'Bias': np.nan, 'RelBias': np.nan, 'EmpSE': np.nan,
                         'RMSE': np.nan, 'Coverage': np.nan, 'Power': np.nan})
            continue
        bv = betas[v]
        bias = np.mean(bv) - beta
        rows.append({
            'Method': method, 'n_valid': nv,
            'Mean_beta': np.mean(bv),
            'Bias': bias,
            'RelBias': bias / abs(beta) * 100,
            'EmpSE': np.std(bv, ddof=1),
            'RMSE': np.sqrt(np.mean((bv - beta) ** 2)),
            'Coverage': np.mean((ci_lo[v] <= beta) & (beta <= ci_hi[v])) * 100,
            'Power': np.mean(pvals[v & ~np.isnan(pvals)] < 0.05) * 100,
        })
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print("=" * 70)
    print("S5 & S6 WITH EXACT DEFF (FIXES MORAN'S I FAILURE)")
    print("=" * 70)
    print(f"  n={N_PATIENTS}, beta={TRUE_BETA}, reps={n_reps}")
    print()

    # Load S1 for comparison
    try:
        s1 = pd.read_csv('/Users/ganeshshiwakoti/Desktop/Biostatistics/simulation/sim_s1_results.csv')
    except:
        s1 = None

    all_scenario_results = {}

    for label, phi_mult in [('S5_exact', 2), ('S6_exact', 5)]:
        print(f"\n{'='*70}")
        print(f"{label}: phi = {phi_mult}x spot spacing, EXACT DEFF")
        print(f"{'='*70}")

        res, diags = run_scenario(n_reps, TRUE_BETA, phi_mult, label)
        df = compute_metrics(res, TRUE_BETA)
        ddf = pd.DataFrame(diags)

        print(f"\n--- {label} Results ---")
        print(df.to_string(index=False, float_format='%.3f'))

        print(f"\n--- Diagnostics ---")
        print(f"  Truncation rate: {ddf['truncated'].mean()*100:.1f}%")
        print(f"  Mean sigma_X^2:  {ddf['sigma_X2'].mean():.4f}")
        print(f"  Mean lambda:     {ddf['lambda_mean'].mean():.3f}")
        print(f"  Lambda range:    {ddf['lambda_min'].mean():.3f} - {ddf['lambda_max'].mean():.3f}")
        print(f"  Mean exact DEFF: {ddf['mean_deff'].mean():.1f}")
        print(f"  Mean m_eff:      {ddf['mean_meff'].mean():.1f}")
        print(f"  Mean pairwise corr: {ddf['mean_corr'].mean():.4f}")

        df.to_csv(f'/Users/ganeshshiwakoti/Desktop/Biostatistics/simulation/sim_{label.lower()}_results.csv', index=False)
        all_scenario_results[label] = df

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("COMPARISON: S1 (exchangeable) vs S5/S6 (exact DEFF)")
    print(f"{'='*70}")

    print(f"\n{'Method':<10} | {'S1 Bias%':>10} {'S1 Cov%':>8} | {'S5 Bias%':>10} {'S5 Cov%':>8} | {'S6 Bias%':>10} {'S6 Cov%':>8}")
    print("-" * 80)

    for method in ['Oracle', 'Naive', 'StdRC', 'GRC', 'SIMEX']:
        parts = [f"{method:<10}"]

        # S1
        if s1 is not None:
            row = s1[s1['Method'] == method]
            if len(row) > 0:
                parts.append(f"| {row.iloc[0]['RelBias']:>9.1f}% {row.iloc[0]['Coverage']:>7.1f}%")
            else:
                parts.append(f"|       N/A      N/A")
        else:
            parts.append(f"|       N/A      N/A")

        for label in ['S5_exact', 'S6_exact']:
            if label in all_scenario_results:
                row = all_scenario_results[label][all_scenario_results[label]['Method'] == method]
                if len(row) > 0 and not np.isnan(row.iloc[0]['RelBias']):
                    parts.append(f"| {row.iloc[0]['RelBias']:>9.1f}% {row.iloc[0]['Coverage']:>7.1f}%")
                else:
                    parts.append(f"|    FAILED      N/A")
            else:
                parts.append(f"|       N/A      N/A")
        print(" ".join(parts))

    # Decision
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    for label in ['S5_exact', 'S6_exact']:
        if label not in all_scenario_results:
            continue
        df = all_scenario_results[label]
        grc = df[df['Method'] == 'GRC']
        if len(grc) == 0 or np.isnan(grc.iloc[0]['RelBias']):
            print(f"\n  {label}: GRC STILL FAILS (NaN results)")
            continue

        grc_bias = abs(grc.iloc[0]['RelBias'])
        grc_cov = grc.iloc[0]['Coverage']
        trunc = pd.DataFrame(diags)['truncated'].mean() * 100

        print(f"\n  {label}:")
        print(f"    GRC relative bias: {grc_bias:.1f}%")
        print(f"    GRC coverage:      {grc_cov:.1f}%")
        print(f"    Truncation rate:   {trunc:.1f}%")

        if s1 is not None:
            s1_grc = s1[s1['Method'] == 'GRC'].iloc[0]
            bias_diff = grc_bias - abs(s1_grc['RelBias'])
            cov_diff = grc_cov - s1_grc['Coverage']
            print(f"    vs S1: bias change = {bias_diff:+.1f}pp, coverage change = {cov_diff:+.1f}pp")

        if grc_bias <= 10 and grc_cov >= 88:
            print(f"    >>> ROBUST: Paper survives <<<")
        elif grc_bias <= 15 and grc_cov >= 85:
            print(f"    >>> PARTIALLY ROBUST: Needs caveat <<<")
        else:
            print(f"    >>> FAILS: Method not usable under this correlation <<<")
