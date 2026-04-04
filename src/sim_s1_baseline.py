"""
Paper 2 Simulation — Scenario S1: Baseline Exchangeable
========================================================
DGM: Logit-normal X_i, Beta-Binomial W_i, Cox survival
Methods: Oracle, Naive, Standard RC, GRC, SIMEX
Metrics: Bias, RelBias, EmpSE, RMSE, Coverage, Power, Truncation

Run with: python sim_s1_baseline.py [n_reps]
Default: 200 reps for debugging, use 2000 for final
"""

import numpy as np
import pandas as pd
from scipy import stats, special, sparse
from scipy.optimize import minimize_scalar
from collections import namedtuple
import warnings
import sys
import time

warnings.filterwarnings('ignore')

# Try importing lifelines for Cox; fall back to manual if needed
try:
    from lifelines import CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("lifelines not installed. Install with: pip install lifelines")
    sys.exit(1)

# ============================================================
# 1. DGM PARAMETERS (S1 Baseline)
# ============================================================
TRUE_BETA = -1.0          # moderate effect
GAMMA = 0.5               # confounder effect
MU_ETA = special.logit(0.35)  # logit(0.35) ≈ -0.619
SIGMA_ETA = 0.8           # spread on logit scale
N_PATIENTS = 50           # sample size
CENSORING_TARGET = 0.40   # 40% censoring
RHO_RANGE = (0.02, 0.10)  # exchangeable correlation range
M_LOG_MU = np.log(2200)   # log-normal center for spots
M_LOG_SD = 0.45           # log-normal spread
M_MIN, M_MAX = 200, 4000  # truncation bounds


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# 2. DATA GENERATION
# ============================================================
def generate_one_dataset(n, beta, rng):
    """Generate one complete dataset under S1 DGM."""

    # 2.1 Latent biomarker (logit-normal)
    eta = rng.normal(MU_ETA, SIGMA_ETA, n)
    X = expit(eta)

    # 2.2 Tissue characteristics
    log_m = rng.normal(M_LOG_MU, M_LOG_SD, n)
    m = np.clip(np.round(np.exp(log_m)), M_MIN, M_MAX).astype(int)
    rho = rng.uniform(RHO_RANGE[0], RHO_RANGE[1], n)

    # 2.3 Effective sample size and true error variance
    DEFF = 1 + (m - 1) * rho
    m_eff = m / DEFF
    sigma2 = X * (1 - X) / m_eff  # theory-based variance

    # 2.4 Generate observed W_i via Beta-Binomial
    # Beta-Binomial(m_i, alpha_i, beta_i) with:
    #   E[W] = X_i
    #   Var[W] = X_i(1-X_i) * [1 + (m_i-1)*rho_i] / m_i
    #
    # For Beta-Binomial: Var = m*alpha*beta*(alpha+beta+m) / ((alpha+beta)^2*(alpha+beta+1))
    # Matching moments: alpha_i = X_i * phi_i, beta_i = (1-X_i) * phi_i
    # where phi_i = (1 - rho_i) / rho_i  (intra-class correlation parameterization)

    W = np.zeros(n)
    for i in range(n):
        if rho[i] < 0.001:
            # Nearly independent: use Binomial
            S = rng.binomial(m[i], X[i])
        else:
            # Beta-Binomial
            phi = (1 - rho[i]) / rho[i]
            alpha_bb = X[i] * phi
            beta_bb = (1 - X[i]) * phi
            # Clamp to avoid degenerate parameters
            alpha_bb = max(alpha_bb, 0.01)
            beta_bb = max(beta_bb, 0.01)
            # Draw p from Beta, then S from Binomial
            p_draw = rng.beta(alpha_bb, beta_bb)
            p_draw = np.clip(p_draw, 1e-6, 1 - 1e-6)
            S = rng.binomial(m[i], p_draw)
        W[i] = S / m[i]

    # 2.5 Confounder
    Z = rng.standard_normal(n)

    # 2.6 Survival times (exponential baseline)
    # h(t|X,Z) = lambda_0 * exp(beta*X + gamma*Z)
    # T ~ Exp(lambda_0 * exp(beta*X + gamma*Z))
    lambda_0 = 0.1  # baseline hazard rate
    rate = lambda_0 * np.exp(beta * X + GAMMA * Z)
    T_event = rng.exponential(1.0 / rate)

    # 2.7 Censoring (exponential, tuned to target rate)
    # Find lambda_C that gives ~40% censoring
    # P(C < T) ≈ target → lambda_C / (lambda_C + mean_rate) ≈ target
    mean_rate = rate.mean()
    lambda_C = CENSORING_TARGET * mean_rate / (1 - CENSORING_TARGET)
    C = rng.exponential(1.0 / lambda_C, n)

    # 2.8 Observed data
    T_obs = np.minimum(T_event, C)
    delta = (T_event <= C).astype(int)

    return {
        'X': X, 'W': W, 'Z': Z, 'T': T_obs, 'delta': delta,
        'm': m, 'rho': rho, 'DEFF': DEFF, 'm_eff': m_eff, 'sigma2': sigma2,
    }


# ============================================================
# 3. CORRECTION METHODS
# ============================================================
def fit_cox(T, delta, covariates, covariate_names):
    """Fit Cox PH model using lifelines. Returns beta_hat, se, p_value, ci_lower, ci_upper."""
    df = pd.DataFrame(covariates, columns=covariate_names)
    df['T'] = T
    df['delta'] = delta

    cph = CoxPHFitter()
    try:
        cph.fit(df, duration_col='T', event_col='delta', show_progress=False)
        beta_hat = cph.params_[covariate_names[0]]
        se = cph.standard_errors_[covariate_names[0]]
        ci = cph.confidence_intervals_.loc[covariate_names[0]]
        p_val = cph.summary['p'].loc[covariate_names[0]]
        return beta_hat, se, float(ci.iloc[0]), float(ci.iloc[1]), p_val
    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan


def method_oracle(data):
    """M0: Cox with true X_i."""
    covs = np.column_stack([data['X'], data['Z']])
    return fit_cox(data['T'], data['delta'], covs, ['X', 'Z'])


def method_naive(data):
    """M1: Cox with observed W_i."""
    covs = np.column_stack([data['W'], data['Z']])
    return fit_cox(data['T'], data['delta'], covs, ['X', 'Z'])


def method_standard_rc(data):
    """M2: Standard RC with one global lambda."""
    W = data['W']
    sigma2_i = estimate_sigma2_theory(W, data['m'], data['rho'])

    # Estimate sigma_X^2
    var_W = np.var(W, ddof=1)
    mean_sigma2 = np.mean(sigma2_i)
    sigma_X2 = max(var_W - mean_sigma2, 0)

    if sigma_X2 == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # One global lambda
    lam = sigma_X2 / (sigma_X2 + mean_sigma2)

    # Calibrated covariate
    mu_W = np.mean(W)
    X_rc = mu_W + lam * (W - mu_W)

    covs = np.column_stack([X_rc, data['Z']])
    return fit_cox(data['T'], data['delta'], covs, ['X', 'Z'])


def method_grc(data):
    """M3: GRC with patient-specific lambda_i."""
    W = data['W']
    sigma2_i = estimate_sigma2_theory(W, data['m'], data['rho'])

    # Estimate sigma_X^2
    var_W = np.var(W, ddof=1)
    mean_sigma2 = np.mean(sigma2_i)
    sigma_X2 = max(var_W - mean_sigma2, 0)

    if sigma_X2 == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Patient-specific lambda_i
    lambda_i = sigma_X2 / (sigma_X2 + sigma2_i)

    # Calibrated covariate
    mu_W = np.mean(W)
    X_grc = mu_W + lambda_i * (W - mu_W)

    covs = np.column_stack([X_grc, data['Z']])
    return fit_cox(data['T'], data['delta'], covs, ['X', 'Z'])


def method_simex(data, n_simex_reps=50):
    """M4: SIMEX with quadratic extrapolation."""
    W = data['W']
    sigma2_i = estimate_sigma2_theory(W, data['m'], data['rho'])
    rng = np.random.default_rng(42)

    zeta_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
    beta_at_zeta = []

    for zeta in zeta_grid:
        betas_b = []
        for b in range(n_simex_reps):
            if zeta == 0:
                W_noisy = W.copy()
            else:
                noise = np.array([rng.normal(0, np.sqrt(zeta * s)) for s in sigma2_i])
                W_noisy = np.clip(W + noise, 0, 1)

            covs = np.column_stack([W_noisy, data['Z']])
            result = fit_cox(data['T'], data['delta'], covs, ['X', 'Z'])
            if not np.isnan(result[0]):
                betas_b.append(result[0])

        if betas_b:
            beta_at_zeta.append(np.mean(betas_b))
        else:
            beta_at_zeta.append(np.nan)

    # Quadratic extrapolation to zeta = -1
    zeta_arr = np.array(zeta_grid)
    beta_arr = np.array(beta_at_zeta)
    valid = ~np.isnan(beta_arr)

    if valid.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    coeffs = np.polyfit(zeta_arr[valid], beta_arr[valid], 2)
    beta_simex = np.polyval(coeffs, -1.0)

    # Use naive SE as approximation (conservative)
    naive_result = method_naive(data)
    se_approx = naive_result[1] if not np.isnan(naive_result[1]) else np.nan

    if np.isnan(se_approx):
        return beta_simex, np.nan, np.nan, np.nan, np.nan

    ci_lo = beta_simex - 1.96 * se_approx
    ci_hi = beta_simex + 1.96 * se_approx
    z_stat = beta_simex / se_approx
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return beta_simex, se_approx, ci_lo, ci_hi, p_val


def estimate_sigma2_theory(W, m, rho):
    """Theory-based variance: sigma_i^2 = W_i(1-W_i) * DEFF_i / m_i"""
    DEFF = 1 + (m - 1) * rho
    sigma2 = W * (1 - W) * DEFF / m
    # Avoid zero variance for W near 0 or 1
    sigma2 = np.clip(sigma2, 1e-8, None)
    return sigma2


# ============================================================
# 4. RUN SIMULATION
# ============================================================
def run_one_replicate(rep_id, beta, rng):
    """Run one replicate: generate data, apply all methods, return results."""
    data = generate_one_dataset(N_PATIENTS, beta, rng)

    results = {}
    for name, method in [('Oracle', method_oracle),
                         ('Naive', method_naive),
                         ('StdRC', method_standard_rc),
                         ('GRC', method_grc)]:
        beta_hat, se, ci_lo, ci_hi, p_val = method(data)
        results[name] = {
            'beta_hat': beta_hat,
            'se': se,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'p_val': p_val,
        }

    # SIMEX (slower — reduced reps for speed)
    beta_hat, se, ci_lo, ci_hi, p_val = method_simex(data, n_simex_reps=30)
    results['SIMEX'] = {
        'beta_hat': beta_hat,
        'se': se,
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'p_val': p_val,
    }

    # Track diagnostics
    W = data['W']
    sigma2_i = estimate_sigma2_theory(W, data['m'], data['rho'])
    var_W = np.var(W, ddof=1)
    sigma_X2_hat = max(var_W - np.mean(sigma2_i), 0)
    truncated = 1 if sigma_X2_hat == 0 else 0

    lambda_i = sigma_X2_hat / (sigma_X2_hat + sigma2_i) if sigma_X2_hat > 0 else np.zeros_like(sigma2_i)

    results['diagnostics'] = {
        'sigma_X2_hat': sigma_X2_hat,
        'truncated': truncated,
        'lambda_mean': np.mean(lambda_i),
        'lambda_min': np.min(lambda_i),
        'lambda_max': np.max(lambda_i),
        'censoring_rate': 1 - data['delta'].mean(),
        'mean_m': np.mean(data['m']),
        'mean_rho': np.mean(data['rho']),
        'mean_meff': np.mean(data['m_eff']),
    }

    return results


def run_simulation(n_reps, beta):
    """Run full simulation with n_reps replicates."""
    all_results = {method: [] for method in ['Oracle', 'Naive', 'StdRC', 'GRC', 'SIMEX']}
    diagnostics = []

    start = time.time()

    for rep in range(n_reps):
        rng = np.random.default_rng(rep * 1000 + 42)
        res = run_one_replicate(rep, beta, rng)

        for method in all_results:
            all_results[method].append(res[method])
        diagnostics.append(res['diagnostics'])

        if (rep + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (rep + 1) / elapsed
            eta = (n_reps - rep - 1) / rate
            print(f"  Rep {rep+1}/{n_reps} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    elapsed = time.time() - start
    print(f"  Completed {n_reps} reps in {elapsed:.1f}s ({elapsed/n_reps:.2f}s/rep)")

    return all_results, diagnostics


# ============================================================
# 5. COMPUTE METRICS
# ============================================================
def compute_metrics(all_results, diagnostics, beta):
    """Compute bias, relative bias, empirical SE, RMSE, coverage, power."""
    methods = ['Oracle', 'Naive', 'StdRC', 'GRC', 'SIMEX']
    rows = []

    for method in methods:
        betas = np.array([r['beta_hat'] for r in all_results[method]])
        ci_los = np.array([r['ci_lo'] for r in all_results[method]])
        ci_his = np.array([r['ci_hi'] for r in all_results[method]])
        p_vals = np.array([r['p_val'] for r in all_results[method]])

        # Remove NaN replicates
        valid = ~np.isnan(betas)
        n_valid = valid.sum()
        betas_v = betas[valid]
        ci_los_v = ci_los[valid]
        ci_his_v = ci_his[valid]
        p_vals_v = p_vals[valid & ~np.isnan(p_vals)]

        if n_valid < 10:
            rows.append({
                'Method': method, 'n_valid': n_valid,
                'Bias': np.nan, 'RelBias': np.nan, 'EmpSE': np.nan,
                'RMSE': np.nan, 'Coverage': np.nan, 'Power': np.nan,
            })
            continue

        bias = np.mean(betas_v) - beta
        rel_bias = bias / abs(beta) * 100 if beta != 0 else np.nan
        emp_se = np.std(betas_v, ddof=1)
        rmse = np.sqrt(np.mean((betas_v - beta) ** 2))
        coverage = np.mean((ci_los_v <= beta) & (beta <= ci_his_v)) * 100
        power = np.mean(p_vals_v < 0.05) * 100 if len(p_vals_v) > 0 else np.nan

        rows.append({
            'Method': method,
            'n_valid': n_valid,
            'Mean_beta': np.mean(betas_v),
            'Bias': bias,
            'RelBias': rel_bias,
            'EmpSE': emp_se,
            'RMSE': rmse,
            'Coverage': coverage,
            'Power': power,
        })

    results_df = pd.DataFrame(rows)

    # Diagnostics summary
    diag_df = pd.DataFrame(diagnostics)
    diag_summary = {
        'mean_sigma_X2': diag_df['sigma_X2_hat'].mean(),
        'truncation_rate': diag_df['truncated'].mean() * 100,
        'mean_lambda': diag_df['lambda_mean'].mean(),
        'lambda_range': f"{diag_df['lambda_min'].mean():.3f} - {diag_df['lambda_max'].mean():.3f}",
        'mean_censoring': diag_df['censoring_rate'].mean() * 100,
        'mean_m': diag_df['mean_m'].mean(),
        'mean_rho': diag_df['mean_rho'].mean(),
        'mean_meff': diag_df['mean_meff'].mean(),
    }

    return results_df, diag_summary


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == '__main__':
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    print("=" * 70)
    print("PAPER 2 SIMULATION — S1: BASELINE EXCHANGEABLE")
    print("=" * 70)
    print(f"  n = {N_PATIENTS}")
    print(f"  beta = {TRUE_BETA}")
    print(f"  censoring target = {CENSORING_TARGET*100:.0f}%")
    print(f"  rho range = {RHO_RANGE}")
    print(f"  n_reps = {n_reps}")
    print()

    all_results, diagnostics = run_simulation(n_reps, TRUE_BETA)
    results_df, diag_summary = compute_metrics(all_results, diagnostics, TRUE_BETA)

    print("\n" + "=" * 70)
    print("RESULTS — S1 BASELINE")
    print("=" * 70)

    print("\n--- Performance Metrics ---")
    print(results_df.to_string(index=False, float_format='%.3f'))

    print("\n--- Diagnostics ---")
    for k, v in diag_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\n--- GO / NO-GO ---")
    naive_row = results_df[results_df['Method'] == 'Naive'].iloc[0]
    grc_row = results_df[results_df['Method'] == 'GRC'].iloc[0]
    rc_row = results_df[results_df['Method'] == 'StdRC'].iloc[0]

    naive_rel_bias = abs(naive_row['RelBias'])
    grc_rel_bias = abs(grc_row['RelBias'])
    grc_coverage = grc_row['Coverage']
    grc_vs_rc_bias = abs(rc_row['RelBias']) - abs(grc_row['RelBias'])
    grc_vs_rc_coverage = grc_row['Coverage'] - rc_row['Coverage']

    print(f"  Naive relative bias:     {naive_rel_bias:.1f}% (need >= 20%)")
    print(f"  GRC relative bias:       {grc_rel_bias:.1f}% (need <= 8%)")
    print(f"  GRC coverage:            {grc_coverage:.1f}% (need >= 90%)")
    print(f"  GRC vs RC bias improve:  {grc_vs_rc_bias:.1f}pp (need >= 5pp)")
    print(f"  GRC vs RC coverage:      {grc_vs_rc_coverage:.1f}pp (need >= 5pp)")

    passes = 0
    if naive_rel_bias >= 20:
        print("  [PASS] Naive bias >= 20%")
        passes += 1
    else:
        print("  [FAIL] Naive bias < 20%")

    if grc_rel_bias <= 8:
        print("  [PASS] GRC bias <= 8%")
        passes += 1
    else:
        print("  [FAIL] GRC bias > 8%")

    if grc_coverage >= 90:
        print("  [PASS] GRC coverage >= 90%")
        passes += 1
    else:
        print("  [FAIL] GRC coverage < 90%")

    if grc_vs_rc_bias >= 5 or grc_vs_rc_coverage >= 5:
        print("  [PASS] GRC beats RC by >= 5pp")
        passes += 1
    else:
        print("  [FAIL] GRC improvement over RC < 5pp")

    print(f"\n  Score: {passes}/4")
    if passes == 4:
        print("  >>> ALL CRITERIA MET. S1 PASSES. <<<")
    elif passes >= 3:
        print("  >>> MOSTLY PASSES. Review the failed criterion. <<<")
    else:
        print("  >>> FAILS. Debug before proceeding. <<<")

    # Save results
    results_df.to_csv('sim_s1_results.csv', index=False)
    print(f"\nResults saved to sim_s1_results.csv")
