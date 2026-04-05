"""
Paper 2 — DECISIVE 2000-REP RUN (S1, S5, S6 only)
=====================================================
Determines paper identity. Saves replicate-level outputs.
"""

import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from sklearn.neighbors import kneighbors_graph
from lifelines import CoxPHFitter
import warnings, sys, time, os, json

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
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT_DIR = '/Users/ganeshshiwakoti/Desktop/Biostatistics/simulation/decisive_2000'
os.makedirs(OUT_DIR, exist_ok=True)

def expit(x): return 1.0 / (1.0 + np.exp(-x))

SCENARIOS = {
    'S1': {'mode': 'exchangeable', 'rho_range': (0.02, 0.10), 'phi_mult': None},
    'S5': {'mode': 'spatial', 'rho_range': None, 'phi_mult': 2},
    'S6': {'mode': 'spatial', 'rho_range': None, 'phi_mult': 5},
}

# ============================================================
# SPATIAL GENERATION
# ============================================================
def make_grid(m):
    side = int(np.ceil(np.sqrt(m * 1.15)))
    c = []
    for r in range(side):
        for col in range(side):
            c.append([col * SPOT_SPACING + (SPOT_SPACING/2 if r%2 else 0), r * SPOT_SPACING])
    c = np.array(c)
    if len(c) > m:
        c = c[np.random.choice(len(c), m, replace=False)]
    return c

def spatial_field(coords, Xi, phi, rng):
    m = len(coords)
    D = squareform(pdist(coords))
    C = np.exp(-D / phi) + np.eye(m) * 1e-6
    try:
        L = cholesky(C, lower=True)
    except:
        ev, evec = np.linalg.eigh(C)
        L = cholesky(evec @ np.diag(np.maximum(ev, 1e-6)) @ evec.T, lower=True)
    Z = L @ rng.standard_normal(m)
    Y = (Z > stats.norm.ppf(1 - Xi)).astype(int)
    return Y, D

def exact_deff(D, phi):
    m = D.shape[0]
    if m < 2: return 1.0
    R = np.exp(-D / phi)
    return 1 + (m - 1) * R[np.triu_indices(m, k=1)].mean()

# ============================================================
# DATA GENERATION
# ============================================================
def gen_data(cfg, rng):
    n = N_PATIENTS
    eta = rng.normal(MU_ETA, SIGMA_ETA, n)
    X = expit(eta)
    m = np.clip(np.round(np.exp(rng.normal(M_LOG_MU, M_LOG_SD, n))), M_MIN, M_MAX).astype(int)
    Z = rng.standard_normal(n)
    W = np.zeros(n); deff = np.zeros(n)

    if cfg['mode'] == 'exchangeable':
        rho = rng.uniform(*cfg['rho_range'], n)
        for i in range(n):
            deff[i] = 1 + (m[i]-1)*rho[i]
            if rho[i] < 0.001:
                S = rng.binomial(m[i], X[i])
            else:
                phi_bb = (1-rho[i])/rho[i]
                a, b = max(X[i]*phi_bb, 0.01), max((1-X[i])*phi_bb, 0.01)
                S = rng.binomial(m[i], np.clip(rng.beta(a, b), 1e-6, 1-1e-6))
            W[i] = S / m[i]
    else:
        m_sim = np.minimum(m, 800)
        phi = cfg['phi_mult'] * SPOT_SPACING
        for i in range(n):
            coords = make_grid(m_sim[i])
            Y, D = spatial_field(coords, X[i], phi, rng)
            W[i] = Y.mean()
            deff[i] = exact_deff(D, phi)

    sigma2 = np.clip(W * (1-W) * deff / m, 1e-8, None)
    meff = m / deff

    rate = 0.1 * np.exp(TRUE_BETA * X + GAMMA * Z)
    T_ev = rng.exponential(1.0 / rate)
    lC = CENSORING_TARGET * rate.mean() / (1 - CENSORING_TARGET)
    C = rng.exponential(1.0 / lC, n)

    return {'X': X, 'W': W, 'Z': Z, 'T': np.minimum(T_ev, C),
            'delta': (T_ev <= C).astype(int), 'm': m, 'sigma2': sigma2,
            'deff': deff, 'meff': meff}

# ============================================================
# COX + METHODS
# ============================================================
def cox(T, d, covs, names):
    df = pd.DataFrame(covs, columns=names); df['T']=T; df['d']=d
    try:
        cph = CoxPHFitter(); cph.fit(df, 'T', 'd', show_progress=False)
        b=cph.params_[names[0]]; se=cph.standard_errors_[names[0]]
        ci=cph.confidence_intervals_.loc[names[0]]; p=cph.summary['p'].loc[names[0]]
        return b, se, float(ci.iloc[0]), float(ci.iloc[1]), p
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def run_methods(data):
    W=data['W']; s2=data['sigma2']; vW=np.var(W, ddof=1)
    ms2=np.mean(s2); sx2=max(vW-ms2, 0); trunc=sx2==0; muW=np.mean(W)
    r = {}

    r['Oracle'] = cox(data['T'], data['delta'], np.column_stack([data['X'], data['Z']]), ['X','Z'])
    r['Naive'] = cox(data['T'], data['delta'], np.column_stack([W, data['Z']]), ['X','Z'])

    if not trunc:
        lam_g = sx2/(sx2+ms2)
        Xrc = muW + lam_g*(W-muW)
        r['StdRC'] = cox(data['T'], data['delta'], np.column_stack([Xrc, data['Z']]), ['X','Z'])
        li = sx2/(sx2+s2)
        Xgrc = muW + li*(W-muW)
        r['GRC'] = cox(data['T'], data['delta'], np.column_stack([Xgrc, data['Z']]), ['X','Z'])
    else:
        r['StdRC'] = (np.nan,)*5
        r['GRC'] = (np.nan,)*5

    diag = {'sx2': sx2, 'trunc': int(trunc), 'lam_mean': np.mean(li) if not trunc else 0,
            'lam_min': np.min(li) if not trunc else 0, 'lam_max': np.max(li) if not trunc else 0,
            'meff_mean': np.mean(data['meff']), 'cens': 1-data['delta'].mean()}
    return r, diag

# ============================================================
# MAIN LOOP
# ============================================================
print("="*70)
print(f"DECISIVE 2000-REP RUN — S1, S5, S6 ({N_REPS} reps)")
print("="*70)

# Save config
with open(f'{OUT_DIR}/config.json', 'w') as f:
    json.dump({'n_reps': N_REPS, 'n_patients': N_PATIENTS, 'beta': TRUE_BETA,
               'gamma': GAMMA, 'scenarios': list(SCENARIOS.keys())}, f, indent=2)

for sname, cfg in SCENARIOS.items():
    print(f"\n{'='*60}")
    print(f"  {sname} ({N_REPS} reps)")
    print(f"{'='*60}")

    # Replicate-level storage
    rep_rows = []
    diag_rows = []
    t0 = time.time()

    for rep in range(N_REPS):
        seed = rep * 1000 + hash(sname) % 10000
        rng = np.random.default_rng(seed)
        data = gen_data(cfg, rng)
        res, diag = run_methods(data)

        for method in ['Oracle', 'Naive', 'StdRC', 'GRC']:
            b, se, ci_lo, ci_hi, p = res[method]
            rep_rows.append({
                'rep': rep, 'seed': seed, 'method': method,
                'beta_hat': b, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p_val': p
            })

        diag_rows.append({'rep': rep, 'seed': seed, **diag})

        if (rep+1) % 200 == 0:
            el = time.time()-t0
            print(f"    {rep+1}/{N_REPS} | {el:.0f}s | {el/(rep+1):.2f}s/rep | ETA {(N_REPS-rep-1)*el/(rep+1):.0f}s")
            sys.stdout.flush()

    el = time.time()-t0
    print(f"    Done: {el:.0f}s ({el/N_REPS:.2f}s/rep)")

    # Save replicate-level data
    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(f'{OUT_DIR}/{sname}_replicates.csv', index=False)

    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(f'{OUT_DIR}/{sname}_diagnostics.csv', index=False)

    # Compute summary
    print(f"\n  --- {sname} SUMMARY ---")
    for method in ['Oracle', 'Naive', 'StdRC', 'GRC']:
        mdf = rep_df[rep_df['method'] == method]
        betas = mdf['beta_hat'].dropna()
        ci_lo = mdf['ci_lo'].dropna()
        ci_hi = mdf['ci_hi'].dropna()
        pvals = mdf['p_val'].dropna()

        nv = len(betas)
        if nv < 50:
            print(f"  {method:8s}: n_valid={nv} — INSUFFICIENT")
            continue

        bias = betas.mean() - TRUE_BETA
        rel_bias = bias / abs(TRUE_BETA) * 100
        emp_se = betas.std(ddof=1)
        rmse = np.sqrt(((betas - TRUE_BETA)**2).mean())

        # Coverage: align indices
        common = betas.index.intersection(ci_lo.index).intersection(ci_hi.index)
        cov = ((ci_lo.loc[common] <= TRUE_BETA) & (TRUE_BETA <= ci_hi.loc[common])).mean() * 100
        power = (pvals < 0.05).mean() * 100

        print(f"  {method:8s}: bias={bias:+.3f} ({rel_bias:+.1f}%) | SE={emp_se:.3f} | RMSE={rmse:.3f} | Cov={cov:.1f}% | Pow={power:.1f}% | n={nv}")

    trunc_rate = diag_df['trunc'].mean() * 100
    print(f"  Truncation: {trunc_rate:.1f}%")
    print(f"  Lambda: {diag_df['lam_mean'].mean():.3f} ({diag_df['lam_min'].mean():.3f}-{diag_df['lam_max'].mean():.3f})")
    print(f"  m_eff: {diag_df['meff_mean'].mean():.1f}")

# ============================================================
# FINAL COMPARISON — THE DECISION
# ============================================================
print(f"\n\n{'='*70}")
print("THE DECISION: DOES GRC BEAT RC IN S1 AND S6?")
print(f"{'='*70}")

for sname in ['S1', 'S5', 'S6']:
    rdf = pd.read_csv(f'{OUT_DIR}/{sname}_replicates.csv')

    rc_betas = rdf[rdf['method']=='StdRC']['beta_hat'].dropna()
    grc_betas = rdf[rdf['method']=='GRC']['beta_hat'].dropna()
    rc_ci_lo = rdf[rdf['method']=='StdRC']['ci_lo'].dropna()
    rc_ci_hi = rdf[rdf['method']=='StdRC']['ci_hi'].dropna()
    grc_ci_lo = rdf[rdf['method']=='GRC']['ci_lo'].dropna()
    grc_ci_hi = rdf[rdf['method']=='GRC']['ci_hi'].dropna()

    rc_bias = abs((rc_betas.mean() - TRUE_BETA) / TRUE_BETA * 100)
    grc_bias = abs((grc_betas.mean() - TRUE_BETA) / TRUE_BETA * 100)

    rc_common = rc_betas.index.intersection(rc_ci_lo.index).intersection(rc_ci_hi.index)
    grc_common = grc_betas.index.intersection(grc_ci_lo.index).intersection(grc_ci_hi.index)
    rc_cov = ((rc_ci_lo.loc[rc_common] <= TRUE_BETA) & (TRUE_BETA <= rc_ci_hi.loc[rc_common])).mean() * 100
    grc_cov = ((grc_ci_lo.loc[grc_common] <= TRUE_BETA) & (TRUE_BETA <= grc_ci_hi.loc[grc_common])).mean() * 100

    bias_diff = rc_bias - grc_bias  # positive = GRC better
    cov_diff = grc_cov - rc_cov     # positive = GRC better

    print(f"\n  {sname}:")
    print(f"    RC  bias: {rc_bias:.1f}%  coverage: {rc_cov:.1f}%")
    print(f"    GRC bias: {grc_bias:.1f}%  coverage: {grc_cov:.1f}%")
    print(f"    Diff: bias {bias_diff:+.1f}pp (+ = GRC better), cov {cov_diff:+.1f}pp (+ = GRC better)")

    if bias_diff >= 3 or cov_diff >= 3:
        print(f"    >>> GRC WINS (by >= 3pp) <<<")
    elif bias_diff >= 0:
        print(f"    >>> GRC MARGINALLY BETTER (< 3pp) <<<")
    else:
        print(f"    >>> RC WINS <<<")

print(f"\n{'='*70}")
print("PAPER IDENTITY DECISION")
print(f"{'='*70}")

# Load S1 and S6 for final call
s1_rdf = pd.read_csv(f'{OUT_DIR}/S1_replicates.csv')
s6_rdf = pd.read_csv(f'{OUT_DIR}/S6_replicates.csv')

s1_rc_bias = abs((s1_rdf[s1_rdf['method']=='StdRC']['beta_hat'].dropna().mean() - TRUE_BETA) / TRUE_BETA * 100)
s1_grc_bias = abs((s1_rdf[s1_rdf['method']=='GRC']['beta_hat'].dropna().mean() - TRUE_BETA) / TRUE_BETA * 100)
s6_rc_bias = abs((s6_rdf[s6_rdf['method']=='StdRC']['beta_hat'].dropna().mean() - TRUE_BETA) / TRUE_BETA * 100)
s6_grc_bias = abs((s6_rdf[s6_rdf['method']=='GRC']['beta_hat'].dropna().mean() - TRUE_BETA) / TRUE_BETA * 100)

s1_wins = (s1_rc_bias - s1_grc_bias) >= 3
s6_wins = (s6_rc_bias - s6_grc_bias) >= 3

if s1_wins and s6_wins:
    print("\n  >>> CASE 1: GRC beats RC in BOTH S1 and S6")
    print("  >>> Paper = MECHANISM + CORRECTION <<<")
elif s1_wins or s6_wins:
    print(f"\n  >>> MIXED: GRC wins in {'S1' if s1_wins else 'S6'} only")
    print("  >>> Paper = MECHANISM + QUALIFIED CORRECTION <<<")
else:
    print("\n  >>> CASE 2: RC matches or beats GRC")
    print("  >>> Paper = MECHANISM + QUANTIFICATION + WARNING <<<")

print(f"\n  S1: RC bias={s1_rc_bias:.1f}% vs GRC bias={s1_grc_bias:.1f}% (diff={s1_rc_bias-s1_grc_bias:+.1f}pp)")
print(f"  S6: RC bias={s6_rc_bias:.1f}% vs GRC bias={s6_grc_bias:.1f}% (diff={s6_rc_bias-s6_grc_bias:+.1f}pp)")
print(f"\nAll results saved to {OUT_DIR}/")
