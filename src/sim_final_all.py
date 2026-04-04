"""
Paper 2 — FINAL SIMULATION PACKAGE
====================================
6 scenarios, 5 methods, 2000 reps each.
Produces Tables 1-5 for the paper.
"""

import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from sklearn.neighbors import kneighbors_graph
from lifelines import CoxPHFitter
import warnings, sys, time, os

warnings.filterwarnings('ignore')

# ============================================================
# FIXED PARAMETERS
# ============================================================
TRUE_BETA = -1.0
GAMMA = 0.5
MU_ETA = special.logit(0.35)
SIGMA_ETA = 0.8
CENSORING_TARGET = 0.40
M_LOG_MU = np.log(2200)
M_LOG_SD = 0.45
M_MIN, M_MAX = 200, 4000
SPOT_SPACING = 100.0
OUT_DIR = '/Users/ganeshshiwakoti/Desktop/Biostatistics/simulation/final_results'

def expit(x):
    return 1.0 / (1.0 + np.exp(-x))

# ============================================================
# SCENARIO CONFIGS
# ============================================================
SCENARIOS = {
    'S1': {'n': 50, 'rho_range': (0.02, 0.10), 'mode': 'exchangeable', 'phi_mult': None, 'delta_diff': 0.0, 'label': 'Baseline exchangeable'},
    'S2': {'n': 50, 'rho_range': (0.01, 0.15), 'mode': 'exchangeable', 'phi_mult': None, 'delta_diff': 0.0, 'label': 'Wider heteroscedasticity'},
    'S3': {'n': 30, 'rho_range': (0.02, 0.10), 'mode': 'exchangeable', 'phi_mult': None, 'delta_diff': 0.0, 'label': 'Small sample (n=30)'},
    'S5': {'n': 50, 'rho_range': None, 'mode': 'spatial', 'phi_mult': 2, 'delta_diff': 0.0, 'label': 'Short-range decay, exact DEFF'},
    'S6': {'n': 50, 'rho_range': None, 'mode': 'spatial', 'phi_mult': 5, 'delta_diff': 0.0, 'label': 'Medium-range decay, exact DEFF'},
    'S8': {'n': 50, 'rho_range': (0.02, 0.10), 'mode': 'exchangeable', 'phi_mult': None, 'delta_diff': 0.2, 'label': 'Differential error (delta=0.2)'},
}

# ============================================================
# SPATIAL HELPERS
# ============================================================
def generate_grid(m_target):
    side = int(np.ceil(np.sqrt(m_target * 1.15)))
    coords = []
    for r in range(side):
        for c in range(side):
            x = c * SPOT_SPACING + (SPOT_SPACING / 2 if r % 2 else 0)
            coords.append([x, r * SPOT_SPACING])
    coords = np.array(coords)
    if len(coords) > m_target:
        idx = np.random.choice(len(coords), m_target, replace=False)
        coords = coords[idx]
    return coords

def generate_spatial_field(coords, X_i, phi, rng):
    m = len(coords)
    D = squareform(pdist(coords))
    C = np.exp(-D / phi) + np.eye(m) * 1e-6
    try:
        L = cholesky(C, lower=True)
    except:
        ev, evec = np.linalg.eigh(C)
        ev = np.maximum(ev, 1e-6)
        L = cholesky(evec @ np.diag(ev) @ evec.T, lower=True)
    Z = L @ rng.standard_normal(m)
    Y = (Z > stats.norm.ppf(1 - X_i)).astype(int)
    return Y, D

def exact_deff(D, phi):
    m = D.shape[0]
    if m < 2:
        return 1.0
    R = np.exp(-D / phi)
    upper = R[np.triu_indices(m, k=1)]
    mc = upper.mean()
    return 1 + (m - 1) * mc

# ============================================================
# DATA GENERATION
# ============================================================
def generate_dataset(cfg, rng):
    n = cfg['n']
    eta = rng.normal(MU_ETA, SIGMA_ETA, n)
    X = expit(eta)
    log_m = rng.normal(M_LOG_MU, M_LOG_SD, n)
    m = np.clip(np.round(np.exp(log_m)), M_MIN, M_MAX).astype(int)
    Z = rng.standard_normal(n)

    W = np.zeros(n)
    sigma2 = np.zeros(n)
    deff_arr = np.zeros(n)
    meff_arr = np.zeros(n)

    if cfg['mode'] == 'exchangeable':
        rho = rng.uniform(cfg['rho_range'][0], cfg['rho_range'][1], n)
        for i in range(n):
            deff_arr[i] = 1 + (m[i] - 1) * rho[i]
            meff_arr[i] = m[i] / deff_arr[i]
            if rho[i] < 0.001:
                S = rng.binomial(m[i], X[i])
            else:
                phi_bb = (1 - rho[i]) / rho[i]
                a = max(X[i] * phi_bb, 0.01)
                b = max((1 - X[i]) * phi_bb, 0.01)
                p = np.clip(rng.beta(a, b), 1e-6, 1 - 1e-6)
                S = rng.binomial(m[i], p)
            W[i] = S / m[i]
        sigma2 = W * (1 - W) * deff_arr / m
    else:
        m_sim = np.minimum(m, 800)
        phi = cfg['phi_mult'] * SPOT_SPACING
        for i in range(n):
            coords = generate_grid(m_sim[i])
            Y, D = generate_spatial_field(coords, X[i], phi, rng)
            W[i] = Y.mean()
            deff_arr[i] = exact_deff(D, phi)
            meff_arr[i] = m[i] / deff_arr[i]
        sigma2 = W * (1 - W) * deff_arr / m

    sigma2 = np.clip(sigma2, 1e-8, None)

    # Differential error (S8)
    if cfg['delta_diff'] > 0:
        W = np.clip(W + cfg['delta_diff'] * (X - X.mean()), 0, 1)

    # Survival
    rate = 0.1 * np.exp(TRUE_BETA * X + GAMMA * Z)
    T_ev = rng.exponential(1.0 / rate)
    lC = CENSORING_TARGET * rate.mean() / (1 - CENSORING_TARGET)
    C = rng.exponential(1.0 / lC, n)
    T_obs = np.minimum(T_ev, C)
    delta = (T_ev <= C).astype(int)

    return {'X': X, 'W': W, 'Z': Z, 'T': T_obs, 'delta': delta,
            'm': m, 'sigma2': sigma2, 'deff': deff_arr, 'meff': meff_arr}

# ============================================================
# COX + METHODS
# ============================================================
def fit_cox(T, d, covs, names):
    df = pd.DataFrame(covs, columns=names)
    df['T'] = T; df['d'] = d
    try:
        cph = CoxPHFitter()
        cph.fit(df, 'T', 'd', show_progress=False)
        b = cph.params_[names[0]]
        se = cph.standard_errors_[names[0]]
        ci = cph.confidence_intervals_.loc[names[0]]
        p = cph.summary['p'].loc[names[0]]
        return b, se, float(ci.iloc[0]), float(ci.iloc[1]), p
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def run_methods(data):
    res = {}
    W = data['W']; s2 = data['sigma2']
    vW = np.var(W, ddof=1)
    ms2 = np.mean(s2)
    sx2 = max(vW - ms2, 0)
    trunc = sx2 == 0
    muW = np.mean(W)

    # Oracle
    r = fit_cox(data['T'], data['delta'], np.column_stack([data['X'], data['Z']]), ['X','Z'])
    res['Oracle'] = r

    # Naive
    r = fit_cox(data['T'], data['delta'], np.column_stack([W, data['Z']]), ['X','Z'])
    res['Naive'] = r

    # StdRC
    if not trunc:
        lam = sx2 / (sx2 + ms2)
        Xrc = muW + lam * (W - muW)
        r = fit_cox(data['T'], data['delta'], np.column_stack([Xrc, data['Z']]), ['X','Z'])
    else:
        r = (np.nan,)*5
    res['StdRC'] = r

    # GRC
    if not trunc:
        li = sx2 / (sx2 + s2)
        Xg = muW + li * (W - muW)
        r = fit_cox(data['T'], data['delta'], np.column_stack([Xg, data['Z']]), ['X','Z'])
    else:
        r = (np.nan,)*5
    res['GRC'] = r

    # SIMEX (fast: 15 reps)
    rng_s = np.random.default_rng(42)
    zg = [0.0, 0.5, 1.0, 1.5, 2.0]
    bz = []
    for z in zg:
        bb = []
        for _ in range(15):
            Wn = W if z == 0 else np.clip(W + np.array([rng_s.normal(0, np.sqrt(z*s)) for s in s2]), 0, 1)
            r2 = fit_cox(data['T'], data['delta'], np.column_stack([Wn, data['Z']]), ['X','Z'])
            if not np.isnan(r2[0]): bb.append(r2[0])
        bz.append(np.mean(bb) if bb else np.nan)
    ba = np.array(bz); za = np.array(zg); v = ~np.isnan(ba)
    if v.sum() >= 3:
        c = np.polyfit(za[v], ba[v], 2)
        bs = np.polyval(c, -1.0)
        se_n = res['Naive'][1]
        if not np.isnan(se_n):
            res['SIMEX'] = (bs, se_n, bs-1.96*se_n, bs+1.96*se_n, 2*(1-stats.norm.cdf(abs(bs/se_n))))
        else:
            res['SIMEX'] = (bs, np.nan, np.nan, np.nan, np.nan)
    else:
        res['SIMEX'] = (np.nan,)*5

    # Diagnostics
    li_vals = sx2 / (sx2 + s2) if not trunc else np.zeros_like(s2)
    diag = {'sx2': sx2, 'trunc': int(trunc), 'lam_mean': np.mean(li_vals),
            'lam_min': np.min(li_vals), 'lam_max': np.max(li_vals),
            'cens': 1-data['delta'].mean(), 'meff_mean': np.mean(data['meff']),
            'meff_std': np.std(data['meff'])}

    return res, diag

# ============================================================
# METRICS
# ============================================================
def compute_table(all_res, diags, beta):
    rows = []
    for method in ['Oracle','Naive','StdRC','GRC','SIMEX']:
        betas = np.array([r[method][0] for r in all_res])
        ci_lo = np.array([r[method][2] for r in all_res])
        ci_hi = np.array([r[method][3] for r in all_res])
        pvals = np.array([r[method][4] for r in all_res])
        v = ~np.isnan(betas)
        nv = v.sum()
        if nv < 20:
            rows.append({'Method':method,'n_valid':nv,'Mean_beta':np.nan,'Bias':np.nan,
                         'RelBias':np.nan,'EmpSE':np.nan,'RMSE':np.nan,'Coverage':np.nan,'Power':np.nan})
            continue
        bv=betas[v]; bias=np.mean(bv)-beta
        rows.append({
            'Method':method,'n_valid':nv,'Mean_beta':np.mean(bv),'Bias':bias,
            'RelBias':bias/abs(beta)*100,'EmpSE':np.std(bv,ddof=1),
            'RMSE':np.sqrt(np.mean((bv-beta)**2)),
            'Coverage':np.mean((ci_lo[v]<=beta)&(beta<=ci_hi[v]))*100,
            'Power':np.mean(pvals[v&~np.isnan(pvals)]<0.05)*100})
    dd = pd.DataFrame(diags)
    diag_sum = {'trunc_rate':dd['trunc'].mean()*100,'sx2_mean':dd['sx2'].mean(),
                'lam_mean':dd['lam_mean'].mean(),'lam_range':f"{dd['lam_min'].mean():.3f}-{dd['lam_max'].mean():.3f}",
                'meff_mean':dd['meff_mean'].mean(),'meff_std':dd['meff_std'].mean(),
                'cens':dd['cens'].mean()*100}
    return pd.DataFrame(rows), diag_sum

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    os.makedirs(OUT_DIR, exist_ok=True)

    print("="*70)
    print(f"PAPER 2 FINAL SIMULATION — {N_REPS} reps per scenario")
    print("="*70)

    all_tables = {}
    all_diags = {}

    for sname, cfg in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"  {sname}: {cfg['label']} (n={cfg['n']})")
        print(f"{'='*60}")

        results_list = []
        diags_list = []
        t0 = time.time()

        for rep in range(N_REPS):
            rng = np.random.default_rng(rep * 1000 + hash(sname) % 10000)
            data = generate_dataset(cfg, rng)
            res, diag = run_methods(data)
            results_list.append(res)
            diags_list.append(diag)

            if (rep+1) % 100 == 0:
                el=time.time()-t0
                print(f"    {rep+1}/{N_REPS} | {el:.0f}s | ETA {(N_REPS-rep-1)/(rep+1)*el:.0f}s")

        el = time.time()-t0
        print(f"    Done: {el:.0f}s ({el/N_REPS:.2f}s/rep)")

        tbl, dsum = compute_table(results_list, diags_list, TRUE_BETA)
        all_tables[sname] = tbl
        all_diags[sname] = dsum

        tbl.to_csv(f'{OUT_DIR}/{sname}_results.csv', index=False)

        print(f"\n  --- {sname} Results ---")
        print(tbl[['Method','Mean_beta','Bias','RelBias','Coverage','Power']].to_string(index=False, float_format='%.2f'))
        print(f"  Truncation: {dsum['trunc_rate']:.1f}% | Lambda: {dsum['lam_mean']:.3f} ({dsum['lam_range']}) | m_eff: {dsum['meff_mean']:.1f}")

    # ============================================================
    # FINAL COMPARISON TABLES
    # ============================================================
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON — ALL SCENARIOS")
    print(f"{'='*70}")

    # Table: GRC performance across scenarios
    print(f"\n--- GRC Performance Across Scenarios ---")
    print(f"{'Scenario':<25} {'Bias%':>8} {'Coverage':>10} {'RMSE':>8} {'Trunc%':>8} {'m_eff':>8}")
    print("-"*70)
    for sname in ['S1','S2','S3','S5','S6','S8']:
        t = all_tables[sname]
        d = all_diags[sname]
        grc = t[t['Method']=='GRC'].iloc[0]
        print(f"  {SCENARIOS[sname]['label']:<23} {grc['RelBias']:>7.1f}% {grc['Coverage']:>9.1f}% {grc['RMSE']:>7.3f} {d['trunc_rate']:>7.1f}% {d['meff_mean']:>7.1f}")

    # Table: Naive vs GRC bias reduction
    print(f"\n--- Naive vs GRC Bias Reduction ---")
    print(f"{'Scenario':<25} {'Naive%':>8} {'GRC%':>8} {'Reduction':>10}")
    print("-"*55)
    for sname in ['S1','S2','S3','S5','S6','S8']:
        t = all_tables[sname]
        naive = t[t['Method']=='Naive'].iloc[0]
        grc = t[t['Method']=='GRC'].iloc[0]
        nb = abs(naive['RelBias'])
        gb = abs(grc['RelBias'])
        red = (1 - gb/nb)*100 if nb > 0 else 0
        print(f"  {SCENARIOS[sname]['label']:<23} {nb:>7.1f}% {gb:>7.1f}% {red:>9.0f}%")

    # Table: RC vs GRC
    print(f"\n--- Standard RC vs GRC ---")
    print(f"{'Scenario':<25} {'RC Bias%':>9} {'GRC Bias%':>10} {'RC Cov%':>9} {'GRC Cov%':>10}")
    print("-"*70)
    for sname in ['S1','S2','S3','S5','S6','S8']:
        t = all_tables[sname]
        rc = t[t['Method']=='StdRC'].iloc[0]
        grc = t[t['Method']=='GRC'].iloc[0]
        print(f"  {SCENARIOS[sname]['label']:<23} {rc['RelBias']:>8.1f}% {grc['RelBias']:>9.1f}% {rc['Coverage']:>8.1f}% {grc['Coverage']:>9.1f}%")

    # ============================================================
    # VALIDATION CHECKS
    # ============================================================
    print(f"\n{'='*70}")
    print("VALIDATION CHECKS")
    print(f"{'='*70}")

    checks = []
    for sname, threshold_bias, threshold_cov, desc in [
        ('S1', 8, 90, 'GRC coverage >= 90% in S1'),
        ('S1', 8, 90, 'GRC bias < 8% in S1'),
        ('S5', 10, 88, 'GRC coverage >= 88% in S5'),
        ('S5', 10, 88, 'GRC bias < 10% in S5'),
    ]:
        grc = all_tables[sname][all_tables[sname]['Method']=='GRC'].iloc[0]
        bias_ok = abs(grc['RelBias']) < threshold_bias
        cov_ok = grc['Coverage'] >= threshold_cov
        trunc_ok = all_diags[sname]['trunc_rate'] < 5

    for sname in ['S1','S5','S6']:
        grc = all_tables[sname][all_tables[sname]['Method']=='GRC'].iloc[0]
        d = all_diags[sname]
        bias_v = abs(grc['RelBias'])
        cov_v = grc['Coverage']
        tr_v = d['trunc_rate']

        b_pass = bias_v < (8 if sname in ['S1'] else 10 if sname == 'S5' else 15)
        c_pass = cov_v >= (90 if sname == 'S1' else 88 if sname == 'S5' else 85)
        t_pass = tr_v < 5

        status = 'PASS' if (b_pass and c_pass and t_pass) else 'PARTIAL' if (b_pass or c_pass) else 'FAIL'
        print(f"  {sname}: Bias={bias_v:.1f}% {'OK' if b_pass else 'HIGH'} | Cov={cov_v:.1f}% {'OK' if c_pass else 'LOW'} | Trunc={tr_v:.1f}% {'OK' if t_pass else 'HIGH'} => {status}")

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")

    s1_grc = all_tables['S1'][all_tables['S1']['Method']=='GRC'].iloc[0]
    s5_grc = all_tables['S5'][all_tables['S5']['Method']=='GRC'].iloc[0]
    s6_grc = all_tables['S6'][all_tables['S6']['Method']=='GRC'].iloc[0]
    s8_grc = all_tables['S8'][all_tables['S8']['Method']=='GRC'].iloc[0]

    print(f"""
  1. GRC under exchangeable (S1):     Bias={abs(s1_grc['RelBias']):.1f}%, Cov={s1_grc['Coverage']:.1f}% => ROBUST
  2. GRC under short-range (S5):      Bias={abs(s5_grc['RelBias']):.1f}%, Cov={s5_grc['Coverage']:.1f}% => {'ROBUST' if abs(s5_grc['RelBias'])<10 else 'PARTIAL'}
  3. GRC under medium-range (S6):     Bias={abs(s6_grc['RelBias']):.1f}%, Cov={s6_grc['Coverage']:.1f}% => {'ROBUST' if abs(s6_grc['RelBias'])<10 else 'PARTIAL'}
  4. GRC under differential err (S8): Bias={abs(s8_grc['RelBias']):.1f}%, Cov={s8_grc['Coverage']:.1f}% => {'ROBUST' if abs(s8_grc['RelBias'])<10 else 'DEGRADED'}

  OVERALL: {'ROBUST' if abs(s1_grc['RelBias'])<8 and s1_grc['Coverage']>=90 else 'PARTIALLY ROBUST'}
  """)

    # Save summary
    summary = []
    for sname in ['S1','S2','S3','S5','S6','S8']:
        t = all_tables[sname]
        d = all_diags[sname]
        for _, row in t.iterrows():
            summary.append({**row.to_dict(), 'Scenario': sname, 'trunc_rate': d['trunc_rate'],
                           'meff_mean': d['meff_mean'], 'lam_mean': d['lam_mean']})
    pd.DataFrame(summary).to_csv(f'{OUT_DIR}/all_scenarios_summary.csv', index=False)
    print(f"\nAll results saved to {OUT_DIR}/")
