"""
Divergence Diagnosis — Single replicate, full diagnostics
"""
import numpy as np
import pandas as pd
from scipy import stats, special
import cmdstanpy
import sys, warnings
warnings.filterwarnings('ignore')

def expit(x): return 1.0/(1.0+np.exp(-x))

def gen_data(seed=42):
    rng=np.random.default_rng(seed); N=50
    eta=rng.normal(special.logit(0.35),0.8,N); X=expit(eta)
    m=np.clip(np.round(np.exp(rng.normal(np.log(2200),0.45,N))),200,4000).astype(int)
    rho=rng.uniform(0.02,0.10,N); Z=rng.standard_normal(N); W=np.zeros(N)
    for i in range(N):
        phi_bb=(1-rho[i])/rho[i]; a,b=max(X[i]*phi_bb,0.01),max((1-X[i])*phi_bb,0.01)
        W[i]=rng.binomial(m[i],np.clip(rng.beta(a,b),1e-6,1-1e-6))/m[i]
    rate=0.1*np.exp(-1.0*X+0.5*Z); T_ev=rng.exponential(1.0/rate)
    lC=0.4*rate.mean()/0.6; C=rng.exponential(1.0/lC,N)
    et=np.minimum(T_ev,C); d=(T_ev<=C).astype(int)
    ci=np.quantile(et[d==1],np.linspace(0,1,6)[1:-1]) if (d==1).sum()>=5 else np.quantile(et,np.linspace(0,1,6)[1:-1])
    return {'N':N,'P':1,'K':5,'W':W.tolist(),'Z':Z.reshape(-1,1).tolist(),
            'T_obs':et.tolist(),'event':d.tolist(),'m_spots':m.astype(float).tolist(),
            'log_m':np.log(m.astype(float)).tolist(),'rho':rho.tolist(),'s':ci.tolist()}

sd = gen_data()
model_v3 = cmdstanpy.CmdStanModel(exe_file='model_v3')

print("="*60)
print("STEP 1: Single rep with v3, adapt_delta=0.99")
print("="*60)

fit = model_v3.sample(data=sd, chains=4, iter_sampling=1000, iter_warmup=1000,
                      adapt_delta=0.99, max_treedepth=15, show_progress=False, show_console=False,
                      save_warmup=True)

# Extract divergences
div = fit.method_variables()['divergent__']
n_div = int(np.sum(div))
n_total = len(div)
print(f"\nDivergences: {n_div} / {n_total} ({100*n_div/n_total:.1f}%)")

# Which iterations have divergences?
div_idx = np.where(div > 0)[0]
if len(div_idx) > 0:
    # Check if divergences are in warmup or sampling
    per_chain = n_total // 4
    warmup_per_chain = 1000
    sampling_per_chain = 1000

    warmup_divs = 0
    sampling_divs = 0
    for idx in div_idx:
        chain = idx // (warmup_per_chain + sampling_per_chain)
        within_chain = idx % (warmup_per_chain + sampling_per_chain)
        if within_chain < warmup_per_chain:
            warmup_divs += 1
        else:
            sampling_divs += 1
    print(f"  Warmup divergences: {warmup_divs}")
    print(f"  Sampling divergences: {sampling_divs}")

# Step sizes and treedepths
stepsize = fit.method_variables()['stepsize__']
treedepth = fit.method_variables()['treedepth__']
energy = fit.method_variables()['energy__']

print(f"\nStep size: mean={np.mean(stepsize):.6f}, min={np.min(stepsize):.6f}")
print(f"Treedepth: mean={np.mean(treedepth):.1f}, max={int(np.max(treedepth))}")
print(f"  At max treedepth: {np.sum(treedepth >= 15)} iterations")

# Rhat and ESS for all parameters
summary = fit.summary()
print(f"\n--- Parameter Summary (worst Rhat) ---")
rhat_col = summary['R_hat']
worst_rhat = rhat_col.nlargest(10)
print(worst_rhat.to_string())

print(f"\n--- Parameters with Rhat > 1.05 ---")
bad = rhat_col[rhat_col > 1.05]
print(f"  Count: {len(bad)}")
if len(bad) > 0:
    print(bad.to_string())

print(f"\n--- Parameters with Rhat > 1.01 ---")
marginal = rhat_col[rhat_col > 1.01]
print(f"  Count: {len(marginal)}")
if len(marginal) > 0 and len(marginal) <= 20:
    print(marginal.to_string())

# ESS
ess_col = summary['N_Eff'] if 'N_Eff' in summary.columns else summary.get('ESS_bulk', summary.get('StdDev'))
if 'N_Eff' in summary.columns:
    print(f"\n--- Lowest ESS ---")
    print(ess_col.nsmallest(10).to_string())

# Key parameters
print(f"\n--- Key Parameter Estimates ---")
for param in ['beta', 'mu_eta', 'sigma_eta', 'alpha0', 'alpha1', 'alpha2', 'tau', 'HR']:
    if param in summary.index:
        row = summary.loc[param]
        print(f"  {param:12s}: mean={row['Mean']:.4f}, sd={row['StdDev']:.4f}, Rhat={row['R_hat']:.4f}")

# Check which latent parameters have worst mixing
print(f"\n--- Latent X[i] Rhat (worst 5) ---")
x_params = [p for p in summary.index if p.startswith('X[')]
if x_params:
    x_rhats = rhat_col[x_params].nlargest(5)
    print(x_rhats.to_string())

print(f"\n--- z_eta Rhat (worst 5) ---")
z_params = [p for p in summary.index if p.startswith('z_eta[')]
if z_params:
    z_rhats = rhat_col[z_params].nlargest(5)
    print(z_rhats.to_string())

print(f"\n--- z_sigma Rhat (worst 5) ---")
zs_params = [p for p in summary.index if p.startswith('z_sigma[')]
if zs_params:
    zs_rhats = rhat_col[zs_params].nlargest(5)
    print(zs_rhats.to_string())

# Divergence association: which parameters have extreme values at divergences
if sampling_divs > 0:
    print(f"\n--- Divergence Association ---")
    draws = fit.draws_pd()
    div_mask = draws['divergent__'] == 1
    nodiv_mask = draws['divergent__'] == 0

    for param in ['beta', 'sigma_eta', 'tau', 'alpha0', 'alpha1', 'alpha2']:
        if param in draws.columns:
            div_vals = draws.loc[div_mask, param]
            nodiv_vals = draws.loc[nodiv_mask, param]
            if len(div_vals) > 0 and len(nodiv_vals) > 0:
                print(f"  {param:12s}: div_mean={div_vals.mean():.4f} vs nodiv_mean={nodiv_vals.mean():.4f} | div_sd={div_vals.std():.4f}")

    # Check sigma floor hits
    sig_params = [p for p in draws.columns if p.startswith('sig[')]
    if sig_params:
        min_sig = draws[sig_params].min(axis=1)
        print(f"\n  Min sigma across patients: mean={min_sig.mean():.6f}, min={min_sig.min():.6f}")
        print(f"  At divergence: {min_sig[div_mask].mean():.6f}")
        print(f"  At non-divergence: {min_sig[nodiv_mask].mean():.6f}")
