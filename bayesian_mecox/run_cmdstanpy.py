"""
Bayesian ME-Cox via CmdStanPy — Full Comparison
"""
import numpy as np
import pandas as pd
from scipy import stats, special
from lifelines import CoxPHFitter
import cmdstanpy
import time, sys, warnings
warnings.filterwarnings('ignore')

TRUE_BETA = -1.0; GAMMA = 0.5
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

def expit(x): return 1.0/(1.0+np.exp(-x))

def gen_data(n=50, seed=42):
    rng = np.random.default_rng(seed)
    eta = rng.normal(special.logit(0.35), 0.8, n)
    X = expit(eta)
    m = np.clip(np.round(np.exp(rng.normal(np.log(2200), 0.45, n))), 200, 4000).astype(int)
    rho = rng.uniform(0.02, 0.10, n)
    DEFF = 1 + (m-1)*rho; Z = rng.standard_normal(n)
    W = np.zeros(n)
    for i in range(n):
        phi = (1-rho[i])/rho[i] if rho[i]>0.001 else 1e6
        a, b = max(X[i]*phi, 0.01), max((1-X[i])*phi, 0.01)
        p = np.clip(rng.beta(a,b), 1e-6, 1-1e-6) if rho[i]>0.001 else X[i]
        W[i] = rng.binomial(m[i], p)/m[i]
    rate = 0.1*np.exp(TRUE_BETA*X + GAMMA*Z)
    T_ev = rng.exponential(1.0/rate)
    lC = 0.4*rate.mean()/0.6; C = rng.exponential(1.0/lC, n)
    return {'X':X,'W':W,'Z':Z.reshape(-1,1),'T':np.minimum(T_ev,C),'delta':(T_ev<=C).astype(int),'m':m,'rho':rho,'DEFF':DEFF}

def cox_fit(T,d,covs,names):
    df = pd.DataFrame(covs, columns=names); df['T']=T; df['d']=d
    try:
        cph = CoxPHFitter(); cph.fit(df,'T','d',show_progress=False)
        b=cph.params_[names[0]]; se=cph.standard_errors_[names[0]]
        ci=cph.confidence_intervals_.loc[names[0]]; p=cph.summary['p'].loc[names[0]]
        return b, se, float(ci.iloc[0]), float(ci.iloc[1]), p
    except: return (np.nan,)*5

def freq_methods(data):
    W=data['W']; s2=np.clip(W*(1-W)*data['DEFF']/data['m'],1e-8,None)
    vW=np.var(W,ddof=1); ms2=np.mean(s2); sx2=max(vW-ms2,0); muW=np.mean(W)
    r={}
    r['Oracle']=cox_fit(data['T'],data['delta'],np.column_stack([data['X'],data['Z']]),['X','Z'])
    r['Naive']=cox_fit(data['T'],data['delta'],np.column_stack([W,data['Z']]),['X','Z'])
    if sx2>0:
        lam=sx2/(sx2+ms2); Xrc=muW+lam*(W-muW)
        r['StdRC']=cox_fit(data['T'],data['delta'],np.column_stack([Xrc,data['Z']]),['X','Z'])
        li=sx2/(sx2+s2); Xg=muW+li*(W-muW)
        r['GRC']=cox_fit(data['T'],data['delta'],np.column_stack([Xg,data['Z']]),['X','Z'])
    else: r['StdRC']=r['GRC']=(np.nan,)*5
    return r

def bayes_fit(data, model, K=5):
    et = data['T'][data['delta']==1]
    if len(et)<K: cuts_inner = np.quantile(data['T'], np.linspace(0,1,K+1)[1:-1])
    else: cuts_inner = np.quantile(et, np.linspace(0,1,K+1)[1:-1])
    cuts = np.concatenate([[0], cuts_inner, [data['T'].max()*2]])
    sd = {
        'N':len(data['W']),'P':data['Z'].shape[1],'K':K,
        'W':data['W'].tolist(),'Z':data['Z'].tolist(),
        'T_obs':data['T'].tolist(),'event':data['delta'].tolist(),
        'm':data['m'].astype(float).tolist(),'log_m':np.log(data['m'].astype(float)).tolist(),
        'rho':data['rho'].tolist(),'s':cuts_inner.tolist()
    }
    fit = model.sample(data=sd, chains=2, iter_sampling=500, iter_warmup=500,
                       adapt_delta=0.99, max_treedepth=14, show_progress=False, show_console=False)
    beta_draws = fit.stan_variable('beta')
    b=np.mean(beta_draws); se=np.std(beta_draws)
    ci=np.percentile(beta_draws,[2.5,97.5])
    p=2*min(np.mean(beta_draws>0),np.mean(beta_draws<0))
    div = fit.diagnose().count('divergent')
    return b, se, ci[0], ci[1], p, div

# MAIN
print("="*60)
print(f"BAYESIAN ME-COX ({N_REPS} reps)")
print("="*60)

model = cmdstanpy.CmdStanModel(exe_file='/Users/ganeshshiwakoti/Desktop/Biostatistics/bayesian_mecox/model')
print("Model loaded.\n")

methods = ['Oracle','Naive','StdRC','GRC','Bayesian']
all_b = {m:[] for m in methods}
all_ci = {m:[] for m in methods}
all_p = {m:[] for m in methods}
t0 = time.time()

for rep in range(N_REPS):
    data = gen_data(seed=rep*1000+42)
    fr = freq_methods(data)
    for m in ['Oracle','Naive','StdRC','GRC']:
        b,se,cl,ch,p = fr[m]
        all_b[m].append(b); all_ci[m].append((cl,ch)); all_p[m].append(p)
    try:
        b,se,cl,ch,p,div = bayes_fit(data, model)
        all_b['Bayesian'].append(b); all_ci['Bayesian'].append((cl,ch)); all_p['Bayesian'].append(p)
        if div > 0: print(f"  Rep {rep+1}: {div} divergences")
    except Exception as e:
        all_b['Bayesian'].append(np.nan); all_ci['Bayesian'].append((np.nan,np.nan)); all_p['Bayesian'].append(np.nan)
        print(f"  Rep {rep+1}: FAIL ({e})")
    if (rep+1)%5==0:
        el=time.time()-t0; print(f"  {rep+1}/{N_REPS} | {el:.0f}s | {el/(rep+1):.1f}s/rep")

# RESULTS
print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"\n{'Method':<12} {'Mean':>8} {'Bias':>8} {'Rel%':>7} {'Cov%':>7} {'n':>5}")
print("-"*50)
for m in methods:
    betas=np.array(all_b[m]); v=~np.isnan(betas); nv=v.sum()
    if nv<3: print(f"{m:<12} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>7} {nv:>5}"); continue
    bv=betas[v]; bias=np.mean(bv)-TRUE_BETA; rel=bias/abs(TRUE_BETA)*100
    cc=0
    for i,vv in enumerate(v):
        if vv:
            cl,ch=all_ci[m][i]
            if not np.isnan(cl) and cl<=TRUE_BETA<=ch: cc+=1
    cov=cc/nv*100
    print(f"{m:<12} {np.mean(bv):>8.3f} {bias:>+8.3f} {rel:>+6.1f}% {cov:>6.1f}% {nv:>5}")

print(f"\n{'='*60}")
print("THE DECISION")
print(f"{'='*60}")
for m in ['Naive','StdRC','GRC','Bayesian']:
    betas=np.array(all_b[m]); v=~np.isnan(betas)
    if v.sum()<3: continue
    bias=abs(np.mean(betas[v])-TRUE_BETA)/abs(TRUE_BETA)*100
    print(f"  {m}: |RelBias| = {bias:.1f}%")
bb=np.array(all_b['Bayesian']); br=np.array(all_b['StdRC']); bg=np.array(all_b['GRC'])
vb,vr,vg = ~np.isnan(bb),~np.isnan(br),~np.isnan(bg)
if vb.sum()>=3 and vr.sum()>=3 and vg.sum()>=3:
    bay=abs(np.mean(bb[vb])-TRUE_BETA)/abs(TRUE_BETA)*100
    rc=abs(np.mean(br[vr])-TRUE_BETA)/abs(TRUE_BETA)*100
    grc=abs(np.mean(bg[vg])-TRUE_BETA)/abs(TRUE_BETA)*100
    print(f"\n  Bayesian vs RC:  {rc-bay:+.1f}pp")
    print(f"  Bayesian vs GRC: {grc-bay:+.1f}pp")
    if rc-bay>=3 and grc-bay>=3: print("\n  >>> BAYESIAN WINS <<<")
    elif rc-bay>=0 and grc-bay>=0: print("\n  >>> BAYESIAN MARGINALLY BETTER <<<")
    else: print("\n  >>> BAYESIAN DOES NOT WIN <<<")
