"""
PRODUCTION RUN v4 — Divergence-free model, 200 reps, S1+S6+S8
Theory-based variance center, only tau estimated.
~3s/rep, total ~30 min for all 3 scenarios.
"""
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from lifelines import CoxPHFitter
import cmdstanpy
import time, sys, os, warnings
warnings.filterwarnings('ignore')

TRUE_BETA = -1.0; GAMMA = 0.5; N = 50; SPOT = 100.0
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUT = '/Users/ganeshshiwakoti/Desktop/Biostatistics/bayesian_mecox/v4_production'
os.makedirs(OUT, exist_ok=True)

def expit(x): return 1.0/(1.0+np.exp(-x))

SCENARIOS = {
    'S1': {'mode':'exch','rho':(0.02,0.10),'phi':None,'delta':0.0,'label':'Baseline'},
    'S6': {'mode':'spatial','rho':None,'phi':5,'delta':0.0,'label':'Medium-range decay'},
    'S8': {'mode':'exch','rho':(0.02,0.10),'phi':None,'delta':0.2,'label':'Differential error'},
}

def make_grid(mt):
    side=int(np.ceil(np.sqrt(mt*1.15))); c=[]
    for r in range(side):
        for col in range(side):
            c.append([col*SPOT+(SPOT/2 if r%2 else 0), r*SPOT])
    c=np.array(c)
    if len(c)>mt: c=c[np.random.choice(len(c),mt,replace=False)]
    return c

def spatial_field(coords, Xi, phi, rng):
    m=len(coords); D=squareform(pdist(coords))
    C=np.exp(-D/phi)+np.eye(m)*1e-6
    try: L=cholesky(C,lower=True)
    except:
        ev,evec=np.linalg.eigh(C); L=cholesky(evec@np.diag(np.maximum(ev,1e-6))@evec.T,lower=True)
    return (L@rng.standard_normal(m)>stats.norm.ppf(1-Xi)).astype(int), D

def exact_deff(D, phi):
    m=D.shape[0]
    if m<2: return 1.0
    return 1+(m-1)*np.exp(-D/phi)[np.triu_indices(m,k=1)].mean()

def gen_data(cfg, seed):
    rng=np.random.default_rng(seed)
    eta=rng.normal(special.logit(0.35),0.8,N); X=expit(eta)
    m=np.clip(np.round(np.exp(rng.normal(np.log(2200),0.45,N))),200,4000).astype(int)
    Z=rng.standard_normal(N); W=np.zeros(N); deff=np.zeros(N); rho_est=np.zeros(N)
    if cfg['mode']=='exch':
        rho=rng.uniform(*cfg['rho'],N)
        for i in range(N):
            deff[i]=1+(m[i]-1)*rho[i]; rho_est[i]=rho[i]
            phi_bb=(1-rho[i])/rho[i] if rho[i]>0.001 else 1e6
            a,b=max(X[i]*phi_bb,0.01),max((1-X[i])*phi_bb,0.01)
            p=np.clip(rng.beta(a,b),1e-6,1-1e-6) if rho[i]>0.001 else X[i]
            W[i]=rng.binomial(m[i],p)/m[i]
    else:
        ms=np.minimum(m,600); phi_val=cfg['phi']*SPOT
        for i in range(N):
            coords=make_grid(ms[i]); Y,D=spatial_field(coords,X[i],phi_val,rng)
            W[i]=Y.mean(); deff[i]=exact_deff(D,phi_val)
            rho_est[i]=max((deff[i]-1)/(m[i]-1),0.001)
    if cfg['delta']>0: W=np.clip(W+cfg['delta']*(X-X.mean()),0,1)
    rate=0.1*np.exp(TRUE_BETA*X+GAMMA*Z)
    T_ev=rng.exponential(1.0/rate); lC=0.4*rate.mean()/0.6; C=rng.exponential(1.0/lC,N)
    # Theory-based log(sigma_i^2)
    log_sig2_theory = np.log(np.clip(W*(1-W)*deff/m, 1e-10, None))
    return {'X':X,'W':W,'Z':Z.reshape(-1,1),'T':np.minimum(T_ev,C),'delta':(T_ev<=C).astype(int),
            'm':m,'rho':rho_est,'deff':deff,'log_sig2_theory':log_sig2_theory}

def cox_fit(T,d,covs,names):
    df=pd.DataFrame(covs,columns=names); df['T']=T; df['d']=d
    try:
        cph=CoxPHFitter(); cph.fit(df,'T','d',show_progress=False)
        b=cph.params_[names[0]]; se=cph.standard_errors_[names[0]]
        ci=cph.confidence_intervals_.loc[names[0]]; p=cph.summary['p'].loc[names[0]]
        return b,se,float(ci.iloc[0]),float(ci.iloc[1]),p
    except: return (np.nan,)*5

def freq_methods(data):
    W=data['W']; s2=np.clip(W*(1-W)*data['deff']/data['m'],1e-8,None)
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
    et=data['T'][data['delta']==1]
    if len(et)<K: ci=np.quantile(data['T'],np.linspace(0,1,K+1)[1:-1])
    else: ci=np.quantile(et,np.linspace(0,1,K+1)[1:-1])
    sd={'N':N,'P':data['Z'].shape[1],'K':K,'W':data['W'].tolist(),'Z':data['Z'].tolist(),
        'T_obs':data['T'].tolist(),'event':data['delta'].tolist(),
        'log_sig2_theory':data['log_sig2_theory'].tolist(),'s':ci.tolist()}
    fit=model.sample(data=sd,chains=4,iter_sampling=1000,iter_warmup=1000,
                     adapt_delta=0.99,max_treedepth=12,show_progress=False,show_console=False)
    bd=fit.stan_variable('beta')
    b=np.mean(bd); se=np.std(bd); c=np.percentile(bd,[2.5,97.5])
    p=2*min(np.mean(bd>0),np.mean(bd<0))
    ndiv=int(np.sum(fit.method_variables()['divergent__']))
    rhat_max=float(fit.summary()['R_hat'].max())
    return b,se,c[0],c[1],p,ndiv,rhat_max

# MAIN
print("="*60)
print(f"v4 PRODUCTION — {N_REPS} reps x {len(SCENARIOS)} scenarios")
print("="*60)

model=cmdstanpy.CmdStanModel(exe_file='model_v4')
print("v4 loaded.\n")

for sname,cfg in SCENARIOS.items():
    print(f"\n{'='*55}")
    print(f"  {sname}: {cfg['label']} ({N_REPS} reps)")
    print(f"{'='*55}")

    rows=[]; t0=time.time(); total_div=0; total_bad=0

    for rep in range(N_REPS):
        seed=rep*1000+hash(sname)%10000
        data=gen_data(cfg,seed)
        fr=freq_methods(data)
        for mt in ['Oracle','Naive','StdRC','GRC']:
            b,se,cl,ch,p=fr[mt]
            rows.append({'rep':rep,'method':mt,'beta':b,'ci_lo':cl,'ci_hi':ch,'p':p,'div':0})
        try:
            b,se,cl,ch,p,ndiv,rhat=bayes_fit(data,model)
            rows.append({'rep':rep,'method':'Bayesian','beta':b,'ci_lo':cl,'ci_hi':ch,'p':p,'div':ndiv})
            total_div+=ndiv
            if rhat>1.05: total_bad+=1
        except:
            rows.append({'rep':rep,'method':'Bayesian','beta':np.nan,'ci_lo':np.nan,'ci_hi':np.nan,'p':np.nan,'div':-1})

        if (rep+1)%50==0:
            el=time.time()-t0
            print(f"    {rep+1}/{N_REPS} | {el:.0f}s | {el/(rep+1):.1f}s/rep | divs={total_div} | bad_rhat={total_bad}")
            sys.stdout.flush()

    el=time.time()-t0
    print(f"    Done: {el:.0f}s ({el/N_REPS:.1f}s/rep)")

    df=pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{sname}_replicates.csv',index=False)

    print(f"\n  {'Method':<12} {'Mean':>8} {'Bias':>8} {'Rel%':>7} {'Cov%':>7} {'n':>5}")
    print(f"  {'-'*48}")
    for mt in ['Oracle','Naive','StdRC','GRC','Bayesian']:
        sub=df[df['method']==mt]; betas=sub['beta'].dropna(); nv=len(betas)
        if nv<10: print(f"  {mt:<12} {'N/A':>40} {nv:>5}"); continue
        bias=betas.mean()-TRUE_BETA; rel=bias/abs(TRUE_BETA)*100
        ci_lo=sub['ci_lo'].dropna(); ci_hi=sub['ci_hi'].dropna()
        common=betas.index.intersection(ci_lo.index).intersection(ci_hi.index)
        cov=((ci_lo.loc[common]<=TRUE_BETA)&(TRUE_BETA<=ci_hi.loc[common])).mean()*100
        print(f"  {mt:<12} {betas.mean():>8.3f} {bias:>+8.3f} {rel:>+6.1f}% {cov:>6.1f}% {nv:>5}")
    print(f"  Divergences: total={total_div}, bad_rhat={total_bad}")

print(f"\n\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}")
for sname in SCENARIOS:
    df=pd.read_csv(f'{OUT}/{sname}_replicates.csv')
    print(f"\n  {sname} ({SCENARIOS[sname]['label']}):")
    for mt in ['Naive','StdRC','GRC','Bayesian']:
        betas=df[df['method']==mt]['beta'].dropna()
        if len(betas)<10: continue
        bias=abs(betas.mean()-TRUE_BETA)/abs(TRUE_BETA)*100
        ci_lo=df[df['method']==mt]['ci_lo'].dropna(); ci_hi=df[df['method']==mt]['ci_hi'].dropna()
        common=betas.index.intersection(ci_lo.index).intersection(ci_hi.index)
        cov=((ci_lo.loc[common]<=TRUE_BETA)&(TRUE_BETA<=ci_hi.loc[common])).mean()*100
        print(f"    {mt:<12} |Bias|={bias:>5.1f}%  Cov={cov:>5.1f}%")

print(f"\nAll saved to {OUT}/")
