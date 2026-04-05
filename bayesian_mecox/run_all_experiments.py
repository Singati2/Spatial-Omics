"""
FULL EXPERIMENT: Bayesian ME-Cox vs Oracle/Naive/RC/GRC
=========================================================
Scenarios: S1 (baseline), S5 (short decay), S6 (medium decay), S8 (differential)
200 reps each. Saves replicate-level data.
"""
import numpy as np
import pandas as pd
from scipy import stats, special
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import cholesky
from sklearn.neighbors import kneighbors_graph
from lifelines import CoxPHFitter
import cmdstanpy
import time, sys, os, json, warnings
warnings.filterwarnings('ignore')

TRUE_BETA = -1.0; GAMMA = 0.5; N = 50; SPOT = 100.0
N_REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
OUT = '/Users/ganeshshiwakoti/Desktop/Biostatistics/bayesian_mecox/experiment_results'
os.makedirs(OUT, exist_ok=True)

def expit(x): return 1.0/(1.0+np.exp(-x))

SCENARIOS = {
    'S1': {'mode':'exch','rho':(0.02,0.10),'phi':None,'delta':0.0,'label':'Baseline'},
    'S5': {'mode':'spatial','rho':None,'phi':2,'delta':0.0,'label':'Short-range decay'},
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
    Z=L@rng.standard_normal(m)
    return (Z>stats.norm.ppf(1-Xi)).astype(int), D

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
    return {'X':X,'W':W,'Z':Z.reshape(-1,1),'T':np.minimum(T_ev,C),'delta':(T_ev<=C).astype(int),
            'm':m,'rho':rho_est,'deff':deff}

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
        'm_spots':data['m'].astype(float).tolist(),'log_m':np.log(data['m'].astype(float)).tolist(),
        'rho':data['rho'].tolist(),'s':ci.tolist()}
    fit=model.sample(data=sd,chains=2,iter_sampling=500,iter_warmup=500,
                     adapt_delta=0.99,max_treedepth=14,show_progress=False,show_console=False)
    bd=fit.stan_variable('beta')
    b=np.mean(bd); se=np.std(bd); c=np.percentile(bd,[2.5,97.5])
    p=2*min(np.mean(bd>0),np.mean(bd<0))
    ndiv=np.sum(fit.method_variables()['divergent__'])
    return b,se,c[0],c[1],p,int(ndiv)

# MAIN
print("="*60)
print(f"FULL BAYESIAN EXPERIMENT — {N_REPS} reps x {len(SCENARIOS)} scenarios")
print("="*60)

model=cmdstanpy.CmdStanModel(exe_file='/Users/ganeshshiwakoti/Desktop/Biostatistics/bayesian_mecox/model_v2')
print("Model v2 loaded.\n")

all_summaries = {}

for sname, cfg in SCENARIOS.items():
    print(f"\n{'='*55}")
    print(f"  {sname}: {cfg['label']} ({N_REPS} reps)")
    print(f"{'='*55}")

    methods=['Oracle','Naive','StdRC','GRC','Bayesian']
    rows=[]; t0=time.time(); n_div_total=0

    for rep in range(N_REPS):
        seed=rep*1000+hash(sname)%10000
        data=gen_data(cfg, seed)
        fr=freq_methods(data)

        for mt in ['Oracle','Naive','StdRC','GRC']:
            b,se,cl,ch,p=fr[mt]
            rows.append({'rep':rep,'method':mt,'beta':b,'se':se,'ci_lo':cl,'ci_hi':ch,'p':p,'div':0})

        try:
            b,se,cl,ch,p,ndiv=bayes_fit(data,model)
            rows.append({'rep':rep,'method':'Bayesian','beta':b,'se':se,'ci_lo':cl,'ci_hi':ch,'p':p,'div':ndiv})
            n_div_total+=ndiv
        except Exception as e:
            rows.append({'rep':rep,'method':'Bayesian','beta':np.nan,'se':np.nan,'ci_lo':np.nan,'ci_hi':np.nan,'p':np.nan,'div':-1})

        if (rep+1)%20==0:
            el=time.time()-t0
            print(f"    {rep+1}/{N_REPS} | {el:.0f}s | {el/(rep+1):.1f}s/rep | divs={n_div_total}")
            sys.stdout.flush()

    el=time.time()-t0
    print(f"    Done: {el:.0f}s ({el/N_REPS:.1f}s/rep)")

    df=pd.DataFrame(rows)
    df.to_csv(f'{OUT}/{sname}_replicates.csv',index=False)

    # Summary
    print(f"\n  {'Method':<12} {'Mean':>8} {'Bias':>8} {'Rel%':>7} {'Cov%':>7} {'n':>5}")
    print(f"  {'-'*48}")
    summary_rows=[]
    for mt in methods:
        sub=df[df['method']==mt]; betas=sub['beta'].dropna()
        nv=len(betas)
        if nv<10:
            print(f"  {mt:<12} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>7} {nv:>5}")
            summary_rows.append({'Method':mt,'Bias':np.nan,'RelBias':np.nan,'Cov':np.nan,'n':nv})
            continue
        bias=betas.mean()-TRUE_BETA; rel=bias/abs(TRUE_BETA)*100
        ci_lo=sub['ci_lo'].dropna(); ci_hi=sub['ci_hi'].dropna()
        common=betas.index.intersection(ci_lo.index).intersection(ci_hi.index)
        cov=((ci_lo.loc[common]<=TRUE_BETA)&(TRUE_BETA<=ci_hi.loc[common])).mean()*100
        print(f"  {mt:<12} {betas.mean():>8.3f} {bias:>+8.3f} {rel:>+6.1f}% {cov:>6.1f}% {nv:>5}")
        summary_rows.append({'Method':mt,'Mean':betas.mean(),'Bias':bias,'RelBias':rel,'Cov':cov,'n':nv})

    bayes_divs=df[df['method']=='Bayesian']['div']
    print(f"  Bayesian divergences: mean={bayes_divs.mean():.1f}, max={bayes_divs.max()}")

    sdf=pd.DataFrame(summary_rows)
    sdf.to_csv(f'{OUT}/{sname}_summary.csv',index=False)
    all_summaries[sname]=sdf

# FINAL COMPARISON
print(f"\n\n{'='*60}")
print("FINAL COMPARISON: ALL SCENARIOS")
print(f"{'='*60}")
print(f"\n{'Scenario':<20} {'Naive%':>8} {'RC%':>6} {'GRC%':>7} {'Bayes%':>8} {'BayesCov':>9}")
print("-"*60)
for sname in ['S1','S5','S6','S8']:
    s=all_summaries[sname]
    def get_val(m,col):
        r=s[s['Method']==m]
        return r.iloc[0][col] if len(r)>0 and not pd.isna(r.iloc[0][col]) else np.nan
    nb=abs(get_val('Naive','RelBias')); rb=abs(get_val('StdRC','RelBias'))
    gb=abs(get_val('GRC','RelBias')); bb=abs(get_val('Bayesian','RelBias'))
    bc=get_val('Bayesian','Cov')
    print(f"  {SCENARIOS[sname]['label']:<18} {nb:>7.1f}% {rb:>5.1f}% {gb:>6.1f}% {bb:>7.1f}% {bc:>8.1f}%")

print(f"\n{'='*60}")
print("VERDICT")
print(f"{'='*60}")
for sname in ['S1','S6']:
    s=all_summaries[sname]
    bb=abs(s[s['Method']=='Bayesian'].iloc[0]['RelBias'])
    rb=abs(s[s['Method']=='StdRC'].iloc[0]['RelBias'])
    gb=abs(s[s['Method']=='GRC'].iloc[0]['RelBias'])
    bc=s[s['Method']=='Bayesian'].iloc[0]['Cov']
    beats_rc=rb-bb; beats_grc=gb-bb
    print(f"\n  {sname}: Bayesian bias={bb:.1f}%, RC={rb:.1f}%, GRC={gb:.1f}%")
    print(f"    vs RC: {beats_rc:+.1f}pp | vs GRC: {beats_grc:+.1f}pp | Coverage: {bc:.1f}%")
    if beats_rc>=3 and beats_grc>=3 and bc>=88:
        print(f"    >>> BAYESIAN WINS IN {sname} <<<")
    elif beats_rc>=0 and beats_grc>=0:
        print(f"    >>> BAYESIAN MARGINALLY BETTER IN {sname} <<<")
    else:
        print(f"    >>> BAYESIAN DOES NOT WIN IN {sname} <<<")

print(f"\nAll results saved to {OUT}/")
