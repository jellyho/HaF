"""Bootstrap 95% CIs + permutation p-values for the key AHA correlations (addresses the 'no CIs' gap).
Small N (12-18 points) -> CIs may be wide; report honestly."""
import json, numpy as np
from scipy.stats import pearsonr, spearmanr
O="/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
rng=np.random.default_rng(0)

def boot(x,y,fn,B=10000):
    x=np.asarray(x);y=np.asarray(y);n=len(x)
    obs=fn(x,y)[0]
    bs=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if np.std(x[i])<1e-9 or np.std(y[i])<1e-9: continue
        bs.append(fn(x[i],y[i])[0])
    lo,hi=np.percentile(bs,[2.5,97.5])
    # permutation p (two-sided): shuffle y
    null=[fn(x, y[rng.permutation(n)])[0] for _ in range(B)]
    p=(np.sum(np.abs(null)>=abs(obs))+1)/(B+1)
    return obs,lo,hi,p,n

def load_measures(tag):
    S=[json.load(open(f"{O}/exp2h_{tag}_s{s}.json")) for s in range(3)]
    names=list(S[0]["objectives"])
    def agg(n,k):
        v=[s["objectives"][n][k] for s in S if s["objectives"][n].get(k) is not None]
        return float(np.mean(v)) if v else None
    gen=[agg(n,"generalization") for n in names]
    return names,agg,gen

print("="*70)
print("MEASURE-COMPARISON: correlation(measure, generalization) with 95% CI + perm-p")
print("="*70)
for tag,lab in [("frac4k","4k"),("frac8k","8k"),("fractal","16k")]:
    try: names,agg,gen=load_measures(tag)
    except: continue
    print(f"\n-- {lab} (N objectives, per-measure) --")
    for key,nm in [("R_linear","linear(frozen)"),("R_deep","policy e2e"),("R_deep_early","policy speed@¼")]:
        xy=[(agg(n,key),gen[i]) for i,n in enumerate(names) if agg(n,key) is not None and gen[i] is not None]
        x=[a for a,_ in xy]; y=[b for _,b in xy]
        r,lo,hi,p,n=boot(x,y,pearsonr)
        print(f"   {nm:16s} n={n}  Pearson {r:+.2f}  95%CI[{lo:+.2f},{hi:+.2f}]  perm-p={p:.3f}")

print("\n"+"="*70)
print("MIXING composition law: correlation(summary, mix-gen) with CI + perm-p (4k)")
print("="*70)
S=[json.load(open(f"{O}/exp2h_mixki_mk_s{s}.json")) for s in range(3)]
nmix=len(S[0]["mixes"])
def mg(i,k): return float(np.mean([s["mixes"][i][k] for s in S]))
gens=[mg(i,"gen") for i in range(nmix)]
for key,nm in [("min_recov","min"),("mean_recov","mean"),("joint_recov","joint")]:
    xs=[mg(i,key) for i in range(nmix)]
    for fn,fnm in [(pearsonr,"Pearson"),(spearmanr,"Spearman")]:
        r,lo,hi,p,n=boot(xs,gens,fn)
        print(f"   {nm:5s} {fnm:8s} n={n}  {r:+.2f}  95%CI[{lo:+.2f},{hi:+.2f}]  perm-p={p:.3f}")
