"""Which recoverability MEASUREMENT method best predicts BC generalization?

For each objective we computed several recoverability estimators (different function classes 𝒱, and
asymptotic vs learning-dynamics). The GOOD measure is the one whose recoverability most strongly &
correct-signly predicts OOD generalization (recov ↑ ⇒ generalization ↓, i.e. negative correlation).
Loads exp2h_{TAG}_s*.json, ranks measures by corr with generalization, writes fig + markdown.
"""
import os, json, sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "fractal"
SEEDS = [0, 1, 2]
MEASURES = [
    ("R_linear",     "linear probe (frozen z_t)",        "asymptotic", "linear"),
    ("R_mlp",        "MLP probe (frozen z_t)",           "asymptotic", "mlp"),
    ("R_mdl",        "prequential MDL (frozen, MLP)",    "dynamics",   "mlp"),
    ("R_deep",       "policy-class end-to-end",          "asymptotic", "policy"),
    ("R_deep_aulc",  "policy-class AULC (learning curve)","dynamics",   "policy"),
    ("R_deep_early", "policy-class @¼ budget (speed)",   "dynamics",   "policy"),
]

S = []
for s in SEEDS:
    p = f"{OUT}/exp2h_{TAG}_s{s}.json"
    if os.path.exists(p):
        S.append(json.load(open(p)))
assert S, "no result jsons found"
names = list(S[0]["objectives"])


def agg(name, field):
    vs = [s["objectives"][name][field] for s in S
          if name in s["objectives"] and s["objectives"][name].get(field) is not None]
    return float(np.mean(vs)) if vs else None


gen = {n: agg(n, "generalization") for n in names}
print(f"=== measure-comparison  TAG={TAG}  seeds={len(S)}  objectives={len(names)} ===\n")
print(f"{'measure':34s} {'class':7s} {'kind':10s} {'n':>3s} {'Pearson':>9s} {'Spearman':>9s} {'sign':>5s}")
results = []
for key, label, kind, cls in MEASURES:
    xs, ys = [], []
    for n in names:
        rv = agg(n, key); gv = gen[n]
        if rv is not None and gv is not None:
            xs.append(rv); ys.append(gv)
    if len(xs) < 4:
        print(f"{label:34s} {cls:7s} {kind:10s} {len(xs):3d}   (too few)"); continue
    xs, ys = np.array(xs), np.array(ys)
    pr, pp = pearsonr(xs, ys); sr, sp = spearmanr(xs, ys)
    ok = "OK" if pr < 0 else "WRONG"
    print(f"{label:34s} {cls:7s} {kind:10s} {len(xs):3d} {pr:+9.3f} {sr:+9.3f} {ok:>5s}")
    results.append(dict(key=key, label=label, cls=cls, kind=kind, n=len(xs),
                        pearson=pr, pp=pp, spearman=sr, sp=sp))

# selectivity control: R_linear vs its shuffled-target control (should be ~0)
ctl = [agg(n, "R_linear_control") for n in names if agg(n, "R_linear_control") is not None]
lin = [agg(n, "R_linear") for n in names if agg(n, "R_linear") is not None]
if ctl:
    print(f"\nselectivity: mean R_linear={np.mean(lin):+.3f}  vs  shuffled-control={np.mean(ctl):+.3f}"
          f"  (control≈0 ⇒ measure reflects genuine extractability)")

# winner
neg = [r for r in results if r["pearson"] < 0]
if neg:
    best = min(neg, key=lambda r: r["pearson"])
    print(f"\n>>> BEST predictor of generalization: {best['label']}  "
          f"(Pearson {best['pearson']:+.3f}, Spearman {best['spearman']:+.3f})")
json.dump(dict(tag=TAG, seeds=len(S), n_obj=len(names), results=results,
               selectivity=dict(R_linear=float(np.mean(lin)) if lin else None,
                                control=float(np.mean(ctl)) if ctl else None)),
          open(f"{OUT}/measure_comparison_{TAG}.json", "w"), indent=2)
print(f"\nSAVED measure_comparison_{TAG}.json")
