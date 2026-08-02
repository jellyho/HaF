"""Aggregate Exp 2d - auxiliary-weight (lambda) dose-response, 6 datasets x 5 seeds.

Regularizer signature = OOD benefit rises with lambda then saturates/reverses (inverted-U), rather than
growing monotonically. Per dataset and pooled: r2_ood vs lambda for retro-obs (single) and mix (retro+fwd),
mean +/- std over seeds. Reports each curve, its argmax lambda, and whether it is non-monotone.
"""
import os, glob, json
import numpy as np

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DATASETS = ["fractal", "droid", "libero_goal", "libero_object", "libero_spatial", "libero_10"]
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]


def load(ds):
    seeds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"exp2d_{ds}_s*.json")))]
    return seeds or None


agg = {}
for ds in DATASETS:
    seeds = load(ds)
    if not seeds:
        continue
    base = np.mean([s["BC-only"]["r2_ood"] for s in seeds])
    agg[ds] = {"n_seeds": len(seeds), "bc_only": float(base), "retro": {}, "mix": {}}
    for fam, pfx in [("retro", "retro"), ("mix", "mix")]:
        for lam in LAMBDAS:
            key = f"{pfx}@{lam}"
            vals = [s[key]["r2_ood"] for s in seeds if key in s]
            if vals:
                agg[ds][fam][lam] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                                     "gain": float(np.mean(vals) - base)}

print("==== Exp 2d: OOD-R2 gain over BC-only vs aux-weight lambda ====")
for ds in agg:
    a = agg[ds]
    print(f"\n{ds}  (BC-only R2={a['bc_only']:+.3f}, seeds={a['n_seeds']})")
    for fam in ["retro", "mix"]:
        gains = [a[fam][l]["gain"] for l in LAMBDAS if l in a[fam]]
        stds = [a[fam][l]["std"] for l in LAMBDAS if l in a[fam]]
        best_l = LAMBDAS[int(np.argmax(gains))]
        nonmono = gains[-1] < max(gains) - 0.01  # falls off after the peak
        cells = "  ".join(f"λ{l}:{g:+.3f}±{s:.2f}" for l, g, s in zip(LAMBDAS, gains, stds))
        print(f"  {fam:5s} {cells}   argmax=λ{best_l} {'[inverted-U]' if nonmono else '[monotone]'}")

# pooled mean curve across datasets
print("\n==== pooled mean gain across datasets ====")
pooled = {}
for fam in ["retro", "mix"]:
    row = []
    for lam in LAMBDAS:
        gs = [agg[ds][fam][lam]["gain"] for ds in agg if lam in agg[ds][fam]]
        row.append(float(np.mean(gs)))
    pooled[fam] = row
    best_l = LAMBDAS[int(np.argmax(row))]
    print(f"  {fam:5s} " + "  ".join(f"λ{l}:{g:+.3f}" for l, g in zip(LAMBDAS, row)) + f"   argmax=λ{best_l}")

json.dump({"per_dataset": agg, "pooled": pooled, "lambdas": LAMBDAS},
          open(os.path.join(OUT, "exp2d_analysis.json"), "w"), indent=2)
print("\nSAVED exp2d_analysis.json")
