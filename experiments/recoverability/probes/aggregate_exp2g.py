"""Aggregate Exp 2g (representation quality) — closes Objective -> Recoverability -> Rep -> Generalization.

(1) per-condition rep-quality table (instr_decod, retrieval, silhouette), seed-mean.
(2) cross-condition CKA: how much each auxiliary MOVES the representation away from BC-only.
(3) paired retro-vs-fwd: does the LOW-recoverability aux (retro) yield more instruction-decodable reps than
    the HIGH-recoverability aux (fwd)?  (sign test across datasets)
(4) does rep-quality predict downstream generalization?  instr_decod  vs  OOD action R2 (from exp2b).
"""
import os, glob, json
import numpy as np
from scipy import stats

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DATASETS = ["fractal", "droid", "libero_goal", "libero_object", "libero_spatial", "libero_10"]
CONDS = ["BC-only", "BC+retro", "BC+fwd", "SG+retro"]
METRICS = ["instr_decod", "task_retrieval", "silhouette"]


def seed_mean(ds):
    files = sorted(glob.glob(os.path.join(OUT, f"exp2g_{ds}_s*.json")))
    seeds = [json.load(open(f)) for f in files]
    if not seeds:
        return None
    return {c: {m: float(np.mean([s[c][m] for s in seeds if c in s and m in s[c] and s[c][m] == s[c][m]]))
                for m in METRICS} for c in CONDS if c in seeds[0]}


def linear_cka(X, Y):
    X = (X - X.mean(0)).astype(np.float32); Y = (Y - Y.mean(0)).astype(np.float32)
    xy = np.linalg.norm(Y.T @ X) ** 2
    xx = np.linalg.norm(X.T @ X); yy = np.linalg.norm(Y.T @ Y)
    return float(xy / (xx * yy + 1e-12))


def cka_vs_bc(ds):
    files = sorted(glob.glob(os.path.join(OUT, f"cache/exp2g_reps_{ds}_s*.npz")))
    out = {c: [] for c in CONDS if c != "BC-only"}
    for f in files:
        z = np.load(f)
        if "R_BC-only" not in z:
            continue
        base = z["R_BC-only"].astype(np.float32)
        for c in out:
            key = f"R_{c}"
            if key in z:
                out[c].append(linear_cka(base, z[key].astype(np.float32)))
    return {c: float(np.mean(v)) for c, v in out.items() if v}


agg = {ds: seed_mean(ds) for ds in DATASETS}
agg = {k: v for k, v in agg.items() if v}

print("==== Exp 2g: representation quality (seed-mean) ====")
for ds in agg:
    print(f"\n{ds}")
    print(f"  {'cond':10s} {'instr_decod':>12s} {'retrieval@10':>13s} {'silhouette':>11s}")
    for c in CONDS:
        if c in agg[ds]:
            a = agg[ds][c]
            print(f"  {c:10s} {a['instr_decod']:+12.3f} {a['task_retrieval']:13.3f} {a['silhouette']:+11.3f}")

print("\n==== CKA vs BC-only (lower = aux moved the representation more) ====")
cka = {ds: cka_vs_bc(ds) for ds in agg}
for ds in agg:
    if cka[ds]:
        print(f"  {ds:15s} " + "  ".join(f"{c}={v:.3f}" for c, v in cka[ds].items()))

# (3) paired retro vs fwd on instruction decodability (low- vs high-recoverability aux)
print("\n==== retro vs fwd — instruction decodability (low- vs high-recoverability aux) ====")
retro_v, fwd_v = [], []
for ds in agg:
    if "BC+retro" in agg[ds] and "BC+fwd" in agg[ds]:
        rv, fv = agg[ds]["BC+retro"]["instr_decod"], agg[ds]["BC+fwd"]["instr_decod"]
        retro_v.append(rv); fwd_v.append(fv)
        print(f"  {ds:15s} retro={rv:+.3f}  fwd={fv:+.3f}  retro-fwd={rv-fv:+.3f}")
if retro_v:
    diff = np.array(retro_v) - np.array(fwd_v)
    wins = int((diff > 0).sum())
    print(f"  MEAN retro={np.mean(retro_v):+.3f} fwd={np.mean(fwd_v):+.3f}  retro>fwd in {wins}/{len(diff)} datasets"
          f"  (mean diff {diff.mean():+.3f})")

# (4) does rep-quality predict generalization? instr_decod vs OOD action R2 (from exp2b)
print("\n==== rep-quality -> generalization: instr_decod vs OOD action R2 (exp2b) ====")
try:
    e2b = json.load(open(os.path.join(OUT, "exp2b_trimodal_agg.json")))
    xs, ys = [], []
    name_map = {"BC-only": "BC-only", "BC+retro": "BC+retro", "BC+fwd": "BC+fwd", "SG+retro": "SG-BC+retro"}
    for ds in agg:
        if ds not in e2b:
            continue
        for c in CONDS:
            e2bc = name_map[c]
            if c in agg[ds] and e2bc in e2b[ds]["cond"]:
                xs.append(agg[ds][c]["instr_decod"])
                ys.append(e2b[ds]["cond"][e2bc]["r2_ood"]["mean"])
    if len(xs) >= 3:
        r, p = stats.pearsonr(xs, ys)
        rs, ps = stats.spearmanr(xs, ys)
        print(f"  n={len(xs)}  pearson r={r:+.3f} (p={p:.3f})  spearman r={rs:+.3f} (p={ps:.3f})")
        print("  (positive => more instruction-decodable representation predicts better OOD generalization)")
except FileNotFoundError:
    print("  exp2b_trimodal_agg.json not found — run aggregate_trimodal.py first")

json.dump({"repquality": agg, "cka_vs_bc": cka},
          open(os.path.join(OUT, "exp2g_analysis.json"), "w"), indent=2)
print("\nSAVED exp2g_analysis.json")
