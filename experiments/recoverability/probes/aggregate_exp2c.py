"""Aggregate Exp 2c (aux-target battery) across 6 datasets x 2 seeds and run the two correlation analyses.

Axis A -- aux-target property -> regularization benefit:
    x = aux target's Exp-1 shortcut-freeness (R_triv, and probe_beyond_trivial) in that dataset
    y = benefit = metric(BC+aux) - metric(BC-only)   [ΔOOD R2, Δcontrib_lang]
    pooled over (aux, dataset); Pearson + Spearman.  HaF prediction: lower R_triv (more shortcut-free)
    and higher probe_beyond_trivial (hard-but-learnable) => larger benefit.

Axis B -- dataset property -> mean aux benefit:
    x = vision-legitimacy (BC-only contrib_vision), instruction-from-scene recoverability (Exp-1 R3
        probe_beyond_trivial), #unique instructions
    y = mean ΔOOD R2 across aux targets;  n = #datasets.
"""
import os, glob, json
import numpy as np

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DATASETS = ["fractal", "droid", "libero_goal", "libero_object", "libero_spatial", "libero_10"]
AUX2EXP1 = {  # aux target -> Exp-1 result key
    "retro-obs": "R1 initial-obs", "farpast-obs": "Mfar past-obs k~45",
    "fwd-obs": "P1s future-obs k~5", "farfwd-obs": "P1l future-obs k~45",
    "progress": "I-progress t/T", "pose0": "R2 initial-pose",
    # dynamics = predict obs CHANGE: shortcut-free by construction, no direct Exp-1 row
}
AUX_ORDER = ["retro-obs", "farpast-obs", "dynamics", "fwd-obs", "farfwd-obs", "progress", "pose0"]
METRICS = ["r2_ood", "contrib_lang", "contrib_vision", "lang_sensitivity", "vision_sensitivity"]


def load_seed_mean(ds):
    files = sorted(glob.glob(os.path.join(OUT, f"exp2c_{ds}_s*.json")))
    seeds = [json.load(open(f)) for f in files]
    if not seeds:
        return None
    conds = seeds[0].keys()
    out = {}
    for c in conds:
        out[c] = {m: float(np.mean([s[c][m] for s in seeds if c in s and m in s[c]])) for m in METRICS}
    return out


from scipy import stats


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    r = lambda v: np.argsort(np.argsort(v))
    return pearson(r(x), r(y))


def pear_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return (float("nan"), float("nan"))
    r, p = stats.pearsonr(x, y)
    return (float(r), float(p))


def spear_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return (float("nan"), float("nan"))
    r, p = stats.spearmanr(x, y)
    return (float(r), float(p))


agg = {ds: load_seed_mean(ds) for ds in DATASETS}
agg = {k: v for k, v in agg.items() if v}
exp1 = {ds: json.load(open(os.path.join(OUT, f"results_{ds}.json"))) for ds in agg
        if os.path.exists(os.path.join(OUT, f"results_{ds}.json"))}

# ---------- benefit table (Δ vs BC-only), per dataset per aux ----------
benefit = {}  # ds -> aux -> {metric: Δ}
for ds, a in agg.items():
    base = a["BC-only"]
    benefit[ds] = {}
    for aux in AUX_ORDER:
        c = f"BC+{aux}"
        if c in a:
            benefit[ds][aux] = {m: a[c][m] - base[m] for m in METRICS}

print("==== Δ OOD R2 vs BC-only (rows=aux target, cols=dataset) ====")
hdr = "aux\\ds".ljust(13) + "".join(ds[:9].rjust(11) for ds in agg)
print(hdr)
for aux in AUX_ORDER:
    row = aux.ljust(13)
    for ds in agg:
        v = benefit[ds].get(aux, {}).get("r2_ood", float("nan"))
        row += (f"{v:+.3f}").rjust(11)
    # mean benefit across datasets
    vals = [benefit[ds][aux]["r2_ood"] for ds in agg if aux in benefit[ds]]
    row += f"   mean={np.mean(vals):+.3f}"
    print(row)

# ---------- Axis A: aux shortcut-freeness -> benefit (pooled over aux,ds) ----------
print("\n==== Axis A: aux-target property -> regularization benefit (pooled aux x ds) ====")
xR, xP, yR2, yCL = [], [], [], []
for ds in agg:
    if ds not in exp1:
        continue
    for aux, key in AUX2EXP1.items():
        if aux in benefit[ds] and key in exp1[ds]:
            xR.append(exp1[ds][key]["R_triv"])
            xP.append(exp1[ds][key]["probe_beyond_trivial"])
            yR2.append(benefit[ds][aux]["r2_ood"])
            yCL.append(benefit[ds][aux]["contrib_lang"])
def fp(x, y, tag):
    rp, pp = pear_p(x, y); rs, ps = spear_p(x, y)
    print(f"  {tag:30s}: pearson r={rp:+.3f} (p={pp:.3f})  spearman r={rs:+.3f} (p={ps:.3f})")
    return rp, pp, rs, ps
print(f"  n points = {len(xR)}")
axisA_stats = {
    "R_triv_vs_dR2": fp(xR, yR2, "R_triv vs ΔOOD_R2 (HaF NEG)"),
    "probe_vs_dR2": fp(xP, yR2, "probe>triv vs ΔOOD_R2 (HaF POS)"),
    "R_triv_vs_dCL": fp(xR, yCL, "R_triv vs ΔcontribL (HaF NEG)"),
    "probe_vs_dCL": fp(xP, yCL, "probe>triv vs ΔcontribL (HaF POS)"),
}

# ---------- Axis B: dataset property -> mean aux benefit ----------
print("\n==== Axis B: dataset property -> mean aux ΔOOD_R2 (n=datasets) ====")
xVis, xInstr, xUniq, yMean = [], [], [], []
dslist = []
for ds in agg:
    mean_ben = np.mean([benefit[ds][aux]["r2_ood"] for aux in AUX_ORDER if aux in benefit[ds]])
    vis_legit = agg[ds]["BC-only"]["contrib_vision"]
    instr_rec = exp1[ds]["R3 instruction"]["probe_beyond_trivial"] if ds in exp1 else float("nan")
    d = np.load(os.path.join(OUT, f"cache/transitions_{ds}.npz"), allow_pickle=True)
    uniq = len(set(str(s) for s in d["instr"]))
    xVis.append(vis_legit); xInstr.append(instr_rec); xUniq.append(uniq); yMean.append(mean_ben)
    dslist.append(ds)
    print(f"  {ds:15s} meanΔR2={mean_ben:+.3f}  vision_legit(cV_BC)={vis_legit:+.3f}  "
          f"instr_recover={instr_rec:+.3f}  uniq_instr={uniq}")
rvis, pvis = pear_p(xVis, yMean); rins, pins = pear_p(xInstr, yMean)
print(f"  vision_legitimacy    vs meanΔR2 : pearson r={rvis:+.3f} (p={pvis:.3f})  (expect NEG)")
print(f"  instr_recoverability vs meanΔR2 : pearson r={rins:+.3f} (p={pins:.3f})  (confounded w/ vision_legit)")

out = {"benefit": benefit, "seed_mean": agg,
       "axisA": {"n": len(xR), "R_triv_vs_dR2_pearson": pearson(xR, yR2),
                 "probe_vs_dR2_pearson": pearson(xP, yR2),
                 "R_triv_vs_dR2_spearman": spearman(xR, yR2),
                 "probe_vs_dR2_spearman": spearman(xP, yR2)},
       "axisB": {"datasets": dslist, "vision_legit": xVis, "instr_recover": xInstr,
                 "uniq_instr": xUniq, "mean_benefit": yMean,
                 "vislegit_vs_benefit_pearson": pearson(xVis, yMean),
                 "instrrec_vs_benefit_pearson": pearson(xInstr, yMean)}}
json.dump(out, open(os.path.join(OUT, "exp2c_analysis.json"), "w"), indent=2)
print("\nSAVED exp2c_analysis.json")
