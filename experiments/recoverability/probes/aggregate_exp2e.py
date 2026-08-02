"""Aggregate Exp 2e - retrospective battery (6 datasets x 5 seeds).

(1) within-retrospective ranking: mean ΔOOD-R2 over BC-only per retro target.
(2) FORM comparison: retro-obs (latent-MSE) vs semantic-retro (discrete-CE) on the same starting-situation
    target -- does the thesis-preferred discrete-semantic form regularize better?
"""
import os, glob, json
import numpy as np

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DATASETS = ["fractal", "droid", "libero_goal", "libero_object", "libero_spatial", "libero_10"]
RETRO = ["retro-obs", "near-past-obs", "farpast-obs", "retro-pose", "retro-action", "displacement", "semantic-retro"]
METRICS = ["r2_ood", "contrib_lang", "contrib_vision", "lang_sensitivity", "vision_sensitivity"]


def load(ds):
    seeds = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, f"exp2e_{ds}_s*.json")))]
    if not seeds:
        return None
    conds = seeds[0].keys()
    return {c: {m: (float(np.mean([s[c][m] for s in seeds if c in s and m in s[c]])),
                    float(np.std([s[c][m] for s in seeds if c in s and m in s[c]]))) for m in METRICS}
            for c in conds}, len(seeds)


agg, benefit = {}, {}
for ds in DATASETS:
    r = load(ds)
    if not r:
        continue
    agg[ds], nseed = r
    base = agg[ds]["BC-only"]["r2_ood"][0]
    benefit[ds] = {a: agg[ds][f"BC+{a}"]["r2_ood"][0] - base for a in RETRO if f"BC+{a}" in agg[ds]}

print("==== Exp 2e: retrospective battery -- Δ OOD R2 vs BC-only (rows=retro target) ====")
print("target".ljust(16) + "".join(ds[:9].rjust(11) for ds in agg) + "     mean")
for a in RETRO:
    row = a.ljust(16)
    vals = []
    for ds in agg:
        v = benefit[ds].get(a, float("nan")); vals.append(v)
        row += (f"{v:+.3f}").rjust(11)
    row += f"   {np.nanmean(vals):+.3f}"
    print(row)

print("\n==== FORM comparison: retro-obs (MSE) vs semantic-retro (CE), same starting-situation target ====")
mse_gains, ce_gains = [], []
for ds in agg:
    mg = benefit[ds].get("retro-obs", float("nan")); cg = benefit[ds].get("semantic-retro", float("nan"))
    mse_gains.append(mg); ce_gains.append(cg)
    print(f"  {ds:15s} retro-obs(MSE) ΔR2={mg:+.3f}   semantic-retro(CE) ΔR2={cg:+.3f}   CE−MSE={cg-mg:+.3f}")
print(f"  MEAN            retro-obs(MSE) ΔR2={np.nanmean(mse_gains):+.3f}   "
      f"semantic-retro(CE) ΔR2={np.nanmean(ce_gains):+.3f}   CE−MSE={np.nanmean(ce_gains)-np.nanmean(mse_gains):+.3f}")

# language-grounding by retro target (does it make BC read language?)
print("\n==== mean Δcontrib_lang by retro target (language grounding gained) ====")
for a in RETRO:
    vals = [agg[ds][f"BC+{a}"]["contrib_lang"][0] - agg[ds]["BC-only"]["contrib_lang"][0]
            for ds in agg if f"BC+{a}" in agg[ds]]
    print(f"  {a:16s} Δcontrib_lang={np.mean(vals):+.4f}")

json.dump({"seed_mean": {ds: {c: {m: agg[ds][c][m] for m in METRICS} for c in agg[ds]} for ds in agg},
           "benefit": benefit,
           "form": {"mse_mean": float(np.nanmean(mse_gains)), "ce_mean": float(np.nanmean(ce_gains))}},
          open(os.path.join(OUT, "exp2e_analysis.json"), "w"), indent=2)
print("\nSAVED exp2e_analysis.json")
