"""Aggregate exp2b tri-modal results across seeds -> exp2b_trimodal_agg.json + printed table.

Only files carrying the tri-modal keys (contrib_lang/contrib_vision) are used; stale grad-only jsons are skipped.
Story: BC-only takes a single-modality shortcut (contrib_lang ~ 0 => ignores language); a shortcut-free aux
co-trained on the shared encoder raises contrib_lang while keeping contrib_vision => the policy models the
JOINT {vision, language} -> action relation, and OOD R2 improves. LIBERO-goal (same scene, 10 goals) is the
sharpest memorization test.
"""
import os, glob, json
import numpy as np

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DATASETS = ["fractal", "droid", "libero_goal", "libero_object"]
METRICS = ["r2_ood", "contrib_lang", "contrib_vision", "lang_sensitivity", "vision_sensitivity",
           "r2_no_lang", "r2_no_vision"]
TRIMODAL_KEY = "contrib_lang"

agg = {}
for ds in DATASETS:
    files = sorted(glob.glob(os.path.join(OUT, f"exp2b_{ds}_s*.json")))
    seeds = []
    for f in files:
        d = json.load(open(f))
        # keep only tri-modal-format files (a condition dict carrying contrib_lang)
        if any(TRIMODAL_KEY in v for v in d.values() if isinstance(v, dict)):
            seeds.append((os.path.basename(f), d))
    if not seeds:
        continue
    conds = list(seeds[0][1].keys())
    agg[ds] = {"n_seeds": len(seeds), "files": [s[0] for s in seeds], "cond": {}}
    for c in conds:
        agg[ds]["cond"][c] = {}
        for m in METRICS:
            vals = [d[c][m] for _, d in seeds if c in d and m in d[c]]
            if vals:
                agg[ds]["cond"][c][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

json.dump(agg, open(os.path.join(OUT, "exp2b_trimodal_agg.json"), "w"), indent=2)

# ---- printed table ----
CONDS = ["BC-only", "BC+retro", "BC+fwd", "BC+retro+fwd",
         "SG-BC+retro", "SG-BC+fwd", "SG-BC+retro+fwd"]
for ds in DATASETS:
    if ds not in agg:
        continue
    a = agg[ds]["cond"]
    print(f"\n=== {ds}  (seeds={agg[ds]['n_seeds']}) ===")
    print(f"{'condition':16s} {'OOD_R2':>8s} {'contribL':>9s} {'contribV':>9s} {'sensL':>8s} {'sensV':>8s}")
    for c in CONDS:
        if c not in a:
            continue
        g = lambda m: a[c].get(m, {}).get("mean", float("nan"))
        print(f"{c:16s} {g('r2_ood'):+8.3f} {g('contrib_lang'):+9.3f} {g('contrib_vision'):+9.3f} "
              f"{g('lang_sensitivity'):8.4f} {g('vision_sensitivity'):8.4f}")
print("\nSAVED exp2b_trimodal_agg.json")
