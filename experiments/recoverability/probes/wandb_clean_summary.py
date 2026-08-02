"""Log a CLEAN wandb dashboard for the AHA recoverability results (from saved jsons).
One run per data scale: a results Table + law/flip/speed scatters + headline correlation scalars.
Plus one 'measure_compare' run with the cross-scale correlation table. Project: aha-recoverability.
"""
import os, json, glob
import numpy as np
import wandb
from scipy.stats import pearsonr, spearmanr

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
ENT, PROJ = "jellyho_", "aha-recoverability"
SCALES = [("frac4k", "N=4000"), ("frac8k", "N=8000"), ("fractal", "N=16000")]
COLS = ["objective", "type", "R_linear", "R_mlp", "R_mdl", "R_deep", "R_deep_aulc", "R_deep_early",
        "generalization", "benefit"]
MEAS = ["R_linear", "R_mlp", "R_mdl", "R_deep", "R_deep_aulc", "R_deep_early"]


def load(scale):
    fs = sorted(glob.glob(f"{OUT}/exp2h_{scale}_s*.json"))
    S = [json.load(open(f)) for f in fs]
    if not S:
        return None
    names = list(S[0]["objectives"])
    def agg(n, k):
        v = [s["objectives"][n][k] for s in S if s["objectives"][n].get(k) is not None]
        return float(np.mean(v)) if v else None
    obj = {n: {k: agg(n, k) for k in COLS[2:]} for n in names}
    for n in names:
        obj[n]["type"] = S[0]["objectives"][n]["type"]
    bc = float(np.mean([s["BC_only_ood"] for s in S]))
    return names, obj, bc, len(S)


compare_rows = []
for scale, tag in SCALES:
    d = load(scale)
    if d is None:
        continue
    names, obj, bc, nseed = d
    run = wandb.init(entity=ENT, project=PROJ, name=f"summary_{scale}", reinit=True,
                     config={"scale": tag, "seeds": nseed, "n_obj": len(names)})
    tbl = wandb.Table(columns=COLS)
    for n in names:
        o = obj[n]
        tbl.add_data(n, o["type"], o["R_linear"], o["R_mlp"], o["R_mdl"], o["R_deep"],
                     o["R_deep_aulc"], o["R_deep_early"], o["generalization"], o["benefit"])
    wandb.log({"results": tbl,
               "law/deep_vs_gen": wandb.plot.scatter(tbl, "R_deep", "generalization", title="LAW: recoverability↓ ⇒ gen↑"),
               "law/linear_vs_gen": wandb.plot.scatter(tbl, "R_linear", "generalization", title="linear probe MISranks (sign flip)"),
               "law/speed_vs_gen": wandb.plot.scatter(tbl, "R_deep_early", "generalization", title="speed (dynamics) = best predictor")})
    for k in MEAS:
        xy = [(obj[n][k], obj[n]["generalization"]) for n in names
              if obj[n].get(k) is not None and obj[n].get("generalization") is not None]
        if len(xy) > 3:
            r = pearsonr([a for a, _ in xy], [b for _, b in xy])[0]
            wandb.summary[f"corr/{k}_vs_gen"] = round(float(r), 3)
            compare_rows.append([tag, k, round(float(r), 3)])
    wandb.summary["BC_only_ood"] = round(bc, 3)
    wandb.finish()
    print(f"logged summary_{scale} ({nseed} seeds, {len(names)} obj)", flush=True)

if compare_rows:
    run = wandb.init(entity=ENT, project=PROJ, name="measure_compare", reinit=True)
    ct = wandb.Table(columns=["scale", "measure", "pearson_vs_gen"], data=compare_rows)
    wandb.log({"measure_comparison": ct,
               "compare/bar": wandb.plot.bar(ct, "measure", "pearson_vs_gen", title="which measure predicts generalization (want negative)")})
    wandb.finish()
    print("logged measure_compare", flush=True)
