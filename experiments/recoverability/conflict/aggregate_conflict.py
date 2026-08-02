"""Aggregate the LA4VLA preliminary probe across seeds → exp_conflict_analysis.json + a printed table.

Reads exp_conflict_s{SEED}.json (results[modality][condition] = {DAR,DCS,SR,SS}).
Reports, per modality, mean±std over seeds for each of the 4 conditions, plus the key derived quantity:
  reliance(m) = DAR(paired) − DAR(conflict)   (how much flipping m breaks instruction-alignment)
High reliance = the policy EXPLOITS m; ~0 = it IGNORES m (action unmoved when m is flipped).
"""
import os, json, glob
import numpy as np

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
MODS = ["vision", "language", "state"]
CONDS = ["paired", "removed", "unaligned", "conflict"]
METS = ["DAR", "DCS", "SR", "SS"]

files = sorted(glob.glob(os.path.join(OUT, "exp_conflict_s*.json")))
runs = [json.load(open(f)) for f in files]
assert runs, "no exp_conflict_s*.json found"


def stack(mod, cond, met):
    return np.array([r["results"][mod][cond][met] for r in runs
                     if not np.isnan(r["results"][mod][cond][met])])


agg = {"n_seeds": len(runs), "modalities": {}}
print(f"LA4VLA preliminary probe — {len(runs)} seeds\n")
hdr = f"{'modality':9s} " + " ".join(f"{c[:8]:>14s}" for c in CONDS) + f"{'reliance':>10s}"
print(hdr); print("-" * len(hdr))
for mod in MODS:
    row = {"conditions": {}}
    for cond in CONDS:
        row["conditions"][cond] = {met: [float(stack(mod, cond, met).mean()),
                                         float(stack(mod, cond, met).std())] for met in METS}
    dar_p = stack(mod, "paired", "DAR").mean()
    dar_c = stack(mod, "conflict", "DAR").mean()
    reliance = float(dar_p - dar_c)
    row["reliance_DAR"] = reliance
    row["exploited"] = reliance > 0.15          # policy leans on this modality
    agg["modalities"][mod] = row
    cells = " ".join(f"{stack(mod,c,'DAR').mean():5.3f}±{stack(mod,c,'DAR').std():4.2f} " for c in CONDS)
    print(f"{mod:9s} {cells}{reliance:+8.3f}")

rank = sorted(MODS, key=lambda x: -agg["modalities"][x]["reliance_DAR"])
agg["exploited_ranking"] = rank
agg["headline"] = (f"paired-trained policy leans on '{rank[0]}' "
                   f"(reliance {agg['modalities'][rank[0]]['reliance_DAR']:+.2f}); "
                   f"'{rank[-1]}' is ignored (reliance {agg['modalities'][rank[-1]]['reliance_DAR']:+.2f})")
print("\nDAR = fraction of actions still aligned with the TRUE direction.")
print("reliance = DAR(paired) − DAR(conflict): high ⇒ exploited, ~0 ⇒ ignored.")
print("\n" + agg["headline"])
json.dump(agg, open(os.path.join(OUT, "exp_conflict_analysis.json"), "w"), indent=2)
print("\nSAVED exp_conflict_analysis.json")
