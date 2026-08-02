"""Change-recoverability: can a probe on o_t recover *what changed* between a past frame and now?

For each past frame p in {near (z_ps), far (z_pl), initial (z0)}: target = (z_p - z_t) (the change),
probe = Ridge(z_t -> change), GroupKFold by episode. R2 = 1 - MSE/Var(change).
High R2 = the current frame carries recoverable info about how the scene got here (retrospective signal),
independent of the strong copy baseline that pins the raw-frame metric to the diagonal.
"""
import json, os
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DS = [("droid", "DROID"), ("bridge", "Bridge"), ("fractal", "RT-1")]


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def r2_change(zt, zp, g):
    change = zp - zt
    var = float(np.mean(np.sum((change - change.mean(0)) ** 2, 1)))
    pred = np.zeros_like(change)
    for tr, te in GroupKFold(5).split(zt, groups=g):
        sc = StandardScaler().fit(zt[tr])
        m = Ridge(alpha=100.0).fit(sc.transform(zt[tr]), change[tr])
        pred[te] = m.predict(sc.transform(zt[te]))
    mse = float(np.mean(np.sum((change - pred) ** 2, 1)))
    return 1 - mse / var


out = {}
for tag, name in DS:
    cache = os.path.join(OUT, f"cache/dino_latents_{tag}.npz")
    npz = os.path.join(OUT, f"cache/transitions_{tag}.npz")
    if not (os.path.exists(cache) and os.path.exists(npz)):
        continue
    lat = np.load(cache)
    g = np.load(npz, allow_pickle=True)["ep_id"]
    zt = l2n(lat["zt"])
    row = {
        "near-past change (t-k~5)": r2_change(zt, l2n(lat["z_ps"]), g),
        "far-past change (t-k~45)": r2_change(zt, l2n(lat["z_pl"]), g),
        "initial change (t-0)": r2_change(zt, l2n(lat["z0"]), g),
    }
    out[name] = row
    print(f"{name}:")
    for k, v in row.items():
        print(f"    {k:26s} R2={v:+.3f}")

json.dump(out, open(os.path.join(OUT, "change_recovery.json"), "w"), indent=2)
print("SAVED change_recovery.json")
