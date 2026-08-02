"""Exp 1b — the BC memorization shortcut, as a data property.

Three panels (GroupKFold by episode; R2 = variance explained on held-out episodes):
  P1 instruction-from-scene : can the task be read off the image? (vision-overrides-language enabler)
  P2 action memorization    : kNN lookup — do visually-similar states share an action? (scene->trajectory)
  P3 language necessity      : Δ_lang = R2(action | image+state+language) − R2(action | image+state)

Frozen DINOv2 features → a lower bound on what an end-to-end-trained policy could memorize;
the *relative* pattern across datasets is the signal.
"""
import os, json
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DS = [("droid", "DROID"), ("bridge", "Bridge"), ("fractal", "RT-1")]
DEV = os.environ.get("DEV", "cpu")


def l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def zs(a):
    a = np.asarray(a, float).reshape(len(a), -1)
    return (a - a.mean(0)) / (a.std(0) + 1e-8)


def embed_text(strings):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            e = tok(list(strings[i:i + 128]), padding=True, truncation=True, return_tensors="pt").to(DEV)
            h = m(**e).last_hidden_state
            mask = e["attention_mask"].unsqueeze(-1).float()
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(out)


def r2(Y, pred):
    var = float(np.mean(np.sum((Y - Y.mean(0)) ** 2, 1)))
    return 1 - float(np.mean(np.sum((Y - pred) ** 2, 1))) / var


def cv_ridge(X, Y, g, std=True):
    p = np.zeros_like(Y, float)
    for tr, te in GroupKFold(5).split(X, groups=g):
        if std:
            sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        else:
            Xtr, Xte = X[tr], X[te]
        p[te] = Ridge(alpha=100.0).fit(Xtr, Y[tr]).predict(Xte).reshape(len(te), -1)
    return r2(Y, p)


def cv_knn(X, Y, g, k=16):
    p = np.zeros_like(Y, float)
    for tr, te in GroupKFold(5).split(X, groups=g):
        m = KNeighborsRegressor(n_neighbors=min(k, len(tr)), metric="cosine", weights="distance").fit(X[tr], Y[tr])
        p[te] = m.predict(X[te]).reshape(len(te), -1)
    return r2(Y, p)


out = {}
for tag, name in DS:
    d = np.load(os.path.join(OUT, f"cache/transitions_{tag}.npz"), allow_pickle=True)
    lat = np.load(os.path.join(OUT, f"cache/dino_latents_{tag}.npz"))
    zt = l2n(lat["zt"]); g = d["ep_id"]
    # action = the 15-step action CHUNK a policy actually outputs (not a single step)
    act = zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1))
    state = zs(np.concatenate([d["cartt"], d["gript"]], axis=1))
    instr = d["instr"]; has = np.array([bool(str(s).strip()) for s in instr])
    lang = l2n(embed_text(instr))

    row = {
        # P1 instruction recoverable from the image
        "instr<-scene (linear)": cv_ridge(zt[has], lang[has], g[has]),
        "instr<-scene (kNN)":    cv_knn(zt[has], lang[has], g[has]),
        # P2 action memorizable from the image (scene -> trajectory lookup)
        "action<-scene (linear)": cv_ridge(zt, act, g),
        "action<-scene (kNN)":    cv_knn(zt, act, g),
        # P3 does language add anything for the action
        "action<-img+state":       cv_ridge(np.concatenate([zt, state], 1)[has], act[has], g[has]),
        "action<-img+state+lang":  cv_ridge(np.concatenate([zt, state, lang], 1)[has], act[has], g[has]),
    }
    row["Δ_lang"] = row["action<-img+state+lang"] - row["action<-img+state"]
    out[name] = row
    print(f"\n{name}  (n={len(zt)}, with-instr={int(has.sum())})")
    for k, v in row.items():
        print(f"    {k:26s} {v:+.3f}")

json.dump(out, open(os.path.join(OUT, "memorization.json"), "w"), indent=2)
print("\nSAVED memorization.json")
