"""Exp 2 — does a shortcut-free (retrospective) pretraining objective give a more transferable
representation than a forward/BC objective?  (small, controlled; frozen DINOv2 + trained adapter.)

Per objective O, train adapter A: DINOv2(o_t) -> rep(256) with a head predicting O's target on TRAIN
episodes. Freeze A. Then linear-probe downstream readouts {action-chunk, future-gripper, progress,
instruction} from rep on HELD-OUT TEST episodes, at data fractions {25,50,100}% of TRAIN. Report R².
Objectives: raw (no adapter), forward-obs (z_{t+k}), retrospective (z_0), BC (action-chunk), mixed (z_0+BC).
"""
import os, json, sys
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "fractal")
DEV = os.environ.get("DEV", "cuda")
torch.manual_seed(0)


def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def zs(a):
    a = np.asarray(a, float).reshape(len(a), -1); return (a - a.mean(0)) / (a.std(0) + 1e-8)
def r2(Y, P):
    v = np.mean(np.sum((Y - Y.mean(0)) ** 2, 1)); return float(1 - np.mean(np.sum((Y - P) ** 2, 1)) / v)


def embed_text(strings):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
    o = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            e = tok(list(strings[i:i + 128]), padding=True, truncation=True, return_tensors="pt").to(DEV)
            h = m(**e).last_hidden_state; mask = e["attention_mask"].unsqueeze(-1).float()
            o.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(o)


class Adapter(torch.nn.Module):
    def __init__(self, din=768, drep=256, outs=()):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(din, 512), torch.nn.GELU(),
                                       torch.nn.Linear(512, drep), torch.nn.GELU())
        self.heads = torch.nn.ModuleList([torch.nn.Linear(drep, o) for o in outs])

    def forward(self, x):
        r = self.enc(x); return r, [h(r) for h in self.heads]


def train_adapter(X, targets, epochs=300):
    """targets: list of ( Y_np ) to predict jointly. Returns rep(X) for all rows."""
    Xt = torch.tensor(X, dtype=torch.float32, device=DEV)
    Ys = [torch.tensor(t, dtype=torch.float32, device=DEV) for t in targets]
    net = Adapter(X.shape[1], 256, tuple(t.shape[1] for t in targets)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    lf = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); _, preds = net(Xt)
        loss = sum(lf(p, y) for p, y in zip(preds, Ys)); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net.enc(Xt).cpu().numpy()


def probe(rep, Y, tr, te, frac, rng):
    ep_tr = np.unique(g[tr])
    k = max(1, int(len(ep_tr) * frac))
    keep = set(rng.choice(ep_tr, k, replace=False).tolist())
    m = np.array([e in keep for e in g[tr]])
    sc = StandardScaler().fit(rep[tr][m])
    mo = Ridge(alpha=100.0).fit(sc.transform(rep[tr][m]), Y[tr][m])
    P = np.asarray(mo.predict(sc.transform(rep[te]))).reshape(int(te.sum()), -1)
    return r2(Y[te], P)


d = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
lat = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
zt, z0, zfs = l2n(lat["zt"]), l2n(lat["z0"]), l2n(lat["z_fs"])
g = d["ep_id"]
act = zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1))
gripf = zs(d["grip_fut"]); prog = zs(d["progress"])
lang = l2n(embed_text(d["instr"]))

# episode split (deterministic): last 25% of episode-ids -> test
eps = np.unique(g); n_te = max(1, int(len(eps) * 0.25))
te_eps = set(eps[-n_te:].tolist())
te = np.array([e in te_eps for e in g]); tr = ~te
print(f"{TAG}: N={len(zt)} train_ep={len(eps)-n_te} test_ep={n_te}", flush=True)

REPS = {
    "raw DINOv2":       zt,
    "forward-obs":      None, "retrospective":    None,
    "BC (chunk)":       None, "mixed (retro+BC)": None,
}
REPS["forward-obs"] = train_adapter(zt, [zfs])
REPS["retrospective"] = train_adapter(zt, [z0])
REPS["BC (chunk)"] = train_adapter(zt, [act])
REPS["mixed (retro+BC)"] = train_adapter(zt, [z0, act])

READOUTS = {"action-chunk": act, "future-gripper": gripf, "progress": prog, "instruction": lang}
FRACS = [0.25, 0.5, 1.0]
rng = np.random.default_rng(0)
res = {}
for rname, rep in REPS.items():
    res[rname] = {}
    for dname, Y in READOUTS.items():
        res[rname][dname] = {f"{int(f*100)}%": probe(rep, Y, tr, te, f, rng) for f in FRACS}
    # transfer score = mean over readouts at 100%
    res[rname]["transfer@100"] = float(np.mean([res[rname][d]["100%"] for d in READOUTS]))
    print(f"  {rname:18s} transfer@100={res[rname]['transfer@100']:+.3f}  "
          f"act@25={res[rname]['action-chunk']['25%']:+.3f} act@100={res[rname]['action-chunk']['100%']:+.3f} "
          f"prog@100={res[rname]['progress']['100%']:+.3f} instr@100={res[rname]['instruction']['100%']:+.3f}", flush=True)

json.dump(res, open(os.path.join(OUT, f"exp2_{TAG}.json"), "w"), indent=2)
print("SAVED", f"exp2_{TAG}.json", flush=True)
