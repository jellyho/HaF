"""Exp 2d - auxiliary-WEIGHT (lambda) dose-response.

Does the regularization behave like a regularizer -- benefit rising then saturating/reversing with strength?
Sweep the aux loss weight lambda for the two best auxiliaries (retro-obs single; retro+fwd mix) on all
datasets. loss = BC + lambda * sum(aux). Tri-modal readout identical to exp2b/2c.
Output: exp2d_{TAG}_s{SEED}.json   (conditions keyed "retro@<lam>", "mix@<lam>").
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "fractal")
DEV = "cuda"
EPOCHS = int(os.environ.get("EPOCHS", 30))
SEED = int(os.environ.get("SEED", 0))
LAMBDAS = [0.25, 0.5, 1.0, 2.0, 4.0]
torch.manual_seed(SEED); np.random.seed(SEED)


def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def zs(a):
    a = np.asarray(a, float).reshape(len(a), -1); m, s = a.mean(0), a.std(0) + 1e-8
    return ((a - m) / s).astype(np.float32)


def embed_text(strings):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    m = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
    o = []
    with torch.no_grad():
        for i in range(0, len(strings), 128):
            e = tok(list(strings[i:i+128]), padding=True, truncation=True, return_tensors="pt").to(DEV)
            h = m(**e).last_hidden_state; mask = e["attention_mask"].unsqueeze(-1).float()
            o.append(((h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)).cpu().numpy())
    return np.concatenate(o).astype(np.float32)


d = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
lat = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
Ft = d["Ft"]
TARGETS = {"retro-obs": l2n(lat["z0"]), "fwd-obs": l2n(lat["z_fs"])}
TGT = {k: torch.tensor(v, device=DEV) for k, v in TARGETS.items()}
TDIM = {k: v.shape[1] for k, v in TARGETS.items()}

act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
lang_np = l2n(embed_text(d["instr"]))
lang = torch.tensor(lang_np, device=DEV)
N = len(Ft)

eps = np.unique(g)
ep_lang = np.stack([lang_np[g == e].mean(0) for e in eps])
K = 8
cl = KMeans(K, n_init=5, random_state=SEED).fit_predict(ep_lang)
ood_clusters = set(np.argsort(np.bincount(cl))[:3].tolist())
ep_is_ood = {e: (cl[i] in ood_clusters) for i, e in enumerate(eps)}
te = np.array([ep_is_ood[e] for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
print(f"{TAG}: N={N} train={tr.sum()} OOD-test={te.sum()}", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def imgs(idx):
    b = torch.from_numpy(Ft[idx]).to(DEV).float().permute(0, 3, 1, 2) / 255.0
    return (b - MEAN) / STD


class Model(nn.Module):
    def __init__(self, sdim, ldim, adim):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained("facebook/dinov2-small")
        r = self.enc.config.hidden_size
        self.bc = nn.Sequential(nn.Linear(r + sdim + ldim, 512), nn.GELU(), nn.Linear(512, adim))
        self.aux = nn.ModuleDict({k: nn.Linear(r, TDIM[k]) for k in TARGETS})


def run(auxset, lam):
    m = Model(state.shape[1], lang.shape[1], act.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
    mse = nn.MSELoss(); bs = 128
    for ep in range(EPOCHS):
        np.random.shuffle(tr_idx); m.train()
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i+bs]
            r = m.enc(pixel_values=imgs(b)).last_hidden_state[:, 0]
            loss = mse(m.bc(torch.cat([r, state[b], lang[b]], 1)), act[b])
            for a in auxset:
                loss = loss + lam * mse(m.aux[a](r), TGT[a][b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        R, S, L = [], [], []
        for i in range(0, len(te_idx), bs):
            b = te_idx[i:i+bs]
            R.append(m.enc(pixel_values=imgs(b)).last_hidden_state[:, 0])
            S.append(state[b]); L.append(lang[b])
        R = torch.cat(R); S = torch.cat(S); L = torch.cat(L)
        Y = act[torch.tensor(te_idx, device=DEV)]
        var = torch.mean((Y - Y.mean(0)) ** 2).item()
        def pred(r, s, l): return m.bc(torch.cat([r, s, l], 1))
        def r2(P): return 1 - torch.mean((P - Y) ** 2).item() / var
        P = pred(R, S, L); r2_ood = r2(P)
        pv = torch.mean(P.var(0)).item() + 1e-9
        perm = torch.randperm(len(L))
        lang_sens = torch.mean((P - pred(R, S, L[perm])) ** 2).item() / pv
        vis_sens = torch.mean((P - pred(R[perm], S, L)) ** 2).item() / pv
        r2_no_lang = r2(pred(R, S, torch.zeros_like(L)))
        r2_no_vision = r2(pred(R.mean(0, keepdim=True).expand_as(R), S, L))
    return dict(r2_ood=r2_ood, contrib_lang=r2_ood - r2_no_lang, contrib_vision=r2_ood - r2_no_vision,
                lang_sensitivity=lang_sens, vision_sensitivity=vis_sens)


CONDS = {"BC-only": ([], 0.0)}
for lam in LAMBDAS:
    CONDS[f"retro@{lam}"] = (["retro-obs"], lam)
    CONDS[f"mix@{lam}"] = (["retro-obs", "fwd-obs"], lam)
res = {}
for name, (auxset, lam) in CONDS.items():
    res[name] = run(auxset, lam); r = res[name]
    print(f"  {name:12s} R2={r['r2_ood']:+.3f} cL={r['contrib_lang']:+.3f} sL={r['lang_sensitivity']:.4f}", flush=True)
json.dump(res, open(os.path.join(OUT, f"exp2d_{TAG}_s{SEED}.json"), "w"), indent=2)
print("SAVED", f"exp2d_{TAG}_s{SEED}.json", flush=True)
