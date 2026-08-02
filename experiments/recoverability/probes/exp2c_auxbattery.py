"""Exp 2c — aux-target BATTERY: which auxiliary property regularizes BC best?

Axis A of the extensive analysis. Same regularizer setup as exp2b (trainable DINOv2-small, BC head
[rep,state,lang]->15-step action chunk, OOD task-cluster split, grad co-training), but we sweep the
AUXILIARY TARGET across a menu that differs in Exp-1 shortcut-freeness:

  retro-obs   (z0)          retrospective initial-obs latent    -- shortcut-FREE  (low R_triv)
  farpast-obs (z_pl)        far past obs latent                 -- mildly redundant retro
  fwd-obs     (z_fs)        near future obs latent              -- copy shortcut   (high R_triv)
  farfwd-obs  (z_fl)        far future obs latent               -- mildly redundant foresight
  dynamics    (z_fs - zt)   change in obs latent                -- shortcut-free foresight (predict CHANGE)
  progress    (t/T)         scalar temporal phase               -- trivially easy
  pose0       (cart0)       initial EEF pose                    -- low-dim retro

Hypothesis (HaF core, at the aux level): the more shortcut-free / hard-but-learnable the aux target
(lower R_triv, higher probe>trivial in Exp 1), the more it forces the shared encoder to build rich
features -> the better it regularizes BC (larger OOD R2 gain, more language grounding). progress/pose0/
fwd-obs (easy or copyable) should regularize weakly; retro-obs / dynamics should regularize strongly.

Tri-modal readout identical to exp2b (r2_ood, contrib_lang/vision, sens_lang/vision).
Output: exp2c_{TAG}_s{SEED}.json.
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


# ---- data ----
d = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
lat = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
Ft = d["Ft"]
zt = l2n(lat["zt"])
# auxiliary-target battery (name -> (numpy target, dim))
TARGETS = {
    "retro-obs":   l2n(lat["z0"]),
    "farpast-obs": l2n(lat["z_pl"]),
    "fwd-obs":     l2n(lat["z_fs"]),
    "farfwd-obs":  l2n(lat["z_fl"]),
    "dynamics":    zs(l2n(lat["z_fs"]) - zt),
    "progress":    zs(d["progress"]),
    "pose0":       zs(d["cart0"]),
}
TGT = {k: torch.tensor(v, device=DEV) for k, v in TARGETS.items()}
TDIM = {k: v.shape[1] for k, v in TARGETS.items()}

act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
lang_np = l2n(embed_text(d["instr"]))
lang = torch.tensor(lang_np, device=DEV)
N = len(Ft)

# ---- OOD split by task cluster ----
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


def run(cond):
    m = Model(state.shape[1], lang.shape[1], act.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
    mse = nn.MSELoss(); bs = 128
    for ep in range(EPOCHS):
        np.random.shuffle(tr_idx); m.train()
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i+bs]
            r = m.enc(pixel_values=imgs(b)).last_hidden_state[:, 0]
            loss = mse(m.bc(torch.cat([r, state[b], lang[b]], 1)), act[b])
            for a in cond:
                loss = loss + mse(m.aux[a](r), TGT[a][b])
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


CONDS = {"BC-only": []}
for a in TARGETS:
    CONDS[f"BC+{a}"] = [a]
res = {}
for name, cond in CONDS.items():
    res[name] = run(cond); r = res[name]
    print(f"  {name:16s} R2={r['r2_ood']:+.3f} cL={r['contrib_lang']:+.3f} cV={r['contrib_vision']:+.3f} "
          f"sL={r['lang_sensitivity']:.4f} sV={r['vision_sensitivity']:.4f}", flush=True)
json.dump(res, open(os.path.join(OUT, f"exp2c_{TAG}_s{SEED}.json"), "w"), indent=2)
print("SAVED", f"exp2c_{TAG}_s{SEED}.json", flush=True)
