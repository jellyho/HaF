"""Exp 2g - REPRESENTATION QUALITY: the missing middle of Objective -> Recoverability -> Rep -> Generalization.

exp2b/2c/2d/2e measured behavior (OOD action R2, language/vision-use). Reviewer's point: we also need
representation-INTRINSIC evidence that a low-recoverability (retrospective) auxiliary yields a RICHER latent.
This trains the same conditions and, on the held-out (OOD) representation, measures:

  instr_decod   linear-probe R2 of  rep -> instruction embedding  (5-fold CV)  -- is language IN the rep?
  task_retrieval  same-instruction rate among the 10 nearest neighbors in rep space -- task-structured?
  silhouette    silhouette of rep clustered by held-out task cluster            -- separable by task?
  + saves the OOD rep matrix (aligned by sample across conditions) for cross-condition CKA.

Conditions: BC-only, BC+retro (low-recoverability aux), BC+fwd (high-recoverability aux), SG+retro (KI).
Output: exp2g_{TAG}_s{SEED}.json  and  exp2g_reps_{TAG}_s{SEED}.npz (R per condition + labels + te_idx).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score

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


d = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
lat = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
Ft = d["Ft"]
z0 = torch.tensor(l2n(lat["z0"]), device=DEV)
zfs = torch.tensor(l2n(lat["z_fs"]), device=DEV)
act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
instr = np.array([str(s) for s in d["instr"]])
lang_np = l2n(embed_text(d["instr"]))
lang = torch.tensor(lang_np, device=DEV)
N = len(Ft)

eps = np.unique(g)
ep_lang = np.stack([lang_np[g == e].mean(0) for e in eps])
K = 8
cl = KMeans(K, n_init=5, random_state=SEED).fit_predict(ep_lang)
ood_clusters = set(np.argsort(np.bincount(cl))[:3].tolist())
ep_cluster = {e: cl[i] for i, e in enumerate(eps)}
ep_is_ood = {e: (cl[i] in ood_clusters) for i, e in enumerate(eps)}
te = np.array([ep_is_ood[e] for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
te_cluster = np.array([ep_cluster[e] for e in g[te_idx]])
te_instr_id = np.unique(instr[te_idx], return_inverse=True)[1]
print(f"{TAG}: N={N} train={tr.sum()} OOD-test={te.sum()} uniq-OOD-instr={te_instr_id.max()+1}", flush=True)

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
        self.retro = nn.Linear(r, 768)
        self.fwd = nn.Linear(r, 768)

    def rep(self, x):
        return self.enc(pixel_values=x).last_hidden_state[:, 0]


def train_and_rep(cond):
    m = Model(state.shape[1], lang.shape[1], act.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
    mse = nn.MSELoss(); bs = 128; sg = "sg" in cond
    for ep in range(EPOCHS):
        np.random.shuffle(tr_idx); m.train()
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i+bs]
            r = m.rep(imgs(b))
            r_bc = r.detach() if sg else r
            loss = mse(m.bc(torch.cat([r_bc, state[b], lang[b]], 1)), act[b])
            if "retro" in cond:
                loss = loss + mse(m.retro(r), z0[b])
            if "fwd" in cond:
                loss = loss + mse(m.fwd(r), zfs[b])
            opt.zero_grad(); loss.backward(); opt.step()
    # extract OOD representation
    m.eval()
    with torch.no_grad():
        R = []
        for i in range(0, len(te_idx), bs):
            R.append(m.rep(imgs(te_idx[i:i+bs])).cpu().numpy())
    return np.concatenate(R).astype(np.float32)


def instr_decod(R, L):
    kf = KFold(5, shuffle=True, random_state=0); r2s = []
    for a, b in kf.split(R):
        p = Ridge(alpha=1.0).fit(R[a], L[a]).predict(R[b])
        ss_res = ((L[b] - p) ** 2).sum(); ss_tot = ((L[b] - L[a].mean(0)) ** 2).sum()
        r2s.append(1 - ss_res / (ss_tot + 1e-9))
    return float(np.mean(r2s))


def retrieval(R, labels, k=10):
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    S = Rn @ Rn.T; np.fill_diagonal(S, -2.0)
    nn_idx = np.argsort(-S, axis=1)[:, :k]
    return float((labels[nn_idx] == labels[:, None]).mean())


CONDS = {"BC-only": [], "BC+retro": ["retro"], "BC+fwd": ["fwd"], "SG+retro": ["sg", "retro"]}
Lte = lang_np[te_idx]
res, reps = {}, {}
for name, cond in CONDS.items():
    R = train_and_rep(cond)
    reps[name] = R.astype(np.float16)
    dec = instr_decod(R, Lte)
    ret = retrieval(R, te_instr_id, k=10)
    try:
        sil = float(silhouette_score(R, te_cluster)) if len(set(te_cluster)) > 1 else float("nan")
    except Exception:
        sil = float("nan")
    res[name] = dict(instr_decod=dec, task_retrieval=ret, silhouette=sil)
    print(f"  {name:10s} instr_decod={dec:+.3f}  retrieval@10={ret:.3f}  silhouette={sil:+.3f}", flush=True)

json.dump(res, open(os.path.join(OUT, f"exp2g_{TAG}_s{SEED}.json"), "w"), indent=2)
np.savez_compressed(os.path.join(OUT, f"cache/exp2g_reps_{TAG}_s{SEED}.npz"),
                    te_instr_id=te_instr_id, te_cluster=te_cluster, **{f"R_{k}": v for k, v in reps.items()})
print("SAVED", f"exp2g_{TAG}_s{SEED}.json (+reps)", flush=True)
