"""Exp 2b — auxiliary as a REGULARIZER on BC (trainable encoder + OOD split).

Not "is the aux representation good" — but "does co-training BC with a shortcut-free aux make the
resulting BC policy generalize better OOD and rely less on the vision shortcut?"

Encoder: fine-tuned DINOv2-small (o_t -> rep).  BC head: [rep, state, language] -> 15-step action chunk.
Aux head (rep only, co-trained): retrospective (predict initial-obs DINOv2-base latent) and/or forward.
OOD split: k-means on instruction embeddings -> hold out task clusters (unseen tasks at test).
Metrics on OOD test: (1) BC action MSE (generalization), (2) language-sensitivity = ||Δ action when the
instruction is swapped|| (high = reads language; low = vision shortcut).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "fractal")
DEV = "cuda"
EPOCHS = int(os.environ.get("EPOCHS", 40))
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
Ft = d["Ft"]  # (N,224,224,3) uint8 current frame
z0 = torch.tensor(l2n(lat["z0"]), device=DEV)        # retrospective target (dino-base 768)
zfs = torch.tensor(l2n(lat["z_fs"]), device=DEV)     # forward target
act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)  # 105
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
lang_np = l2n(embed_text(d["instr"]))
lang = torch.tensor(lang_np, device=DEV)
N = len(Ft)

# ---- OOD split by task cluster (k-means on per-episode instruction) ----
eps = np.unique(g)
ep_lang = np.stack([lang_np[g == e].mean(0) for e in eps])
K = 8
cl = KMeans(K, n_init=5, random_state=SEED).fit_predict(ep_lang)
ood_clusters = set(np.argsort(np.bincount(cl))[:3].tolist())  # 3 clusters -> OOD test
ep_is_ood = {e: (cl[i] in ood_clusters) for i, e in enumerate(eps)}
te = np.array([ep_is_ood[e] for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
print(f"{TAG}: N={N} train={tr.sum()} OOD-test={te.sum()} ({len(ood_clusters)}/{K} task clusters held out)", flush=True)

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
        r = self.enc.config.hidden_size  # 384
        self.bc = nn.Sequential(nn.Linear(r + sdim + ldim, 512), nn.GELU(), nn.Linear(512, adim))
        self.retro = nn.Linear(r, 768)
        self.fwd = nn.Linear(r, 768)

    def rep(self, x):
        return self.enc(pixel_values=x).last_hidden_state[:, 0]

    def forward(self, x, s, l):
        r = self.rep(x)
        return r, self.bc(torch.cat([r, s, l], 1))


def run(cond):
    m = Model(state.shape[1], lang.shape[1], act.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
    mse = nn.MSELoss()
    bs = 128
    for ep in range(EPOCHS):
        np.random.shuffle(tr_idx)
        m.train()
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i+bs]
            r = m.enc(pixel_values=imgs(b)).last_hidden_state[:, 0]
            # KI variant: with "sg", BC loss does NOT backprop into the encoder (like the SG-isolated
            # continuous action head); the encoder is then shaped ONLY by the auxiliary objectives.
            r_bc = r.detach() if "sg" in cond else r
            bc = m.bc(torch.cat([r_bc, state[b], lang[b]], 1))
            loss = mse(bc, act[b])
            if "retro" in cond:
                loss = loss + mse(m.retro(r), z0[b])
            if "fwd" in cond:
                loss = loss + mse(m.fwd(r), zfs[b])
            opt.zero_grad(); loss.backward(); opt.step()
    # eval on OOD test — tri-modal relational analysis (not just "reads language"):
    # does the BC policy model the JOINT {vision, language} -> action relation, or a single-modality shortcut?
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

        P = pred(R, S, L)
        r2_ood = r2(P); mse_ood = torch.mean((P - Y) ** 2).item()
        pvar = torch.mean(P.var(0)).item() + 1e-9
        perm = torch.randperm(len(L))
        # sensitivity: how much the action MOVES when one modality is swapped across the batch
        lang_sens = torch.mean((P - pred(R, S, L[perm])) ** 2).item() / pvar
        vis_sens = torch.mean((P - pred(R[perm], S, L)) ** 2).item() / pvar
        # contribution: how much OOD accuracy each modality actually BUYS (ablate -> R2 drop).
        # lang ablated -> zero the instruction vector; vision ablated -> replace rep with its batch mean.
        r2_no_lang = r2(pred(R, S, torch.zeros_like(L)))
        r2_no_vision = r2(pred(R.mean(0, keepdim=True).expand_as(R), S, L))
        contrib_lang = r2_ood - r2_no_lang
        contrib_vision = r2_ood - r2_no_vision
    return dict(r2_ood=r2_ood, mse_ood=mse_ood,
                lang_sensitivity=lang_sens, vision_sensitivity=vis_sens,
                r2_no_lang=r2_no_lang, r2_no_vision=r2_no_vision,
                contrib_lang=contrib_lang, contrib_vision=contrib_vision)


CONDS = {
    "BC-only": [], "BC+retro": ["retro"], "BC+fwd": ["fwd"], "BC+retro+fwd": ["retro", "fwd"],
    "SG-BC+retro": ["sg", "retro"], "SG-BC+fwd": ["sg", "fwd"], "SG-BC+retro+fwd": ["sg", "retro", "fwd"],
}
res = {}
for name, cond in CONDS.items():
    res[name] = run(cond)
    r = res[name]
    print(f"  {name:16s} OOD R2={r['r2_ood']:+.3f}  contrib[L={r['contrib_lang']:+.3f} V={r['contrib_vision']:+.3f}]"
          f"  sens[L={r['lang_sensitivity']:.4f} V={r['vision_sensitivity']:.4f}]", flush=True)
json.dump(res, open(os.path.join(OUT, f"exp2b_{TAG}_s{SEED}.json"), "w"), indent=2)
print("SAVED", f"exp2b_{TAG}_s{SEED}.json", flush=True)
