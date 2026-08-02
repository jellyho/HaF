"""Exp 2f - FUSED-backbone probe: a more faithful VLA mock.

exp2b/2c/2d/2e used a vision-only shared representation (r = encoder(image)); state and language were
late-fused only at the BC head. A real VLA (PaliGemma) fuses vision + language IN the backbone, and the
action head reads the fused multimodal representation. This experiment closes that gap:

    r_img = DINOv2-small(image)
    tokens = [proj(r_img), proj(state), proj(language)]   (+ modality-type embeddings)
    r_fused = TransformerEncoder(tokens).mean(1)          <-- shared MULTIMODAL backbone rep
    BC head:  r_fused -> action ;   aux head: r_fused -> aux target

Now BC, aux, and KI all act on a representation that CONTAINS language. Two things this lets us check that
the vision-only probe could not:
  (1) representation-level grounding (does language actually live in r_fused?),
  (2) whether "hard KI loses vision" survives when the insulated backbone is multimodal AND when the aux is
      RICH (retro+fwd) rather than a single narrow target.

Conditions: BC-only, BC+retro (grad), BC+retro+fwd (grad), SG+retro (KI single), SG+retro+fwd (KI rich).
Tri-modal readout (r2_ood, contrib_lang/vision, sensL/sensV) computed on the fused backbone.
Output: exp2f_{TAG}_s{SEED}.json.
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
DMODEL = 256
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
print(f"{TAG}: N={N} train={tr.sum()} OOD-test={te.sum()}  fused d={DMODEL}", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def imgs(idx):
    b = torch.from_numpy(Ft[idx]).to(DEV).float().permute(0, 3, 1, 2) / 255.0
    return (b - MEAN) / STD


class Model(nn.Module):
    def __init__(self, sdim, ldim, adim, d=DMODEL):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained("facebook/dinov2-small")
        ri = self.enc.config.hidden_size  # 384
        self.p_img = nn.Linear(ri, d)
        self.p_state = nn.Linear(sdim, d)
        self.p_lang = nn.Linear(ldim, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))  # image/state/lang type embeddings
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        self.bc = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, adim))
        self.retro = nn.Linear(d, 768)
        self.fwd = nn.Linear(d, 768)

    def img_feat(self, x):
        return self.enc(pixel_values=x).last_hidden_state[:, 0]  # (B,384)

    def fuse(self, rimg, s, l):
        toks = torch.stack([self.p_img(rimg), self.p_state(s), self.p_lang(l)], dim=1)  # (B,3,d)
        toks = toks + self.type_emb.unsqueeze(0)
        return self.backbone(toks).mean(1)  # (B,d) shared multimodal rep


def run(cond):
    m = Model(state.shape[1], lang.shape[1], act.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
    mse = nn.MSELoss(); bs = 128
    sg = "sg" in cond
    for ep in range(EPOCHS):
        np.random.shuffle(tr_idx); m.train()
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i+bs]
            rimg = m.img_feat(imgs(b))
            rf = m.fuse(rimg, state[b], lang[b])          # fused backbone rep (shapes encoder+fusion)
            rf_bc = rf.detach() if sg else rf              # KI: BC insulated from the fused backbone
            loss = mse(m.bc(rf_bc), act[b])
            if "retro" in cond:
                loss = loss + mse(m.retro(rf), z0[b])
            if "fwd" in cond:
                loss = loss + mse(m.fwd(rf), zfs[b])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        RI, S, L = [], [], []
        for i in range(0, len(te_idx), bs):
            b = te_idx[i:i+bs]
            RI.append(m.img_feat(imgs(b))); S.append(state[b]); L.append(lang[b])
        RI = torch.cat(RI); S = torch.cat(S); L = torch.cat(L)
        Y = act[torch.tensor(te_idx, device=DEV)]
        var = torch.mean((Y - Y.mean(0)) ** 2).item()
        def act_of(ri, s, l): return m.bc(m.fuse(ri, s, l))
        def r2(P): return 1 - torch.mean((P - Y) ** 2).item() / var
        P = act_of(RI, S, L); r2_ood = r2(P)
        pv = torch.mean(P.var(0)).item() + 1e-9
        perm = torch.randperm(len(L))
        lang_sens = torch.mean((P - act_of(RI, S, L[perm])) ** 2).item() / pv       # swap language
        vis_sens = torch.mean((P - act_of(RI[perm], S, L)) ** 2).item() / pv        # swap image
        r2_no_lang = r2(act_of(RI, S, torch.zeros_like(L)))                          # ablate language
        r2_no_vision = r2(act_of(RI.mean(0, keepdim=True).expand_as(RI), S, L))      # ablate vision
    return dict(r2_ood=r2_ood, contrib_lang=r2_ood - r2_no_lang, contrib_vision=r2_ood - r2_no_vision,
                lang_sensitivity=lang_sens, vision_sensitivity=vis_sens)


CONDS = {
    "BC-only": [], "BC+retro": ["retro"], "BC+retro+fwd": ["retro", "fwd"],
    "SG+retro": ["sg", "retro"], "SG+retro+fwd": ["sg", "retro", "fwd"],
}
res = {}
for name, cond in CONDS.items():
    res[name] = run(cond); r = res[name]
    print(f"  {name:14s} R2={r['r2_ood']:+.3f} cL={r['contrib_lang']:+.3f} cV={r['contrib_vision']:+.3f} "
          f"sL={r['lang_sensitivity']:.4f} sV={r['vision_sensitivity']:.4f}", flush=True)
json.dump(res, open(os.path.join(OUT, f"exp2f_{TAG}_s{SEED}.json"), "w"), indent=2)
print("SAVED", f"exp2f_{TAG}_s{SEED}.json", flush=True)
