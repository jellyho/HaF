"""Train a mini-VLA policy and SAVE it (weights + normalization stats) for SimplerEnv rollout.
ARM=bc  -> BC only ; ARM=aha -> BC + a low-recoverability aux (far-fut-obs). Output: outputs/policy_{ARM}.pt
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "fractal")
DEV = "cuda"
EPOCHS = int(os.environ.get("EPOCHS", 25))
SEED = int(os.environ.get("SEED", 0))
SUBN = int(os.environ.get("SUBN", 0))         # 0 = full 16k
ARM = os.environ.get("ARM", "bc")
DMODEL = 256
torch.manual_seed(SEED); np.random.seed(SEED)


def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def zs_stats(a):
    a = np.asarray(a, float).reshape(len(a), -1); m, s = a.mean(0), a.std(0) + 1e-8
    return ((a - m) / s).astype(np.float32), m.astype(np.float32), s.astype(np.float32)


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


_d = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
_l = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
_N0 = len(_d["Ft"])
_sel = (np.random.default_rng(123).permutation(_N0)[:SUBN] if (SUBN and SUBN < _N0) else np.arange(_N0))
d = {k: _d[k][_sel] for k in _d.files}
lat = {k: _l[k][_sel] for k in _l.files}
Ft = d["Ft"]
act_np, act_mean, act_std = zs_stats(d["act_chunk"].reshape(len(d["act_chunk"]), -1))   # BC target [N,105]
state_np, state_mean, state_std = zs_stats(np.concatenate([d["cartt"], d["gript"]], 1))  # [N,8]
act = torch.tensor(act_np, device=DEV); state = torch.tensor(state_np, device=DEV)
lang = torch.tensor(l2n(embed_text(d["instr"])), device=DEV)
N = len(Ft); adim = act.shape[1]
CHUNK = 15; ADOF = adim // CHUNK
print(f"{TAG}: N={N} adim={adim} ({CHUNK}x{ADOF}) ARM={ARM}", flush=True)

# aux target for the AHA arm (far-fut-obs = a low-recov hard question)
aux_tgt = torch.tensor(l2n(lat["z_fl"]), device=DEV) if ARM == "aha" else None

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def imgs(idx):
    b = torch.from_numpy(Ft[idx]).to(DEV).float().permute(0, 3, 1, 2) / 255.0
    return (b - MEAN) / STD


class Model(nn.Module):
    def __init__(self, sdim, ldim, adim, auxdim, dm=DMODEL):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained("facebook/dinov2-small")
        ri = self.enc.config.hidden_size
        self.p_img = nn.Linear(ri, dm); self.p_state = nn.Linear(sdim, dm); self.p_lang = nn.Linear(ldim, dm)
        self.type_emb = nn.Parameter(torch.zeros(3, dm))
        layer = nn.TransformerEncoderLayer(dm, nhead=4, dim_feedforward=4*dm, batch_first=True, dropout=0.0)
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        self.adim = adim
        self.t_emb = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))
        self.flow = nn.Sequential(nn.Linear(adim + dm + 64, 512), nn.GELU(),
                                  nn.Linear(512, 512), nn.GELU(), nn.Linear(512, adim))
        self.aux = nn.Linear(dm, auxdim) if auxdim else None

    def rep(self, x, s, l):
        rimg = self.enc(pixel_values=x).last_hidden_state[:, 0]
        toks = torch.stack([self.p_img(rimg), self.p_state(s), self.p_lang(l)], 1) + self.type_emb.unsqueeze(0)
        return self.backbone(toks).mean(1)

    def velocity(self, x_t, t, cond):
        return self.flow(torch.cat([x_t, cond, self.t_emb(t)], -1))


def fm_loss(m, cond, a):
    eps = torch.randn_like(a); t = torch.rand(a.shape[0], 1, device=DEV)
    return nn.functional.mse_loss(m.velocity((1-t)*eps+t*a, t, cond), a - eps)


auxdim = aux_tgt.shape[1] if aux_tgt is not None else 0
m = Model(state.shape[1], lang.shape[1], adim, auxdim).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); mse = nn.MSELoss(); bs = 128
idx = np.arange(N)
for ep in range(EPOCHS):
    np.random.shuffle(idx); m.train(); tot = 0.0; nb = 0
    for i in range(0, N, bs):
        b = idx[i:i+bs]; x = imgs(b); loss = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rep = m.rep(x, state[b], lang[b])
            loss = loss + fm_loss(m, rep, act[b])
            if aux_tgt is not None:
                loss = loss + mse(m.aux(rep), aux_tgt[torch.tensor(b, device=DEV)])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    if ep % 5 == 0 or ep == EPOCHS-1:
        print(f"  ep {ep} loss {tot/nb:.4f}", flush=True)

ckpt = dict(state_dict={k: v.half().cpu() for k, v in m.state_dict().items()},
            arm=ARM, adim=adim, chunk=CHUNK, adof=ADOF, dmodel=DMODEL, auxdim=auxdim,
            act_mean=act_mean, act_std=act_std, state_mean=state_mean, state_std=state_std)
path = f"{OUT}/policy_{ARM}.pt"
torch.save(ckpt, path)
print(f"SAVED {path}  (adim={adim}, aux={ARM=='aha'})", flush=True)
import os as _os; _os._exit(0)
