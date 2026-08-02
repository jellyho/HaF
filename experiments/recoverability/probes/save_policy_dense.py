"""Train the mini-VLA on the DENSE scaled cache with a TRAINABLE DINOv2-small encoder (like a real VLA),
streaming JPEG-compressed frames and encoding them on the fly. Saves a checkpoint 100% compatible with
mini_policy_server.py (same Model). ARM=bc|aha. Reads cache/dense_fractal_{MAX_EP}.npz.
Output: outputs/policy_{ARM}.pt   env: ARM, MAX_EP, EPOCHS(default 12), SEED, SUBN, BS(default 128), WORKERS(default 8).
"""
import os, numpy as np, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", 12))
SEED   = int(os.environ.get("SEED", 0))
SUBN   = int(os.environ.get("SUBN", 0))
ARM    = os.environ.get("ARM", "bc")
MAX_EP = int(os.environ.get("MAX_EP", 20000))
BS     = int(os.environ.get("BS", 128))
WORKERS= int(os.environ.get("WORKERS", 8))
DMODEL = 256
torch.manual_seed(SEED); np.random.seed(SEED)

def l2n(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
def zs_stats(a):
    a = np.asarray(a, float).reshape(len(a), -1); m, s = a.mean(0), a.std(0) + 1e-8
    return ((a - m) / s).astype(np.float32), m.astype(np.float32), s.astype(np.float32)

def embed_text_unique(strings):
    from transformers import AutoTokenizer, AutoModel
    uniq = sorted(set(strings)); u2i = {u: i for i, u in enumerate(uniq)}
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    lm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(uniq), 256):
            e = tok(uniq[i:i+256], padding=True, truncation=True, return_tensors="pt").to(DEV)
            h = lm(**e).last_hidden_state; mask = e["attention_mask"].unsqueeze(-1).float()
            embs.append(((h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)).cpu().numpy())
    U = l2n(np.concatenate(embs).astype(np.float32))
    return U[np.array([u2i[s] for s in strings])]                 # [N,384]

# ---- data ----
cache = os.path.join(OUT, f"cache/dense_fractal_{MAX_EP}.npz")
_d = np.load(cache, allow_pickle=True)
_N0 = len(_d["Ft_jpg"])
_sel = (np.random.default_rng(123).permutation(_N0)[:SUBN] if (SUBN and SUBN < _N0) else np.arange(_N0))
ft_jpg = _d["Ft_jpg"][_sel]
act_np, act_mean, act_std = zs_stats(_d["act_chunk"][_sel].reshape(len(_sel), -1))
state_np, state_mean, state_std = zs_stats(np.concatenate([_d["cartt"][_sel], _d["gript"][_sel]], 1))
lang_np = embed_text_unique([str(s) for s in _d["instr"][_sel]])
act = torch.tensor(act_np, device=DEV); state = torch.tensor(state_np, device=DEV)
lang = torch.tensor(lang_np, device=DEV)
N = len(ft_jpg); adim = act.shape[1]; CHUNK = 15; ADOF = adim // CHUNK
aux_tgt = torch.tensor(l2n(_d["z_fl"][_sel]), device=DEV) if ARM == "aha" else None
print(f"DENSE-trainable: N={N} adim={adim} ARM={ARM} cache={MAX_EP}ep epochs={EPOCHS} bs={BS}", flush=True)

class JpgDS(Dataset):
    def __init__(self, jpgs): self.j = jpgs
    def __len__(self): return len(self.j)
    def __getitem__(self, i):
        im = cv2.imdecode(np.frombuffer(self.j[i], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]  # BGR->RGB
        return i, torch.from_numpy(np.ascontiguousarray(im))                                  # idx, [224,224,3] uint8
loader = DataLoader(JpgDS(ft_jpg), batch_size=BS, shuffle=True, num_workers=WORKERS,
                    pin_memory=True, drop_last=False, persistent_workers=(WORKERS > 0))

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

MEAN = torch.tensor([0.485,0.456,0.406], device=DEV).view(1,3,1,1)
STD  = torch.tensor([0.229,0.224,0.225], device=DEV).view(1,3,1,1)
auxdim = aux_tgt.shape[1] if aux_tgt is not None else 0
m = Model(state.shape[1], lang.shape[1], adim, auxdim).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); mse = nn.MSELoss()
for ep in range(EPOCHS):
    m.train(); tot = 0.0; nb = 0
    for idx, im in loader:
        idx = idx.to(DEV); x = im.to(DEV, non_blocking=True).float().permute(0,3,1,2)/255.0; x = (x-MEAN)/STD
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=(DEV=="cuda")):
            rep = m.rep(x, state[idx], lang[idx])
            loss = fm_loss(m, rep, act[idx])
            if aux_tgt is not None: loss = loss + mse(m.aux(rep), aux_tgt[idx])
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    print(f"  ep {ep} loss {tot/max(nb,1):.4f}", flush=True)

ckpt = dict(state_dict={k: v.half().cpu() for k, v in m.state_dict().items()},
            arm=ARM, adim=adim, chunk=CHUNK, adof=ADOF, dmodel=DMODEL, auxdim=auxdim,
            act_mean=act_mean, act_std=act_std, state_mean=state_mean, state_std=state_std)
path = f"{OUT}/policy_{ARM}.pt"
torch.save(ckpt, path)
print(f"SAVED {path}  (adim={adim}, aux={ARM=='aha'}, N={N})", flush=True)
import os as _os; _os._exit(0)
