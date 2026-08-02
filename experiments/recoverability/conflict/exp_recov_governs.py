"""Completing the LA4VLA line: RECOVERABILITY governs WHICH modality a policy exploits.

The preliminary probe (exp_conflict.py) showed a paired-trained policy commits to vision and ignores language
AND state. The open question it left: is that because vision is special, or because vision was the most
RECOVERABLE (cheapest-to-read) path? Here we answer it. All three modalities ALWAYS carry the true direction θ
(no reliability trick — this is not the old tautological regime); we only vary each modality's RECOVERABILITY =
its signal-to-noise (how cheaply θ can be extracted). In three regimes a different modality is made the cheapest:
  V : vision high-SNR,   language low-SNR, state low-SNR
  L : vision low-SNR,    language high-SNR, state low-SNR
  S : vision low-SNR,    language low-SNR, state high-SNR
If recoverability governs, the EXPLOITED modality (max reliance = DAR(paired) − DAR(conflict)) rotates V→L→S,
even though every modality still fully determines θ. Output: exp_recov_{REGIME}_s{SEED}.json.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
SEED = int(os.environ.get("SEED", 0))
REGIME = os.environ.get("REGIME", "V")
DEV = os.environ.get("DEV", "cuda")
EPOCHS = int(os.environ.get("EPOCHS", 25))
N_TRAIN, N_TEST, RES = 4000, 1600, 224
torch.manual_seed(SEED); np.random.seed(SEED)

DIRS = np.arange(8) * (np.pi / 4)
DIRNAMES = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
MODS = ["vision", "language", "state"]
# per-regime (bright, lang_scale, state_amp) — HIGH ≈ recoverable, LOW ≈ buried in noise (still carries θ)
# vision's trainable DINOv2 path is high-capacity, so its LOW must be aggressive to drop below language/state.
HI, LO = 1.0, 0.035
SNR = {"V": (HI, LO, LO), "L": (LO, HI, LO), "S": (LO, LO, HI)}[REGIME]
BRIGHT, LSCALE, SAMP = SNR
_YY, _XX = np.mgrid[0:RES, 0:RES].astype(np.float32)
VEMB = None


def render_blob(theta, rng, bright=1.0, blank=False):
    img = np.full((RES, RES, 3), 30, np.float32) + rng.normal(0, 20, (RES, RES, 3)).astype(np.float32)
    if not blank:
        R = 70 + rng.integers(-8, 9)
        cx = RES/2 + R*np.cos(theta) + rng.integers(-6, 7)
        cy = RES/2 - R*np.sin(theta) + rng.integers(-6, 7)
        sig = 16 + rng.integers(-3, 4)
        blob = np.exp(-((_XX-cx)**2 + (_YY-cy)**2)/(2*sig**2))
        col = (200 + rng.integers(-40, 40, 3)).astype(np.float32)
        img = img + bright * blob[..., None] * col[None, None, :]
    return img.clip(0, 255).astype(np.uint8)


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


def lang_emb(thetas, scale, rng, removed=False):
    if removed:
        return np.zeros((len(thetas), VEMB.shape[1]), np.float32)
    idx = [int(round(t/(np.pi/4))) % 8 for t in thetas]
    return (scale * VEMB[idx] + rng.normal(0, 0.10, (len(thetas), VEMB.shape[1]))).astype(np.float32)


def state_vec(thetas, amp, rng, removed=False):
    n = len(thetas); s = np.zeros((n, 8), np.float32)
    if removed:
        return s
    s[:, 0] = amp*np.cos(thetas); s[:, 1] = amp*np.sin(thetas)
    return (s + rng.normal(0, 0.30, s.shape)).astype(np.float32)


MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def to_img(a):
    b = torch.from_numpy(a).to(DEV).float().permute(0, 3, 1, 2) / 255.0
    return (b - MEAN) / STD


class Model(nn.Module):
    def __init__(self, ldim):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained("facebook/dinov2-small")
        r = self.enc.config.hidden_size
        self.head = nn.Sequential(nn.Linear(r + ldim + 8, 256), nn.GELU(), nn.Linear(256, 2))

    def forward(self, img, lang, state):
        r = self.enc(pixel_values=img).last_hidden_state[:, 0]
        return self.head(torch.cat([r, lang, state], 1))


rng = np.random.default_rng(SEED)
theta = DIRS[rng.integers(0, 8, N_TRAIN)]
imgs = np.stack([render_blob(t, rng, bright=BRIGHT) for t in theta]).astype(np.uint8)
VEMB = embed_text([f"move {d}" for d in DIRNAMES]); VEMB = VEMB/(np.linalg.norm(VEMB, axis=1, keepdims=True)+1e-8)
lang = torch.tensor(lang_emb(theta, LSCALE, rng), device=DEV)
state = torch.tensor(state_vec(theta, SAMP, rng), device=DEV)
act = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32), device=DEV)
print(f"REGIME={REGIME} SEED={SEED} SNR(bright,lscale,samp)={SNR}", flush=True)

m = Model(lang.shape[1]).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
mse = nn.MSELoss(); bs = 128; idx = np.arange(N_TRAIN)
for ep in range(EPOCHS):
    np.random.shuffle(idx); m.train()
    for i in range(0, N_TRAIN, bs):
        b = idx[i:i+bs]
        opt.zero_grad(); mse(m(to_img(imgs[b]), lang[b], state[b]), act[b]).backward(); opt.step()

rng_te = np.random.default_rng(SEED + 999)
theta_te = DIRS[rng_te.integers(0, 8, N_TEST)]


def predict(mod, cond):
    r = np.random.default_rng(SEED + 7)
    def pt(base):
        return base + np.pi if cond == "conflict" else base
    tv, tl, ts = (pt(theta_te) if mod == "vision" else theta_te), \
                 (pt(theta_te) if mod == "language" else theta_te), \
                 (pt(theta_te) if mod == "state" else theta_te)
    im = np.stack([render_blob(t, np.random.default_rng(k), bright=BRIGHT) for k, t in enumerate(tv)]).astype(np.uint8)
    L = torch.tensor(lang_emb(tl, LSCALE, np.random.default_rng(1)), device=DEV)
    S = torch.tensor(state_vec(ts, SAMP, np.random.default_rng(2)), device=DEV)
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, N_TEST, bs):
            out.append(m(to_img(im[i:i+bs]), L[i:i+bs], S[i:i+bs]).cpu().numpy())
    return np.concatenate(out)


def dar(pred, th):
    u = pred/(np.linalg.norm(pred, axis=1, keepdims=True)+1e-8)
    return float(np.mean(np.sum(u*np.stack([np.cos(th), np.sin(th)], 1), 1) > 0))


res = {"seed": SEED, "regime": REGIME, "snr": {"bright": BRIGHT, "lscale": LSCALE, "samp": SAMP}, "reliance": {}}
for mod in MODS:
    rl = dar(predict(mod, "paired"), theta_te) - dar(predict(mod, "conflict"), theta_te)
    res["reliance"][mod] = rl
    print(f"  {mod:8s} reliance={rl:+.3f}", flush=True)
res["exploited"] = max(MODS, key=lambda x: res["reliance"][x])
print(f"  -> exploited: {res['exploited']}", flush=True)
json.dump(res, open(os.path.join(OUT, f"exp_recov_{REGIME}_s{SEED}.json"), "w"), indent=2)
print("SAVED", flush=True)
os._exit(0)
