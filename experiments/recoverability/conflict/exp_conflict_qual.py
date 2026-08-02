"""Qualitative dump for the conflict probe — save the ACTUAL predicted action vectors per condition.

Like LA4VLA Fig 5 (left): show how the action is actually generated. Train the paired policy, then for the
aligned baseline and a single-modality conflict on vision / language / state, save the raw predicted 2-D action
vectors (with the true commanded angle) so plot_conflict_qual.py can draw them.
Output: exp_conflict_qual.npz
"""
import os
import numpy as np
import torch
import torch.nn as nn

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
SEED = int(os.environ.get("SEED", 0))
DEV = os.environ.get("DEV", "cuda")
EPOCHS = int(os.environ.get("EPOCHS", 25))
N_TRAIN, N_TEST, RES = 4000, 1600, 224
torch.manual_seed(SEED); np.random.seed(SEED)
DIRS = np.arange(8) * (np.pi / 4)
DIRNAMES = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
_YY, _XX = np.mgrid[0:RES, 0:RES].astype(np.float32)
VEMB = None


def render_blob(theta, rng, blank=False):
    img = np.full((RES, RES, 3), 30, np.float32)
    if blank:
        return img.clip(0, 255).astype(np.uint8)
    R = 70 + rng.integers(-8, 9)
    cx = RES/2 + R*np.cos(theta) + rng.integers(-6, 7)
    cy = RES/2 - R*np.sin(theta) + rng.integers(-6, 7)
    sig = 16 + rng.integers(-3, 4)
    blob = np.exp(-((_XX-cx)**2 + (_YY-cy)**2)/(2*sig**2))
    col = (200 + rng.integers(-40, 40, 3)).clip(0, 255).astype(np.float32)
    return (img + blob[..., None]*col[None, None, :]).clip(0, 255).astype(np.uint8)


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


def lang_emb(thetas, removed=False):
    if removed:
        return np.zeros((len(thetas), VEMB.shape[1]), np.float32)
    idx = [int(round(t/(np.pi/4))) % 8 for t in thetas]
    return VEMB[idx]


def state_vec(thetas, removed=False, rng=None):
    n = len(thetas); s = np.zeros((n, 8), np.float32)
    if removed:
        return s
    s[:, 0] = np.cos(thetas); s[:, 1] = np.sin(thetas)
    r = rng if rng is not None else np.random.default_rng(0)
    return s + r.normal(0, 0.05, s.shape).astype(np.float32)


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
imgs = np.stack([render_blob(t, rng) for t in theta]).astype(np.uint8)
VEMB = embed_text([f"move {d}" for d in DIRNAMES]); VEMB = VEMB/(np.linalg.norm(VEMB, axis=1, keepdims=True)+1e-8)
lang = torch.tensor(lang_emb(theta), device=DEV)
state = torch.tensor(state_vec(theta, rng=rng), device=DEV)
act = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32), device=DEV)
print(f"qual dump SEED={SEED}", flush=True)

m = Model(lang.shape[1]).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
mse = nn.MSELoss(); bs = 128; idx = np.arange(N_TRAIN)
for ep in range(EPOCHS):
    np.random.shuffle(idx); m.train()
    for i in range(0, N_TRAIN, bs):
        b = idx[i:i+bs]
        opt.zero_grad(); mse(m(to_img(imgs[b]), lang[b], state[b]), act[b]).backward(); opt.step()

# fixed test set
rng_te = np.random.default_rng(SEED + 999)
theta_te = DIRS[rng_te.integers(0, 8, N_TEST)]


def predict(mod):  # mod=None -> aligned; else flip that modality to theta+pi
    tv = theta_te + np.pi if mod == "vision" else theta_te
    tl = theta_te + np.pi if mod == "language" else theta_te
    ts = theta_te + np.pi if mod == "state" else theta_te
    im = np.stack([render_blob(t, np.random.default_rng(k)) for k, t in enumerate(tv)]).astype(np.uint8)
    L = torch.tensor(lang_emb(tl), device=DEV)
    S = torch.tensor(state_vec(ts, rng=np.random.default_rng(0)), device=DEV)
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, N_TEST, bs):
            out.append(m(to_img(im[i:i+bs]), L[i:i+bs], S[i:i+bs]).cpu().numpy())
    return np.concatenate(out)


preds = {"theta_true": theta_te, "aligned": predict(None),
         "vision_flip": predict("vision"), "language_flip": predict("language"),
         "state_flip": predict("state")}
np.savez(os.path.join(OUT, "exp_conflict_qual.npz"), **preds)
for k in ["aligned", "vision_flip", "language_flip", "state_flip"]:
    u = preds[k] / (np.linalg.norm(preds[k], axis=1, keepdims=True) + 1e-8)
    tvec = np.stack([np.cos(theta_te), np.sin(theta_te)], 1)
    print(f"  {k:13s} mean cos(pred, true) = {np.mean(np.sum(u*tvec, 1)):+.3f}", flush=True)
print("SAVED exp_conflict_qual.npz", flush=True)
os._exit(0)
