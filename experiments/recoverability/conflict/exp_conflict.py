"""PRELIMINARY EXPERIMENT — LA4VLA §3, reproduced exactly, then generalized from vision to every modality.

This is HaF's motivating (preliminary) probe, in the spirit of LA4VLA §3. LA4VLA trains a policy on paired
(instruction, vision) atomic direction-following, then perturbs the VISUAL input in four controlled ways and
measures whether the action still follows the instruction — showing the policy leans on the paired visual cue
rather than the language. We reproduce that protocol faithfully and apply the SAME four perturbations to EACH
input modality {vision, language, state}, so whichever modality the policy actually leans on is exposed — not
just vision.

Task (as in LA4VLA): predict a 2-D unit action = one of 8 commanded atomic directions theta ("move in
direction theta"). PAIRED training — all three inputs carry the TRUE theta, always; there is no noise regime,
so the policy *could* read theta from any modality. The perturbations reveal which one it committed to.
  vision  : a blob at angle theta on a ring        (trainable DINOv2-small)
  language: "move {east|northeast|...}" at theta    (MiniLM embedding)
  state   : [cos theta, sin theta, 0..] (+ noise)   (proprio-like)

Four perturbation conditions (LA4VLA), applied to ONE modality m at a time, others kept at the true theta:
  +  paired    : m carries the true theta                       (V+ baseline)
  0  removed   : m masked out (blank image / zero embed / zero state)     (the ∅ condition)
  ~  unaligned : m carries an INDEPENDENT random direction      (same "scene", wrong pairing; Ṽ)
  -  conflict  : m carries the OPPOSITE direction theta+pi       (V-)
If the policy EXPLOITS m, its action is dragged toward the perturbed cue (DAR falls, < 0.5 under conflict);
if it IGNORES m, the action stays aligned to the true theta across all four conditions.

Metrics (LA4VLA): DAR = fraction of predicted actions in the instruction-aligned half-space (cos>0);
DCS = mean cosine(a_hat, theta_hat); SR = between-direction / within-direction spread of predicted actions
grouped by true theta; SS = silhouette of predicted actions clustered by true theta.
Output: exp_conflict_s{SEED}.json  (results[modality][condition] = {DAR,DCS,SR,SS}).
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
SEED = int(os.environ.get("SEED", 0))
DEV = os.environ.get("DEV", "cuda")
EPOCHS = int(os.environ.get("EPOCHS", 25))
N_TRAIN, N_TEST, RES = 4000, 1600, 224
torch.manual_seed(SEED); np.random.seed(SEED)

DIRS = np.arange(8) * (np.pi / 4)
DIRNAMES = ["east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast"]
MODS = ["vision", "language", "state"]
CONDS = ["paired", "removed", "unaligned", "conflict"]
_YY, _XX = np.mgrid[0:RES, 0:RES].astype(np.float32)
VEMB = None  # instruction-vocabulary embeddings, filled after embed


def render_blob(theta, rng, blank=False):
    img = np.full((RES, RES, 3), 30, np.float32)
    if blank:                              # 'removed' vision: background only, no directional blob
        return img.clip(0, 255).astype(np.uint8)
    R = 70 + rng.integers(-8, 9)
    cx = RES / 2 + R * np.cos(theta) + rng.integers(-6, 7)
    cy = RES / 2 - R * np.sin(theta) + rng.integers(-6, 7)
    sig = 16 + rng.integers(-3, 4)
    blob = np.exp(-((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * sig ** 2))
    col = (200 + rng.integers(-40, 40, 3)).clip(0, 255).astype(np.float32)
    return (img + blob[..., None] * col[None, None, :]).clip(0, 255).astype(np.uint8)


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
    idx = [int(round(t / (np.pi/4))) % 8 for t in thetas]
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


def to_img(arr):
    b = torch.from_numpy(arr).to(DEV).float().permute(0, 3, 1, 2) / 255.0
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


# ---------- PAIRED training data (all cues = true theta; LA4VLA's paired setup) ----------
rng = np.random.default_rng(SEED)
theta = DIRS[rng.integers(0, 8, N_TRAIN)]
imgs = np.stack([render_blob(t, rng) for t in theta]).astype(np.uint8)
VEMB = embed_text([f"move {d}" for d in DIRNAMES])
VEMB = VEMB / (np.linalg.norm(VEMB, axis=1, keepdims=True) + 1e-8)
lang = torch.tensor(lang_emb(theta), device=DEV)
state = torch.tensor(state_vec(theta, rng=rng), device=DEV)
act = torch.tensor(np.stack([np.cos(theta), np.sin(theta)], 1).astype(np.float32), device=DEV)
print(f"SEED={SEED} paired training N={N_TRAIN}", flush=True)

m = Model(lang.shape[1]).to(DEV)
opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2)
mse = nn.MSELoss(); bs = 128; idx = np.arange(N_TRAIN)
for ep in range(EPOCHS):
    np.random.shuffle(idx); m.train()
    for i in range(0, N_TRAIN, bs):
        b = idx[i:i+bs]
        opt.zero_grad(); mse(m(to_img(imgs[b]), lang[b], state[b]), act[b]).backward(); opt.step()


# ---------- evaluation: 4 conditions x 3 modalities on a fresh test set ----------
rng_te = np.random.default_rng(SEED + 999)
theta_te = DIRS[rng_te.integers(0, 8, N_TEST)]


def perturbed_theta(base, cond, rng):
    if cond in ("paired", "removed"):
        return base                                    # removed handled by masking; cue angle irrelevant
    if cond == "conflict":
        return base + np.pi                            # opposite direction
    return DIRS[rng.integers(0, 8, len(base))]         # unaligned: independent random direction


def predict(mod, cond):
    r = np.random.default_rng(SEED + 7)
    tv = perturbed_theta(theta_te, cond, r) if mod == "vision" else theta_te
    tl = perturbed_theta(theta_te, cond, r) if mod == "language" else theta_te
    ts = perturbed_theta(theta_te, cond, r) if mod == "state" else theta_te
    rm_v = cond == "removed" and mod == "vision"
    rm_l = cond == "removed" and mod == "language"
    rm_s = cond == "removed" and mod == "state"
    imgs_e = np.stack([render_blob(t, np.random.default_rng(k), blank=rm_v)
                       for k, t in enumerate(tv)]).astype(np.uint8)
    L = torch.tensor(lang_emb(tl, removed=rm_l), device=DEV)
    S = torch.tensor(state_vec(ts, removed=rm_s, rng=np.random.default_rng(0)), device=DEV)
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, N_TEST, bs):
            out.append(m(to_img(imgs_e[i:i+bs]), L[i:i+bs], S[i:i+bs]).cpu().numpy())
    return np.concatenate(out)


def metrics(pred, theta_true):
    u = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    tvec = np.stack([np.cos(theta_true), np.sin(theta_true)], 1)
    cos = np.sum(u * tvec, axis=1)
    DAR = float(np.mean(cos > 0)); DCS = float(np.mean(cos))
    lab = (theta_true / (np.pi/4)).round().astype(int) % 8
    cents = np.stack([pred[lab == c].mean(0) if (lab == c).any() else np.zeros(2) for c in range(8)])
    within = float(np.mean([np.linalg.norm(pred[lab == c] - cents[c], axis=1).mean()
                            for c in range(8) if (lab == c).any()]))
    between = float(np.mean([np.linalg.norm(cents[a] - cents[b])
                             for a in range(8) for b in range(8) if a < b]))
    SR = float(between / (within + 1e-8))
    try:
        SS = float(silhouette_score(pred, lab)) if len(set(lab)) > 1 else float("nan")
    except Exception:
        SS = float("nan")
    return dict(DAR=DAR, DCS=DCS, SR=SR, SS=SS)


res = {"seed": SEED, "protocol": "LA4VLA 4-condition (paired/removed/unaligned/conflict), per modality",
       "results": {}}
for mod in MODS:
    res["results"][mod] = {}
    for cond in CONDS:
        mt = metrics(predict(mod, cond), theta_te)
        res["results"][mod][cond] = mt
        print(f"  {mod:8s} {cond:9s} DAR={mt['DAR']:.3f} DCS={mt['DCS']:+.3f} "
              f"SR={mt['SR']:.2f} SS={mt['SS']:+.3f}", flush=True)

json.dump(res, open(os.path.join(OUT, f"exp_conflict_s{SEED}.json"), "w"), indent=2)
print("SAVED", f"exp_conflict_s{SEED}.json", flush=True)
os._exit(0)
