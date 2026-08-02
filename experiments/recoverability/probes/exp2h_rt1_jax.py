"""Stage 1 (JAX): RT-1-style real VLA on fractal — does a LOW-recoverability aux improve OOD generalization on a
GENUINE, VLA-faithful setup? All-JAX so it lines up with JAX-based pi0.5 (openpi) later.

VLA-faithful choices (per user, 2026-08-02): NO embedding cache — encode images DYNAMICALLY in the forward pass;
TRAIN the vision encoder (not frozen). The AHA aux is therefore SELF-PREDICTIVE (SPR/BYOL/DINO-WM style): predict
the FUTURE frame's representation computed ONLINE by the SAME trainable backbone, with a stop-gradient target. No
frozen DINO, no cache.

Arms (same RT-1 backbone + RT-1/OpenVLA 256-bin action objective):
  BC   : bin256 action cross-entropy only.
  AHA  : BC + self-predictive aux (predict stop-grad(backbone(future_frame)) from the current representation).
Report: OOD generalization = action R^2 on held-out task clusters (bins decoded to continuous), per arm.

Data: outputs/cache/transitions_fractal.npz (Ft + Ffl raw frames, act_chunk, instr). Only RAW FRAMES are read; all
embeddings are computed online. Language = MiniLM (torch, CPU, one-time). Run in .venv.
env: SEED, EPOCHS(10), SUBN(6000), AUX(1=AHA,0=BC), LAMBDA(1.0), SMOKE(1=synthetic tiny).
"""
import os, numpy as np, jax, jax.numpy as jnp, flax.linen as nn, optax
import sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
from rt1_vla_jax import RT1BackboneJAX, BinActionHeadJAX

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
SEED = int(os.environ.get("SEED", 0)); EPOCHS = int(os.environ.get("EPOCHS", 10))
SUBN = int(os.environ.get("SUBN", 6000)); AUX = int(os.environ.get("AUX", 1))
LAMBDA = float(os.environ.get("LAMBDA", 1.0)); SMOKE = int(os.environ.get("SMOKE", 0))
BINS = 256; ADIM = 105; LANGD = 384; DMODEL = 512
rng = np.random.default_rng(SEED)

# ---------- data (RAW FRAMES ONLY; embeddings computed online) ----------
if SMOKE:
    N = 240
    Ft = rng.integers(0, 255, (N, 224, 224, 3), np.uint8); Ffl = rng.integers(0, 255, (N, 224, 224, 3), np.uint8)
    act = rng.standard_normal((N, ADIM)).astype(np.float32); lang = rng.standard_normal((N, LANGD)).astype(np.float32)
    g = rng.integers(0, 8, N)
else:
    _d = np.load(f"{OUT}/cache/transitions_fractal.npz", allow_pickle=True)
    N0 = len(_d["Ft"]); sel = rng.permutation(N0)[:SUBN] if SUBN and SUBN < N0 else np.arange(N0)
    Ft = _d["Ft"][sel]; Ffl = _d["Ffl"][sel]                      # current + far-future RAW frames
    act = _d["act_chunk"][sel].reshape(len(sel), -1).astype(np.float32); g = _d["ep_id"][sel]
    import torch
    from transformers import AutoTokenizer, AutoModel
    tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    lm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
    embs = []
    with torch.no_grad():
        instr = [str(s) for s in _d["instr"][sel]]
        for i in range(0, len(instr), 256):
            e = tk(instr[i:i+256], padding=True, truncation=True, return_tensors="pt")
            h = lm(**e).last_hidden_state; m = e["attention_mask"][..., None].float()
            embs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy())
    lang = np.concatenate(embs).astype(np.float32); lang = lang / (np.linalg.norm(lang, 1, keepdims=True) + 1e-8)

N = len(Ft)
from sklearn.cluster import KMeans
eps = np.unique(g); ep_lang = np.stack([lang[g == e].mean(0) for e in eps])
cl = KMeans(8, n_init=5, random_state=SEED).fit_predict(ep_lang); ood = set(np.argsort(np.bincount(cl))[:3].tolist())
epi = {e: i for i, e in enumerate(eps)}; te = np.array([cl[epi[e]] in ood for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
lo = np.percentile(act[tr_idx], 1, 0); hi = np.percentile(act[tr_idx], 99, 0)
bins = np.clip(((act - lo) / (hi - lo + 1e-8) * BINS).astype(np.int32), 0, BINS - 1)
centers = lo[:, None] + (np.arange(BINS) + 0.5) / BINS * (hi - lo)[:, None]
print(f"N={N} tr={len(tr_idx)} ood={len(te_idx)} AUX={AUX} lambda={LAMBDA} SMOKE={SMOKE}", flush=True)

MEAN = jnp.array([0.485, 0.456, 0.406]); STD = jnp.array([0.229, 0.224, 0.225])
def norm(x): return (jnp.asarray(x, jnp.float32) / 255.0 - MEAN) / STD
def prep(idx): return norm(Ft[idx]), norm(Ffl[idx]), jnp.asarray(lang[idx]), jnp.asarray(bins[idx])

# ---------- model: shared trainable backbone, called twice (current + future) ----------
class RT1SelfPred(nn.Module):
    def setup(self):
        self.bb = RT1BackboneJAX(d_model=DMODEL, k_tokens=8, depth=4)
        self.binhead = BinActionHeadJAX(n_dims=ADIM, bins=BINS)
        self.pred = nn.Dense(DMODEL)                              # self-predictive predictor head
    def __call__(self, im, lng, im_fut):
        tok = self.bb(im, lng)
        logits = self.binhead(tok)
        cur = tok.mean(1)
        fut = jax.lax.stop_gradient(self.bb(im_fut, lng).mean(1))  # online future rep, stop-grad target
        return logits, self.pred(cur), fut
    def act_logits(self, im, lng):
        return self.binhead(self.bb(im, lng))

model = RT1SelfPred()
key = jax.random.PRNGKey(SEED)
im0, imf0, ln0, _ = prep(tr_idx[:2]); params = model.init(key, im0, ln0, imf0)
tx = optax.adamw(3e-4, weight_decay=1e-2); opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, im, lng, imf, bn):
    def L(p):
        logits, ap, fut = model.apply(p, im, lng, imf)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, -1), bn[..., None], -1).mean()
        aux = ((ap - fut) ** 2).mean()
        return ce + (LAMBDA * aux if AUX else 0.0), (ce, aux)
    (l, (ce, aux)), g = jax.value_and_grad(L, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, ce, aux

BS = 32 if SMOKE else 64
for ep in range(EPOCHS):
    p = rng.permutation(tr_idx)
    for i in range(0, len(p), BS):
        b = p[i:i+BS]; im, imf, lng, bn = prep(b)
        params, opt_state, ce, aux = train_step(params, opt_state, im, lng, imf, bn)
    if ep % 3 == 0 or ep == EPOCHS - 1:
        print(f"  ep{ep} CE={float(ce):.3f} auxMSE={float(aux):.3f}", flush=True)

# ---------- eval: OOD action R^2 (decode bins) ----------
@jax.jit
def logits_fn(params, im, lng): return model.apply(params, im, lng, method=RT1SelfPred.act_logits)
def action_r2(idx):
    preds = []
    for i in range(0, len(idx), 128):
        b = idx[i:i+128]; lg = logits_fn(params, norm(Ft[b]), jnp.asarray(lang[b]))
        preds.append(np.asarray(lg.argmax(-1)))
    pb = np.concatenate(preds); yhat = np.stack([centers[d][pb[:, d]] for d in range(ADIM)], 1)
    Y = act[idx]; mu = act[tr_idx].mean(0)
    return 1 - ((yhat - Y) ** 2).mean() / (((Y - mu) ** 2).mean() + 1e-9)

res = dict(seed=SEED, aux=AUX, lam=LAMBDA, gen_ood=float(action_r2(te_idx)))
print(f"RESULT arm={'AHA' if AUX else 'BC'}  OOD action R2 = {res['gen_ood']:+.4f}", flush=True)
if not SMOKE:
    import json; json.dump(res, open(f"{OUT}/exp2h_rt1_jax_a{AUX}_s{SEED}.json", "w"), indent=1); print("SAVED", flush=True)
