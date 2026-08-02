"""Stage 1 (JAX): RT-1-style backbone + FLOW-MATCHING action loss (pi0/pi0.5/SmolVLA style), VLA-faithful.

Backbone: RT-1 recipe (TRAINABLE image encoder -> FiLM(language) -> TokenLearner -> Transformer -> tokens).
Action objective: FLOW MATCHING on the continuous action chunk (not discrete bins).
AHA aux: predict the future frame's embedding from a FROZEN pretrained encoder (DINOv2), computed ONLINE (no cache).
Rationale (user): we don't want to LEARN a good image representation (so not SPR/BYOL/DINO-WM self-prediction) — we
want a stable, meaningful, low-recoverability grounding TARGET that helps VLA training. A fixed encoder gives that;
computing it online (per batch) keeps it VLA-faithful (no embedding cache) while the VLA's OWN encoder is trained.

Arms (same backbone):  BC = flow-matching action only.   AHA = BC + predict frozen-DINO(future frame).
Report: OOD generalization = action R^2 on held-out task clusters (flow-sampled), per arm.
Data: outputs/cache/transitions_fractal.npz (Ft + Ffl RAW frames, act_chunk, instr). Language = MiniLM (torch CPU).
env: SEED, EPOCHS(10), SUBN(6000), AUX(1|0), LAMBDA(1.0), SMOKE(1). Run in .venv.
"""
import os, numpy as np, jax, jax.numpy as jnp, flax.linen as nn, optax
import sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
from rt1_vla_jax import RT1BackboneJAX

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
SEED = int(os.environ.get("SEED", 0)); EPOCHS = int(os.environ.get("EPOCHS", 10))
SUBN = int(os.environ.get("SUBN", 6000)); AUX = int(os.environ.get("AUX", 1))
LAMBDA = float(os.environ.get("LAMBDA", 1.0)); SMOKE = int(os.environ.get("SMOKE", 0))
ADIM = 105; LANGD = 384; DMODEL = 512
rng = np.random.default_rng(SEED)

# ---------- data (RAW FRAMES ONLY; embeddings online) ----------
if SMOKE:
    N = 240
    Ft = rng.integers(0, 255, (N, 224, 224, 3), np.uint8); Ffl = rng.integers(0, 255, (N, 224, 224, 3), np.uint8)
    act = rng.standard_normal((N, ADIM)).astype(np.float32); lang = rng.standard_normal((N, LANGD)).astype(np.float32)
    g = rng.integers(0, 8, N)
else:
    _d = np.load(f"{OUT}/cache/transitions_fractal.npz", allow_pickle=True)
    N0 = len(_d["Ft"]); sel = rng.permutation(N0)[:SUBN] if SUBN and SUBN < N0 else np.arange(N0)
    Ft = _d["Ft"][sel]; Ffl = _d["Ffl"][sel]
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
eps_u = np.unique(g); ep_lang = np.stack([lang[g == e].mean(0) for e in eps_u])
cl = KMeans(8, n_init=5, random_state=SEED).fit_predict(ep_lang); ood = set(np.argsort(np.bincount(cl))[:3].tolist())
epi = {e: i for i, e in enumerate(eps_u)}; te = np.array([cl[epi[e]] in ood for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
amu, asd = act[tr_idx].mean(0), act[tr_idx].std(0) + 1e-6
act_n = ((act - amu) / asd).astype(np.float32)
print(f"N={N} tr={len(tr_idx)} ood={len(te_idx)} AUX={AUX} lambda={LAMBDA} SMOKE={SMOKE}", flush=True)

MEAN = jnp.array([0.485, 0.456, 0.406]); STD = jnp.array([0.229, 0.224, 0.225])
def norm(x): return (jnp.asarray(x, jnp.float32) / 255.0 - MEAN) / STD          # NHWC, ImageNet norm

# ---------- FROZEN target encoder (online, no cache): DINOv2 embedding of the future frame ----------
AUXTGT_DIM = 384
if SMOKE:
    _W = jax.random.normal(jax.random.PRNGKey(7), (3 * 14 * 14, AUXTGT_DIM)) * 0.02   # deterministic frozen stub
    def dino_embed(nhwc):
        B = nhwc.shape[0]; x = jax.image.resize(nhwc, (B, 14, 14, 3), "linear").reshape(B, -1)
        return jax.lax.stop_gradient(x @ _W)
else:
    from transformers import FlaxDinov2Model
    _dino = FlaxDinov2Model.from_pretrained("facebook/dinov2-small", from_pt=True)
    @jax.jit
    def dino_embed(nhwc):                                                        # frozen, online
        nchw = jnp.transpose(nhwc, (0, 3, 1, 2))
        return jax.lax.stop_gradient(_dino(pixel_values=nchw).last_hidden_state[:, 0])

def time_emb(t, dim=64):                                          # t [B,1] -> [B,dim]
    half = dim // 2; fr = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / half)
    a = t * fr[None]; return jnp.concatenate([jnp.cos(a), jnp.sin(a)], -1)


class FlowActionHead(nn.Module):
    adim: int
    @nn.compact
    def __call__(self, x_t, t, cond):                            # x_t[B,adim] t[B,1] cond[B,d] -> velocity[B,adim]
        h = jnp.concatenate([x_t, cond, time_emb(t, 64)], -1)
        h = nn.gelu(nn.Dense(512)(h)); h = nn.gelu(nn.Dense(512)(h))
        return nn.Dense(self.adim)(h)


class RT1Flow(nn.Module):
    def setup(self):
        self.bb = RT1BackboneJAX(d_model=DMODEL, k_tokens=8, depth=4)
        self.flow = FlowActionHead(ADIM)
        self.pred = nn.Dense(AUXTGT_DIM)                          # predicts the frozen-DINO future embedding
    def __call__(self, im, lng, x_t, t):                         # training: velocity + aux prediction
        cond = self.bb(im, lng).mean(1)
        return self.flow(x_t, t, cond), self.pred(cond)
    def cond(self, im, lng): return self.bb(im, lng).mean(1)
    def vel(self, x_t, t, cond): return self.flow(x_t, t, cond)

model = RT1Flow()
key = jax.random.PRNGKey(SEED)
im0, ln0 = norm(Ft[tr_idx[:2]]), jnp.asarray(lang[tr_idx[:2]])
params = model.init(key, im0, ln0, jnp.zeros((2, ADIM)), jnp.zeros((2, 1)))
tx = optax.adamw(3e-4, weight_decay=1e-2); opt_state = tx.init(params)

@jax.jit
def train_step(params, opt_state, key, im, lng, a, fut):
    k1, k2 = jax.random.split(key)
    epsn = jax.random.normal(k1, a.shape); t = jax.random.uniform(k2, (a.shape[0], 1))
    x_t = (1 - t) * epsn + t * a
    def L(p):
        vel, ap = model.apply(p, im, lng, x_t, t)
        fm = ((vel - (a - epsn)) ** 2).mean()
        aux = ((ap - fut) ** 2).mean()
        return fm + (LAMBDA * aux if AUX else 0.0), (fm, aux)
    (l, (fm, aux)), g = jax.value_and_grad(L, has_aux=True)(params)
    upd, opt_state = tx.update(g, opt_state, params)
    return optax.apply_updates(params, upd), opt_state, fm, aux

BS = 32 if SMOKE else 64
for ep in range(EPOCHS):
    p = rng.permutation(tr_idx)
    for i in range(0, len(p), BS):
        b = p[i:i+BS]; key, sk = jax.random.split(key)
        fut = dino_embed(norm(Ffl[b])) if AUX else jnp.zeros((len(b), AUXTGT_DIM))   # frozen DINO, ONLINE
        params, opt_state, fm, aux = train_step(params, opt_state, sk, norm(Ft[b]), jnp.asarray(lang[b]), jnp.asarray(act_n[b]), fut)
    if ep % 3 == 0 or ep == EPOCHS - 1:
        print(f"  ep{ep} flow={float(fm):.3f} auxMSE={float(aux):.3f}", flush=True)

# ---------- eval: flow-sample action, OOD R^2 ----------
@jax.jit
def cond_fn(params, im, lng): return model.apply(params, im, lng, method=RT1Flow.cond)
@jax.jit
def step_fn(params, x, t, cond): return model.apply(params, x, t, cond, method=RT1Flow.vel)
def sample(params, cond, key, K=10, S=8):
    B = cond.shape[0]; c = jnp.repeat(cond, S, 0); x = jax.random.normal(key, (B * S, ADIM))
    for k in range(K):
        t = jnp.full((B * S, 1), k / K); x = x + (1.0 / K) * step_fn(params, x, t, c)
    return x.reshape(B, S, ADIM).mean(1)
def action_r2(idx):
    preds = []; kk = jax.random.PRNGKey(123)
    for i in range(0, len(idx), 128):
        b = idx[i:i+128]; kk, sk = jax.random.split(kk)
        cond = cond_fn(params, norm(Ft[b]), jnp.asarray(lang[b]))
        preds.append(np.asarray(sample(params, cond, sk)))
    yhat = np.concatenate(preds) * asd + amu; Y = act[idx]; mu = act[tr_idx].mean(0)
    return 1 - ((yhat - Y) ** 2).mean() / (((Y - mu) ** 2).mean() + 1e-9)

res = dict(seed=SEED, aux=AUX, lam=LAMBDA, gen_ood=float(action_r2(te_idx)))
print(f"RESULT arm={'AHA' if AUX else 'BC'}  OOD action R2 (flow) = {res['gen_ood']:+.4f}", flush=True)
if not SMOKE:
    import json; json.dump(res, open(f"{OUT}/exp2h_rt1_jax_a{AUX}_s{SEED}.json", "w"), indent=1); print("SAVED", flush=True)
