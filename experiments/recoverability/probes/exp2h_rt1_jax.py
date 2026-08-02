"""Stage 1 (JAX): RT-1-style real VLA on fractal — does a LOW-recoverability aux improve OOD generalization on a
GENUINE architecture (not just the mini-VLA)? All-JAX so it lines up with JAX-based pi0.5 (openpi) later.

Arms (same RT-1 backbone, RT-1/OpenVLA 256-bin action objective):
  BC     : bin256 action cross-entropy only.
  AHA    : BC + a low-recoverability aux (predict far-future DINO latent z_fl, mse) shaping the backbone.
Report: OOD generalization = action R^2 on held-out task clusters (bins decoded to continuous), per arm.
        recoverability of the action (1 - CE_val/CE_marg) and of the aux (1 - MSE_val/MSE_marg).

Data: outputs/cache/transitions_fractal.npz (Ft frames, act_chunk, cartt, gript, instr) +
      dino_latents_fractal.npz (z_fl). Language = MiniLM embeddings (torch, CPU, one-time). Run in .venv.
env: SEED, EPOCHS(10), SUBN(6000), AUX(1=AHA,0=BC), SMOKE(1=synthetic tiny, no cache/GPU).
"""
import os, numpy as np, jax, jax.numpy as jnp, flax.linen as nn, optax
import sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
from rt1_vla_jax import RT1BackboneJAX, BinActionHeadJAX

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
SEED = int(os.environ.get("SEED", 0)); EPOCHS = int(os.environ.get("EPOCHS", 10))
SUBN = int(os.environ.get("SUBN", 6000)); AUX = int(os.environ.get("AUX", 1)); SMOKE = int(os.environ.get("SMOKE", 0))
BINS = 256; ADIM = 105; LANGD = 384; AUXD = 768
rng = np.random.default_rng(SEED)

# ---------- data ----------
if SMOKE:
    N = 240; Ft = rng.integers(0, 255, (N, 224, 224, 3), np.uint8)
    act = rng.standard_normal((N, ADIM)).astype(np.float32); lang = rng.standard_normal((N, LANGD)).astype(np.float32)
    z_fl = rng.standard_normal((N, AUXD)).astype(np.float32); g = rng.integers(0, 8, N)
else:
    _d = np.load(f"{OUT}/cache/transitions_fractal.npz", allow_pickle=True)
    _l = np.load(f"{OUT}/cache/dino_latents_fractal.npz")
    N0 = len(_d["Ft"]); sel = rng.permutation(N0)[:SUBN] if SUBN and SUBN < N0 else np.arange(N0)
    Ft = _d["Ft"][sel]
    act = _d["act_chunk"][sel].reshape(len(sel), -1).astype(np.float32)
    z_fl = (_l["z_fl"][sel] / (np.linalg.norm(_l["z_fl"][sel], axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    g = _d["ep_id"][sel]
    # MiniLM language embeddings (torch, CPU, one-time)
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
            embs.append(((h*m).sum(1)/m.sum(1).clamp(min=1e-9)).numpy())
    lang = np.concatenate(embs).astype(np.float32)
    lang = lang / (np.linalg.norm(lang, axis=1, keepdims=True) + 1e-8)

N = len(Ft)
# z-score actions; bin to 256 uniform bins on [1,99]pct of the fit split
# OOD split: cluster episodes by mean language, hold out 3 smallest clusters
from sklearn.cluster import KMeans
eps = np.unique(g); ep_lang = np.stack([lang[g == e].mean(0) for e in eps])
cl = KMeans(8, n_init=5, random_state=SEED).fit_predict(ep_lang); ood = set(np.argsort(np.bincount(cl))[:3].tolist())
epidx = {e: i for i, e in enumerate(eps)}; te = np.array([cl[epidx[e]] in ood for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
pm = rng.permutation(tr_idx); nval = max(1, int(0.15*len(pm))); fit_idx, val_idx = pm[nval:], pm[:nval]

amu, asd = act[fit_idx].mean(0), act[fit_idx].std(0) + 1e-6
lo = np.percentile(act[fit_idx], 1, 0); hi = np.percentile(act[fit_idx], 99, 0)
bins = np.clip(((act - lo) / (hi - lo + 1e-8) * BINS).astype(np.int32), 0, BINS - 1)     # [N,105]
centers = lo[:, None] + (np.arange(BINS) + 0.5) / BINS * (hi - lo)[:, None]                # [105,BINS]
print(f"N={N} fit={len(fit_idx)} val={len(val_idx)} ood={len(te_idx)} AUX={AUX} SMOKE={SMOKE}", flush=True)

MEAN = jnp.array([0.485, 0.456, 0.406]); STD = jnp.array([0.229, 0.224, 0.225])
def prep(idx):
    im = (jnp.asarray(Ft[idx], jnp.float32) / 255.0 - MEAN) / STD                          # NHWC
    return im, jnp.asarray(lang[idx]), jnp.asarray(bins[idx]), jnp.asarray(z_fl[idx])

# ---------- model ----------
class RT1VLA(nn.Module):
    @nn.compact
    def __call__(self, img, lng):
        tok = RT1BackboneJAX(d_model=512, k_tokens=8, depth=4)(img, lng)
        logits = BinActionHeadJAX(n_dims=ADIM, bins=BINS)(tok)
        aux = nn.Dense(AUXD)(tok.mean(1))
        return logits, aux

model = RT1VLA()
key = jax.random.PRNGKey(SEED)
im0, ln0, _, _ = prep(fit_idx[:2])
params = model.init(key, im0, ln0)
tx = optax.adamw(3e-4, weight_decay=1e-2); opt_state = tx.init(params)

def losses(params, im, lng, bn, zt):
    logits, aux = model.apply(params, im, lng)
    lp = jax.nn.log_softmax(logits, -1)
    ce = -jnp.take_along_axis(lp, bn[..., None], -1).mean()
    amse = ((aux - zt) ** 2).mean()
    return ce, amse

@jax.jit
def train_step(params, opt_state, im, lng, bn, zt):
    def L(p):
        ce, amse = losses(p, im, lng, bn, zt)
        return ce + (amse if AUX else 0.0), (ce, amse)
    (l, (ce, amse)), grads = jax.value_and_grad(L, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, ce, amse

BS = 32 if SMOKE else 64
for ep in range(EPOCHS):
    p = rng.permutation(fit_idx)
    for i in range(0, len(p), BS):
        b = p[i:i+BS]; im, lng, bn, zt = prep(b)
        params, opt_state, ce, amse = train_step(params, opt_state, im, lng, bn, zt)
    if ep % 3 == 0 or ep == EPOCHS-1:
        print(f"  ep{ep} CE={float(ce):.3f} auxMSE={float(amse):.3f}", flush=True)

# ---------- eval ----------
def gather(idx):
    L=[]
    for i in range(0, len(idx), 128):
        b=idx[i:i+128]; im,lng,bn,zt=prep(b); lg,ax=model.apply(params,im,lng); L.append((lg,ax,bn,zt))
    return L
def action_r2(idx):                                   # decode argmax bin -> continuous, R^2 vs true action
    num=den=0.0; ytot=[]
    for lg,ax,bn,zt in gather(idx):
        pred_bin=np.asarray(lg.argmax(-1)); ytot.append(pred_bin)
    pred=np.concatenate(ytot); yhat=np.take_along_axis(centers.T[None].repeat(len(pred),0), pred[...,None],2)[...,0] if False else \
         np.stack([centers[d][pred[:,d]] for d in range(ADIM)],1)
    Y=act[idx]; mu=act[fit_idx].mean(0)
    return 1 - ((yhat-Y)**2).mean() / (((Y-mu)**2).mean()+1e-9)
def ce_recov(idx):
    tot=n=0.0
    for lg,ax,bn,zt in gather(idx):
        lp=jax.nn.log_softmax(lg,-1); tot+=float(-jnp.take_along_axis(lp,bn[...,None],-1).sum()); n+=bn.size
    CEval=tot/n
    uni=np.bincount(bins[fit_idx].reshape(-1),minlength=BINS)+1; CEm=float(-np.log(uni/uni.sum())[bins[val_idx]].mean())
    return 1-CEval/CEm

res=dict(seed=SEED, aux=AUX, gen_ood=float(action_r2(te_idx)), recov_action=float(ce_recov(val_idx)))
print(f"RESULT arm={'AHA' if AUX else 'BC'}  OOD action R2 = {res['gen_ood']:+.4f}  |  action recoverability = {res['recov_action']:+.4f}", flush=True)
if not SMOKE:
    import json; json.dump(res, open(f"{OUT}/exp2h_rt1_jax_a{AUX}_s{SEED}.json","w"), indent=1)
    print("SAVED", flush=True)
