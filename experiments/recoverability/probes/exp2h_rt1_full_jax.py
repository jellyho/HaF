"""RT-1-style VLA (JAX), FULL fractal dataset, STREAMING (no cache) — trains a policy competent enough for a real
SimplerEnv rollout. All continuous (flow-matching action + frozen-DINO future-frame aux). Shared trunk, multi-objective.

Data: streams the FULL RT-1/fractal TFRecords directly (no npz cache) -> raw frames encoded online. Language via
MiniLM (torch CPU) with a small instruction->embedding lookup. AHA aux target = frozen DINOv2(future frame), online.
OOD: instructions hashed to a held-out 20% for an offline generalization read; the real test is SimplerEnv (Stage 2).

env: MAX_EP(0=all 87k), EPOCHS(2), N_T(24), AUX(1|0), LAMBDA(1.0), BS(64), SMOKE(1=synthetic). Run in .venv.
Saves params to outputs/rt1_full_a{AUX}.msgpack for serving.
"""
import os, glob, numpy as np, jax, jax.numpy as jnp, flax.linen as nn, optax
import tensorflow as tf, tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')                          # TF reads on CPU; GPU is JAX's
import cv2, sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
from rt1_vla_jax import RT1BackboneJAX

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
DATA = "/data5/jellyho/Hindsight/fractal_rlds/fractal20220817_data/0.1.0"
MAX_EP = int(os.environ.get("MAX_EP", 0)); EPOCHS = int(os.environ.get("EPOCHS", 2))
N_T = int(os.environ.get("N_T", 24)); AUX = int(os.environ.get("AUX", 1)); LAMBDA = float(os.environ.get("LAMBDA", 1.0))
BS = int(os.environ.get("BS", 64)); SMOKE = int(os.environ.get("SMOKE", 0)); SEED = int(os.environ.get("SEED", 0))
ADIM, LANGD, DMODEL, AUXTGT_DIM, HCHUNK, MINLEN = 105, 384, 512, 384, 15, 30
KLARGE = int(os.environ.get("AUX_OFFSET", 30))                   # future-frame offset for AHA aux (t+KLARGE); 5=near, 30=far
rng = np.random.default_rng(SEED)

# ---------- language: MiniLM embedding lookup (torch CPU, lazily cached per unique instruction) ----------
import torch
from transformers import AutoTokenizer, AutoModel
_tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_lm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
_lang_cache = {}
def lang_emb(instr):
    if instr not in _lang_cache:
        with torch.no_grad():
            e = _tk([instr], padding=True, truncation=True, return_tensors="pt")
            h = _lm(**e).last_hidden_state; m = e["attention_mask"][..., None].float()
            v = ((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).numpy()[0]
        _lang_cache[instr] = (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
    return _lang_cache[instr]

def resize(im): return cv2.resize(np.asarray(im), (224, 224), interpolation=cv2.INTER_AREA)
def act_vec(a): return np.concatenate([np.asarray(a["world_vector"], np.float32),
    np.asarray(a["rotation_delta"], np.float32), np.asarray(a["gripper_closedness_action"], np.float32).reshape(-1)[:1]])
def chunk(acts, t):
    c = acts[t:t+HCHUNK]
    if len(c) < HCHUNK: c = np.concatenate([c, np.repeat(c[-1:], HCHUNK-len(c), 0)], 0)
    return c.astype(np.float32).reshape(-1)
def is_ood(instr): return (hash(instr) % 5 == 0)                  # ~20% instructions held out

# ---------- streaming transition generator over the FULL dataset ----------
def transitions(max_ep):
    if SMOKE:
        for _ in range(240):
            yield (rng.integers(0,255,(224,224,3),np.uint8), rng.integers(0,255,(224,224,3),np.uint8),
                   rng.standard_normal(ADIM).astype(np.float32), "pick object", False)
        return
    shards = [s for s in sorted(glob.glob(DATA + "/fractal*train.tfrecord-*")) if ".gstmp" not in s]
    feats = tfds.builder_from_directory(DATA).info.features
    ds = tf.data.TFRecordDataset(shards).map(feats.deserialize_example)
    used = 0
    for ep in ds:
        if max_ep and used >= max_ep: break
        steps = list(ep["steps"]); T = len(steps)
        if T < MINLEN: continue
        frames = [s["observation"]["image"].numpy() for s in steps]
        acts = np.stack([act_vec(s["action"]) for s in steps])
        instr = steps[0]["observation"]["natural_language_instruction"].numpy().decode("utf-8", "ignore").strip()
        for t in np.unique(np.linspace(3, T-3, N_T).astype(int)):
            t = int(t); tfl = min(t + KLARGE, T - 1)
            yield resize(frames[t]), resize(frames[tfl]), chunk(acts, t), instr, is_ood(instr)
        used += 1

OOD_EV = []                                                       # held-out (instruction-hash) transitions for eval
def batches(max_ep, bufsize=4096):
    buf = []
    for tr in transitions(max_ep):
        if tr[4]:                                                 # OOD -> hold out (never trained on)
            if len(OOD_EV) < 2000: OOD_EV.append((tr[0], tr[2], lang_emb(tr[3])))
            continue
        buf.append(tr)
        if len(buf) >= bufsize:
            rng.shuffle(buf)
            while len(buf) >= BS:
                b = buf[:BS]; buf = buf[BS:]; yield collate(b)
    rng.shuffle(buf)
    for i in range(0, len(buf) - BS + 1, BS): yield collate(buf[i:i+BS])
def collate(b):
    return (np.stack([x[0] for x in b]), np.stack([x[1] for x in b]),
            np.stack([x[2] for x in b]), np.stack([lang_emb(x[3]) for x in b]))

# ---------- model (continuous: flow action + frozen-DINO aux) ----------
MEAN = jnp.array([0.485,0.456,0.406]); STD = jnp.array([0.229,0.224,0.225])
def norm(x): return (jnp.asarray(x, jnp.float32)/255.0 - MEAN)/STD
def temb(t, d=64):
    half=d//2; fr=jnp.exp(-jnp.log(10000.0)*jnp.arange(half)/half); a=t*fr[None]
    return jnp.concatenate([jnp.cos(a),jnp.sin(a)],-1)
if SMOKE:
    _W = jax.random.normal(jax.random.PRNGKey(7),(3*14*14,AUXTGT_DIM))*0.02
    def dino_embed(nhwc):
        B=nhwc.shape[0]; return jax.lax.stop_gradient(jax.image.resize(nhwc,(B,14,14,3),"linear").reshape(B,-1)@_W)
else:
    from transformers import FlaxDinov2Model
    _dino = FlaxDinov2Model.from_pretrained("facebook/dinov2-small", from_pt=True)
    @jax.jit
    def dino_embed(nhwc): return jax.lax.stop_gradient(_dino(pixel_values=jnp.transpose(nhwc,(0,3,1,2))).last_hidden_state[:,0])

class FlowHead(nn.Module):
    @nn.compact
    def __call__(self,x,t,cond):
        h=jnp.concatenate([x,cond,temb(t)],-1); h=nn.gelu(nn.Dense(512)(h)); h=nn.gelu(nn.Dense(512)(h)); return nn.Dense(ADIM)(h)
class RT1Flow(nn.Module):
    def setup(self):
        self.bb=RT1BackboneJAX(d_model=DMODEL,k_tokens=8,depth=4); self.flow=FlowHead(); self.pred=nn.Dense(AUXTGT_DIM)
    def __call__(self,im,lng,x,t): cond=self.bb(im,lng).mean(1); return self.flow(x,t,cond), self.pred(cond)
    def cond(self,im,lng): return self.bb(im,lng).mean(1)
    def vel(self,x,t,cond): return self.flow(x,t,cond)

model=RT1Flow(); key=jax.random.PRNGKey(SEED)
params=model.init(key, jnp.zeros((2,224,224,3)), jnp.zeros((2,LANGD)), jnp.zeros((2,ADIM)), jnp.zeros((2,1)))
tx=optax.adamw(3e-4,weight_decay=1e-2); opt_state=tx.init(params)
# action normalization: running estimate from the first buffer (streaming) — collect a warmup sample
warm=[]
for Ft,Ffl,act,lang in batches(min(MAX_EP,200) if not SMOKE else 0):
    warm.append(act)
    if sum(len(w) for w in warm)>=4000: break
warm=np.concatenate(warm); amu,asd=warm.mean(0),warm.std(0)+1e-6
print(f"warmup {len(warm)} trans for act-norm; streaming full (MAX_EP={MAX_EP}) AUX={AUX} SMOKE={SMOKE}", flush=True)

@jax.jit
def train_step(params, opt_state, key, im, lng, a, fut):
    k1,k2=jax.random.split(key); e=jax.random.normal(k1,a.shape); t=jax.random.uniform(k2,(a.shape[0],1))
    xt=(1-t)*e+t*a
    def L(p):
        vel,ap=model.apply(p,im,lng,xt,t); fm=((vel-(a-e))**2).mean(); aux=((ap-fut)**2).mean()
        return fm+(LAMBDA*aux if AUX else 0.0),(fm,aux)
    (l,(fm,aux)),g=jax.value_and_grad(L,has_aux=True)(params); upd,opt_state=tx.update(g,opt_state,params)
    return optax.apply_updates(params,upd),opt_state,fm,aux

step=0
for ep in range(EPOCHS):
    for Ft,Ffl,act,lang in batches(MAX_EP):
        an=(act-amu)/asd; fut=dino_embed(norm(Ffl)) if AUX else jnp.zeros((len(act),AUXTGT_DIM))
        key,sk=jax.random.split(key)
        params,opt_state,fm,aux=train_step(params,opt_state,sk,norm(Ft),jnp.asarray(lang),jnp.asarray(an),fut)
        step+=1
        if step%200==0: print(f"  ep{ep} step{step} flow={float(fm):.3f} aux={float(aux):.3f} uniq_instr={len(_lang_cache)} ood_ev={len(OOD_EV)}", flush=True)

# ---------- offline OOD generalization: flow-sample actions on held-out instructions, R^2 ----------
@jax.jit
def cond_fn(params, im, lng): return model.apply(params, im, lng, method=RT1Flow.cond)
@jax.jit
def vel_fn(params, x, t, c): return model.apply(params, x, t, c, method=RT1Flow.vel)
def sample(params, cond, k, K=10, S=8):
    B=cond.shape[0]; c=jnp.repeat(cond,S,0); x=jax.random.normal(k,(B*S,ADIM))
    for i in range(K):
        t=jnp.full((B*S,1),i/K); x=x+(1.0/K)*vel_fn(params,x,t,c)
    return x.reshape(B,S,ADIM).mean(1)
gen=None
if OOD_EV:
    Fo=np.stack([x[0] for x in OOD_EV]); Ao=np.stack([x[1] for x in OOD_EV]); Lo=np.stack([x[2] for x in OOD_EV])
    preds=[]; kk=jax.random.PRNGKey(123)
    for i in range(0,len(Fo),128):
        kk,sk=jax.random.split(kk); cond=cond_fn(params,norm(Fo[i:i+128]),jnp.asarray(Lo[i:i+128]))
        preds.append(np.asarray(sample(params,cond,sk))*asd+amu)
    yhat=np.concatenate(preds); mu=warm.mean(0)
    gen=float(1-((yhat-Ao)**2).mean()/(((Ao-mu)**2).mean()+1e-9))
print(f"RESULT arm={'AHA' if AUX else 'BC'}  steps={step}  OOD action R2 = {gen}  (uniq_instr={len(_lang_cache)}, ood_ev={len(OOD_EV)})", flush=True)

if not SMOKE:
    import json
    from flax.serialization import to_bytes
    open(f"{OUT}/rt1_full_a{AUX}.msgpack","wb").write(to_bytes({"params":params,"amu":amu,"asd":asd}))
    tag=f"a{AUX}_l{LAMBDA}_o{KLARGE}_s{SEED}"
    json.dump({"arm":"AHA" if AUX else "BC","aux":AUX,"lam":LAMBDA,"offset":KLARGE,"steps":step,"gen_ood":gen,
               "uniq_instr":len(_lang_cache),"ood_ev":len(OOD_EV),"max_ep":MAX_EP,"epochs":EPOCHS},
              open(f"{OUT}/rt1_full_{tag}.json","w"), indent=1)
    print(f"SAVED params + json ({tag})", flush=True)
