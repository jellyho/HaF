"""M1 — mini V-L-A: (A) objective-MIXING composition law + (B) Knowledge-Insulation (schedule) ablation.

Reuses the exp2h fused mini-VLA + flow-matching BC. TEMPORAL objectives only (no masking, so a mix = a single
aux head predicting the CONCATENATION of member targets = the mix's JOINT recoverability).

(A) Mixing: for each mix, gen = BC+concat-aux OOD R²; joint recov = 1−L_val/L_marg on the concat; member
    R_deep pulled from the single-objective run (exp2h_fractal). Ask: does min / mean / joint recoverability
    predict mix generalization? (hypothesis: the LOWEST-recoverability member dominates.)
(B) KI: BC's cond into the flow head is detached from the backbone for the first τ·EPOCHS epochs (insulate),
    then attached (joint). τ∈{0 always-joint, 0.5 late-release, 1.0 always-KI} × aux∈{low-recov, high-recov}.
    hypothesis: with a low-recov aux, late-release (τ≈0.5) ≥ always-KI > always-joint.

Output: exp2h_mixki_{RUNTAG}_s{SEED}.json
"""
import os, json
import numpy as np
import torch
import torch.nn as nn

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "fractal")
DEV = "cuda"
EPOCHS = int(os.environ.get("EPOCHS", 20))
SEED = int(os.environ.get("SEED", 0))
SUBN = int(os.environ.get("SUBN", 6000))
RUNTAG = os.environ.get("RUNTAG", "mk")
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


_d = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
_l = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
_N0 = len(_d["Ft"])
_sel = (np.random.default_rng(123).permutation(_N0)[:SUBN] if (SUBN and SUBN < _N0) else np.arange(_N0))
d = {k: _d[k][_sel] for k in _d.files}
lat = {k: _l[k][_sel] for k in _l.files}
Ft = d["Ft"]
act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
lang_np = l2n(embed_text(d["instr"])); lang = torch.tensor(lang_np, device=DEV)
N = len(Ft)
print(f"{TAG}: N={N} of {_N0} (SUBN={SUBN})  EPOCHS={EPOCHS} SEED={SEED}", flush=True)


def T(a): return torch.tensor(a, device=DEV, dtype=torch.float32)
def lz(k): return T(l2n(lat[k]))
def vec(k): return T(zs(d[k]))
disp = T(zs(lat["zt"] - lat["z0"]))

# temporal objective target library (name -> [N, dim] tensor)   (matches exp2h names)
TGT = {
    "near-past-obs": lz("z_ps"), "far-past-obs": lz("z_pl"), "initial-obs": lz("z0"),
    "prev-action": vec("act_prev"), "initial-pose": vec("cart0"), "displacement": disp,
    "near-fut-obs": lz("z_fs"), "far-fut-obs": lz("z_fl"), "final-obs": lz("z_last"),
    "fut-action": vec("act_fut"), "fut-gripper": vec("grip_fut"), "final-pose": vec("cart_last"),
    "cur-action": vec("actt"),
}

# ---- OOD task-cluster split + fit/val (same protocol as exp2h) ----
from sklearn.cluster import KMeans
eps = np.unique(g); ep_lang = np.stack([lang_np[g == e].mean(0) for e in eps])
cl = KMeans(8, n_init=5, random_state=SEED).fit_predict(ep_lang)
ood = set(np.argsort(np.bincount(cl))[:3].tolist())
epidx = {e: i for i, e in enumerate(eps)}
te = np.array([cl[epidx[e]] in ood for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
rng = np.random.default_rng(SEED); pm = rng.permutation(tr_idx)
nval = max(1, int(0.15 * len(pm))); fit_idx, val_idx = pm[nval:], pm[:nval]

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
    x_t = (1 - t) * eps + t * a
    return nn.functional.mse_loss(m.velocity(x_t, t, cond), a - eps)


@torch.no_grad()
def fm_sample(m, cond, K=20, S=16):
    B = cond.shape[0]; c = cond.repeat_interleave(S, 0)
    x = torch.randn(B * S, m.adim, device=DEV)
    for k in range(K):
        t = torch.full((B * S, 1), k / K, device=DEV)
        x = x + (1.0 / K) * m.velocity(x, t, c)
    return x.view(B, S, m.adim).mean(1)


def train(do_bc, aux_tgt=None, ki_tau=0.0, log=""):
    auxdim = aux_tgt.shape[1] if aux_tgt is not None else 0
    m = Model(state.shape[1], lang.shape[1], act.shape[1], auxdim).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); mse = nn.MSELoss(); bs = 128
    fit = tr_idx.copy() if do_bc else fit_idx.copy()
    for ep in range(EPOCHS):
        ki_now = ep < ki_tau * EPOCHS          # KI on during first τ fraction (insulate), then release
        np.random.shuffle(fit); m.train()
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]; x = imgs(b); loss = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep = m.rep(x, state[b], lang[b])
                if do_bc:
                    cond = rep.detach() if ki_now else rep         # KI = stop-grad BC head -> backbone
                    loss = loss + fm_loss(m, cond, act[b])
                if aux_tgt is not None:
                    loss = loss + mse(m.aux(rep), aux_tgt[torch.tensor(b, device=DEV)])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def recov_eval(m, tgt):
    m.eval(); P = []
    for i in range(0, len(val_idx), 128):
        b = val_idx[i:i+128]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(m.aux(m.rep(imgs(b), state[b], lang[b])).float())
    P = torch.cat(P); Yv = tgt[torch.tensor(val_idx, device=DEV)]
    Lval = torch.mean((P - Yv) ** 2).item()
    Lmarg = torch.mean((Yv - tgt[torch.tensor(fit_idx, device=DEV)].mean(0)) ** 2).item()
    return 1 - Lval / (Lmarg + 1e-9)


@torch.no_grad()
def bc_ood(m):
    m.eval(); P = []
    for i in range(0, len(te_idx), 128):
        b = te_idx[i:i+128]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond = m.rep(imgs(b), state[b], lang[b])
        P.append(fm_sample(m, cond.float()))
    P = torch.cat(P); Y = act[torch.tensor(te_idx, device=DEV)]
    return 1 - torch.mean((P - Y) ** 2).item() / (torch.mean((Y - Y.mean(0)) ** 2).item() + 1e-9)


def concat(names): return torch.cat([TGT[n] for n in names], dim=1)


# member single-objective R_deep from the full exp2h run (fallback: compute here)
def member_recov():
    r = {}
    try:
        S = [json.load(open(f"{OUT}/exp2h_{TAG}_s{s}.json")) for s in range(3)]
        for n in TGT:
            v = [s["objectives"][n]["R_deep"] for s in S if n in s["objectives"] and s["objectives"][n].get("R_deep") is not None]
            if v: r[n] = float(np.mean(v))
    except Exception as e:
        print("member recov json missing, computing:", e, flush=True)
    for n in TGT:
        if n not in r:
            r[n] = recov_eval(train(False, TGT[n]), TGT[n])
    return r


res = {"config": dict(tag=TAG, seed=SEED, epochs=EPOCHS, subn=SUBN)}
memberR = member_recov()
res["member_R_deep"] = memberR
bc0 = bc_ood(train(True, None, 0.0))
res["BC_only_ood"] = bc0
print(f"BC-only OOD R2 = {bc0:+.3f}", flush=True)

# ---- (A) mixing ----
_MIXBASE = [
    ["far-past-obs"], ["final-pose"], ["final-obs"], ["fut-gripper"],
    ["far-past-obs", "far-fut-obs"], ["far-past-obs", "final-pose"],
    ["final-pose", "initial-pose"], ["far-past-obs", "far-fut-obs", "near-fut-obs"],
    ["far-past-obs", "far-fut-obs", "final-pose"], ["final-obs", "far-past-obs"],
    ["final-pose", "fut-gripper"], ["far-fut-obs", "final-obs", "near-past-obs"],
]
# deep: (i) size-sweep of LOW-recov obs objectives -> interference/optimal-count;
#       (ii) diversity: 2 low-recov same-modality (obs+obs) vs cross (obs+action, obs+pose-low)
_LOWOBS = ["far-past-obs", "far-fut-obs", "near-past-obs", "near-fut-obs", "final-obs", "initial-obs"]
_MIXDEEP = [ _LOWOBS[:k] for k in range(1, 7) ] + [
    ["far-past-obs", "far-fut-obs"],            # obs + obs (same modality)
    ["far-past-obs", "fut-action"],             # obs + action (cross)
    ["far-past-obs", "prev-action"],            # obs + action (cross)
    ["far-past-obs", "displacement"],           # obs + displacement
    ["far-fut-obs", "fut-action", "far-past-obs"],
]
MIXES = _MIXDEEP if os.environ.get("MIXSET") == "deep" else _MIXBASE
res["mixes"] = []
if not os.environ.get("KI_ONLY"):
    print("\n--- (A) MIXING ---", flush=True)
    for mix in MIXES:
        ct = concat(mix)
        joint = recov_eval(train(False, ct), ct)
        gen = bc_ood(train(True, ct, 0.0))
        ms = [memberR[n] for n in mix]
        o = dict(members=mix, joint_recov=joint, min_recov=float(min(ms)), mean_recov=float(np.mean(ms)),
                 gen=gen, benefit=gen - bc0)
        res["mixes"].append(o)
        print(f"  {'+'.join(mix):45s} min={min(ms):+.2f} mean={np.mean(ms):+.2f} joint={joint:+.2f} | gen={gen:+.3f}", flush=True)

# ---- (B) KI schedule ----
print("\n--- (B) KI SCHEDULE ---", flush=True)
KI = {"unrelated,low (far-past-obs)": TGT["far-past-obs"],
      "unrelated,high (final-pose)": TGT["final-pose"],
      "action-related (cur-action)": TGT["cur-action"],
      "action-related (fut-action)": TGT["fut-action"]}
res["ki"] = []
for name, tgt in KI.items():
    for tau in [0.0, 0.5, 1.0]:
        gen = bc_ood(train(True, tgt, tau))
        res["ki"].append(dict(aux=name, tau=tau, gen=gen, benefit=gen - bc0))
        print(f"  aux={name:22s} tau={tau:.1f}  gen={gen:+.3f}", flush=True)

json.dump(res, open(f"{OUT}/exp2h_mixki_{RUNTAG}_s{SEED}.json", "w"), indent=2)
print(f"\nSAVED exp2h_mixki_{RUNTAG}_s{SEED}.json", flush=True)
import os as _os; _os._exit(0)
