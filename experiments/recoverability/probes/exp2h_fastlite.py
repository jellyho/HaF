"""Phase 2 — FAST-lite: is DISCRETE action-token prediction lower-recoverability than CONTINUOUS action
regression, and does it ground the backbone under KI? (π0.5 KI+FAST ↔ AHA connection.)

Same mini V-L-A. Target = current action a_t (7-d). Two aux forms of the SAME quantity:
  continuous: predict a_t with MSE  → R_cont = 1 − MSE_val/MSE_marg
  discrete  : bin each dim into K bins (FAST-lite tokenization), predict tokens with cross-entropy
              → R_disc = 1 − CE_val/CE_marg   (uncertainty coefficient / discrete 𝒱-information, in [0,1])
Then KI: shape the backbone with {continuous, discrete} action aux under τ∈{0,1} and read BC OOD gen.

Hypothesis: R_disc < R_cont (discretization raises prediction difficulty / matches NTP) → discrete grounds
better under KI. If R_disc ≈ R_cont, FAST's benefit is training-dynamics/modality, not recoverability (report honestly).
Output: exp2h_fastlite_{RUNTAG}_s{SEED}.json
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
SUBN = int(os.environ.get("SUBN", 4000))
RUNTAG = os.environ.get("RUNTAG", "fl")
K = int(os.environ.get("KBINS", 16))
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
act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)   # BC target
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
lang_np = l2n(embed_text(d["instr"])); lang = torch.tensor(lang_np, device=DEV)
N = len(Ft)

# ---- action target (continuous + discrete tokens) ----
actt_np = zs(d["actt"])              # [N, 7]
Dact = actt_np.shape[1]
ACT = torch.tensor(actt_np, device=DEV)

from sklearn.cluster import KMeans
eps = np.unique(g); ep_lang = np.stack([lang_np[g == e].mean(0) for e in eps])
cl = KMeans(8, n_init=5, random_state=SEED).fit_predict(ep_lang)
ood = set(np.argsort(np.bincount(cl))[:3].tolist())
epidx = {e: i for i, e in enumerate(eps)}
te = np.array([cl[epidx[e]] in ood for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
rng = np.random.default_rng(SEED); pm = rng.permutation(tr_idx)
nval = max(1, int(0.15 * len(pm))); fit_idx, val_idx = pm[nval:], pm[:nval]

# FAST-lite tokenization: per-dim quantile bins fit on the fit split
edges = [np.quantile(actt_np[fit_idx, j], np.linspace(0, 1, K + 1)[1:-1]) for j in range(Dact)]
TOK = np.stack([np.clip(np.digitize(actt_np[:, j], edges[j]), 0, K - 1) for j in range(Dact)], 1)  # [N,7] 0..K-1
TOK = torch.tensor(TOK, device=DEV, dtype=torch.long)
print(f"{TAG}: N={N} SUBN={SUBN} EPOCHS={EPOCHS} SEED={SEED} K={K} Dact={Dact}", flush=True)

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


@torch.no_grad()
def fm_sample(m, cond, Kn=20, Sn=16):
    B = cond.shape[0]; c = cond.repeat_interleave(Sn, 0); x = torch.randn(B*Sn, m.adim, device=DEV)
    for k in range(Kn):
        t = torch.full((B*Sn, 1), k/Kn, device=DEV); x = x + (1.0/Kn)*m.velocity(x, t, c)
    return x.view(B, Sn, m.adim).mean(1)


def _auxloss(m, rep, kind, b):
    bt = torch.tensor(b, device=DEV)
    if kind == "cont":
        return nn.functional.mse_loss(m.aux(rep), ACT[bt])
    logits = m.aux(rep).view(-1, Dact, K)
    return nn.functional.cross_entropy(logits.reshape(-1, K), TOK[bt].reshape(-1))


def train(do_bc, aux_kind=None, ki_tau=0.0):
    auxdim = (Dact if aux_kind == "cont" else Dact*K) if aux_kind else 0
    m = Model(state.shape[1], lang.shape[1], act.shape[1], auxdim).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); bs = 128
    fit = tr_idx.copy() if do_bc else fit_idx.copy()
    for ep in range(EPOCHS):
        ki_now = ep < ki_tau * EPOCHS
        np.random.shuffle(fit); m.train()
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]; x = imgs(b); loss = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep = m.rep(x, state[b], lang[b])
                if do_bc:
                    cond = rep.detach() if ki_now else rep
                    loss = loss + fm_loss(m, cond, act[b])
                if aux_kind is not None:
                    loss = loss + _auxloss(m, rep, aux_kind, b)
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def recov(m, kind):
    m.eval()
    if kind == "cont":
        P = []
        for i in range(0, len(val_idx), 128):
            b = val_idx[i:i+128]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                P.append(m.aux(m.rep(imgs(b), state[b], lang[b])).float())
        P = torch.cat(P); Yv = ACT[torch.tensor(val_idx, device=DEV)]
        Lval = torch.mean((P-Yv)**2).item()
        Lmarg = torch.mean((Yv - ACT[torch.tensor(fit_idx, device=DEV)].mean(0))**2).item()
        return 1 - Lval/(Lmarg+1e-9)
    # discrete: CE-based uncertainty coefficient
    ce = 0.0; nb = 0
    for i in range(0, len(val_idx), 128):
        b = val_idx[i:i+128]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m.aux(m.rep(imgs(b), state[b], lang[b])).float().view(-1, Dact, K)
        ce += nn.functional.cross_entropy(logits.reshape(-1, K), TOK[torch.tensor(b, device=DEV)].reshape(-1),
                                          reduction="sum").item()
        nb += len(b)*Dact
    CEval = ce/nb
    # marginal CE = mean per-dim entropy of fit bin distribution
    fit_t = torch.tensor(fit_idx, device=DEV); ent = 0.0
    for j in range(Dact):
        c = torch.bincount(TOK[fit_t, j], minlength=K).float() + 1e-6; p = c/c.sum()
        ent += (-(p*torch.log(p)).sum()).item()
    CEmarg = ent/Dact
    return 1 - CEval/(CEmarg+1e-9)


@torch.no_grad()
def bc_ood(m):
    m.eval(); P = []
    for i in range(0, len(te_idx), 128):
        b = te_idx[i:i+128]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cond = m.rep(imgs(b), state[b], lang[b])
        P.append(fm_sample(m, cond.float()))
    P = torch.cat(P); Y = act[torch.tensor(te_idx, device=DEV)]
    return 1 - torch.mean((P-Y)**2).item()/(torch.mean((Y-Y.mean(0))**2).item()+1e-9)


res = {"config": dict(tag=TAG, seed=SEED, epochs=EPOCHS, subn=SUBN, K=K)}
bc0 = bc_ood(train(True, None, 0.0)); res["BC_only_ood"] = bc0
print(f"BC-only OOD R2 = {bc0:+.3f}", flush=True)

# recoverability of continuous vs discrete action prediction
Rc = recov(train(False, "cont"), "cont")
Rd = recov(train(False, "disc"), "disc")
res["R_cont_action"] = Rc; res["R_disc_action"] = Rd
print(f"R_cont(action)={Rc:+.3f}   R_disc(action-tokens,K={K})={Rd:+.3f}", flush=True)

# KI: shape backbone with each action aux, hard-KI vs joint
res["ki"] = []
for kind in ["cont", "disc"]:
    for tau in [0.0, 1.0]:
        gen = bc_ood(train(True, kind, tau))
        res["ki"].append(dict(aux=kind, tau=tau, gen=gen, benefit=gen-bc0))
        print(f"  KI aux={kind:5s} tau={tau:.1f}  gen={gen:+.3f}  Δ={gen-bc0:+.3f}", flush=True)

json.dump(res, open(f"{OUT}/exp2h_fastlite_{RUNTAG}_s{SEED}.json", "w"), indent=2)
print(f"SAVED exp2h_fastlite_{RUNTAG}_s{SEED}.json", flush=True)
import os as _os; _os._exit(0)
