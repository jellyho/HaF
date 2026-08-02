"""Exp 2h — the LAW: objective recoverability (training-based) vs OOD generalization, on a small V-L-A model.

Three objective TYPES (clean taxonomy):
  prospective   : predict the FUTURE from the input        (future obs / actions)     — full input
  retrospective : predict the PAST from the input          (past/initial obs, actions)— full input
  introspective : MASK part of the current input, reconstruct it   (MAE/JEPA on now)  — masked image → clean z_t

For each objective:
  cheap recoverability  = rule-based (no deep train): frozen z_t → Ridge → target   (temporal objs only)
  deep  recoverability  = 1 − L_val/L_marg of the mini-VLA trained END-TO-END to predict the target
  generalization        = OOD action R² of BC co-trained with the objective (held-out task clusters)

The law: deep recoverability ↓  ⇒  generalization ↑ .  Everything logged to wandb. Output: exp2h_{TAG}_s{SEED}.json.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
TAG = os.environ.get("TAG", "fractal")
DEV = "cuda"
EPOCHS = int(os.environ.get("EPOCHS", 25))
SEED = int(os.environ.get("SEED", 0))
RUNTAG = os.environ.get("RUNTAG", TAG)
DMODEL = 256
torch.manual_seed(SEED); np.random.seed(SEED)

CFG = dict(tag=TAG, seed=SEED, epochs=EPOCHS, d_model=DMODEL, model="fused DINOv2-small+MiniLM+state")
_logf = open(os.path.join(OUT, f"exp2h_{RUNTAG}_s{SEED}.log.jsonl"), "w")
def logj(rec): _logf.write(json.dumps(rec) + "\n"); _logf.flush()
logj({"event": "config", **CFG})
try:
    import wandb
    _wb = wandb.init(entity="jellyho_", project="aha-recoverability", name=f"{RUNTAG}_s{SEED}", config=CFG)
    print("wandb: live", flush=True)
except Exception as e:
    _wb = None; print(f"wandb disabled ({type(e).__name__}); logging to JSONL only", flush=True)


def wlog(rec, step=None):
    logj({"event": "metric", **rec})
    if _wb is not None:
        try:
            _wb.log(rec)
        except Exception:
            pass


def wsummary(k, v):
    logj({"event": "summary", k: v})
    if _wb is not None:
        try:
            _wb.summary[k] = v
        except Exception:
            pass


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


_draw = np.load(os.path.join(OUT, f"cache/transitions_{TAG}.npz"), allow_pickle=True)
_lraw = np.load(os.path.join(OUT, f"cache/dino_latents_{TAG}.npz"))
_N0 = len(_draw["Ft"])
SUBN = int(os.environ.get("SUBN", 0))
_sel = (np.random.default_rng(123).permutation(_N0)[:SUBN] if (SUBN and SUBN < _N0) else np.arange(_N0))
d = {k: _draw[k][_sel] for k in _draw.files}
lat = {k: _lraw[k][_sel] for k in _lraw.files}
CFG["N"] = int(len(_sel)); print(f"{TAG}: using N={len(_sel)} of {_N0} (SUBN={SUBN})", flush=True)
Ft = d["Ft"]
act = torch.tensor(zs(d["act_chunk"].reshape(len(d["act_chunk"]), -1)), device=DEV)   # BC target (15-step)
state = torch.tensor(zs(np.concatenate([d["cartt"], d["gript"]], 1)), device=DEV)
g = d["ep_id"]
lang_np = l2n(embed_text(d["instr"])); lang = torch.tensor(lang_np, device=DEV)
zt_tgt = torch.tensor(l2n(lat["zt"]), device=DEV)   # clean current-frame latent (introspective target)
N = len(Ft)


def T(a): return torch.tensor(a, device=DEV, dtype=torch.float32)
def lz(k): return T(l2n(lat[k])) if k in lat else None
def vec(k): return T(zs(d[k])) if k in d else None


disp = T(zs(lat["zt"] - lat["z0"])) if "zt" in lat else None
# battery: name -> dict(target, type, mode['temporal'|'masked'], ratio)
def temporal(target, typ): return dict(target=target, type=typ, mode="temporal", ratio=0.0)
def masked(ratio): return dict(target=zt_tgt, type="intro", mode="masked", mask="img", ratio=ratio)
def introspect(mask, target, typ="intro"): return dict(target=target, type=typ, mode="masked", mask=mask, ratio=0.0)


RAW = {
    # retrospective (predict the past from the full input)
    "near-past-obs": temporal(lz("z_ps"), "retro"),
    "far-past-obs":  temporal(lz("z_pl"), "retro"),
    "initial-obs":   temporal(lz("z0"), "retro"),
    "prev-action":   temporal(vec("act_prev"), "retro"),
    "initial-pose":  temporal(vec("cart0"), "retro"),
    "displacement":  temporal(disp, "retro"),
    # prospective (predict the future; actions are forward-directed)
    "near-fut-obs":  temporal(lz("z_fs"), "prosp"),
    "far-fut-obs":   temporal(lz("z_fl"), "prosp"),
    "final-obs":     temporal(lz("z_last"), "prosp"),
    "fut-action":    temporal(vec("act_fut"), "prosp"),
    "fut-gripper":   temporal(vec("grip_fut"), "prosp"),
    "final-pose":    temporal(vec("cart_last"), "prosp"),
    "cur-action":    temporal(vec("actt"), "prosp"),
    # introspective (mask the current image, reconstruct the clean latent z_t) — MAE/JEPA lever
    "mae-mask25":    masked(0.25),
    "mae-mask50":    masked(0.50),
    "mae-mask75":    masked(0.75),
    # mask a whole non-vision modality of o_t and reconstruct it from the others
    "state-infer":   introspect("state", state, "intro"),   # mask proprio -> infer pose+gripper from image+lang
    "instr-infer":   introspect("lang",  lang,  "intro"),   # mask language -> infer instruction from image+state  (label: intro | retro — TBD)
}
OBJ = {k: v for k, v in RAW.items() if v["target"] is not None}

# ---- splits: OOD task clusters for generalization; in-distribution fit/val for recoverability ----
eps = np.unique(g)
ep_lang = np.stack([lang_np[g == e].mean(0) for e in eps])
cl = KMeans(8, n_init=5, random_state=SEED).fit_predict(ep_lang)
ood_clusters = set(np.argsort(np.bincount(cl))[:3].tolist())
epidx = {e: i for i, e in enumerate(eps)}
te = np.array([cl[epidx[e]] in ood_clusters for e in g]); tr = ~te
tr_idx = np.where(tr)[0]; te_idx = np.where(te)[0]
rng = np.random.default_rng(SEED); pm = rng.permutation(tr_idx)
nval = max(1, int(0.15 * len(pm))); fit_idx, val_idx = pm[nval:], pm[:nval]
print(f"{TAG}: N={N} fit={len(fit_idx)} iid-val={len(val_idx)} OOD={len(te_idx)}  objs={list(OBJ)}", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def imgs(idx):
    b = torch.from_numpy(Ft[idx]).to(DEV).float().permute(0, 3, 1, 2) / 255.0
    return (b - MEAN) / STD


def mask_img(x, ratio, gen):  # zero a fraction of 16x16 patches (MAE-style)
    B = x.shape[0]; P = 16; G = 224 // P
    keep = (torch.rand(B, G, G, generator=gen, device=DEV) > ratio).float()
    m = keep.repeat_interleave(P, 1).repeat_interleave(P, 2).unsqueeze(1)
    return x * m


def apply_mask(x, sb, lb, spec, gen):  # mask the modality this objective hides, leave the rest intact
    if spec.get("mode") != "masked":
        return x, sb, lb
    mk = spec.get("mask", "img")
    if mk == "img":   return mask_img(x, spec["ratio"], gen), sb, lb
    if mk == "state": return x, torch.zeros_like(sb), lb
    if mk == "lang":  return x, sb, torch.zeros_like(lb)
    return x, sb, lb


class Model(nn.Module):
    def __init__(self, sdim, ldim, adim, auxdim, d=DMODEL):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained("facebook/dinov2-small")
        ri = self.enc.config.hidden_size
        self.p_img = nn.Linear(ri, d); self.p_state = nn.Linear(sdim, d); self.p_lang = nn.Linear(ldim, d)
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=4 * d, batch_first=True, dropout=0.0)
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        # flow-matching action head (velocity field), pi0/pi0.5-style, conditioned on the fused rep
        self.adim = adim
        self.t_emb = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))
        self.flow = nn.Sequential(nn.Linear(adim + d + 64, 512), nn.GELU(),
                                  nn.Linear(512, 512), nn.GELU(), nn.Linear(512, adim))
        self.aux = nn.Linear(d, auxdim) if auxdim else None

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
    # S independent noise draws → Euler ODE integrate → average = conditional-mean estimate E[a|cond]
    B = cond.shape[0]; c = cond.repeat_interleave(S, 0)
    x = torch.randn(B * S, m.adim, device=DEV)
    for k in range(K):
        t = torch.full((B * S, 1), k / K, device=DEV)
        x = x + (1.0 / K) * m.velocity(x, t, c)
    return x.view(B, S, m.adim).mean(1)


def train(do_bc, spec=None, log_prefix="", curve=None):
    auxdim = (spec["target"].shape[1] if spec and spec["target"].dim() > 1 else 0) if spec else 0
    m = Model(state.shape[1], lang.shape[1], act.shape[1], auxdim).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); mse = nn.MSELoss(); bs = 128
    fit = tr_idx.copy() if do_bc else fit_idx.copy()   # gen uses all train; recov uses fit split
    gmask = torch.Generator(device=DEV).manual_seed(SEED)
    for ep in range(EPOCHS):
        np.random.shuffle(fit); m.train(); bcl = axl = 0.0; nb = 0
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]; x = imgs(b); loss = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if do_bc:
                    lb = fm_loss(m, m.rep(x, state[b], lang[b]), act[b]); loss = loss + lb; bcl += lb.item()
                if spec is not None:
                    xi, si, li = apply_mask(x, state[b], lang[b], spec, gmask)
                    la = mse(m.aux(m.rep(xi, si, li)), spec["target"][torch.tensor(b, device=DEV)])
                    loss = loss + la; axl += la.item()
            opt.zero_grad(); loss.backward(); opt.step(); nb += 1
        if curve is not None and spec is not None and not do_bc:
            curve.append(recov_eval(m, spec))   # learning curve: val 𝒱-info R per epoch (cheapness/dynamics)
    return m


@torch.no_grad()
def recov_eval(m, spec):  # normalized empirical 𝒱-information R = 1 - L_val/L_marg (policy class 𝒱) on iid val
    was = m.training; m.eval(); Y = spec["target"]; gmask = torch.Generator(device=DEV).manual_seed(SEED + 1); P = []
    for i in range(0, len(val_idx), 128):
        b = val_idx[i:i+128]; x = imgs(b)
        xi, si, li = apply_mask(x, state[b], lang[b], spec, gmask)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(m.aux(m.rep(xi, si, li)).float())
    P = torch.cat(P); Yv = Y[torch.tensor(val_idx, device=DEV)]
    Lval = torch.mean((P - Yv) ** 2).item(); Lmarg = torch.mean((Yv - Y[torch.tensor(fit_idx, device=DEV)].mean(0)) ** 2).item()
    if was: m.train()
    return 1 - Lval / (Lmarg + 1e-9)


def recov_of(m, spec):
    return recov_eval(m, spec)


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


zt_np = l2n(lat["zt"])


def cheap_recov(spec):  # rule-based estimate, temporal objectives only (target not in input)
    if spec["mode"] != "temporal":
        return None
    Y = spec["target"].detach().cpu().numpy().reshape(N, -1)
    r = Ridge(alpha=100.0).fit(zt_np[fit_idx], Y[fit_idx])
    P = np.asarray(r.predict(zt_np[val_idx])).reshape(len(val_idx), -1)
    Lval = float(np.mean((P - Y[val_idx]) ** 2)); Lmarg = float(np.mean((Y[val_idx] - Y[fit_idx].mean(0)) ** 2))
    return 1 - Lval / (Lmarg + 1e-9)


# ---- frozen-feature probe measures (different function classes 𝒱) + prequential MDL (cheapness) ----
def _probe_fit(Xtr, Ytr, Xva, kind):
    if kind == "linear":
        r = Ridge(alpha=100.0).fit(Xtr, Ytr); return np.asarray(r.predict(Xva)).reshape(len(Xva), -1)
    xt = torch.tensor(Xtr, device=DEV); yt = torch.tensor(Ytr, device=DEV, dtype=torch.float32)
    xv = torch.tensor(Xva, device=DEV)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, Ytr.shape[1])).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for _ in range(300):
        idx = torch.randint(len(xt), (256,), device=DEV)
        loss = nn.functional.mse_loss(net(xt[idx]), yt[idx]); opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(xv).float().cpu().numpy()


def probe_recov(spec, kind="linear", control=False):  # asymptotic 𝒱-info R under a frozen-feature probe class
    if spec["mode"] != "temporal":
        return None
    Y = spec["target"].detach().cpu().numpy().reshape(N, -1).astype(np.float32)
    if control:
        Y = Y[np.random.default_rng(7).permutation(N)]   # selectivity control (shuffled target)
    Xtr, Xva, Ytr, Yva = zt_np[fit_idx], zt_np[val_idx], Y[fit_idx], Y[val_idx]
    P = _probe_fit(Xtr, Ytr, Xva, kind)
    Lval = float(np.mean((P - Yva) ** 2)); Lmarg = float(np.mean((Yva - Ytr.mean(0)) ** 2))
    return 1 - Lval / (Lmarg + 1e-9)


def prequential_mdl(spec, kind="mlp"):  # online codelength ~ area under learning curve (cheapness/dynamics)
    if spec["mode"] != "temporal":
        return None
    Y = spec["target"].detach().cpu().numpy().reshape(N, -1).astype(np.float32)
    idx = fit_idx.copy(); n = len(idx)
    fr = [0.0625, 0.125, 0.25, 0.5, 1.0]; prev = max(16, int(0.03 * n)); Lsum = Msum = 0.0; nb = 0
    for j, fj in enumerate(fr):
        cut = int(fj * n)
        tr, bl = idx[:prev], idx[prev:cut]
        if len(bl) == 0:
            continue
        Ytr, Ybl = Y[tr], Y[bl]
        P = _probe_fit(zt_np[tr], Ytr, zt_np[bl], kind)
        Lsum += float(np.mean((P - Ybl) ** 2)) * len(bl); Msum += float(np.mean((Ybl - Ytr.mean(0)) ** 2)) * len(bl)
        nb += len(bl); prev = cut
    return 1 - (Lsum / nb) / ((Msum / nb) + 1e-9)


def _f(v): return ("%.3f" % v) if v is not None else " n/a "


# BC-only generalization baseline
bc0 = bc_ood(train(True, None, log_prefix="BC-only"))
print(f"  BC-only OOD R2 = {bc0:+.3f}", flush=True)
wsummary("BC_only_ood", bc0)

res = {"BC_only_ood": bc0, "objectives": {}}
for name, spec in OBJ.items():
    # frozen-feature probe classes (asymptotic) + prequential MDL (dynamics) + selectivity control
    r_lin = probe_recov(spec, "linear"); r_mlp = probe_recov(spec, "mlp")
    r_mdl = prequential_mdl(spec, "mlp"); r_lin_ctl = probe_recov(spec, "linear", control=True)
    # policy-class 𝒱: asymptotic R + learning curve (dynamics)
    curve = []
    m_rec = train(False, spec, log_prefix=f"recov/{name}", curve=curve)
    deep = recov_eval(m_rec, spec)
    aulc = float(np.mean(curve)) if curve else None                       # area under val-R curve (cheapness)
    early = curve[max(0, len(curve) // 4 - 1)] if curve else None         # val-R at ~1/4 budget (speed)
    # generalization outcome: BC (flow-matching) co-trained with the objective, OOD action R2
    gen = bc_ood(train(True, spec, log_prefix=f"gen/{name}"))
    o = {"type": spec["type"], "mode": spec["mode"], "ratio": spec["ratio"],
         "generalization": gen, "benefit": gen - bc0,
         "R_linear": r_lin, "R_mlp": r_mlp, "R_mdl": r_mdl, "R_linear_control": r_lin_ctl,
         "R_deep": deep, "R_deep_aulc": aulc, "R_deep_early": early,
         "cheap_recoverability": r_lin, "deep_recoverability": deep}   # legacy keys for existing plots
    res["objectives"][name] = o
    logj({"event": "objective_done", "name": name, **o})
    print(f"  {name:13s}[{spec['type']:5s}] lin={_f(r_lin)} mlp={_f(r_mlp)} mdl={_f(r_mdl)} "
          f"deep={deep:+.3f} aulc={_f(aulc)} early={_f(early)} | gen={gen:+.3f} ctl={_f(r_lin_ctl)}", flush=True)

json.dump(res, open(os.path.join(OUT, f"exp2h_{RUNTAG}_s{SEED}.json"), "w"), indent=2)
print("SAVED", f"exp2h_{RUNTAG}_s{SEED}.json", flush=True)

# ---- clean wandb summary: one results Table + law/flip scatters + headline scalars ----
if _wb is not None:
    try:
        from scipy.stats import pearsonr
        cols = ["objective", "type", "R_linear", "R_mlp", "R_mdl", "R_deep", "R_deep_aulc", "R_deep_early",
                "generalization", "benefit"]
        tbl = wandb.Table(columns=cols)
        for nm, o in res["objectives"].items():
            tbl.add_data(nm, o["type"], o["R_linear"], o["R_mlp"], o["R_mdl"], o["R_deep"],
                         o["R_deep_aulc"], o["R_deep_early"], o["generalization"], o["benefit"])
        logs = {"results": tbl,
                "law/deep_vs_gen": wandb.plot.scatter(tbl, "R_deep", "generalization", title="LAW: recoverability↓ ⇒ gen↑"),
                "law/linear_vs_gen": wandb.plot.scatter(tbl, "R_linear", "generalization", title="linear probe MISranks (flip)"),
                "law/speed_vs_gen": wandb.plot.scatter(tbl, "R_deep_early", "generalization", title="speed (dynamics) best predictor")}
        _wb.log(logs)
        # headline correlations as summary scalars
        def corr(key):
            xy = [(o[key], o["generalization"]) for o in res["objectives"].values()
                  if o.get(key) is not None and o.get("generalization") is not None]
            return pearsonr([a for a, _ in xy], [b for _, b in xy])[0] if len(xy) > 3 else None
        for k in ["R_linear", "R_mlp", "R_mdl", "R_deep", "R_deep_aulc", "R_deep_early"]:
            c = corr(k)
            if c is not None:
                wsummary(f"corr/{k}_vs_gen", round(c, 3))
        wsummary("BC_only_ood", bc0)
    except Exception as e:
        print("wandb summary skipped:", type(e).__name__, e, flush=True)
    try:
        _wb.finish()
    except Exception:
        pass
_logf.close()
os._exit(0)
