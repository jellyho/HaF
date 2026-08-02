"""Corrected count-sweep: does stacking DIVERSE hard questions keep helping, or does it really saturate?

Diagnosis of the old exp2h_mixki deep run: a mix was ONE linear head over the CONCATENATION of member targets
with a single mean-MSE at fixed 1:1 weight => the aux gradient budget is CONSTANT in the member count K (each
member gets 1/K of it). Saturation was baked in by the loss normalization, not a property of the world.

Fix + fair controls. For K = 1..6 diverse low-recoverability members (ordered, distinct groundings):
  SUM      : K independent heads, EACH full weight w=1  -> total aux pressure = K   (the honest count-sweep)
  MATCHED  : 1 head on the single hardest member, weight = K -> SAME total pressure as SUM, but one question
  MEAN     : old behavior (concat, single head, mean-MSE, weight 1) -> should reproduce the flat ~0.04 artifact
Read: SUM rising with K AND SUM > MATCHED  => diverse hard questions add complementary grounding (NOT saturation).
      SUM ~ MATCHED                          => it was only total aux weight; count/diversity per se inert.
      SUM rises then falls                   => over-regularization (BC crowded out) at high K.
Output: exp2h_countfix_s{SEED}.json    (reuses exp2h_mixki data/model setup)
"""
import os, json, numpy as np, torch, torch.nn as nn

# ---- reuse the exact data/model/split setup from exp2h_mixki ----
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "mixki", "/data5/jellyho/Hindsight/HaF/experiments/recoverability/probes/exp2h_mixki.py")
# We don't exec the whole file (it runs experiments); instead replicate the header via exec of lines up to setup.
# Simpler: import the shared pieces by executing the file's setup guarded by __name__.
# exp2h_mixki runs at import; to avoid that, we re-implement the minimal setup here by importing its module-level
# objects through a controlled exec of only the top (data + Model). To keep it robust we just import selected names.

# --- minimal re-exec of setup block (no experiment loops) ---
G = {}
with open("/data5/jellyho/Hindsight/HaF/experiments/recoverability/probes/exp2h_mixki.py") as f:
    src = f.read()
# cut the file at the first experiment marker so only setup (data, TGT, Model, fm_loss, fm_sample, imgs, indices) runs
cut = src.index("# member single-objective R_deep")
exec(compile(src[:cut], "mixki_setup", "exec"), G)

TGT = G["TGT"]; Model = G["Model"]; fm_loss = G["fm_loss"]; fm_sample = G["fm_sample"]
imgs = G["imgs"]; state = G["state"]; lang = G["lang"]; act = G["act"]
fit_idx = G["fit_idx"]; val_idx = G["val_idx"]; te_idx = G["te_idx"]; tr_idx = G["tr_idx"]
DEV = G["DEV"]; EPOCHS = G["EPOCHS"]; SEED = G["SEED"]; DMODEL = G["DMODEL"]; OUT = G["OUT"]

# diverse low-recoverability members, ordered hardest-first (measured R_deep in comment)
ORDER = ["fut-action",   # 0.04  dynamics
         "far-fut-obs",  # 0.14  future scene
         "far-past-obs", # 0.16  past scene
         "near-fut-obs", # 0.19  near-future scene
         "initial-obs",  # 0.22  initial scene
         "displacement"] # 0.49  geometry (kept last: highest recov)


class MultiHeadModel(Model):
    """Same backbone/rep/flow as Model; aux = a ModuleList of independent linear heads (one per member)."""
    def __init__(self, sdim, ldim, adim, head_dims):
        super().__init__(sdim, ldim, adim, auxdim=0)     # no single aux head
        self.auxheads = nn.ModuleList([nn.Linear(DMODEL, d) for d in head_dims])


def train_arms(arms, mean_concat=False):
    """arms = list of (target_tensor[N,dk], weight).  mean_concat=True reproduces the OLD single-head mean-MSE."""
    mse = nn.MSELoss()
    if mean_concat:
        cat = torch.cat([t for t, _ in arms], dim=1)
        m = Model(state.shape[1], lang.shape[1], act.shape[1], cat.shape[1]).to(DEV)
    else:
        m = MultiHeadModel(state.shape[1], lang.shape[1], act.shape[1], [t.shape[1] for t, _ in arms]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); bs = 128
    fit = tr_idx.copy()
    for ep in range(EPOCHS):
        np.random.shuffle(fit); m.train()
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]; x = imgs(b); bt = torch.tensor(b, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep = m.rep(x, state[b], lang[b])
                loss = fm_loss(m, rep, act[b])
                if mean_concat:
                    loss = loss + mse(m.aux(rep), cat[bt])
                else:
                    for h, (t, w) in zip(m.auxheads, arms):
                        loss = loss + w * mse(h(rep), t[bt])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


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


def train_bc_only():
    m = Model(state.shape[1], lang.shape[1], act.shape[1], 0).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); bs = 128; fit = tr_idx.copy()
    for ep in range(EPOCHS):
        np.random.shuffle(fit); m.train()
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = fm_loss(m, m.rep(imgs(b), state[b], lang[b]), act[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m


res = {"config": {"seed": SEED, "epochs": EPOCHS, "order": ORDER}, "sweep": []}
bc0 = bc_ood(train_bc_only())
print(f"BC-only gen = {bc0:+.4f}", flush=True)
hardest = TGT[ORDER[0]]

KMAX = int(os.environ.get("KMAX", len(ORDER)))
for K in range(1, min(KMAX, len(ORDER)) + 1):
    members = ORDER[:K]
    arms_sum = [(TGT[n], 1.0) for n in members]                 # K heads, each weight 1  (total = K)
    g_sum = bc_ood(train_arms(arms_sum, mean_concat=False))
    g_match = bc_ood(train_arms([(hardest, float(K))], mean_concat=False))  # 1 head, weight K (same total)
    g_mean = bc_ood(train_arms([(TGT[n], 1.0) for n in members], mean_concat=True))  # old artifact
    row = dict(K=K, members=members,
               sum=g_sum, matched=g_match, mean=g_mean,
               b_sum=g_sum - bc0, b_matched=g_match - bc0, b_mean=g_mean - bc0)
    res["sweep"].append(row)
    print(f"K={K}  SUM {g_sum-bc0:+.4f} | MATCHED {g_match-bc0:+.4f} | MEAN {g_mean-bc0:+.4f}  [{'+'.join(members)}]", flush=True)

res["bc0"] = bc0
path = f"{OUT}/exp2h_countfix_s{SEED}.json"
json.dump(res, open(path, "w"), indent=1)
print(f"SAVED {path}", flush=True)
import os as _os; _os._exit(0)
