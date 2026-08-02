"""Dynamic Knowledge Insulation: release the BC-head -> backbone stop-grad when the BC head's gradient into the
backbone has DECAYED (i.e. the action/flow expert has warmed up), instead of at a fixed epoch fraction tau.

Rationale (user's idea): early, the flow expert is from scratch -> its gradient w.r.t. the backbone rep is large &
noisy -> insulate. As it converges, that gradient shrinks -> safe to release SG -> joint training. So gate the
release on the *magnitude* of the BC-head gradient reaching the backbone, measured by a cheap probe.

Compares, per aux, BC-only vs {fixed tau=0,0.5,1.0} vs {dynamic release @ grad-decay}. Logs the release epoch.
Reuses the exp2h_mixki mini-VLA setup. Output: exp2h_dynki_s{SEED}.json
env: SEED, EPOCHS, SUBN, RELFRAC (release when EMA grad-norm < RELFRAC*peak, default 0.3), PROBE_EVERY (default 15).
"""
import os, json, numpy as np, torch, torch.nn as nn

G = {}
with open("/data5/jellyho/Hindsight/HaF/experiments/recoverability/probes/exp2h_mixki.py") as f:
    src = f.read()
exec(compile(src[:src.index("# member single-objective R_deep")], "mixki_setup", "exec"), G)
TGT = G["TGT"]; Model = G["Model"]; fm_loss = G["fm_loss"]; fm_sample = G["fm_sample"]
imgs = G["imgs"]; state = G["state"]; lang = G["lang"]; act = G["act"]
fit_idx = G["fit_idx"]; te_idx = G["te_idx"]; DEV = G["DEV"]; EPOCHS = G["EPOCHS"]; SEED = G["SEED"]; OUT = G["OUT"]
RELFRAC = float(os.environ.get("RELFRAC", 0.3)); PROBE_EVERY = int(os.environ.get("PROBE_EVERY", 15))
mse = nn.MSELoss()


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


def train(aux_tgt=None, mode="joint", tau=0.0):
    """mode: 'joint'(tau=0) | 'fixed'(release at tau*EPOCHS) | 'dynamic'(release when grad decays)."""
    m = Model(state.shape[1], lang.shape[1], act.shape[1], aux_tgt.shape[1] if aux_tgt is not None else 0).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); bs = 128; fit = fit_idx.copy()
    released = (mode == "joint"); rel_ep = 0 if released else None
    peak = 0.0; ema = None; step = 0
    for ep in range(EPOCHS):
        if mode == "fixed" and not released and ep >= tau * EPOCHS: released, rel_ep = True, ep
        np.random.shuffle(fit); m.train()
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]; x = imgs(b); bt = torch.tensor(b, device=DEV)
            # dynamic probe: measure BC-head gradient magnitude into the backbone (rep), decide release
            if mode == "dynamic" and not released and step % PROBE_EVERY == 0:
                rep_p = m.rep(x, state[b], lang[b])
                bcl = fm_loss(m, rep_p, act[b])
                gnorm = torch.autograd.grad(bcl, rep_p, retain_graph=False)[0].norm().item()
                ema = gnorm if ema is None else 0.8 * ema + 0.2 * gnorm
                peak = max(peak, ema)
                if peak > 0 and ema < RELFRAC * peak: released, rel_ep = True, ep
                opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                rep = m.rep(x, state[b], lang[b])
                cond = rep if released else rep.detach()
                loss = fm_loss(m, cond, act[b])
                if aux_tgt is not None: loss = loss + mse(m.aux(rep), aux_tgt[bt])
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
    return m, (rel_ep if rel_ep is not None else EPOCHS)


AUX = {"low (far-past-obs)": TGT["far-past-obs"], "action-rel (fut-action)": TGT["fut-action"]}
res = {"config": {"seed": SEED, "epochs": EPOCHS, "relfrac": RELFRAC}, "runs": []}
bc0 = bc_ood(train(None, "joint")[0]); print(f"BC-only gen = {bc0:+.4f}", flush=True); res["bc0"] = bc0
for name, tgt in AUX.items():
    for mode, tau in [("joint", 0.0), ("fixed", 0.5), ("fixed", 1.0), ("dynamic", 0.0)]:
        m, rel = train(tgt, mode, tau)
        gen = bc_ood(m); lab = f"{mode}" + (f"@{tau}" if mode == "fixed" else f"(rel@ep{rel})")
        res["runs"].append(dict(aux=name, mode=mode, tau=tau, rel_ep=rel, gen=gen, benefit=gen - bc0))
        print(f"  {name:26s} {lab:16s} gen {gen:+.4f}  benefit {gen-bc0:+.4f}", flush=True)

path = f"{OUT}/exp2h_dynki_s{SEED}.json"; json.dump(res, open(path, "w"), indent=1)
print(f"SAVED {path}", flush=True); import os as _os; _os._exit(0)
