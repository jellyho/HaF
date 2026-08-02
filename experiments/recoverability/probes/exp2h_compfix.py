"""Corrected composition (re-test R3): is the mix's benefit really dominated by its SINGLE lowest-recoverability
member (min), or do diverse hard members add complementarily? R3's "min dominates" was measured under the OLD
concat + mean-MSE head (the loss-normalization artifact of R3b). Here we measure each mix with PER-MEMBER
independent heads + SUMMED loss (each full weight), then recompute which summary (min / mean / joint) predicts benefit.

If min still best-predicts -> R3 robust. If mean/joint rise (complementarity) -> R3 shifts from "hardest member alone
decides" to "hardest member sets the level, diverse hard questions add on top".
Reuses exp2h_mixki mini-VLA. Output: exp2h_compfix_s{SEED}.json. env: SEED, EPOCHS(20), SUBN(4000).
"""
import os, json, numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr

G = {}
with open("/data5/jellyho/Hindsight/HaF/experiments/recoverability/probes/exp2h_mixki.py") as f:
    src = f.read()
exec(compile(src[:src.index("# member single-objective R_deep")], "mixki_setup", "exec"), G)
TGT = G["TGT"]; Model = G["Model"]; fm_loss = G["fm_loss"]; fm_sample = G["fm_sample"]
imgs = G["imgs"]; state = G["state"]; lang = G["lang"]; act = G["act"]
fit_idx = G["fit_idx"]; te_idx = G["te_idx"]; tr_idx = G["tr_idx"]; DEV = G["DEV"]; EPOCHS = G["EPOCHS"]; SEED = G["SEED"]; OUT = G["OUT"]; DMODEL = G["DMODEL"]
mse = nn.MSELoss()

# measured per-member recoverability (exp2h fractal 4k, deep_recoverability)
RECOV = {"far-past-obs":-.152,"final-obs":-.150,"initial-obs":-.134,"far-fut-obs":-.111,"near-fut-obs":-.106,
         "near-past-obs":-.100,"cur-action":-.081,"fut-action":-.006,"prev-action":.101,"fut-gripper":.357,
         "displacement":.372,"initial-pose":.411,"final-pose":.924}
# 12 mixes spanning a range of min-recoverability (some contain a hard obs member, some only easy/shortcut members)
MIXES = [["far-past-obs","far-fut-obs"], ["far-past-obs","final-pose"], ["final-pose","initial-pose"],
         ["far-past-obs","far-fut-obs","near-fut-obs"], ["far-past-obs","far-fut-obs","final-pose"],
         ["final-obs","far-past-obs"], ["final-pose","fut-gripper"], ["far-fut-obs","final-obs","near-past-obs"],
         ["displacement","initial-pose"], ["far-fut-obs","displacement"], ["final-pose","displacement","initial-pose"],
         ["far-past-obs","near-fut-obs","initial-obs"]]

class MultiHead(Model):
    def __init__(self, sdim, ldim, adim, dims):
        super().__init__(sdim, ldim, adim, auxdim=0)
        self.heads = nn.ModuleList([nn.Linear(DMODEL, d) for d in dims])

def train_sum(members):
    tgts = [TGT[m] for m in members]
    m = MultiHead(state.shape[1], lang.shape[1], act.shape[1], [t.shape[1] for t in tgts]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=1e-2); bs=128; fit=tr_idx.copy()
    for ep in range(EPOCHS):
        np.random.shuffle(fit); m.train()
        for i in range(0,len(fit),bs):
            b=fit[i:i+bs]; bt=torch.tensor(b,device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                rep=m.rep(imgs(b),state[b],lang[b]); loss=fm_loss(m,rep,act[b])
                for h,t in zip(m.heads,tgts): loss=loss+mse(h(rep),t[bt])   # per-member, each full weight (SUM)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def bc_ood(m):
    m.eval(); P=[]
    for i in range(0,len(te_idx),128):
        b=te_idx[i:i+128]
        with torch.autocast("cuda",dtype=torch.bfloat16): cond=m.rep(imgs(b),state[b],lang[b])
        P.append(fm_sample(m,cond.float()))
    P=torch.cat(P); Y=act[torch.tensor(te_idx,device=DEV)]
    return 1-torch.mean((P-Y)**2).item()/(torch.mean((Y-Y.mean(0))**2).item()+1e-9)

def train_bc():
    m=Model(state.shape[1],lang.shape[1],act.shape[1],0).to(DEV)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=1e-2); bs=128; fit=tr_idx.copy()
    for ep in range(EPOCHS):
        np.random.shuffle(fit); m.train()
        for i in range(0,len(fit),bs):
            b=fit[i:i+bs]
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=fm_loss(m,m.rep(imgs(b),state[b],lang[b]),act[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m

bc0=bc_ood(train_bc()); print(f"BC-only gen={bc0:+.4f}",flush=True)
rows=[]
for mix in MIXES:
    gen=bc_ood(train_sum(mix)); rs=[RECOV[x] for x in mix]
    rows.append(dict(members=mix, gen=gen, benefit=gen-bc0, min=min(rs), mean=float(np.mean(rs)), max=max(rs)))
    print(f"  {'+'.join(mix):40s} benefit={gen-bc0:+.4f} min={min(rs):+.2f} mean={np.mean(rs):+.2f}",flush=True)

mins=[r["min"] for r in rows]; means=[r["mean"] for r in rows]; bens=[r["benefit"] for r in rows]
res=dict(seed=SEED, bc0=bc0, rows=rows,
         pearson_min=pearsonr(mins,bens)[0], pearson_mean=pearsonr(means,bens)[0],
         spearman_min=spearmanr(mins,bens)[0], spearman_mean=spearmanr(means,bens)[0])
print(f"\nCORRECTED composition (per-member SUM): Pearson min={res['pearson_min']:+.3f} mean={res['pearson_mean']:+.3f}"
      f"  |  Spearman min={res['spearman_min']:+.3f} mean={res['spearman_mean']:+.3f}",flush=True)
json.dump(res, open(f"{OUT}/exp2h_compfix_s{SEED}.json","w"), indent=1); print("SAVED",flush=True)
import os as _os; _os._exit(0)
