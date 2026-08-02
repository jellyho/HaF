"""Where do the DIFFERENT TARGET FORMS sit on the recoverability axis? — measured the "real VLA" way.

For the ACTION target, three forms sharing ONE backbone (mini-VLA fused token sequence as cross-attn context):
  cont-flow   : continuous flow-matching regression (pi0 action-expert style)      R = 1 - MSE_val/MSE_marg
  FAST-NTP    : official FAST tokens, autoregressive NTP (OpenVLA/pi0.5-FAST style) R = 1 - CE_val/CE_marg
  bin256-NTP  : OpenVLA-style 256 uniform bins per dim, NTP                          R = 1 - CE_val/CE_marg
For the INSTRUCTION target:
  text-embed  : predict the MiniLM sentence embedding (regression)                   R = 1 - MSE_val/MSE_marg
  text-NTP    : autoregressive NTP of the instruction's WordPiece tokens (RT-2 style) R = 1 - CE_val/CE_marg
Recoverability is normalized (fraction of the marginal-baseline loss removed) so forms are comparable.
Reuses the exp2h_mixki mini-VLA. Run in .venv (torch+transformers+sklearn+FAST). Output: exp2h_actiontok_s{SEED}.json
env: SEED, EPOCHS(default 15), SUBN(default 6000).
"""
import os, json, math, numpy as np, torch, torch.nn as nn
import sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
from token_ntp_expert import TokenNTPExpert

G = {}
with open("/data5/jellyho/Hindsight/HaF/experiments/recoverability/probes/exp2h_mixki.py") as f:
    src = f.read()
exec(compile(src[:src.index("# member single-objective R_deep")], "mixki_setup", "exec"), G)
Model = G["Model"]; imgs = G["imgs"]; state = G["state"]; lang = G["lang"]; act = G["act"]
d = G["d"]; fit_idx = G["fit_idx"]; val_idx = G["val_idx"]; DEV = G["DEV"]; EPOCHS = G["EPOCHS"]
SEED = G["SEED"]; OUT = G["OUT"]; DMODEL = G["DMODEL"]
mse = nn.MSELoss()


class Backbone(Model):
    """Expose the fused TOKEN SEQUENCE [B,3,dm] (img,state,lang) as cross-attn context for the experts."""
    def rep_seq(self, x, s, l):
        rimg = self.enc(pixel_values=x).last_hidden_state[:, 0]
        toks = torch.stack([self.p_img(rimg), self.p_state(s), self.p_lang(l)], 1) + self.type_emb.unsqueeze(0)
        return self.backbone(toks)                                   # [B,3,dm]


# ---- build the target tokenizations (once) ----
raw_act = d["act_chunk"].astype(np.float32)                          # [N,15,7]
N = len(raw_act)

# FAST tokens
from transformers import AutoProcessor, AutoTokenizer
_fast = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
fast_ids = _fast(raw_act)                                            # list of variable-length id lists
FAST_V = 2048; FLmax = min(160, max(len(t) for t in fast_ids))
def pad_ids(ids, L, PAD):
    out = np.full((len(ids), L), PAD, np.int64); m = np.ones((len(ids), L), bool)
    for i, t in enumerate(ids):
        t = list(t)[:L]; out[i, :len(t)] = t; m[i, :len(t)] = False
    return out, m
fast_tok, fast_pad = pad_ids(fast_ids, FLmax, FAST_V + 1)
fast_tok = torch.tensor(fast_tok, device=DEV); fast_pad = torch.tensor(fast_pad, device=DEV)
fast_uni = np.bincount(np.concatenate([np.asarray(t) for t in fast_ids]), minlength=FAST_V).astype(np.float64) + 1
fast_logp = torch.tensor(np.log(fast_uni / fast_uni.sum()), device=DEV, dtype=torch.float32)

# OpenVLA 256 uniform bins per dim (on the flat 105-d chunk), NTP
BIN_V = 256; flat = raw_act.reshape(N, -1)                           # [N,105]
lo = np.percentile(flat[fit_idx], 1, 0); hi = np.percentile(flat[fit_idx], 99, 0)
binned = np.clip(((flat - lo) / (hi - lo + 1e-8) * BIN_V).astype(np.int64), 0, BIN_V - 1)  # [N,105] 0..255
bin_tok = torch.tensor(binned, device=DEV); bin_pad = torch.zeros_like(bin_tok, dtype=torch.bool)
bin_uni = np.bincount(binned[fit_idx].reshape(-1), minlength=BIN_V).astype(np.float64) + 1
bin_logp = torch.tensor(np.log(bin_uni / bin_uni.sum()), device=DEV, dtype=torch.float32)

# instruction: MiniLM WordPiece ids (reuse the loaded tokenizer's vocab) + its embedding target
_tk = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
enc_txt = _tk([str(s) for s in d["instr"]], padding="max_length", truncation=True, max_length=16, return_tensors="np")
TXT_V = _tk.vocab_size
txt_ids = enc_txt["input_ids"].astype(np.int64); txt_attn = enc_txt["attention_mask"].astype(bool)
txt_tok = torch.tensor(txt_ids, device=DEV); txt_pad = torch.tensor(~txt_attn, device=DEV)
txt_uni = np.bincount(txt_ids[fit_idx][txt_attn[fit_idx]], minlength=TXT_V).astype(np.float64) + 1
txt_logp = torch.tensor(np.log(txt_uni / txt_uni.sum()), device=DEV, dtype=torch.float32)
lang_tgt = lang                                                     # MiniLM embedding target (for text-embed)

print(f"N={N} FAST_L={FLmax} bins={BIN_V} txt_V={TXT_V}", flush=True)


def marg_ce(logp, tok, pad):
    v = tok[torch.tensor(val_idx, device=DEV)][~pad[torch.tensor(val_idx, device=DEV)]]
    return float(-logp[v].mean())

def marg_mse(tgt):
    Yv = tgt[torch.tensor(val_idx, device=DEV)]; mu = tgt[torch.tensor(fit_idx, device=DEV)].mean(0)
    return float(torch.mean((Yv - mu) ** 2))


def train_measure(kind, tgt=None, tok=None, pad=None, logp=None, vocab=None):
    """Train backbone + one head on a single objective; return recoverability 1 - L_val/L_marg."""
    m = Backbone(state.shape[1], lang.shape[1], act.shape[1], 0).to(DEV)
    if kind == "ntp":
        head = TokenNTPExpert(DMODEL, vocab=vocab, d_model=256, depth=2, nhead=4, max_len=tok.shape[1] + 2).to(DEV)
    elif kind == "mse":
        head = nn.Linear(DMODEL, tgt.shape[1]).to(DEV)
    params = list(m.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=3e-5, weight_decay=1e-2); bs = 96; fit = fit_idx.copy()
    for ep in range(EPOCHS):
        np.random.shuffle(fit); m.train(); head.train()
        for i in range(0, len(fit), bs):
            b = fit[i:i+bs]; bt = torch.tensor(b, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = m.rep_seq(imgs(b), state[b], lang[b])
                if kind == "ntp":
                    loss = head.ntp_loss(ctx, tok[bt], pad[bt])
                else:
                    loss = mse(head(ctx.mean(1)), tgt[bt])
            opt.zero_grad(); loss.backward(); opt.step()
    # val loss
    m.eval(); head.eval(); tot = 0.0; nb = 0
    with torch.no_grad():
        for i in range(0, len(val_idx), bs):
            b = val_idx[i:i+bs]; bt = torch.tensor(b, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ctx = m.rep_seq(imgs(b), state[b], lang[b])
                if kind == "ntp":
                    l = head.ntp_loss(ctx, tok[bt], pad[bt]).item()
                else:
                    l = mse(head(ctx.mean(1)), tgt[bt]).item()
            tot += l * len(b); nb += len(b)
    Lval = tot / nb
    Lmarg = marg_ce(logp, tok, pad) if kind == "ntp" else marg_mse(tgt)
    return 1 - Lval / (Lmarg + 1e-9), Lval, Lmarg


res = {"config": {"seed": SEED, "epochs": EPOCHS, "N": int(N)}, "recov": {}}
JOBS = [
    ("action:FAST-NTP",   dict(kind="ntp", tok=fast_tok, pad=fast_pad, logp=fast_logp, vocab=FAST_V)),
    ("action:bin256-NTP", dict(kind="ntp", tok=bin_tok,  pad=bin_pad,  logp=bin_logp,  vocab=BIN_V)),
    ("instr:text-NTP",    dict(kind="ntp", tok=txt_tok,  pad=txt_pad,  logp=txt_logp,  vocab=TXT_V)),
    ("action:cont-flow",  dict(kind="mse", tgt=act)),           # continuous chunk regression (flow proxy: MSE recov)
    ("instr:text-embed",  dict(kind="mse", tgt=lang_tgt)),
]
for name, kw in JOBS:
    R, Lv, Lm = train_measure(**kw)
    res["recov"][name] = dict(R=R, Lval=Lv, Lmarg=Lm)
    print(f"  {name:20s} R={R:+.4f}  (Lval={Lv:.4f} Lmarg={Lm:.4f})", flush=True)

path = f"{OUT}/exp2h_actiontok_s{SEED}.json"; json.dump(res, open(path, "w"), indent=1)
print(f"SAVED {path}", flush=True); import os as _os; _os._exit(0)
