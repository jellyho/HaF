"""Token-NTP Expert — GENERIC autoregressive next-token prediction over ANY discrete token stream, conditioned on
the backbone (VLM) features via cross-attention. Mirrors how real VLAs train every discrete objective by NTP
through the transformer (cross-entropy), rather than a bespoke regression head. Vocab-agnostic:

  * ACTION objective  -> official FAST tokens  (AutoProcessor("physical-intelligence/fast"), DCT+BPE, vocab 2048,
    action chunk [B,H,Dact] -> variable-length ids, near-lossless decode) = OpenVLA/RT-2/pi0.5-FAST style.
  * TEXT objectives   -> a text tokenizer's ids (instruction / subtask / caption / CoT) = RT-2/ECoT/pi0.5-subtask
    style. instr-infer, subtask-predict, etc. all become NTP here (just pass a different tokenizer's ids).

Recoverability of the target, measured the "real VLA" way:  R = 1 - CE_val / CE_marg   (CE_marg = unigram baseline).
Companion to `latent_expert.py` (the AHA future-latent expert; continuous flow/mse or discrete VQ). Same
cross-attn-to-backbone coupling => gradients ground the backbone and can be stop-gradded / dynamically released
(KI). Scale via d_model/depth/nhead so the same module serves the mini-VLA and a pi0.5-scale backbone.
"""
import torch, torch.nn as nn


class CausalCrossBlock(nn.Module):
    def __init__(self, dm, nhead, ff_mult=4, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dm); self.sa = nn.MultiheadAttention(dm, nhead, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dm); self.ca = nn.MultiheadAttention(dm, nhead, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dm)
        self.mlp = nn.Sequential(nn.Linear(dm, ff_mult * dm), nn.GELU(), nn.Linear(ff_mult * dm, dm))

    def forward(self, x, ctx, causal_mask, key_pad, ctx_mask=None):
        h = self.n1(x)
        x = x + self.sa(h, h, h, attn_mask=causal_mask, key_padding_mask=key_pad, need_weights=False)[0]
        h = self.n2(x)
        x = x + self.ca(h, ctx, ctx, key_padding_mask=ctx_mask, need_weights=False)[0]
        x = x + self.mlp(self.n3(x))
        return x


class TokenNTPExpert(nn.Module):
    def __init__(self, backbone_dim, vocab=2048, d_model=512, depth=6, nhead=8, max_len=160):
        super().__init__()
        self.vocab = vocab; self.BOS = vocab; self.PAD = vocab + 1
        self.emb = nn.Embedding(vocab + 2, d_model, padding_idx=self.PAD)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        self.ctx_proj = nn.Linear(backbone_dim, d_model)
        self.blocks = nn.ModuleList([CausalCrossBlock(d_model, nhead) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.lm = nn.Linear(d_model, vocab)

    def _run(self, inp, ctx, key_pad, ctx_mask):
        L = inp.shape[1]
        cm = torch.triu(torch.full((L, L), float("-inf"), device=inp.device), 1)
        x = self.emb(inp) + self.pos[:L].unsqueeze(0)
        ctx = self.ctx_proj(ctx)
        for b in self.blocks: x = b(x, ctx, cm, key_pad, ctx_mask)
        return self.lm(self.norm(x))                                    # [B,L,vocab]

    def ntp_loss(self, ctx, tokens, tok_pad, ctx_mask=None):
        """ctx:[B,Tkv,Dbb]  tokens:[B,L] long (padded with PAD)  tok_pad:[B,L] bool (True=pad)."""
        B, L = tokens.shape
        bos = torch.full((B, 1), self.BOS, device=tokens.device, dtype=tokens.dtype)
        inp = torch.cat([bos, tokens[:, :-1]], 1)                       # teacher forcing
        inp_pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device), tok_pad[:, :-1]], 1)
        logits = self._run(inp, ctx, inp_pad, ctx_mask)
        tgt = tokens.clone(); tgt[tok_pad] = -100                       # ignore PAD in CE
        return nn.functional.cross_entropy(logits.reshape(-1, self.vocab), tgt.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def ce_marginal(self, tokens, tok_pad, unigram_logp):
        """CE of the unigram baseline (predict marginal token freq) — the recoverability denominator."""
        valid = tokens[~tok_pad]
        return float(-unigram_logp[valid].mean())


if __name__ == "__main__":
    torch.manual_seed(0); B, Tkv, Dbb = 4, 20, 768
    ctx = torch.randn(B, Tkv, Dbb)
    L = 40; toks = torch.randint(0, 2048, (B, L)); pad = torch.zeros(B, L, dtype=torch.bool); pad[:, 30:] = True
    ex = TokenNTPExpert(Dbb, vocab=2048, d_model=256, depth=2, nhead=4)   # ACTION: FAST vocab
    print("params(mini d=256,depth=2):", sum(p.numel() for p in ex.parameters()) // 1000, "K")
    loss = ex.ntp_loss(ctx, toks, pad)
    print("ntp CE loss:", float(loss), " (random init ~ ln(2048)=%.2f)" % __import__("math").log(2048))
    # grad flows expert->backbone (grounding); detach(ctx) => KI
    ctx2 = torch.randn(B, Tkv, Dbb, requires_grad=True)
    ex.ntp_loss(ctx2, toks, pad).backward()
    print("grad to backbone norm:", float(ctx2.grad.norm()), "(>0 => grounds backbone)")
    # TEXT objective via the SAME module (e.g. instruction NTP, text tokenizer vocab ~32k)
    tx = TokenNTPExpert(Dbb, vocab=32000, d_model=256, depth=2, nhead=4)
    ttoks = torch.randint(0, 32000, (B, 24)); tpad = torch.zeros(B, 24, dtype=torch.bool); tpad[:, 18:] = True
    print("text NTP CE loss:", float(tx.ntp_loss(ctx, ttoks, tpad)), "(random ~ ln32000=%.2f)" % __import__("math").log(32000))
    print("FASTNTPExpert = TokenNTPExpert  # alias kept")
    print("SMOKE OK")

FASTNTPExpert = TokenNTPExpert  # backward-compat alias
