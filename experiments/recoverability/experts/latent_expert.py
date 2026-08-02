"""Latent-Prediction Expert — a future-frame-latent expert analogous to the pi0/pi0.5 flow-matching ACTION expert.

Design mirrors the action expert: a separate transformer ("expert") that READS the VLM/backbone features and
produces a prediction. Here the target is a FUTURE-FRAME LATENT (the AHA low-recoverability hard question), not an
action. It plugs into any backbone that exposes a feature sequence [B, T_kv, D_bb] (mini-VLA fused tokens OR a
pi0.5-scale PaliGemma hidden-state sequence), so the same module scales from the mini-VLA to full VLA.

Coupling to the backbone = cross-attention (the expert queries attend to projected backbone features). This is the
standalone analogue of pi0's joint MoE attention; crucially it lets gradients flow expert->backbone so the
low-recoverability target GROUNDS the backbone — and that flow can be stop-gradded / dynamically released (KI).

Heads (choose by `head=`), all on the paper's recoverability axis:
  'flow' : flow-matching regression of the latent (like the action expert). Default.
  'mse'  : plain L2 regression of the latent (continuous embedding prediction).
  'vq'   : discrete codebook cross-entropy (VQ-image / discrete frame tokens). Needs `vq_ncode`, `vq_ntok`.

Scale by (d_model, depth, heads, num_queries). num_queries = # latent tokens to predict (1 for a pooled CLS latent;
=grid size for a VQ token grid).
"""
import math, torch, torch.nn as nn


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float().view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2: emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], -1)
    return emb


class ExpertBlock(nn.Module):
    """One expert layer: self-attn over expert tokens + cross-attn into backbone features + MLP (pre-norm)."""
    def __init__(self, dm, nhead, ff_mult=4, dropout=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dm); self.sa = nn.MultiheadAttention(dm, nhead, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(dm); self.ca = nn.MultiheadAttention(dm, nhead, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(dm)
        self.mlp = nn.Sequential(nn.Linear(dm, ff_mult * dm), nn.GELU(), nn.Linear(ff_mult * dm, dm))

    def forward(self, x, ctx, ctx_mask=None):
        h = self.n1(x); x = x + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(x); x = x + self.ca(h, ctx, ctx, key_padding_mask=ctx_mask, need_weights=False)[0]
        x = x + self.mlp(self.n3(x))
        return x


class LatentPredictionExpert(nn.Module):
    def __init__(self, backbone_dim, latent_dim, d_model=1024, depth=6, nhead=8, num_queries=1,
                 head="flow", vq_ncode=None, vq_ntok=None, time_dim=256):
        super().__init__()
        assert head in ("flow", "mse", "vq")
        self.head_kind, self.num_queries, self.latent_dim = head, num_queries, latent_dim
        self.ctx_proj = nn.Linear(backbone_dim, d_model)                 # project backbone features -> expert width
        self.query = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)
        self.blocks = nn.ModuleList([ExpertBlock(d_model, nhead) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(d_model)
        if head == "flow":
            self.in_proj = nn.Linear(latent_dim, d_model)               # noised latent -> token
            self.t_mlp = nn.Sequential(nn.Linear(time_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.time_dim = time_dim
            self.out = nn.Linear(d_model, latent_dim)                    # velocity
        elif head == "mse":
            self.out = nn.Linear(d_model, latent_dim)
        else:  # vq
            assert vq_ncode and vq_ntok and num_queries == vq_ntok, "vq: num_queries must equal vq_ntok (grid size)"
            self.vq_ncode = vq_ncode
            self.out = nn.Linear(d_model, vq_ncode)                     # per-token codebook logits

    def _trunk(self, tokens, ctx, ctx_mask):
        ctx = self.ctx_proj(ctx)
        for blk in self.blocks: tokens = blk(tokens, ctx, ctx_mask)
        return self.out_norm(tokens)

    # ---- flow-matching (like the action expert) ----
    def flow_loss(self, ctx, target, ctx_mask=None):
        """ctx:[B,Tkv,Dbb]  target:[B,num_queries,latent_dim] (or [B,latent_dim] if num_queries==1)."""
        B = ctx.shape[0]; a = target.view(B, self.num_queries, self.latent_dim)
        eps = torch.randn_like(a); t = torch.rand(B, 1, 1, device=a.device)
        x_t = (1 - t) * eps + t * a
        tok = self.in_proj(x_t) + self.t_mlp(timestep_embedding(t.view(B), self.time_dim)).unsqueeze(1)
        v = self.out(self._trunk(tok, ctx, ctx_mask))
        return nn.functional.mse_loss(v, a - eps)

    @torch.no_grad()
    def sample(self, ctx, steps=10, ctx_mask=None):
        B = ctx.shape[0]; x = torch.randn(B, self.num_queries, self.latent_dim, device=ctx.device)
        for k in range(steps):
            t = torch.full((B,), k / steps, device=ctx.device)
            tok = self.in_proj(x) + self.t_mlp(timestep_embedding(t, self.time_dim)).unsqueeze(1)
            x = x + (1.0 / steps) * self.out(self._trunk(tok, ctx, ctx_mask))
        return x

    # ---- plain regression ----
    def mse_loss(self, ctx, target, ctx_mask=None):
        B = ctx.shape[0]; q = self.query.unsqueeze(0).expand(B, -1, -1)
        pred = self.out(self._trunk(q, ctx, ctx_mask))
        return nn.functional.mse_loss(pred, target.view(B, self.num_queries, self.latent_dim))

    # ---- discrete VQ codes ----
    def vq_loss(self, ctx, codes, ctx_mask=None):
        """codes:[B,num_queries] long in [0,vq_ncode)."""
        B = ctx.shape[0]; q = self.query.unsqueeze(0).expand(B, -1, -1)
        logits = self.out(self._trunk(q, ctx, ctx_mask))               # [B,ntok,ncode]
        return nn.functional.cross_entropy(logits.reshape(-1, self.vq_ncode), codes.reshape(-1))

    def loss(self, ctx, target, ctx_mask=None):
        return {"flow": self.flow_loss, "mse": self.mse_loss, "vq": self.vq_loss}[self.head_kind](ctx, target, ctx_mask)


if __name__ == "__main__":
    torch.manual_seed(0); B, Tkv, Dbb, Dlat = 4, 20, 768, 768
    ctx = torch.randn(B, Tkv, Dbb)
    print("params(mini d=256,depth=2):",
          sum(p.numel() for p in LatentPredictionExpert(Dbb, Dlat, d_model=256, depth=2, num_queries=1).parameters()) // 1000, "K")
    # flow head (like action expert)
    ex = LatentPredictionExpert(Dbb, Dlat, d_model=256, depth=2, num_queries=1, head="flow")
    tgt = torch.randn(B, Dlat)
    print("flow_loss", float(ex.flow_loss(ctx, tgt)), " sample", tuple(ex.sample(ctx, steps=4).shape))
    # mse head
    exm = LatentPredictionExpert(Dbb, Dlat, d_model=256, depth=2, num_queries=1, head="mse")
    print("mse_loss ", float(exm.mse_loss(ctx, tgt)))
    # vq head (256-code grid of 16 tokens)
    exv = LatentPredictionExpert(Dbb, Dlat, d_model=256, depth=2, num_queries=16, head="vq", vq_ncode=256, vq_ntok=16)
    codes = torch.randint(0, 256, (B, 16))
    print("vq_loss  ", float(exv.vq_loss(ctx, codes)))
    # grad flows expert->backbone (grounding); can be detached for KI
    ctx2 = torch.randn(B, Tkv, Dbb, requires_grad=True)
    ex.flow_loss(ctx2, tgt).backward()
    print("grad to backbone ctx norm:", float(ctx2.grad.norm()), "(>0 => expert grounds backbone; detach() => KI)")
    print("SMOKE OK")
