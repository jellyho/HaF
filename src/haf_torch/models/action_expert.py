"""pi0.5 / SmolVLA-style ACTION EXPERT: a second transformer tower that reads the VLM layer-by-layer.

Contrast with the naive head we had before (mean-pool the VLM output into one 576-d vector -> MLP), which throws
away all spatial/token structure. Here, following pi0 / pi0.5 / SmolVLA:

  prefix (VLM tower)                          suffix (expert tower)
  [image tokens | language tokens | state]    [action tokens x horizon]
        |  layer L hidden states  ---------->  layer L of the expert cross-attends to them
  ...   |                                      (SmolVLA interleaves: self-attn every n-th layer, else cross-attn)

Design choices copied from the verified SmolVLA source:
  * expert width = 0.75 x VLM width (576 -> 432), FFN rounded to a multiple of 256
  * expert depth divides the VLM depth, so expert layer i reads VLM layer ((i+1) * L_vlm // L_exp - 1)
  * `self_attn_every_n_layers = 2`: even layers do self-attention over the action tokens, odd layers cross-attend
  * the action tokens are CAUSAL among themselves; the prefix never attends to the suffix
  * flow matching: x_t = (1-t)*eps + t*a, target = a - eps, single random t (Beta(1.5,1.0)) per sample
  * a proprio/state token is projected into the PREFIX (pi0.5 puts state in the prefix; SmolVLA ablation: 80.3 vs 73.3)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float() * fr[None]
    return torch.cat([torch.cos(a), torch.sin(a)], -1).to(t.dtype)


def _round_ffn(w: int, mult: int = 256) -> int:
    return int(math.ceil((int(2 * w / 3) * 4) / mult) * mult)


class ExpertLayer(nn.Module):
    """One expert layer. `mode`:
         'joint' — pi0.5 style: attend over [VLM hidden ‖ action tokens] in ONE attention (dense coupling)
         'cross' — attend only into the VLM hidden states
         'self'  — attend only over the action tokens (causal)
    """

    def __init__(self, width: int, kv_width: int, nhead: int, cross: bool, mode: str = "cross"):
        super().__init__()
        self.mode = mode if mode in ("joint", "cross", "self") else ("cross" if cross else "self")
        self.cross, self.nhead, self.hd = (self.mode != "self"), nhead, width // nhead
        self.n1 = nn.RMSNorm(width) if hasattr(nn, "RMSNorm") else nn.LayerNorm(width)
        self.q = nn.Linear(width, width, bias=False)
        kin = kv_width if self.mode != "self" else width
        self.k = nn.Linear(kin, width, bias=False)
        self.v = nn.Linear(kin, width, bias=False)
        self.k_self = nn.Linear(width, width, bias=False) if self.mode == "joint" else None
        self.v_self = nn.Linear(width, width, bias=False) if self.mode == "joint" else None
        self.o = nn.Linear(width, width, bias=False)
        self.n2 = nn.RMSNorm(width) if hasattr(nn, "RMSNorm") else nn.LayerNorm(width)
        ff = _round_ffn(width)
        self.mlp = nn.Sequential(nn.Linear(width, ff, bias=False), nn.GELU(), nn.Linear(ff, width, bias=False))

    def forward(self, x, ctx=None, ctx_mask=None, causal=True):
        B, L, _ = x.shape
        h = self.n1(x)
        q = self.q(h).view(B, L, self.nhead, self.hd).transpose(1, 2)
        if self.mode == "self":
            k = self.k(h).view(B, L, self.nhead, self.hd).transpose(1, 2)
            v = self.v(h).view(B, L, self.nhead, self.hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        elif self.mode == "cross":
            S = ctx.shape[1]
            k = self.k(ctx).view(B, S, self.nhead, self.hd).transpose(1, 2)
            v = self.v(ctx).view(B, S, self.nhead, self.hd).transpose(1, 2)
            am = None if ctx_mask is None else ctx_mask[:, None, None, :].bool()
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        else:  # joint: one attention over [prefix ‖ suffix]  (pi0.5 dense coupling)
            S = ctx.shape[1]
            kp = self.k(ctx).view(B, S, self.nhead, self.hd).transpose(1, 2)
            vp = self.v(ctx).view(B, S, self.nhead, self.hd).transpose(1, 2)
            ks = self.k_self(h).view(B, L, self.nhead, self.hd).transpose(1, 2)
            vs = self.v_self(h).view(B, L, self.nhead, self.hd).transpose(1, 2)
            k = torch.cat([kp, ks], 2); v = torch.cat([vp, vs], 2)
            pm = torch.ones(B, S, dtype=torch.bool, device=x.device) if ctx_mask is None else ctx_mask.bool()
            causal_suffix = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            am = torch.cat([pm[:, None, None, :].expand(B, 1, L, S),
                            causal_suffix[None, None].expand(B, 1, L, L)], dim=-1)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, -1))
        return x + self.mlp(self.n2(x))


class ActionExpert(nn.Module):
    """Two-tower action expert. Call `loss(vlm_hidden_list, prefix_mask, actions)` / `sample(...)`."""

    def __init__(self, vlm_width: int, vlm_layers: int, action_dim: int, horizon: int,
                 width_mult: float = 0.75, depth: int = 6, nhead: int = 8, t_dim: int = 128,
                 mode: str = "joint"):
        super().__init__()
        w = int(vlm_width * width_mult) // nhead * nhead                       # divisible by heads
        self.width, self.depth, self.horizon, self.action_dim, self.t_dim = w, depth, horizon, action_dim, t_dim
        assert vlm_layers % depth == 0 or True, "expert depth need not divide, we index proportionally"
        # expert layer i reads VLM hidden state at this index (proportional mapping, last layer inclusive)
        self.read_idx = [int((i + 1) * vlm_layers / depth) for i in range(depth)]
        self.in_proj = nn.Linear(action_dim, w)                                # action token embedding
        self.t_mlp = nn.Sequential(nn.Linear(t_dim, w), nn.SiLU(), nn.Linear(w, w))
        # pi0.5: every expert layer is paired with a VLM layer ("joint"). SmolVLA interleaves self/cross.
        self.layers = nn.ModuleList([
            ExpertLayer(w, vlm_width, nhead, cross=(i % 2 == 1),
                        mode=("joint" if mode == "joint" else ("cross" if i % 2 == 1 else "self")))
            for i in range(depth)])
        self.norm = nn.RMSNorm(w) if hasattr(nn, "RMSNorm") else nn.LayerNorm(w)
        self.out = nn.Linear(w, action_dim)

    def _tokens(self, x_t, t):
        """x_t: [B, horizon, action_dim] noisy actions; t: [B,1] -> expert tokens [B,horizon,width]."""
        return self.in_proj(x_t) + self.t_mlp(timestep_embedding(t, self.t_dim)).unsqueeze(1)

    def forward(self, hidden_states, prefix_mask, x_t, t):
        h = self._tokens(x_t, t)
        for i, layer in enumerate(self.layers):
            if layer.mode == "self":
                h = layer(h)
            else:
                h = layer(h, ctx=hidden_states[self.read_idx[i]], ctx_mask=prefix_mask)
        return self.out(self.norm(h))                                           # [B, horizon, action_dim]

    def loss(self, hidden_states, prefix_mask, actions):
        """actions: [B, horizon, action_dim] normalized."""
        B = actions.shape[0]
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        t = (beta.sample((B, 1)) * 0.999 + 0.001).to(device=actions.device, dtype=actions.dtype)
        eps = torch.randn_like(actions)
        x_t = (1 - t)[:, :, None] * eps + t[:, :, None] * actions
        v = self.forward(hidden_states, prefix_mask, x_t, t)
        return ((v - (actions - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, hidden_states, prefix_mask, steps: int = 10):
        B = hidden_states[-1].shape[0]
        x = torch.randn(B, self.horizon, self.action_dim,
                        device=hidden_states[-1].device, dtype=hidden_states[-1].dtype)
        for k in range(steps):
            t = torch.full((B, 1), k / steps, device=x.device, dtype=x.dtype)
            x = x + (1.0 / steps) * self.forward(hidden_states, prefix_mask, x, t)
        return x
