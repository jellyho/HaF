"""SmolVLM-VLA (fast path) — SmolVLA's engineering recipe, without their LM truncation.

Key wins copied from LeRobot/SmolVLA (verified against their source):
  * BYPASS the HF image processor entirely. Images stay uint8 tensors, are resized+normalized ON GPU, and go
    straight into vision_model -> connector. This removes the CPU PIL/processor bottleneck (~130 samp/s ceiling,
    ~65 with a second aux forward) and all Idefics3 tiling bookkeeping.
  * 512x512 with connector scale_factor=4 -> exactly 64 visual tokens per image.
  * FREEZE the SigLIP vision tower (requires_grad=False AND forced .eval()) — ~93M params off the optimizer.
  * Flow matching with Beta(1.5,1.0) timesteps, single random t per sample, 10 Euler steps at inference.
  * SDPA attention (their eager fp32 path is O(L^2) and is the one thing NOT to copy).
Deliberately NOT copied (they break the LM's pretrained behaviour, which our auxiliary objectives depend on):
  their hardcoded RoPE max_wavelength=10_000 (we keep the checkpoint's rope_theta=100_000), the sqrt(dim)
  embedding rescaling, the PaliGemma prefix-LM mask, and the LM layer truncation (we keep all 30 layers).

Text is tokenized ONCE per unique instruction (fractal has ~500 unique strings) and cached — no per-step tokenizer.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HAFTorchConfig

IMAGENET_MEAN = None  # SigLIP path uses [-1,1], not ImageNet stats


def resize_with_pad(img: torch.Tensor, h: int, w: int, pad_value: float = 0.0) -> torch.Tensor:
    """Aspect-preserving resize + pad (SmolVLA convention: pad on left/top). img: [B,3,H,W] float."""
    b, c, ih, iw = img.shape
    r = min(h / ih, w / iw)
    nh, nw = max(1, int(round(ih * r))), max(1, int(round(iw * r)))
    x = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)
    out = img.new_full((b, c, h, w), pad_value)
    out[:, :, h - nh:, w - nw:] = x
    return out


class FlowActionExpert(nn.Module):
    """Flow-matching action head over the pooled VLM representation (pi0/SmolVLA-style, MSE on velocity)."""

    def __init__(self, cond_dim: int, action_dim: int, horizon: int, width: int = 512, depth: int = 2, t_dim: int = 64):
        super().__init__()
        self.adim = action_dim * horizon
        self.action_dim, self.horizon, self.t_dim = action_dim, horizon, t_dim
        layers, d_in = [], self.adim + cond_dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.GELU()]; d_in = width
        layers += [nn.Linear(d_in, self.adim)]
        self.net = nn.Sequential(*layers)

    def _temb(self, t):
        half = self.t_dim // 2
        fr = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        a = t.float() * fr[None]
        return torch.cat([torch.cos(a), torch.sin(a)], -1).to(t.dtype)

    def velocity(self, x_t, t, cond):
        return self.net(torch.cat([x_t, cond, self._temb(t)], -1))

    def loss(self, cond, actions):
        # SmolVLA samples t ~ Beta(1.5,1.0)*0.999+0.001 (favours the noisy end)
        b = actions.shape[0]
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        t = (beta.sample((b, 1)).to(actions.device) * 0.999 + 0.001).to(actions.dtype)
        eps = torch.randn_like(actions)
        x_t = (1 - t) * eps + t * actions
        return ((self.velocity(x_t, t, cond) - (actions - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond, steps: int = 10, n_samples: int = 8):
        B = cond.shape[0]
        c = cond.repeat_interleave(n_samples, 0)
        x = torch.randn(B * n_samples, self.adim, device=cond.device, dtype=cond.dtype)
        for k in range(steps):
            t = torch.full((B * n_samples, 1), k / steps, device=cond.device, dtype=cond.dtype)
            x = x + (1.0 / steps) * self.velocity(x, t, c)
        return x.view(B, n_samples, self.adim).mean(1)


class SmolVLMFastVLA(nn.Module):
    """Inputs are GPU tensors: images uint8 [B,H,W,3] and pre-tokenized instruction ids [B,L]."""

    def __init__(self, cfg: HAFTorchConfig):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.cfg = cfg
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[cfg.dtype]
        full = AutoModelForImageTextToText.from_pretrained(cfg.vlm_id, torch_dtype=dtype,
                                                           attn_implementation="sdpa")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.vlm_id)
        inner = full.model
        self.vision = inner.vision_model          # SigLIP tower (frozen)
        self.connector = inner.connector          # pixel-shuffle -> 64 tokens/img
        self.lm = inner.text_model                # ALL 30 layers (we need them for NTP/aux)
        self.width = int(self.lm.config.hidden_size)

        for p in self.vision.parameters():        # SmolVLA freezes the vision tower
            p.requires_grad_(False)
        if cfg.freeze_vlm:
            for p in self.lm.parameters():
                p.requires_grad_(False)
            for p in self.connector.parameters():
                p.requires_grad_(False)

        self.expert = FlowActionExpert(self.width, cfg.action_dim, cfg.action_horizon,
                                       cfg.expert_width, cfg.expert_depth)
        self.aux_head = nn.Linear(self.width, self.width) if cfg.aux_loss_weight > 0 else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision.eval()                        # keep the frozen tower in eval (SmolVLA does this too)
        return self

    # ---- image path: no processor, all on GPU ----
    def encode_images(self, images_u8: torch.Tensor) -> torch.Tensor:
        """images_u8: [B,H,W,3] uint8 on GPU -> [B, 64, width] visual tokens."""
        x = images_u8.permute(0, 3, 1, 2).float() / 255.0
        x = resize_with_pad(x, self.cfg.vlm_image_size, self.cfg.vlm_image_size)
        x = (x * 2.0 - 1.0).to(next(self.connector.parameters()).dtype)      # SigLIP expects [-1,1]
        with torch.no_grad() if not any(p.requires_grad for p in self.vision.parameters()) else torch.enable_grad():
            v = self.vision(pixel_values=x).last_hidden_state
        return self.connector(v)

    def rep(self, images_u8: torch.Tensor, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """Joint V-L representation: [visual tokens ‖ text embeddings] through the LM, mean-pooled."""
        vis = self.encode_images(images_u8)                                   # [B,64,D]
        txt = self.lm.embed_tokens(text_ids)                                  # [B,L,D]
        h = torch.cat([vis, txt], dim=1)
        mask = torch.cat([torch.ones(vis.shape[:2], device=vis.device, dtype=text_mask.dtype), text_mask], dim=1)
        out = self.lm(inputs_embeds=h, attention_mask=mask).last_hidden_state  # [B,64+L,D]
        m = mask[..., None].to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1e-6)

    # ---- objectives ----
    def compute_loss(self, images_u8, text_ids, text_mask, actions,
                     aux_images_u8=None, future_images_u8=None):
        cfg = self.cfg
        rep = self.rep(images_u8, text_ids, text_mask)
        parts, total = {}, rep.new_zeros(())
        if cfg.enable_action_training:
            cond = rep.detach() if cfg.stop_bc_to_vlm_grad else rep
            l = self.expert.loss(cond, actions); parts["bc"] = l.detach()
            total = total + cfg.action_loss_weight * l
        if cfg.aux_loss_weight > 0 and self.aux_head is not None:
            if cfg.aux_family == "mask":
                src = self.rep(aux_images_u8, text_ids, text_mask); tgt = rep.detach()
            elif cfg.aux_family == "noise":
                src = rep; tgt = rep.detach()
                if cfg.aux_noise_sigma > 0:
                    tgt = tgt + cfg.aux_noise_sigma * torch.randn_like(tgt)
            elif cfg.aux_family == "future":
                src = rep
                with torch.no_grad():
                    tgt = self.rep(future_images_u8, text_ids, text_mask)
            else:
                raise ValueError(cfg.aux_family)
            if cfg.stop_aux_to_vlm_grad:
                src = src.detach()
            pred = self.aux_head(src)
            pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
            tn = tgt / (tgt.norm(dim=-1, keepdim=True) + 1e-8)
            l = (1.0 - (pn * tn).sum(-1)).mean(); parts["aux"] = l.detach()
            total = total + cfg.aux_loss_weight * l
        return total, parts

    @torch.no_grad()
    def sample_actions(self, images_u8, text_ids, text_mask):
        rep = self.rep(images_u8, text_ids, text_mask)
        return self.expert.sample(rep, self.cfg.flow_steps_sample, self.cfg.flow_samples)
