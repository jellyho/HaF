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
from .action_expert import ActionExpert

IMAGENET_MEAN = None  # SigLIP path uses [-1,1], not ImageNet stats


def resize_for_vlm(img: torch.Tensor, size: int, mode: str = "bicubic") -> torch.Tensor:
    """Match SmolVLM's processor EXACTLY: it does NOT preserve aspect ratio — it stretches to size x size.
    (Measured against the processor: stretch+bicubic gives mean|diff| 0.0006 vs 0.467 for aspect-preserving pad.)
    img: [B,3,H,W] float in [0,1] -> [B,3,size,size] in [-1,1] (processor mean/std = 0.5 => x*2-1)."""
    x = F.interpolate(img, size=(size, size), mode=mode, align_corners=False, antialias=True)
    return x * 2.0 - 1.0


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
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        inner = full.model
        self.vision = inner.vision_model          # SigLIP tower (frozen)
        self.connector = inner.connector          # pixel-shuffle -> 64 tokens/img
        self.lm = inner.text_model                # ALL 30 layers (we need them for NTP/aux)
        self.lm_head = full.lm_head               # kept so token-level (NTP) objectives are possible
        self.width = int(self.lm.config.hidden_size)

        if cfg.freeze_vision:                     # SmolVLA freezes it; pi0.5/LAP (our default) trains it
            for p in self.vision.parameters():
                p.requires_grad_(False)
        if cfg.freeze_vlm:
            for p in self.lm.parameters():
                p.requires_grad_(False)
            for p in self.connector.parameters():
                p.requires_grad_(False)

        # proprio/state token lives in the PREFIX (pi0.5; SmolVLA ablation: prefix 80.3 vs suffix 73.3)
        self.state_proj = nn.Linear(cfg.state_dim, self.width).to(dtype)
        self.expert = ActionExpert(self.width, len(self.lm.layers), cfg.action_dim, cfg.action_horizon,
                                   cfg.expert_width_mult, cfg.expert_depth, cfg.expert_heads,
                                   mode=cfg.expert_mode).to(dtype)
        self.aux_head = nn.Linear(self.width, self.width).to(dtype) if cfg.aux_loss_weight > 0 else None
        self.param_dtype = dtype

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_vision:
            self.vision.eval()                    # only force eval when the tower is actually frozen
        return self

    # ---- image path: no processor, all on GPU ----
    def encode_images(self, images_u8: torch.Tensor) -> torch.Tensor:
        """images_u8: [B,H,W,3] uint8 on GPU -> [B, 64, width] visual tokens."""
        x = images_u8.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.cfg.vlm_image_size or x.shape[-2] != self.cfg.vlm_image_size:
            x = resize_for_vlm(x, self.cfg.vlm_image_size, self.cfg.resize_mode)   # fallback (GPU bicubic)
        else:
            x = x * 2.0 - 1.0                                                # workers already did LANCZOS
        x = x.to(next(self.connector.parameters()).dtype)
        if self.cfg.freeze_vision:
            with torch.no_grad():
                v = self.vision(pixel_values=x).last_hidden_state
        else:
            v = self.vision(pixel_values=x).last_hidden_state       # trained end-to-end (pi0.5/LAP style)
        return self.connector(v)

    def embed_joint(self, images_u8: torch.Tensor, text_ids: torch.Tensor):
        """Build the SAME sequence the processor produces (no-split template):
             <|im_start|>User:<fake_token_around_image><global-img>[<image> x64]<fake...>TEXT<end_of_utterance>\nAssistant:
        i.e. text embeddings with our 64 visual tokens SCATTERED into the <image> placeholder positions.
        Prepending them instead (what we did first) yields garbage — the template structure matters."""
        vis = self.encode_images(images_u8).to(self.param_dtype)              # [B,n_img,D]
        emb = self.lm.embed_tokens(text_ids).to(self.param_dtype)             # [B,L,D]
        img_pos = (text_ids == self.image_token_id)                           # [B,L] bool
        n_slots = int(img_pos[0].sum())
        assert n_slots == vis.shape[1], f"<image> slots {n_slots} != visual tokens {vis.shape[1]}"
        emb = emb.clone()
        emb[img_pos] = vis.reshape(-1, vis.shape[-1])                          # scatter in order
        return emb

    def prefix(self, images_u8, text_ids, text_mask, state=None):
        """Run the VLM prefix (image+language[+state token]) and return (per-layer hidden states, mask, pooled rep)."""
        h = self.embed_joint(images_u8, text_ids)
        mask = text_mask
        if state is not None:
            st = self.state_proj(state.to(self.param_dtype)).unsqueeze(1)      # [B,1,D] appended to the prefix
            h = torch.cat([h, st], 1)
            mask = torch.cat([mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)], 1)
        out = self.lm(inputs_embeds=h, attention_mask=mask, output_hidden_states=True)
        hs = out.hidden_states                                                 # tuple len = n_layers+1
        m = mask[..., None].to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return hs, mask, pooled

    def rep(self, images_u8, text_ids, text_mask, state=None):
        """Pooled joint V-L representation (used by the auxiliary heads)."""
        return self.prefix(images_u8, text_ids, text_mask, state)[2]

    # ---- objectives ----
    def compute_loss(self, images_u8, text_ids, text_mask, actions, state=None,
                     aux_images_u8=None, future_images_u8=None):
        cfg = self.cfg
        hs, mask, rep = self.prefix(images_u8, text_ids, text_mask, state)
        parts, total = {}, rep.new_zeros(())
        if cfg.enable_action_training:
            hs_bc = tuple(h.detach() for h in hs) if cfg.stop_bc_to_vlm_grad else hs   # KI: insulate the VLM
            a = actions.view(actions.shape[0], cfg.action_horizon, cfg.action_dim)
            l = self.expert.loss(hs_bc, mask, a); parts["bc"] = l.detach()
            total = total + cfg.action_loss_weight * l
        if cfg.aux_loss_weight > 0 and self.aux_head is not None:
            if cfg.aux_family == "mask":
                src = self.rep(aux_images_u8, text_ids, text_mask, state); tgt = rep.detach()
            elif cfg.aux_family == "noise":
                src = rep; tgt = rep.detach()
                if cfg.aux_noise_sigma > 0:
                    tgt = tgt + cfg.aux_noise_sigma * torch.randn_like(tgt)
            elif cfg.aux_family == "future":
                src = rep
                with torch.no_grad():
                    tgt = self.rep(future_images_u8, text_ids, text_mask, state)
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
    def sample_actions(self, images_u8, text_ids, text_mask, state=None):
        hs, mask, _ = self.prefix(images_u8, text_ids, text_mask, state)
        a = self.expert.sample(hs, mask, self.cfg.flow_steps_sample)           # [B,horizon,action_dim]
        return a.reshape(a.shape[0], -1)
