"""SmolVLM-VLA — the HaF/AHA architecture with a pretrained SmolVLM backbone (PyTorch).

Mirrors src/haf (JAX, PaliGemma/pi0.5):
  obs(image, instruction) -> VLM joint V-L hidden states -> pooled rep
      -> flow-matching action expert  (continuous action chunk)      [BC]
      -> aux head                     (AHA, recoverability-selected) [regularizer]
KI toggles mirror HAFConfig: `stop_bc_to_vlm_grad` (insulate the VLM from BC, pi0.5-style) and
`stop_aux_to_vlm_grad` (control: aux does NOT shape the VLM).

Auxiliary families (the confound-free recoverability dial — see fair-rigorous-experiments memory):
  'mask'   structured-hard: forward a MASKED image through the VLM, predict the CLEAN pooled rep (stop-grad target).
           mask_ratio dials recoverability while the target stays learnable + task-relevant.
  'noise'  noise-hard: predict the clean rep corrupted by gaussian noise (sigma dials it toward the signal floor).
           Same target semantics, difficulty added by UNLEARNABLE noise -> separates "hard" from "noisy".
  'future' natural: predict a frozen VLM embedding of a future frame (offset dials it) — the natural-objective arm.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from .config import HAFTorchConfig


def timestep_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    a = t * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class FlowActionExpert(nn.Module):
    """Flow-matching action expert over the pooled VLM representation (pi0-style, continuous chunk)."""

    def __init__(self, cond_dim: int, action_dim: int, horizon: int, width: int = 512, depth: int = 2, t_dim: int = 64):
        super().__init__()
        self.adim = action_dim * horizon
        self.action_dim, self.horizon, self.t_dim = action_dim, horizon, t_dim
        layers, d_in = [], self.adim + cond_dim + t_dim
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.GELU()]
            d_in = width
        layers += [nn.Linear(d_in, self.adim)]
        self.net = nn.Sequential(*layers)

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_t, cond, timestep_embedding(t, self.t_dim)], dim=-1))

    def loss(self, cond: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """actions: [B, horizon*action_dim] (normalized)."""
        eps = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], 1, device=actions.device)
        x_t = (1 - t) * eps + t * actions
        return ((self.velocity(x_t, t, cond) - (actions - eps)) ** 2).mean()

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int = 10, n_samples: int = 8) -> torch.Tensor:
        B = cond.shape[0]
        c = cond.repeat_interleave(n_samples, 0)
        x = torch.randn(B * n_samples, self.adim, device=cond.device, dtype=cond.dtype)
        for k in range(steps):
            t = torch.full((B * n_samples, 1), k / steps, device=cond.device, dtype=cond.dtype)
            x = x + (1.0 / steps) * self.velocity(x, t, c)
        return x.view(B, n_samples, self.adim).mean(1)


class SmolVLMVLA(nn.Module):
    def __init__(self, config: HAFTorchConfig):
        super().__init__()
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.cfg = config
        self.processor = AutoProcessor.from_pretrained(config.vlm_id)
        # SmolVLM/Idefics3 splits each image into ~17 high-res tiles by default -> ~1139 tokens/sample and OOM.
        # A robot policy needs one view: disable splitting (79 tokens/sample) and cap the tile size.
        if not config.image_splitting:
            self.processor.image_processor.do_image_splitting = False
        self.processor.image_processor.size = {"longest_edge": config.vlm_image_size}
        self.processor.image_processor.max_image_size = {"longest_edge": config.vlm_image_size}
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
        self.vlm = AutoModelForImageTextToText.from_pretrained(config.vlm_id, torch_dtype=dtype)
        if config.freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad_(False)
        self.width = int(self.vlm.config.text_config.hidden_size)
        self.expert = FlowActionExpert(self.width, config.action_dim, config.action_horizon,
                                       config.expert_width, config.expert_depth)
        self.aux_head = nn.Linear(self.width, self.width) if config.aux_loss_weight > 0 else None

    # ---- inputs ----
    def build_inputs(self, images, instructions, device):
        """images: list of PIL.Image; instructions: list[str] -> processor batch on `device`."""
        prompts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": (self.cfg.prompt_prefix + t)}]}],
                add_generation_prompt=True)
            for t in instructions
        ]
        batch = self.processor(text=prompts, images=[[im] for im in images], return_tensors="pt", padding=True)
        return {k: v.to(device) for k, v in batch.items()}

    def pooled_rep(self, inputs) -> torch.Tensor:
        """Joint V-L representation used to condition the policy: mean-pooled last hidden state."""
        out = self.vlm(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1]                                        # [B, T, D]
        mask = inputs.get("attention_mask")
        if mask is not None:
            m = mask[..., None].to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return h.mean(1)

    # ---- objectives ----
    def bc_loss(self, rep: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cond = rep.detach() if self.cfg.stop_bc_to_vlm_grad else rep     # KI: insulate VLM from the BC head
        return self.expert.loss(cond, actions)

    def aux_loss(self, rep: torch.Tensor, aux_inputs=None, target_rep: torch.Tensor | None = None) -> torch.Tensor:
        """AHA auxiliary. `rep` = clean pooled rep (the conditioning). Behaviour by family:
             mask   : `aux_inputs` = processor batch of the MASKED image -> predict stop-grad(clean rep)
             noise  : predict stop-grad(clean rep) + sigma*noise, from the clean rep itself
             future : `target_rep` = frozen pooled rep of the future frame -> predict it from `rep`
        Returns cosine-distance loss (mirrors haf.retro_embedding_loss)."""
        cfg = self.cfg
        if cfg.aux_family == "mask":
            src = self.pooled_rep(aux_inputs)                            # masked forward (through the VLM)
            tgt = rep.detach()
        elif cfg.aux_family == "noise":
            src = rep
            tgt = rep.detach()
            if cfg.aux_noise_sigma > 0:
                tgt = tgt + cfg.aux_noise_sigma * torch.randn_like(tgt)  # unlearnable corruption
        elif cfg.aux_family == "future":
            src = rep
            assert target_rep is not None, "future aux needs target_rep"
            tgt = target_rep.detach()
        else:
            raise ValueError(cfg.aux_family)
        if cfg.stop_aux_to_vlm_grad:                                     # KI control: aux must not shape the VLM
            src = src.detach()
        pred = self.aux_head(src)
        pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
        tn = tgt / (tgt.norm(dim=-1, keepdim=True) + 1e-8)
        return (1.0 - (pn * tn).sum(-1)).mean()

    def compute_loss(self, inputs, actions, aux_inputs=None, target_rep=None):
        """Returns (total, dict of parts). Mirrors HaF's weighted multi-objective composition."""
        cfg = self.cfg
        rep = self.pooled_rep(inputs)
        parts, total = {}, rep.new_zeros(())
        if cfg.enable_action_training:
            l = self.bc_loss(rep, actions); parts["bc"] = l.detach()
            total = total + cfg.action_loss_weight * l
        if cfg.aux_loss_weight > 0 and self.aux_head is not None:
            l = self.aux_loss(rep, aux_inputs=aux_inputs, target_rep=target_rep); parts["aux"] = l.detach()
            total = total + cfg.aux_loss_weight * l
        return total, parts

    @torch.no_grad()
    def sample_actions(self, inputs) -> torch.Tensor:
        rep = self.pooled_rep(inputs)
        return self.expert.sample(rep, self.cfg.flow_steps_sample, self.cfg.flow_samples)


def mask_image_array(arr: np.ndarray, ratio: float, patch: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly zero `ratio` of patch-grid cells (structured-hard aux input). arr: HWC uint8."""
    if ratio <= 0:
        return arr
    rng = rng or np.random.default_rng()
    out = arr.copy()
    H, W = arr.shape[:2]
    gh, gw = H // patch, W // patch
    n = gh * gw
    idx = rng.permutation(n)[: int(round(ratio * n))]
    for i in idx:
        r, c = divmod(int(i), gw)
        out[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = 0
    return out
