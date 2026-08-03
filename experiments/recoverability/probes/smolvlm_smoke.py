"""Mechanics smoke for the haf_torch SmolVLM-VLA stack (GPU): does the whole contract work and fit?
  (1) pooled joint V-L rep, (2) BC flow loss, (3) each aux family (mask/noise/future), (4) grad reaches the VLM,
  (5) KI toggles, (6) step time + peak VRAM. Run via slurm (L40S)."""
import sys, time, numpy as np, torch
sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/src")
from PIL import Image
from haf_torch.models.config import HAFTorchConfig
from haf_torch.models.smolvlm_vla import SmolVLMVLA, mask_image_array

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
rng = np.random.default_rng(0)

cfg = HAFTorchConfig(aux_loss_weight=0.1, aux_family="mask", aux_mask_ratio=0.5, dtype="bfloat16", batch_size=B)
print(cfg.describe(), flush=True)
m = SmolVLMVLA(cfg).to(DEV); m.train()
print(f"VLM params {sum(p.numel() for p in m.vlm.parameters())/1e6:.0f}M | width {m.width}", flush=True)

imgs_np = [(rng.random((224, 224, 3)) * 255).astype(np.uint8) for _ in range(B)]
imgs = [Image.fromarray(a) for a in imgs_np]
masked = [Image.fromarray(mask_image_array(a, cfg.aux_mask_ratio, rng=rng)) for a in imgs_np]
instr = ["pick coke can"] * B
inp = m.build_inputs(imgs, instr, DEV)
print("input_ids:", tuple(inp["input_ids"].shape), "pixel_values:", tuple(inp["pixel_values"].shape), flush=True)

acts = torch.randn(B, cfg.action_horizon * cfg.action_dim, device=DEV, dtype=torch.bfloat16)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=cfg.lr)

t0 = time.time()
aux_inp = m.build_inputs(masked, instr, DEV)
total, parts = m.compute_loss(inp, acts, aux_inputs=aux_inp)
opt.zero_grad(); total.backward()
gv = sum((p.grad.float() ** 2).sum().item() for p in m.vlm.parameters() if p.grad is not None) ** 0.5
opt.step()
if DEV == "cuda": torch.cuda.synchronize()
peak = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
print(f"[mask]  bc={parts['bc'].item():.3f} aux={parts['aux'].item():.3f} grad->VLM={gv:.4f} "
      f"step={time.time()-t0:.2f}s peakVRAM={peak:.1f}GB", flush=True)

for fam, kw in [("noise", dict(aux_noise_sigma=1.0)), ("future", dict(aux_future_offset=5))]:
    c2 = HAFTorchConfig(aux_loss_weight=0.1, aux_family=fam, dtype="bfloat16", **kw)
    m2 = SmolVLMVLA(c2).to(DEV); m2.train()
    i2 = m2.build_inputs(imgs, instr, DEV)
    tgt = m2.pooled_rep(m2.build_inputs(imgs, instr, DEV)).detach() if fam == "future" else None
    tot, pr = m2.compute_loss(i2, acts, aux_inputs=None, target_rep=tgt)
    tot.backward()
    g2 = sum((p.grad.float() ** 2).sum().item() for p in m2.vlm.parameters() if p.grad is not None) ** 0.5
    print(f"[{fam:6s}] bc={pr['bc'].item():.3f} aux={pr['aux'].item():.3f} grad->VLM={g2:.4f}", flush=True)
    del m2, i2
    if DEV == "cuda": torch.cuda.empty_cache()

for name, kw in [("BC-KI ", dict(stop_bc_to_vlm_grad=True)), ("aux-KI", dict(stop_aux_to_vlm_grad=True))]:
    c3 = HAFTorchConfig(aux_loss_weight=0.1, aux_family="mask", aux_mask_ratio=0.5, dtype="bfloat16", **kw)
    m3 = SmolVLMVLA(c3).to(DEV); m3.train()
    i3 = m3.build_inputs(imgs, instr, DEV); a3 = m3.build_inputs(masked, instr, DEV)
    tot, pr = m3.compute_loss(i3, acts, aux_inputs=a3); tot.backward()
    g3 = sum((p.grad.float() ** 2).sum().item() for p in m3.vlm.parameters() if p.grad is not None) ** 0.5
    print(f"[{name}] grad->VLM={g3:.4f}", flush=True)
    del m3
    if DEV == "cuda": torch.cuda.empty_cache()

with torch.no_grad():
    a = m.sample_actions(inp)
print("sampled action chunk:", tuple(a.shape), "expect", (B, cfg.action_horizon * cfg.action_dim), flush=True)
print("SMOLVLM_MECHANICS_OK", flush=True)
