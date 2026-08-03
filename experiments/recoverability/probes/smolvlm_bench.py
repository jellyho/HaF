"""Throughput/VRAM benchmark for the haf_torch SmolVLM-VLA stack, with an explicit bottleneck breakdown.

Measures, per batch size:
  - peak VRAM (train step, fwd+bwd+opt)
  - time split: processor/preprocess (CPU) | VLM forward | backward | optimizer
  - samples/sec, and the aux overhead (BC-only vs +mask-aux which needs a SECOND VLM forward)
  - effect of gradient checkpointing and of the data pipeline (synthetic vs real fractal decode)
Goal: find where the time actually goes so we don't ship a preprocessing-bound trainer.

Usage (slurm):  python smolvlm_bench.py [bs_list]   e.g.  "4,8,16,32"
"""
import sys, time, json, os
import numpy as np
import torch
sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/src")
from PIL import Image
from haf_torch.models.config import HAFTorchConfig
from haf_torch.models.smolvlm_vla import SmolVLMVLA, mask_image_array

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
BSL = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "4,8,16,32").split(",")]
STEPS = int(os.environ.get("BENCH_STEPS", 6))
rng = np.random.default_rng(0)


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def bench(bs, aux_w, grad_ckpt=False, label=""):
    cfg = HAFTorchConfig(aux_loss_weight=aux_w, aux_family="mask", aux_mask_ratio=0.5,
                         dtype="bfloat16", batch_size=bs)
    m = SmolVLMVLA(cfg).to(DEV); m.train()
    if grad_ckpt:
        try:
            m.vlm.gradient_checkpointing_enable()
        except Exception as e:
            print(f"   (grad-ckpt unavailable: {e})", flush=True)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()

    imgs_np = [(rng.random((224, 224, 3)) * 255).astype(np.uint8) for _ in range(bs)]
    imgs = [Image.fromarray(a) for a in imgs_np]
    instr = ["pick coke can"] * bs
    acts = torch.randn(bs, cfg.action_horizon * cfg.action_dim, device=DEV, dtype=torch.bfloat16)

    t_pre = t_fwd = t_bwd = t_opt = 0.0
    for i in range(STEPS):
        # --- preprocessing (CPU): masking + processor (tokenize + image transform) ---
        t0 = time.time()
        masked = [Image.fromarray(mask_image_array(a, 0.5, rng=rng)) for a in imgs_np] if aux_w > 0 else None
        inp = m.build_inputs(imgs, instr, DEV)
        aux_inp = m.build_inputs(masked, instr, DEV) if aux_w > 0 else None
        sync(); t1 = time.time()
        # --- forward ---
        total, parts = m.compute_loss(inp, acts, aux_inputs=aux_inp)
        sync(); t2 = time.time()
        # --- backward ---
        opt.zero_grad(set_to_none=True); total.backward()
        sync(); t3 = time.time()
        # --- optimizer ---
        opt.step()
        sync(); t4 = time.time()
        if i >= 1:                      # skip warmup step
            t_pre += t1 - t0; t_fwd += t2 - t1; t_bwd += t3 - t2; t_opt += t4 - t3
    n = max(STEPS - 1, 1)
    step_s = (t_pre + t_fwd + t_bwd + t_opt) / n
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
    row = dict(label=label, bs=bs, aux=aux_w > 0, grad_ckpt=grad_ckpt, peak_vram_gb=round(peak, 2),
               step_s=round(step_s, 3), samples_s=round(bs / step_s, 1),
               pre_s=round(t_pre / n, 3), fwd_s=round(t_fwd / n, 3), bwd_s=round(t_bwd / n, 3),
               opt_s=round(t_opt / n, 3), pre_pct=round(100 * (t_pre / n) / step_s, 1),
               tokens=int(inp["input_ids"].shape[1]))
    print(f"  bs={bs:<3} aux={int(aux_w>0)} ckpt={int(grad_ckpt)} | VRAM {row['peak_vram_gb']:>5.2f}GB | "
          f"{row['step_s']:.3f}s/step {row['samples_s']:>5.1f} samp/s | "
          f"pre {row['pre_s']:.3f}({row['pre_pct']}%) fwd {row['fwd_s']:.3f} bwd {row['bwd_s']:.3f} opt {row['opt_s']:.3f} "
          f"| tok/sample {row['tokens']}", flush=True)
    del m, opt
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return row


rows = []
print("=== BC-only (no aux) ===", flush=True)
for bs in BSL:
    try:
        rows.append(bench(bs, 0.0, label="bc"))
    except torch.OutOfMemoryError:
        print(f"  bs={bs} OOM (bc)", flush=True); torch.cuda.empty_cache(); break

print("=== +mask aux (2 VLM forwards) ===", flush=True)
for bs in BSL:
    try:
        rows.append(bench(bs, 0.1, label="aux"))
    except torch.OutOfMemoryError:
        print(f"  bs={bs} OOM (aux)", flush=True); torch.cuda.empty_cache(); break

print("=== gradient checkpointing (largest bs that fit) ===", flush=True)
big = max([r["bs"] for r in rows if r["label"] == "aux"], default=BSL[0])
for bs in sorted({big, big * 2}):
    try:
        rows.append(bench(bs, 0.1, grad_ckpt=True, label="aux+ckpt"))
    except torch.OutOfMemoryError:
        print(f"  bs={bs} OOM (aux+ckpt)", flush=True); torch.cuda.empty_cache()

# --- real data pipeline: is fractal decode the bottleneck? ---
print("=== data pipeline (real fractal decode throughput) ===", flush=True)
try:
    from haf_torch.data import fractal as fx
    t0 = time.time(); n = 0
    for tr in fx.stream_transitions(max_ep=20, n_t=24, mask_ratio=0.5):
        n += 1
        if n >= 200:
            break
    dt = time.time() - t0
    print(f"  fractal stream: {n} transitions in {dt:.1f}s = {n/dt:.1f} trans/s (single-process decode)", flush=True)
    rows.append(dict(label="data", trans_per_s=round(n / dt, 1)))
except Exception as e:
    print(f"  data pipeline bench failed: {e}", flush=True)

json.dump(rows, open(f"{OUT}/smolvlm_bench.json", "w"), indent=1)
print(f"SAVED {OUT}/smolvlm_bench.json", flush=True)
print("BENCH_OK", flush=True)
