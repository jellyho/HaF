"""Benchmark the FAST path (processor bypass, GPU resize, frozen vision, SDPA) vs batch size.
Reports peak VRAM, s/step, samples/s, and the fwd/bwd/opt split. Synthetic GPU tensors => pure model cost."""
import sys, time, json, os
import numpy as np, torch
sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/src")
from haf_torch.models.config import HAFTorchConfig
from haf_torch.models.smolvlm_fast import SmolVLMFastVLA

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
BSL = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "8,16,32,64").split(",")]
STEPS = int(os.environ.get("BENCH_STEPS", 6))

def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def bench(bs, aux_w, freeze_vlm=False, label=""):
    cfg = HAFTorchConfig(aux_loss_weight=aux_w, aux_family="mask", aux_mask_ratio=0.5, dtype="bfloat16",
                         batch_size=bs, freeze_vlm=freeze_vlm, lr=1e-4)
    m = SmolVLMFastVLA(cfg).to(DEV); m.train()
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    if DEV == "cuda": torch.cuda.reset_peak_memory_stats()
    img = torch.randint(0, 255, (bs, 224, 224, 3), dtype=torch.uint8, device=DEV)
    aimg = torch.randint(0, 255, (bs, 224, 224, 3), dtype=torch.uint8, device=DEV)
    ids = torch.randint(0, 49000, (bs, 48), device=DEV); am = torch.ones(bs, 48, dtype=torch.long, device=DEV)
    act = torch.randn(bs, cfg.action_horizon * cfg.action_dim, device=DEV, dtype=torch.bfloat16)
    tf = tb = to = 0.0
    for i in range(STEPS):
        sync(); t0 = time.time()
        total, parts = m.compute_loss(img, ids, am, act, aux_images_u8=aimg if aux_w > 0 else None)
        sync(); t1 = time.time()
        opt.zero_grad(set_to_none=True); total.backward()
        sync(); t2 = time.time()
        torch.nn.utils.clip_grad_norm_(ps, 10.0); opt.step()
        sync(); t3 = time.time()
        if i >= 1: tf += t1-t0; tb += t2-t1; to += t3-t2
    n = max(STEPS-1, 1); step_s = (tf+tb+to)/n
    peak = torch.cuda.max_memory_allocated()/1e9 if DEV=="cuda" else 0
    ntr = sum(p.numel() for p in ps)/1e6
    print(f"  bs={bs:<3} aux={int(aux_w>0)} frozenVLM={int(freeze_vlm)} | VRAM {peak:5.2f}GB | {step_s:.3f}s/step "
          f"{bs/step_s:6.1f} samp/s | fwd {tf/n:.3f} bwd {tb/n:.3f} opt {to/n:.3f} | trainable {ntr:.0f}M", flush=True)
    r = dict(label=label, bs=bs, aux=aux_w>0, frozen=freeze_vlm, vram=round(peak,2), step_s=round(step_s,3),
             samples_s=round(bs/step_s,1), trainable_M=round(ntr,1))
    del m, opt
    if DEV=="cuda": torch.cuda.empty_cache()
    return r

rows=[]
for label, aux, fr in [("bc", 0.0, False), ("aux", 0.1, False), ("aux+frozenVLM", 0.1, True)]:
    print(f"=== {label} ===", flush=True)
    for bs in BSL:
        try: rows.append(bench(bs, aux, fr, label))
        except torch.OutOfMemoryError:
            print(f"  bs={bs} OOM", flush=True); torch.cuda.empty_cache(); break
json.dump(rows, open(f"{OUT}/smolvlm_fast_bench.json","w"), indent=1)
print("FAST_BENCH_OK", flush=True)
