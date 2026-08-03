"""haf_torch training loop — fine-tune SmolVLM-VLA on RT-1/fractal with BC (+ optional AHA aux).

Mirrors the HaF training contract (weighted multi-objective over a shared backbone, KI toggles) in PyTorch.
Streams fractal (no cache), fine-tunes the pretrained VLM, evaluates OOD action R^2 on held-out instructions.

Usage (env-driven so slurm can sweep):
  AUX_W=0.1 AUX_FAMILY=mask AUX_MASK=0.5 MAX_EP=2000 STEPS=2000 python -m haf_torch.training.train
Env: AUX_W, AUX_FAMILY(mask|noise|future), AUX_MASK, AUX_SIGMA, AUX_OFFSET, BC_KI, AUX_KI, LR, BS,
     MAX_EP, STEPS, SEED, TAG, N_T, EVAL_EVERY.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/src")
from haf_torch.models.config import HAFTorchConfig
from haf_torch.models.smolvlm_fast import SmolVLMFastVLA
from haf_torch.data import fast_loader as dl

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"


def build_config() -> HAFTorchConfig:
    return HAFTorchConfig(
        aux_loss_weight=float(os.environ.get("AUX_W", 0.0)),
        aux_family=os.environ.get("AUX_FAMILY", "mask"),
        aux_mask_ratio=float(os.environ.get("AUX_MASK", 0.5)),
        aux_noise_sigma=float(os.environ.get("AUX_SIGMA", 0.0)),
        aux_future_offset=int(os.environ.get("AUX_OFFSET", 5)),
        stop_bc_to_vlm_grad=bool(int(os.environ.get("BC_KI", 0))),
        stop_aux_to_vlm_grad=bool(int(os.environ.get("AUX_KI", 0))),
        lr=float(os.environ.get("LR", 1e-4)),
        batch_size=int(os.environ.get("BS", 8)),
        max_steps=int(os.environ.get("STEPS", 2000)),
        seed=int(os.environ.get("SEED", 0)),
    )


@torch.no_grad()
def eval_ood(model, held, amu, asd, dev, bs=8):
    """OOD action R^2 on held-out-instruction transitions (flow-sampled). `held` items are pre-processed."""
    if not held:
        return None
    model.eval()
    preds, ys = [], []
    for i in range(0, len(held), bs):
        b = dl.collate(held[i:i + bs])
        a = model.sample_actions(b["image"].to(dev), b["text_ids"].to(dev), b["text_mask"].to(dev),
                                 b["state"].to(dev))
        preds.append(a.float().cpu().numpy() * asd + amu)
        ys.append(b["actions"].numpy())
    model.train()
    P, Y = np.concatenate(preds), np.concatenate(ys)
    return float(1 - ((P - Y) ** 2).mean() / (((Y - amu) ** 2).mean() + 1e-9))


def main():
    cfg = build_config()
    tag = os.environ.get("TAG", f"a{cfg.aux_loss_weight}_{cfg.aux_family}_s{cfg.seed}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print(cfg.describe(), f"| tag={tag} dev={dev}", flush=True)

    model = SmolVLMFastVLA(cfg).to(dev); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=1e-10)   # SmolVLA recipe
    warmup = int(os.environ.get("WARMUP", 300))
    def lr_at(step):                                    # linear warmup -> cosine decay to 2.5e-6
        if step < warmup:
            return cfg.lr * (step + 1) / warmup
        import math
        p = (step - warmup) / max(1, cfg.max_steps - warmup)
        return 2.5e-6 + 0.5 * (cfg.lr - 2.5e-6) * (1 + math.cos(math.pi * min(p, 1.0)))

    max_ep = int(os.environ.get("MAX_EP", 0))
    n_t = int(os.environ.get("N_T", 24))
    eval_every = int(os.environ.get("EVAL_EVERY", 1000))
    workers = int(os.environ.get("WORKERS", 8))

    # OOD holdout (preprocessed once, never trained on)
    t_ood = time.time()
    held = dl.collect_ood(cfg, max_ep=int(os.environ.get("OOD_EP", 300)), limit=int(os.environ.get("OOD_N", 384)),
                          seed=cfg.seed)
    print(f"held-out OOD transitions: {len(held)} ({time.time()-t_ood:.0f}s)", flush=True)

    # training loader: ALL preprocessing (decode + mask + processor) happens in worker processes
    loader = dl.make_loader(cfg, max_ep=max_ep, n_t=n_t, num_workers=workers, seed=cfg.seed)

    step, hist = 0, []
    amu = asd = None
    warm_actions = []
    t0 = time.time(); t_data = 0.0; t_step = 0.0; tlast = time.time()

    for batch in loader:
        if step >= cfg.max_steps:
            break
        t_data += time.time() - tlast
        if amu is None:                                   # action normalization from the first batches
            warm_actions.append(batch["actions"].numpy())
            if sum(len(a) for a in warm_actions) < 512:
                tlast = time.time(); continue
            A = np.concatenate(warm_actions); amu, asd = A.mean(0), A.std(0) + 1e-6
            print(f"action-norm from {len(A)} transitions", flush=True)

        ts = time.time()
        img = batch["image"].to(dev, non_blocking=True)
        tid = batch["text_ids"].to(dev, non_blocking=True)
        tms = batch["text_mask"].to(dev, non_blocking=True)
        acts = torch.tensor((batch["actions"].numpy() - amu) / asd, device=dev,
                            dtype=torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32)
        sta = batch["state"].to(dev, non_blocking=True)
        aux_img = batch["image_masked"].to(dev, non_blocking=True) if "image_masked" in batch else None
        fut_img = batch["image_future"].to(dev, non_blocking=True) if "image_future" in batch else None
        total, parts = model.compute_loss(img, tid, tms, acts, state=sta,
                                          aux_images_u8=aux_img, future_images_u8=fut_img)
        opt.zero_grad(set_to_none=True); total.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)                       # SmolVLA clip
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step(); step += 1
        t_step += time.time() - ts

        if step % 50 == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
            el = time.time() - t0
            print(f"  step {step} {msg} | {el/step:.2f}s/step "
                  f"(gpu {t_step/step:.2f}s, data-wait {t_data/step:.2f}s = {100*t_data/max(el,1e-9):.0f}%) "
                  f"| {cfg.batch_size*step/el:.1f} samp/s", flush=True)
        if step % eval_every == 0:
            r2 = eval_ood(model, held[:256], amu, asd, dev, cfg.batch_size)
            hist.append({"step": step, "ood_r2": r2}); print(f"  [eval] step {step} OOD R2 = {r2}", flush=True)
        tlast = time.time()

    r2 = eval_ood(model, held[:512], amu, asd, dev, cfg.batch_size)
    res = {"tag": tag, "config": cfg.describe(), "steps": step, "final_ood_r2": r2, "history": hist,
           "held_ood": len(held), "aux_w": cfg.aux_loss_weight, "aux_family": cfg.aux_family,
           "aux_mask": cfg.aux_mask_ratio, "aux_sigma": cfg.aux_noise_sigma, "aux_offset": cfg.aux_future_offset,
           "bc_ki": cfg.stop_bc_to_vlm_grad, "aux_ki": cfg.stop_aux_to_vlm_grad, "seed": cfg.seed}
    json.dump(res, open(f"{OUT}/smolvlm_{tag}.json", "w"), indent=1)
    torch.save({"model": model.state_dict(), "amu": amu, "asd": asd, "cfg": cfg.describe()},
               f"{OUT}/smolvlm_{tag}.pt")
    print(f"RESULT tag={tag} steps={step} OOD_R2={r2}", flush=True)
    print(f"SAVED {OUT}/smolvlm_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
