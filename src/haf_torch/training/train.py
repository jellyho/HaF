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
from haf_torch.models.smolvlm_vla import SmolVLMVLA
from haf_torch.data import fractal as fx

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
        lr=float(os.environ.get("LR", 1e-5)),
        batch_size=int(os.environ.get("BS", 8)),
        max_steps=int(os.environ.get("STEPS", 2000)),
        seed=int(os.environ.get("SEED", 0)),
    )


@torch.no_grad()
def eval_ood(model, held, amu, asd, dev, bs=8):
    """OOD action R^2 on held-out-instruction transitions (flow-sampled)."""
    if not held:
        return None
    model.eval()
    preds, ys = [], []
    for i in range(0, len(held), bs):
        b = held[i:i + bs]
        inp = model.build_inputs([x["image"] for x in b], [x["instruction"] for x in b], dev)
        a = model.sample_actions(inp).float().cpu().numpy()
        preds.append(a * asd + amu)
        ys.append(np.stack([x["actions"] for x in b]))
    model.train()
    P, Y = np.concatenate(preds), np.concatenate(ys)
    return float(1 - ((P - Y) ** 2).mean() / (((Y - amu) ** 2).mean() + 1e-9))


def main():
    cfg = build_config()
    tag = os.environ.get("TAG", f"a{cfg.aux_loss_weight}_{cfg.aux_family}_s{cfg.seed}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    print(cfg.describe(), f"| tag={tag} dev={dev}", flush=True)

    model = SmolVLMVLA(cfg).to(dev); model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    max_ep = int(os.environ.get("MAX_EP", 0))
    n_t = int(os.environ.get("N_T", 24))
    eval_every = int(os.environ.get("EVAL_EVERY", 1000))
    held: list = []
    need_mask = cfg.aux_loss_weight > 0 and cfg.aux_family == "mask"

    gen = fx.stream_transitions(max_ep=max_ep, n_t=n_t, future_offset=cfg.aux_future_offset,
                                image_size=cfg.image_size,
                                mask_ratio=cfg.aux_mask_ratio if need_mask else 0.0, seed=cfg.seed)
    loader = fx.batched(gen, cfg.batch_size, seed=cfg.seed, hold_ood=held)

    # action normalization from a warmup buffer
    warm, warm_batches = [], []
    for batch in loader:
        warm.append(batch["actions"]); warm_batches.append(batch)
        if sum(len(w) for w in warm) >= 512:
            break
    A = np.concatenate(warm); amu, asd = A.mean(0), A.std(0) + 1e-6
    print(f"warmup {len(A)} transitions for action-norm; held-OOD so far {len(held)}", flush=True)

    step, t0, hist = 0, time.time(), []
    def run_batch(batch):
        nonlocal step
        imgs, instr = batch["images"], batch["instructions"]
        inp = model.build_inputs(imgs, instr, dev)
        acts = torch.tensor((batch["actions"] - amu) / asd, device=dev,
                            dtype=torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32)
        aux_inp = target_rep = None
        if cfg.aux_loss_weight > 0:
            if cfg.aux_family == "mask":
                aux_inp = model.build_inputs(batch["images_masked"], instr, dev)
            elif cfg.aux_family == "future":
                with torch.no_grad():
                    target_rep = model.pooled_rep(model.build_inputs(batch["images_future"], instr, dev))
        total, parts = model.compute_loss(inp, acts, aux_inputs=aux_inp, target_rep=target_rep)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step(); step += 1
        if step % 50 == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
            print(f"  step {step} {msg} ({(time.time()-t0)/step:.2f}s/step, heldOOD={len(held)})", flush=True)
        if step % eval_every == 0:
            r2 = eval_ood(model, held[:256], amu, asd, dev, cfg.batch_size)
            hist.append({"step": step, "ood_r2": r2}); print(f"  [eval] step {step} OOD R2 = {r2}", flush=True)

    for b in warm_batches:
        if step >= cfg.max_steps: break
        run_batch(b)
    for b in loader:
        if step >= cfg.max_steps: break
        run_batch(b)

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
