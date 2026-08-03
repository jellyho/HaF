"""Parallel, prefetching data pipeline for haf_torch.

Why: the Idefics3/SmolVLM processor costs ~123 ms per 16 samples on one CPU core (~130 samples/s), and the
mask-aux arm needs TWO processor calls per batch (clean + masked) -> ~65 samples/s. Doing that inline in the train
loop makes training preprocessing-bound. Here the fractal decode + masking + processor all run in DataLoader
worker processes, so the main process only moves ready tensors to the GPU.

Design:
  - `FractalIterable` is an IterableDataset that shards the TFRecord stream across workers (worker i takes every
    n-th episode), decodes frames, applies the mask, runs the processor, and yields ready tensor dicts.
  - `make_loader` wraps it with num_workers + prefetch + pin_memory.
  - OOD (held-out instruction) transitions are NOT yielded for training; collect them separately with
    `collect_ood()` (single pass, small).
"""
from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from . import fractal as fx


def _build_processor(vlm_id: str, image_splitting: bool, vlm_image_size: int):
    from transformers import AutoProcessor
    p = AutoProcessor.from_pretrained(vlm_id, use_fast=True)          # fast processor: ~1.16x
    if not image_splitting:
        p.image_processor.do_image_splitting = False                  # 1139 -> 79 tokens/sample
    p.image_processor.size = {"longest_edge": vlm_image_size}
    p.image_processor.max_image_size = {"longest_edge": vlm_image_size}
    return p


def _prompt(processor, instruction: str, prefix: str = "") -> str:
    return processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prefix + instruction}]}],
        add_generation_prompt=True)


class FractalIterable(IterableDataset):
    """Streams fractal, does ALL preprocessing in the worker, yields tensor dicts ready for the GPU."""

    def __init__(self, cfg, max_ep: int = 0, n_t: int = 24, need_mask: bool = False, need_future: bool = False,
                 seed: int = 0, skip_ood: bool = True):
        self.cfg, self.max_ep, self.n_t = cfg, max_ep, n_t
        self.need_mask, self.need_future, self.seed, self.skip_ood = need_mask, need_future, seed, skip_ood

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        proc = _build_processor(self.cfg.vlm_id, self.cfg.image_splitting, self.cfg.vlm_image_size)
        gen = fx.stream_transitions(max_ep=self.max_ep, n_t=self.n_t,
                                    future_offset=self.cfg.aux_future_offset,
                                    image_size=self.cfg.image_size,
                                    mask_ratio=self.cfg.aux_mask_ratio if self.need_mask else 0.0,
                                    seed=self.seed + wid)
        i = 0
        for tr in gen:
            i += 1
            if (i % nw) != wid:                       # shard across workers
                continue
            if self.skip_ood and tr["ood"]:
                continue
            prompt = _prompt(proc, tr["instruction"], self.cfg.prompt_prefix)
            main = proc(text=prompt, images=[tr["image"]], return_tensors="pt")
            item = {"input_ids": main["input_ids"][0], "attention_mask": main["attention_mask"][0],
                    "pixel_values": main["pixel_values"][0],
                    "actions": torch.from_numpy(tr["actions"]), "instruction": tr["instruction"]}
            if self.need_mask and tr["image_masked"] is not None:
                aux = proc(text=prompt, images=[tr["image_masked"]], return_tensors="pt")
                item["aux_input_ids"] = aux["input_ids"][0]
                item["aux_attention_mask"] = aux["attention_mask"][0]
                item["aux_pixel_values"] = aux["pixel_values"][0]
            if self.need_future:
                fut = proc(text=prompt, images=[tr["image_future"]], return_tensors="pt")
                item["fut_input_ids"] = fut["input_ids"][0]
                item["fut_attention_mask"] = fut["attention_mask"][0]
                item["fut_pixel_values"] = fut["pixel_values"][0]
            yield item


def collate(items):
    """Pad token sequences, stack pixels/actions."""
    out = {}
    keys = [k for k in items[0] if k.endswith("input_ids")]
    for k in keys:
        base = k[:-len("input_ids")]
        L = max(x[k].shape[0] for x in items)
        ids = torch.zeros(len(items), L, dtype=items[0][k].dtype)
        am = torch.zeros(len(items), L, dtype=torch.long)
        for j, x in enumerate(items):
            n = x[k].shape[0]
            ids[j, :n] = x[k]; am[j, :n] = x[base + "attention_mask"][:n]
        out[base + "input_ids"] = ids
        out[base + "attention_mask"] = am
        out[base + "pixel_values"] = torch.stack([x[base + "pixel_values"] for x in items])
    out["actions"] = torch.stack([x["actions"] for x in items])
    out["instructions"] = [x["instruction"] for x in items]
    return out


def make_loader(cfg, max_ep=0, n_t=24, num_workers=8, prefetch=4, seed=0):
    need_mask = cfg.aux_loss_weight > 0 and cfg.aux_family == "mask"
    need_future = cfg.aux_loss_weight > 0 and cfg.aux_family == "future"
    ds = FractalIterable(cfg, max_ep=max_ep, n_t=n_t, need_mask=need_mask, need_future=need_future, seed=seed)
    return DataLoader(ds, batch_size=cfg.batch_size, num_workers=num_workers, collate_fn=collate,
                      pin_memory=True, prefetch_factor=(prefetch if num_workers > 0 else None),
                      persistent_workers=(num_workers > 0))


def collect_ood(cfg, max_ep=200, n_t=8, limit=512, seed=0):
    """One pass to gather held-out-instruction transitions for OOD eval (preprocessed once)."""
    proc = _build_processor(cfg.vlm_id, cfg.image_splitting, cfg.vlm_image_size)
    held = []
    for tr in fx.stream_transitions(max_ep=max_ep, n_t=n_t, future_offset=cfg.aux_future_offset,
                                    image_size=cfg.image_size, mask_ratio=0.0, seed=seed):
        if not tr["ood"]:
            continue
        b = proc(text=_prompt(proc, tr["instruction"], cfg.prompt_prefix), images=[tr["image"]], return_tensors="pt")
        held.append({"input_ids": b["input_ids"][0], "attention_mask": b["attention_mask"][0],
                     "pixel_values": b["pixel_values"][0], "actions": torch.from_numpy(tr["actions"]),
                     "instruction": tr["instruction"]})
        if len(held) >= limit:
            break
    return held


def to_device(batch, device, prefix=""):
    """Slice a collated batch into the model's inputs dict for a given prefix ('', 'aux_', 'fut_')."""
    return {"input_ids": batch[prefix + "input_ids"].to(device, non_blocking=True),
            "attention_mask": batch[prefix + "attention_mask"].to(device, non_blocking=True),
            "pixel_values": batch[prefix + "pixel_values"].to(device, non_blocking=True)}
