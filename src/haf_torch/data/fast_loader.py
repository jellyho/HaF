"""Fast data pipeline for the SmolVLM fast path: workers only decode + resize + mask; NO processor, NO tokenizer.

Instruction tokenization is done ONCE per unique string (fractal has ~500) and cached, so the per-step CPU cost is
just a dict lookup. Images stay uint8 [H,W,3] and are normalized/resized on the GPU inside the model.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

from . import fractal as fx


class TokenCache:
    """instruction -> (ids, mask) for the PROCESSOR-IDENTICAL prompt, tokenized once per unique string.

    Important: the <image> placeholder is only EXPANDED (1 -> 64 tokens) when the processor is called WITH an
    image. apply_chat_template alone leaves a single <image>. So we build the cache by calling the processor once
    per unique instruction with a dummy image (no pixels are kept) — the ids are then exactly what the official
    path produces, and our model scatters the 64 visual tokens into those slots.
    """

    def __init__(self, vlm_id: str, max_len: int = 160):
        from transformers import AutoProcessor
        from PIL import Image as _I
        self.proc = AutoProcessor.from_pretrained(vlm_id)
        self.proc.image_processor.do_image_splitting = False          # 1 tile -> 64 <image> tokens
        self.max_len = max_len
        self._dummy = _I.fromarray(np.zeros((8, 8, 3), np.uint8))
        self._c: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, instr: str):
        if instr not in self._c:
            tpl = self.proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}],
                add_generation_prompt=True)
            e = self.proc(text=tpl, images=[self._dummy], return_tensors="np",
                          padding="max_length", truncation=True, max_length=self.max_len)
            self._c[instr] = (e["input_ids"][0].astype(np.int64), e["attention_mask"][0].astype(np.int64))
        return self._c[instr]


class FastFractal(IterableDataset):
    def __init__(self, cfg, max_ep=0, n_t=24, need_mask=False, need_future=False, seed=0, skip_ood=True):
        self.cfg, self.max_ep, self.n_t = cfg, max_ep, n_t
        self.need_mask, self.need_future, self.seed, self.skip_ood = need_mask, need_future, seed, skip_ood

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        tc = TokenCache(self.cfg.vlm_id)
        gen = fx.stream_transitions(max_ep=self.max_ep, n_t=self.n_t,
                                    future_offset=self.cfg.aux_future_offset,
                                    image_size=self.cfg.image_size,
                                    mask_ratio=self.cfg.aux_mask_ratio if self.need_mask else 0.0,
                                    seed=self.seed, shard_id=wid, num_shards=nw)   # own files per worker
        for tr in gen:
            if self.skip_ood and tr["ood"]:
                continue
            ids, am = tc(tr["instruction"])
            item = {"image": torch.from_numpy(np.asarray(tr["image"], dtype=np.uint8)),
                    "text_ids": torch.from_numpy(ids), "text_mask": torch.from_numpy(am),
                    "actions": torch.from_numpy(tr["actions"]),
                    "state": torch.from_numpy(tr["state"])}
            if self.need_mask and tr["image_masked"] is not None:
                item["image_masked"] = torch.from_numpy(np.asarray(tr["image_masked"], dtype=np.uint8))
            if self.need_future:
                item["image_future"] = torch.from_numpy(np.asarray(tr["image_future"], dtype=np.uint8))
            yield item


def collate(items):
    out = {k: torch.stack([x[k] for x in items]) for k in items[0]}
    return out


def make_loader(cfg, max_ep=0, n_t=24, num_workers=8, prefetch=6, seed=0):
    need_mask = cfg.aux_loss_weight > 0 and cfg.aux_family == "mask"
    need_future = cfg.aux_loss_weight > 0 and cfg.aux_family == "future"
    ds = FastFractal(cfg, max_ep=max_ep, n_t=n_t, need_mask=need_mask, need_future=need_future, seed=seed)
    return DataLoader(ds, batch_size=cfg.batch_size, num_workers=num_workers, collate_fn=collate,
                      pin_memory=True, prefetch_factor=(prefetch if num_workers > 0 else None),
                      persistent_workers=(num_workers > 0))


def collect_ood(cfg, max_ep=300, n_t=8, limit=384, seed=0):
    tc = TokenCache(cfg.vlm_id)
    held = []
    for tr in fx.stream_transitions(max_ep=max_ep, n_t=n_t, future_offset=cfg.aux_future_offset,
                                    image_size=cfg.image_size, mask_ratio=0.0, seed=seed):
        if not tr["ood"]:
            continue
        ids, am = tc(tr["instruction"])
        held.append({"image": torch.from_numpy(np.asarray(tr["image"], dtype=np.uint8)),
                     "text_ids": torch.from_numpy(ids), "text_mask": torch.from_numpy(am),
                     "actions": torch.from_numpy(tr["actions"]),
                     "state": torch.from_numpy(tr["state"])})
        if len(held) >= limit:
            break
    return held
