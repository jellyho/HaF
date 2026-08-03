"""RLDS / tf.data pipeline for haf_torch — the STANDARD path (dlimp), not a hand-rolled python loop.

Why this replaces `fast_loader.py`'s custom worker loop:
  * the custom loop had NO shuffling (batches were consecutive frames of one episode -> correlated gradients),
    no parallel decode, and needed manual worker sharding (which stalled: 10 workers decoded the same 1024
    shards and discarded 9/10 of the data, 1h54m at 0% CPU without a single step).
  * tf.data + dlimp gives, for free and battle-tested on OXE/RT-1:
      - `DLataset.from_rlds(builder, split=..., shuffle=True)`  -> parallel shard interleave
      - `.repeat().shuffle(buffer)`                             -> real shuffling across episodes
      - `.flatten()`                                            -> episode -> frame stream
      - `.frame_map(fn, num_parallel_calls=AUTOTUNE)`            -> parallel per-frame work
      - `.batch().prefetch(AUTOTUNE)`                            -> overlap with the GPU step
All heavy per-frame work (resize to the VLM size with LANCZOS-equivalent, masking for the aux) happens INSIDE
tf.data ops, so it is parallel and pipelined. The torch side just converts ready numpy batches to tensors.

Instruction tokens are still cached per unique string (fractal has ~500), attached after tf.data via a lookup.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")      # TF must NOT take GPU memory — the GPU belongs to torch.
                                              # (Without this TF grabbed ~42GB and torch OOM'd at 2GB allocated.)
import tensorflow_datasets as tfds
import dlimp as dl
import torch

DATA_DIR = "/data5/jellyho/Hindsight"
DATASET = "fractal20220817_data"
HCHUNK = 15


def _act_vec(action):
    return tf.concat([
        tf.cast(action["world_vector"], tf.float32),
        tf.cast(action["rotation_delta"], tf.float32),
        tf.reshape(tf.cast(action["gripper_closedness_action"], tf.float32), [-1])[:1],
    ], axis=0)


def build_rlds(cfg, split: str = "train", shuffle_buffer: int = 8000, seed: int = 0,
               need_mask: bool = False, need_future: bool = False, ood: bool = False):
    """Returns a tf.data pipeline yielding per-frame dicts (numpy-ready)."""
    builder = tfds.builder(DATASET, data_dir=f"{DATA_DIR}/fractal_rlds")
    ds = dl.DLataset.from_rlds(builder, split=split, shuffle=not ood)

    H, off, size = HCHUNK, cfg.aux_future_offset, cfg.vlm_image_size
    mask_ratio = cfg.aux_mask_ratio if need_mask else 0.0

    def restructure(traj):
        """episode -> per-frame tensors: image, future image, action chunk, state, instruction."""
        obs, act = traj["observation"], traj["action"]
        T = tf.shape(obs["image"])[0]
        idx = tf.range(T)
        fut_idx = tf.minimum(idx + off, T - 1)
        # action chunk [T, H*7] with last-value padding
        a = tf.map_fn(_act_vec, act, fn_output_signature=tf.float32)          # [T,7]
        chunk_idx = tf.minimum(idx[:, None] + tf.range(H)[None, :], T - 1)     # [T,H]
        chunk = tf.reshape(tf.gather(a, chunk_idx), [T, H * 7])
        state = tf.concat([tf.cast(obs["base_pose_tool_reached"], tf.float32),
                           tf.cast(obs["gripper_closed"], tf.float32)[:, :1]], axis=-1)   # [T,8]
        out = {"image": obs["image"], "image_future": tf.gather(obs["image"], fut_idx),
               "actions": chunk, "state": state,
               "instruction": obs["natural_language_instruction"]}
        return out

    ds = ds.traj_map(restructure, tf.data.AUTOTUNE)
    ds = ds.flatten(num_parallel_calls=tf.data.AUTOTUNE)                       # episodes -> frames

    def hold_or_train(f):
        """~20% of instructions are held out (deterministic hash on the string)."""
        h = tf.strings.to_hash_bucket_fast(f["instruction"], 5)
        return tf.equal(h, 0) if ood else tf.not_equal(h, 0)

    ds = ds.filter(hold_or_train)

    def _rs(x):
        """Decode (dlimp keeps images as encoded tf.string for lazy, parallel decoding) then resize one frame
        to size x size — processor-equivalent stretch (no aspect preserve) with LANCZOS3."""
        x = tf.io.decode_image(x, channels=3, expand_animations=False)
        x = tf.cast(x, tf.float32)
        x = tf.image.resize(x[None], [size, size], method=tf.image.ResizeMethod.LANCZOS3, antialias=True)[0]
        x = tf.cast(tf.clip_by_value(x, 0, 255), tf.uint8)
        return tf.ensure_shape(x, [size, size, 3])

    def prep(f):
        img = _rs(f["image"])
        out = {"image": img, "actions": f["actions"], "state": f["state"], "instruction": f["instruction"]}
        if need_future:
            out["image_future"] = _rs(f["image_future"])
        if mask_ratio > 0:                                                     # structured-hard aux input
            g = size // 16
            keep = tf.cast(tf.random.stateless_uniform([g, g], seed=[seed, 0]) >= mask_ratio, tf.uint8)
            keep = tf.repeat(tf.repeat(keep, 16, axis=0), 16, axis=1)[..., None]
            out["image_masked"] = img * keep
        return out

    ds = ds.frame_map(prep, tf.data.AUTOTUNE)
    if not ood:
        ds = ds.repeat().shuffle(shuffle_buffer, seed=seed)                    # REAL shuffling across episodes
    ds = ds.batch(cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    opts = tf.data.Options(); opts.autotune.enabled = True
    opts.experimental_optimization.map_parallelization = True
    return ds.with_options(opts)


class TorchRLDS:
    """Iterate the tf.data pipeline, attach cached instruction tokens, yield torch tensors."""

    def __init__(self, cfg, split="train", seed=0, need_mask=False, need_future=False, ood=False,
                 shuffle_buffer=8000):
        from .fast_loader import TokenCache
        self.cfg, self.tc = cfg, TokenCache(cfg.vlm_id)
        self.ds = build_rlds(cfg, split=split, seed=seed, need_mask=need_mask,
                             need_future=need_future, ood=ood, shuffle_buffer=shuffle_buffer)

    def __iter__(self):
        for b in self.ds.as_numpy_iterator():
            instrs = [s.decode() if isinstance(s, bytes) else str(s) for s in b["instruction"]]
            toks = [self.tc(s) for s in instrs]
            out = {
                "image": torch.from_numpy(b["image"]),
                "text_ids": torch.from_numpy(np.stack([t[0] for t in toks])),
                "text_mask": torch.from_numpy(np.stack([t[1] for t in toks])),
                "actions": torch.from_numpy(b["actions"]),
                "state": torch.from_numpy(b["state"]),
            }
            if "image_masked" in b:
                out["image_masked"] = torch.from_numpy(b["image_masked"])
            if "image_future" in b:
                out["image_future"] = torch.from_numpy(b["image_future"])
            yield out


def make_loader(cfg, seed=0, **_):
    need_mask = cfg.aux_loss_weight > 0 and cfg.aux_family == "mask"
    need_future = cfg.aux_loss_weight > 0 and cfg.aux_family == "future"
    return TorchRLDS(cfg, seed=seed, need_mask=need_mask, need_future=need_future)


def collect_ood(cfg, limit=384, seed=0, **_):
    """Held-out-instruction frames for OOD eval (single pass, no shuffle/repeat)."""
    it = TorchRLDS(cfg, seed=seed, ood=True)
    held, n = [], 0
    for b in it:
        for i in range(b["image"].shape[0]):
            held.append({k: v[i] for k, v in b.items()})
            n += 1
            if n >= limit:
                return held
    return held


def collate(items):
    return {k: torch.stack([x[k] for x in items]) for k in items[0]}
