"""RT-1/fractal streaming dataset for haf_torch (no cache — decode TFRecords on the fly, VLA-faithful).

Yields, per transition: current frame (PIL), masked frame (PIL, for the structured aux), future frame (PIL, for the
natural aux), the 15-step action chunk, the instruction, and an OOD flag (instruction-hash holdout).
TensorFlow is forced CPU-only so the GPU belongs to torch.
"""
from __future__ import annotations
import os, glob
import numpy as np
from PIL import Image

DATA = os.environ.get("FRACTAL_DIR", "/data5/jellyho/Hindsight/fractal_rlds/fractal20220817_data/0.1.0")
HCHUNK, MINLEN = 15, 30


def _act_vec(a):
    return np.concatenate([np.asarray(a["world_vector"], np.float32),
                           np.asarray(a["rotation_delta"], np.float32),
                           np.asarray(a["gripper_closedness_action"], np.float32).reshape(-1)[:1]])


def _chunk(acts, t, h=HCHUNK):
    c = acts[t:t + h]
    if len(c) < h:
        c = np.concatenate([c, np.repeat(c[-1:], h - len(c), 0)], 0)
    return c.astype(np.float32).reshape(-1)


def is_ood(instr: str) -> bool:
    """Deterministic ~20% instruction holdout (stable across processes, unlike hash())."""
    return (sum(instr.encode()) % 5) == 0


def stream_transitions(max_ep: int = 0, n_t: int = 24, future_offset: int = 5, image_size: int = 0,
                       mask_ratio: float = 0.0, seed: int = 0, shard_id: int = 0, num_shards: int = 1):
    """Generator of dicts: image, image_masked, image_future, actions[105], instruction, ood."""
    import tensorflow as tf, tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    import cv2
    from haf_torch.models.smolvlm_vla import mask_image_array
    rng = np.random.default_rng(seed)

    def rs(im):
        """Resize EXACTLY like SmolVLM's processor: stretch (no aspect preserve) to image_size with PIL LANCZOS.
        Measured: mean|diff| vs processor = 0.00012 (GPU bicubic = 0.00061); cost 3.1 ms/img => 326 img/s per
        worker, ~2600 img/s on 8 workers, i.e. far above the ~200-380 img/s the GPU can consume."""
        a = np.asarray(im)
        if not image_size:
            return a
        return np.asarray(Image.fromarray(a).resize((image_size, image_size), Image.LANCZOS), dtype=np.uint8)

    shards = [s for s in sorted(glob.glob(DATA + "/fractal*train.tfrecord-*")) if ".gstmp" not in s]
    if num_shards > 1:                      # give each worker its OWN files (no duplicate decoding, no I/O fight)
        rs = np.random.default_rng(seed).permutation(len(shards))
        shards = [shards[i] for i in rs[shard_id::num_shards]]
    feats = tfds.builder_from_directory(DATA).info.features
    ds = tf.data.TFRecordDataset(shards).map(feats.deserialize_example)
    used = 0
    for ep in ds:
        if max_ep and used >= max_ep:
            break
        steps = list(ep["steps"]); T = len(steps)
        if T < MINLEN:
            continue
        frames = [s["observation"]["image"].numpy() for s in steps]
        acts = np.stack([_act_vec(s["action"]) for s in steps])
        # proprio state: 7-d tool pose (pos+quat) + 1-d gripper — same 8-d vector SimplerEnv exposes
        states = np.stack([np.concatenate([
            np.asarray(s["observation"]["base_pose_tool_reached"], np.float32),
            np.asarray(s["observation"]["gripper_closed"], np.float32).reshape(-1)[:1]]) for s in steps])
        instr = steps[0]["observation"]["natural_language_instruction"].numpy().decode("utf-8", "ignore").strip()
        ood = is_ood(instr)
        for t in np.unique(np.linspace(3, T - 3, n_t).astype(int)):
            t = int(t); tf_ = min(t + future_offset, T - 1)
            cur = rs(frames[t])
            yield {
                "image": Image.fromarray(cur),
                "image_masked": Image.fromarray(mask_image_array(cur, mask_ratio, rng=rng)) if mask_ratio > 0 else None,
                "image_future": Image.fromarray(rs(frames[tf_])),
                "actions": _chunk(acts, t),
                "state": states[t].astype(np.float32),
                "instruction": instr,
                "ood": ood,
            }
        used += 1


def batched(gen, batch_size: int, buf_size: int = 2048, seed: int = 0, hold_ood: list | None = None,
            max_ood: int = 512):
    """Shuffle-buffer batching. OOD transitions are diverted into `hold_ood` (never trained on)."""
    rng = np.random.default_rng(seed)
    buf = []
    for tr in gen:
        if tr["ood"]:
            if hold_ood is not None and len(hold_ood) < max_ood:
                hold_ood.append(tr)
            continue
        buf.append(tr)
        if len(buf) >= buf_size:
            rng.shuffle(buf)
            while len(buf) >= batch_size:
                yield collate(buf[:batch_size]); buf = buf[batch_size:]
    rng.shuffle(buf)
    for i in range(0, len(buf) - batch_size + 1, batch_size):
        yield collate(buf[i:i + batch_size])


def collate(items):
    return {
        "images": [x["image"] for x in items],
        "images_masked": [x["image_masked"] for x in items],
        "images_future": [x["image_future"] for x in items],
        "actions": np.stack([x["actions"] for x in items]),
        "instructions": [x["instruction"] for x in items],
    }
