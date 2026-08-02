"""Exp 1 — Stage 1: extract transitions from the DROID subset (complete shards) to npz.

Reads whatever full-DROID shards are already downloaded (skips .gstmp) via
features.deserialize_example, so it works while the download is still running.

Per transition we store 6 frames (resized 224 uint8):
  o_{t-kl}, o_{t-ks}, o_t, o_{t+ks}, o_{t+kl}, o_0     (past-far..future-far + initial)
plus vectors: cartesian_position at 0/t, gripper at t, action at t,
              future action & gripper (t+ks) ; meta: progress, ep_id, t, T, deltas, instruction.
"""
import os, glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

DATA = os.environ.get("DROID_DIR", "/data5/jellyho/Hindsight/droid_rlds/droid/1.0.0")
OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
os.makedirs(OUT, exist_ok=True)

CAM = "exterior_image_1_left"
RES = 224
K_SMALL, K_LARGE = 5, 45
N_T = 8
MIN_LEN = 40
MAX_EP = int(os.environ.get("MAX_EP", 500))


H_CHUNK = 15


def resize(img):
    return cv2.resize(np.asarray(img), (RES, RES), interpolation=cv2.INTER_AREA)


def chunk(acts, t, H=H_CHUNK):
    c = acts[t:t + H]
    if len(c) < H:
        c = np.concatenate([c, np.repeat(c[-1:], H - len(c), axis=0)], axis=0)
    return c.astype(np.float32)


def first_instr(step):
    for k in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        s = step[k].numpy().decode("utf-8", "ignore").strip()
        if s:
            return s
    return ""


def main():
    shards = sorted(glob.glob(DATA + "/r2d2_faceblur-train.tfrecord-*-of-02048"))
    shards = [s for s in shards if ".gstmp" not in s]
    print(f"complete shards available: {len(shards)}", flush=True)
    feats = tfds.builder_from_directory(DATA).info.features
    ds = tf.data.TFRecordDataset(shards).map(feats.deserialize_example)

    cols = {k: [] for k in
            ["Fpl", "Fps", "Ft", "Ffs", "Ffl", "F0", "Flast",
             "cart0", "cart_last", "cartt", "gript", "actt", "act_fut", "grip_fut",
             "act_prev", "grip_prev", "act_chunk",
             "progress", "ep_id", "t", "T", "instr"]}

    ep_id = 0
    used = 0
    for ep in ds:
        if used >= MAX_EP:
            break
        steps = list(ep["steps"])
        T = len(steps)
        if T < MIN_LEN:
            ep_id += 1
            continue
        frames = [s["observation"][CAM].numpy() for s in steps]
        carts = np.stack([s["observation"]["cartesian_position"].numpy() for s in steps])
        grips = np.stack([s["observation"]["gripper_position"].numpy() for s in steps])
        acts = np.stack([s["action"].numpy() for s in steps])
        instr = first_instr(steps[0])
        f0 = resize(frames[0]); flast = resize(frames[-1]); cart_0 = carts[0]; cart_last = carts[-1]

        lo, hi = 3, T - 3
        for t in np.unique(np.linspace(lo, hi, N_T).astype(int)):
            tps, tpl = max(t - K_SMALL, 0), max(t - K_LARGE, 0)
            tfs, tfl = min(t + K_SMALL, T - 1), min(t + K_LARGE, T - 1)
            cols["Fpl"].append(resize(frames[tpl])); cols["Fps"].append(resize(frames[tps]))
            cols["Ft"].append(resize(frames[t]))
            cols["Ffs"].append(resize(frames[tfs])); cols["Ffl"].append(resize(frames[tfl]))
            cols["F0"].append(f0); cols["Flast"].append(flast)
            cols["cart0"].append(cart_0); cols["cart_last"].append(cart_last); cols["cartt"].append(carts[t])
            cols["gript"].append(grips[t]); cols["actt"].append(acts[t])
            cols["act_fut"].append(acts[tfs]); cols["grip_fut"].append(grips[tfs])
            cols["act_prev"].append(acts[tps]); cols["grip_prev"].append(grips[tps])
            cols["act_chunk"].append(chunk(acts, t))
            cols["progress"].append(t / (T - 1)); cols["ep_id"].append(ep_id)
            cols["t"].append(int(t)); cols["T"].append(int(T)); cols["instr"].append(instr)
        used += 1; ep_id += 1
        if used % 25 == 0:
            print(f"  used {used} episodes, {len(cols['Ft'])} transitions", flush=True)

    out = {}
    for k, v in cols.items():
        if k == "instr":
            out[k] = np.array(v, dtype=object)
        elif k.startswith("F"):
            out[k] = np.stack(v).astype(np.uint8)
        elif k in ("ep_id", "t", "T"):
            out[k] = np.array(v, np.int32)
        else:
            out[k] = np.array(v, np.float32)
    path = os.path.join(OUT, "cache/transitions_droid.npz")
    np.savez_compressed(path, **out)
    print(f"SAVED {path}  N={len(cols['Ft'])}  episodes={used}  "
          f"non-empty-instr={sum(1 for s in cols['instr'] if s)}", flush=True)


if __name__ == "__main__":
    main()
