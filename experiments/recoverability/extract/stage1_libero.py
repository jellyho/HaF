"""Exp 1 — Stage 1 for LIBERO (openvla/modified_libero_rlds). Writes transitions_libero_<suite>.npz.

LIBERO is the memorization testbed (②): near-identical scenes, language distinguishes tasks — a policy
that memorizes scene->action while ignoring language CANNOT solve held-out instructions. libero_goal is
the sharpest (same scene, goal varies). Schema per step:
  action (7,) float32  [6-DoF EEF delta + gripper]
  observation.image (256,256,3) uint8   (main cam)
  observation.state (8,) float32        [EEF pose (7) + gripper (1)]
  language_instruction: text
Suite chosen via env SUITE (default libero_goal). ~500 demos/suite (10 tasks), plenty for GroupKFold.
"""
import os, glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

SUITE = os.environ.get("SUITE", "libero_goal")
ROOT = "/data5/jellyho/Hindsight/libero_rlds"
DATA = os.path.join(ROOT, f"{SUITE}_no_noops", "1.0.0")
OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CAM = "image"
RES = 224
K_SMALL, K_LARGE = 5, 30
N_T = 8
MIN_LEN = 30
MAX_EP = int(os.environ.get("MAX_EP", 600))
H_CHUNK = 15


def resize(img):
    return cv2.resize(np.asarray(img), (RES, RES), interpolation=cv2.INTER_AREA)


def chunk(acts, t, H=H_CHUNK):
    c = acts[t:t + H]
    if len(c) < H:
        c = np.concatenate([c, np.repeat(c[-1:], H - len(c), axis=0)], axis=0)
    return c.astype(np.float32)


def main():
    # NB: openvla names libero_10 shards "liber_o10-train..." (upstream typo) -> match any *-train prefix
    shards = [s for s in sorted(glob.glob(DATA + "/*-train.tfrecord-*")) if ".gstmp" not in s]
    print(f"[{SUITE}] shards: {len(shards)}", flush=True)
    feats = tfds.builder_from_directory(DATA).info.features
    ds = tf.data.TFRecordDataset(shards).map(feats.deserialize_example)

    cols = {k: [] for k in
            ["Fpl", "Fps", "Ft", "Ffs", "Ffl", "F0", "Flast",
             "cart0", "cart_last", "cartt", "gript", "actt", "act_fut", "grip_fut",
             "act_prev", "grip_prev", "act_chunk", "progress", "ep_id", "t", "T", "instr"]}
    ep_id = 0; used = 0
    for ep in ds:
        if used >= MAX_EP:
            break
        steps = list(ep["steps"])
        T = len(steps)
        if T < MIN_LEN:
            ep_id += 1; continue
        frames = [s["observation"][CAM].numpy() for s in steps]
        state = np.stack([s["observation"]["state"].numpy() for s in steps]).astype(np.float32)  # (T,8)
        poses = state[:, :7]                    # EEF pose proxy
        grips = state[:, 7:8]                   # gripper proxy
        acts = np.stack([np.asarray(s["action"].numpy(), np.float32).reshape(-1)[:7] for s in steps])
        instr = steps[0]["language_instruction"].numpy().decode("utf-8", "ignore").strip()
        f0 = resize(frames[0]); flast = resize(frames[-1]); p0 = poses[0]; p_last = poses[-1]
        for t in np.unique(np.linspace(3, T - 3, N_T).astype(int)):
            tps, tpl = max(t - K_SMALL, 0), max(t - K_LARGE, 0)
            tfs, tfl = min(t + K_SMALL, T - 1), min(t + K_LARGE, T - 1)
            cols["Fpl"].append(resize(frames[tpl])); cols["Fps"].append(resize(frames[tps]))
            cols["Ft"].append(resize(frames[t]))
            cols["Ffs"].append(resize(frames[tfs])); cols["Ffl"].append(resize(frames[tfl]))
            cols["F0"].append(f0); cols["Flast"].append(flast)
            cols["cart0"].append(p0); cols["cart_last"].append(p_last); cols["cartt"].append(poses[t])
            cols["gript"].append(grips[t]); cols["actt"].append(acts[t])
            cols["act_fut"].append(acts[tfs]); cols["grip_fut"].append(grips[tfs])
            cols["act_prev"].append(acts[tps]); cols["grip_prev"].append(grips[tps])
            cols["act_chunk"].append(chunk(acts, t))
            cols["progress"].append(t / (T - 1)); cols["ep_id"].append(ep_id)
            cols["t"].append(int(t)); cols["T"].append(int(T)); cols["instr"].append(instr)
        used += 1; ep_id += 1
        if used % 50 == 0:
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
    path = os.path.join(OUT, f"cache/transitions_{SUITE}.npz")
    np.savez_compressed(path, **out)
    uniq = len(set(s for s in cols["instr"]))
    print(f"SAVED {path}  N={len(cols['Ft'])}  episodes={used}  unique-instr={uniq}", flush=True)


if __name__ == "__main__":
    main()
