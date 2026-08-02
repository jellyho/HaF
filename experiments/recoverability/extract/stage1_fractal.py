"""Exp 1 — Stage 1 for RT-1 / fractal20220817_data (Google robot). Writes transitions_fractal.npz.

obs: image(256x320), base_pose_tool_reached(7)=pose, gripper_closed(1), natural_language_instruction.
action dict: world_vector(3)+rotation_delta(3)+gripper_closedness_action(1) -> 7-d. Episodes ~115.
"""
import os, glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

DATA = "/data5/jellyho/Hindsight/fractal_rlds/fractal20220817_data/0.1.0"
OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CAM = "image"
RES = 224
K_SMALL, K_LARGE = 5, 30
N_T = 8
MIN_LEN = 30
MAX_EP = int(os.environ.get("MAX_EP", 500))


H_CHUNK = 15


def resize(img):
    return cv2.resize(np.asarray(img), (RES, RES), interpolation=cv2.INTER_AREA)


def chunk(acts, t, H=H_CHUNK):
    c = acts[t:t + H]
    if len(c) < H:
        c = np.concatenate([c, np.repeat(c[-1:], H - len(c), axis=0)], axis=0)
    return c.astype(np.float32)


def act_vec(a):
    return np.concatenate([
        np.asarray(a["world_vector"], np.float32),
        np.asarray(a["rotation_delta"], np.float32),
        np.asarray(a["gripper_closedness_action"], np.float32).reshape(-1)[:1],
    ])


def main():
    shards = [s for s in sorted(glob.glob(DATA + "/fractal*train.tfrecord-*")) if ".gstmp" not in s]
    print(f"complete fractal shards: {len(shards)}", flush=True)
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
        poses = np.stack([s["observation"]["base_pose_tool_reached"].numpy() for s in steps]).astype(np.float32)
        grips = np.stack([np.asarray(s["observation"]["gripper_closed"].numpy(), np.float32).reshape(-1)[:1] for s in steps])
        acts = np.stack([act_vec(s["action"]) for s in steps])
        instr = steps[0]["observation"]["natural_language_instruction"].numpy().decode("utf-8", "ignore").strip()
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
    path = os.path.join(OUT, "cache/transitions_fractal.npz")
    np.savez_compressed(path, **out)
    print(f"SAVED {path}  N={len(cols['Ft'])}  episodes={used}  "
          f"non-empty-instr={sum(1 for s in cols['instr'] if s)}", flush=True)


if __name__ == "__main__":
    main()
