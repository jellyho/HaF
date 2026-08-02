"""Dense extract for the SCALED closed-loop rollout with a TRAINABLE encoder (disk-safe).

Disk on /data5 is ~98% full, so we do NOT store raw Ft (120GB at 20k dense). Instead:
  Ft_jpg = JPEG-compressed current frame (224x224)  -> ~15GB for 20k dense; decoded per-batch at train time
           so the DINOv2-small encoder is TRAINED on it (like a real VLA), on the fly.
  z_fl   = DINOv2-base(Ffl, t+30) latent (FROZEN aux target for the AHA arm) -> encoded here, raw discarded.
plus state(pose7+grip1), 15-step act_chunk, instr, ep/t/T. Peak RAM bounded by CHUNK_EP.

Output: cache/dense_fractal_{MAX_EP}.npz
env: MAX_EP (default 20000), N_T (default 40), STRIDE (>0 overrides N_T with every-STRIDE sampling),
     CHUNK_EP (episodes per z_fl encode flush), JPEG_Q (default 95).
"""
import os, glob, numpy as np, cv2, torch
import tensorflow as tf, tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')   # TF reads TFRecords on CPU only; leave the GPU entirely to torch (DINO-base)

DATA = "/data5/jellyho/Hindsight/fractal_rlds/fractal20220817_data/0.1.0"
OUT  = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
CAM, RES, K_LARGE, H_CHUNK, MIN_LEN = "image", 224, 30, 15, 30
MAX_EP  = int(os.environ.get("MAX_EP", 20000))
N_T     = int(os.environ.get("N_T", 40))
STRIDE  = int(os.environ.get("STRIDE", 0))
CHUNK_EP= int(os.environ.get("CHUNK_EP", 400))
JPEG_Q  = int(os.environ.get("JPEG_Q", 95))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def resize(img): return cv2.resize(np.asarray(img), (RES, RES), interpolation=cv2.INTER_AREA)
def jpg(img):    return cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()  # RGB->BGR for cv2
def act_vec(a):
    return np.concatenate([np.asarray(a["world_vector"], np.float32),
                           np.asarray(a["rotation_delta"], np.float32),
                           np.asarray(a["gripper_closedness_action"], np.float32).reshape(-1)[:1]])
def chunk(acts, t, H=H_CHUNK):
    c = acts[t:t+H]
    if len(c) < H: c = np.concatenate([c, np.repeat(c[-1:], H-len(c), axis=0)], axis=0)
    return c.astype(np.float32)

print(f"loading DINOv2-base (for z_fl aux target) on {DEV} ...", flush=True)
from transformers import Dinov2Model
_m = Dinov2Model.from_pretrained("facebook/dinov2-base").to(DEV).eval()
_mean = torch.tensor([0.485,0.456,0.406], device=DEV).view(1,3,1,1)
_std  = torch.tensor([0.229,0.224,0.225], device=DEV).view(1,3,1,1)
@torch.no_grad()
def enc(frames):
    out=[]; bs=512 if DEV=="cuda" else 128
    for i in range(0,len(frames),bs):
        b=torch.from_numpy(frames[i:i+bs]).to(DEV).float().permute(0,3,1,2)/255.0
        b=(b-_mean)/_std
        out.append(_m(pixel_values=b).last_hidden_state[:,0].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,768),np.float32)

def main():
    shards=[s for s in sorted(glob.glob(DATA+"/fractal*train.tfrecord-*")) if ".gstmp" not in s]
    print(f"shards={len(shards)}  MAX_EP={MAX_EP}  N_T={N_T} STRIDE={STRIDE} JPEG_Q={JPEG_Q}", flush=True)
    feats=tfds.builder_from_directory(DATA).info.features
    ds=tf.data.TFRecordDataset(shards).map(feats.deserialize_example)

    ft_jpg=[]; zfl_parts=[]; buf_ffl=[]
    meta={k:[] for k in ["cartt","gript","act_chunk","instr","ep_id","t","T","progress"]}
    ep_id=0; used=0
    def flush():
        if not buf_ffl: return
        zfl_parts.append(enc(np.stack(buf_ffl))); buf_ffl.clear()
    for ep in ds:
        if used>=MAX_EP: break
        steps=list(ep["steps"]); T=len(steps)
        if T<MIN_LEN: ep_id+=1; continue
        frames=[s["observation"][CAM].numpy() for s in steps]
        poses=np.stack([s["observation"]["base_pose_tool_reached"].numpy() for s in steps]).astype(np.float32)
        grips=np.stack([np.asarray(s["observation"]["gripper_closed"].numpy(),np.float32).reshape(-1)[:1] for s in steps])
        acts=np.stack([act_vec(s["action"]) for s in steps])
        instr=steps[0]["observation"]["natural_language_instruction"].numpy().decode("utf-8","ignore").strip()
        ts=(np.arange(3,T-3,STRIDE) if STRIDE>0 else np.unique(np.linspace(3,T-3,N_T).astype(int)))
        for t in ts:
            t=int(t); tfl=min(t+K_LARGE, T-1)
            ft=resize(frames[t])
            ft_jpg.append(jpg(ft)); buf_ffl.append(resize(frames[tfl]))
            meta["cartt"].append(poses[t]); meta["gript"].append(grips[t])
            meta["act_chunk"].append(chunk(acts,t)); meta["instr"].append(instr)
            meta["ep_id"].append(ep_id); meta["t"].append(t); meta["T"].append(int(T)); meta["progress"].append(t/(T-1))
        used+=1; ep_id+=1
        if used % CHUNK_EP == 0:
            flush(); print(f"  used {used} ep, {len(ft_jpg)} transitions", flush=True)
    flush()

    out={}
    out["Ft_jpg"]=np.array(ft_jpg, dtype=object)
    out["z_fl"]=np.concatenate(zfl_parts).astype(np.float32) if zfl_parts else np.zeros((0,768),np.float32)
    out["cartt"]=np.array(meta["cartt"],np.float32); out["gript"]=np.array(meta["gript"],np.float32)
    out["act_chunk"]=np.array(meta["act_chunk"],np.float32); out["instr"]=np.array(meta["instr"],dtype=object)
    for k in ("ep_id","t","T"): out[k]=np.array(meta[k],np.int32)
    out["progress"]=np.array(meta["progress"],np.float32)
    path=os.path.join(OUT,f"cache/dense_fractal_{MAX_EP}.npz")
    np.savez_compressed(path,**out)
    mb=sum(len(b) for b in ft_jpg)/1e6
    print(f"SAVED {path}  N={len(ft_jpg)}  episodes={used}  jpg~{mb:.0f}MB  z_fl={out['z_fl'].shape}", flush=True)

if __name__=="__main__": main()
