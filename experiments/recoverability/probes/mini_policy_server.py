"""Torch mini-VLA policy server for SimplerEnv rollout (runs in enc_venv).
Speaks the openpi_client WebsocketClientPolicy protocol: on connect send msgpack metadata; then per request
unpack obs {image[224,224,3] uint8, state[8], prompt str} -> return {"actions": [chunk,7]} (denormalized RT-1).
Env: ARM=bc|aha  PORT=8000. Loads outputs/policy_{ARM}.pt (weights + norm stats)."""
import os
import numpy as np
import torch
import torch.nn as nn
from openpi_client import msgpack_numpy
import websockets.sync.server

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
ARM = os.environ.get("ARM", "bc")
PORT = int(os.environ.get("PORT", "8000"))
DEV = "cuda"
DMODEL = 256

ck = torch.load(f"{OUT}/policy_{ARM}.pt", map_location=DEV, weights_only=False)


class Model(nn.Module):
    def __init__(self, sdim, ldim, adim, auxdim, dm=DMODEL):
        super().__init__()
        from transformers import Dinov2Model
        self.enc = Dinov2Model.from_pretrained("facebook/dinov2-small")
        ri = self.enc.config.hidden_size
        self.p_img = nn.Linear(ri, dm); self.p_state = nn.Linear(sdim, dm); self.p_lang = nn.Linear(ldim, dm)
        self.type_emb = nn.Parameter(torch.zeros(3, dm))
        layer = nn.TransformerEncoderLayer(dm, nhead=4, dim_feedforward=4*dm, batch_first=True, dropout=0.0)
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        self.adim = adim
        self.t_emb = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))
        self.flow = nn.Sequential(nn.Linear(adim + dm + 64, 512), nn.GELU(),
                                  nn.Linear(512, 512), nn.GELU(), nn.Linear(512, adim))
        self.aux = nn.Linear(dm, auxdim) if auxdim else None

    def rep(self, x, s, l):
        rimg = self.enc(pixel_values=x).last_hidden_state[:, 0]
        toks = torch.stack([self.p_img(rimg), self.p_state(s), self.p_lang(l)], 1) + self.type_emb.unsqueeze(0)
        return self.backbone(toks).mean(1)

    def velocity(self, x_t, t, cond):
        return self.flow(torch.cat([x_t, cond, self.t_emb(t)], -1))


m = Model(len(ck["state_mean"]), 384, ck["adim"], ck["auxdim"]).to(DEV)
m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
m.eval()

from transformers import AutoTokenizer, AutoModel
_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_lm = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEV).eval()


@torch.no_grad()
def embed(text):
    e = _tok([text], padding=True, truncation=True, return_tensors="pt").to(DEV)
    h = _lm(**e).last_hidden_state; mask = e["attention_mask"].unsqueeze(-1).float()
    v = (h*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
    return v / (v.norm(dim=1, keepdim=True) + 1e-8)          # l2-normalized [1,384]


act_mean = torch.tensor(ck["act_mean"], device=DEV); act_std = torch.tensor(ck["act_std"], device=DEV)
state_mean = torch.tensor(ck["state_mean"], device=DEV); state_std = torch.tensor(ck["state_std"], device=DEV)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
CH, DOF = ck["chunk"], ck["adof"]


@torch.no_grad()
def fm_sample(cond, K=20, S=16):
    B = cond.shape[0]; c = cond.repeat_interleave(S, 0); x = torch.randn(B*S, m.adim, device=DEV)
    for k in range(K):
        t = torch.full((B*S, 1), k/K, device=DEV); x = x + (1.0/K)*m.velocity(x, t, c)
    return x.view(B, S, m.adim).mean(1)


@torch.no_grad()
def infer(obs):
    o = obs.get("observation", obs)                          # accept nested {observation:{...}} or flat
    img_np = o.get("base_0_rgb", o.get("image"))
    img = torch.from_numpy(np.asarray(img_np)).to(DEV).float().permute(2, 0, 1)[None] / 255.0
    img = (img - MEAN) / STD
    st = (torch.tensor(np.asarray(o["state"], np.float32), device=DEV)[None] - state_mean) / state_std
    lang = embed(str(obs.get("prompt", "")))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cond = m.rep(img, st, lang)
    a = fm_sample(cond.float())[0] * act_std + act_mean        # denorm z-score -> RT-1 units [adim]
    return {"actions": a.reshape(CH, DOF).cpu().numpy().astype(np.float32)}


_packer = msgpack_numpy.Packer()


def handler(conn):
    conn.send(_packer.pack({"model": f"mini-vla-{ARM}", "action_horizon": CH, "action_dim": DOF}))
    while True:
        try:
            data = conn.recv()
        except Exception:
            break
        try:
            res = infer(msgpack_numpy.unpackb(data))
            conn.send(_packer.pack(res))
        except Exception as e:
            conn.send(f"infer error: {e}")


if __name__ == "__main__":
    print(f"mini-policy server ARM={ARM} chunk={CH}x{DOF} on :{PORT}  (waiting for SimplerEnv client)", flush=True)
    with websockets.sync.server.serve(handler, "0.0.0.0", PORT) as server:
        server.serve_forever()
