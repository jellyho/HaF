"""JAX RT-1 policy server for SimplerEnv closed-loop rollout (Stage 2). Loads a trained flax RT-1 policy
(rt1_full_a{ARM}.msgpack: {params, amu, asd}), speaks the openpi_client WebsocketClientPolicy protocol
(msgpack_numpy; metadata on connect; per request obs -> {"actions":[chunk,7]}), and flow-samples the action chunk.

RT-1 uses image + language only (no proprio). Language = MiniLM (torch, CPU). Runs in .venv (jax + websockets +
openpi_client + transformers). env: ARM (0=BC | 1=AHA), PORT (8000).
"""
import os, numpy as np, jax, jax.numpy as jnp, flax.linen as nn
from flax.serialization import from_bytes
import websockets.sync.server
from openpi_client import msgpack_numpy
import sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
from rt1_vla_jax import RT1BackboneJAX

OUT = "/data5/jellyho/Hindsight/HaF/experiments/recoverability/outputs"
ARM = os.environ.get("ARM", "1"); PORT = int(os.environ.get("PORT", "8000")); CKPT = os.environ.get("CKPT", "")
ADIM, LANGD, DMODEL, CH, DOF = 105, 384, 512, 15, 7

def temb(t, d=64):
    half=d//2; fr=jnp.exp(-jnp.log(10000.0)*jnp.arange(half)/half); a=t*fr[None]
    return jnp.concatenate([jnp.cos(a),jnp.sin(a)],-1)
class FlowHead(nn.Module):
    @nn.compact
    def __call__(self,x,t,cond):
        h=jnp.concatenate([x,cond,temb(t)],-1); h=nn.gelu(nn.Dense(512)(h)); h=nn.gelu(nn.Dense(512)(h)); return nn.Dense(ADIM)(h)
class RT1Flow(nn.Module):
    def setup(self):
        self.bb=RT1BackboneJAX(d_model=DMODEL,k_tokens=8,depth=4); self.flow=FlowHead(); self.pred=nn.Dense(384)
    def __call__(self,im,lng,x,t): cond=self.bb(im,lng).mean(1); return self.flow(x,t,cond), self.pred(cond)
    def cond(self,im,lng): return self.bb(im,lng).mean(1)
    def vel(self,x,t,cond): return self.flow(x,t,cond)

model=RT1Flow()
# init template params, then overwrite with loaded ones
tmpl=model.init(jax.random.PRNGKey(0), jnp.zeros((1,224,224,3)), jnp.zeros((1,LANGD)), jnp.zeros((1,ADIM)), jnp.zeros((1,1)))
ck=from_bytes({"params":tmpl,"amu":np.zeros(ADIM,np.float32),"asd":np.ones(ADIM,np.float32)},
              open(f"{OUT}/rt1_full_a{ARM}{CKPT}.msgpack","rb").read())
params=ck["params"]; amu=jnp.asarray(ck["amu"]); asd=jnp.asarray(ck["asd"])
print(f"loaded rt1_full_a{ARM}.msgpack (ARM={'AHA' if ARM=='1' else 'BC'})", flush=True)

from transformers import AutoTokenizer, AutoModel
import torch
_tk=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
_lm=AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").eval()
def embed(text):
    with torch.no_grad():
        e=_tk([text],padding=True,truncation=True,return_tensors="pt"); h=_lm(**e).last_hidden_state
        m=e["attention_mask"][...,None].float(); v=((h*m).sum(1)/m.sum(1).clamp(min=1e-9)).numpy()[0]
    return (v/(np.linalg.norm(v)+1e-8)).astype(np.float32)

MEAN=jnp.array([0.485,0.456,0.406]); STD=jnp.array([0.229,0.224,0.225])
@jax.jit
def cond_fn(im,lng): return model.apply(params,im,lng,method=RT1Flow.cond)
@jax.jit
def vel_fn(x,t,c): return model.apply(params,x,t,c,method=RT1Flow.vel)
def sample(cond,key,K=10,S=8):
    B=cond.shape[0]; c=jnp.repeat(cond,S,0); x=jax.random.normal(key,(B*S,ADIM))
    for i in range(K):
        t=jnp.full((B*S,1),i/K); x=x+(1.0/K)*vel_fn(x,t,c)
    return x.reshape(B,S,ADIM).mean(1)

import cv2
_key=[jax.random.PRNGKey(0)]
def infer(obs):
    o=obs.get("observation",obs); img=np.asarray(o.get("base_0_rgb",o.get("image")))
    if img.shape[:2]!=(224,224): img=cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
    im=(jnp.asarray(img,jnp.float32)/255.0-MEAN)/STD
    lng=jnp.asarray(embed(str(obs.get("prompt","")))[None])
    cond=cond_fn(im[None],lng)
    _key[0],sk=jax.random.split(_key[0])
    a=np.asarray(sample(cond,sk))[0]*np.asarray(asd)+np.asarray(amu)
    return {"actions": a.reshape(CH,DOF).astype(np.float32)}

_packer=msgpack_numpy.Packer()
def handler(conn):
    conn.send(_packer.pack({"model":f"rt1-{'AHA' if ARM=='1' else 'BC'}","action_horizon":CH,"action_dim":DOF}))
    while True:
        try: data=conn.recv()
        except Exception: break
        try: conn.send(_packer.pack(infer(msgpack_numpy.unpackb(data))))
        except Exception as e: conn.send(f"infer error: {e}")

if __name__=="__main__":
    print(f"RT-1 JAX server ARM={ARM} chunk={CH}x{DOF} on :{PORT}", flush=True)
    with websockets.sync.server.serve(handler,"0.0.0.0",PORT) as srv: srv.serve_forever()
