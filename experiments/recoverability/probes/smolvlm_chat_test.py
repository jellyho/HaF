"""Does SmolVLM actually SEE and CHAT — and does our fast path preserve that?

Three ways to feed the SAME real fractal frame, then greedy-decode the same question:
  (A) OFFICIAL   : AutoProcessor(image+text) -> model.generate                        [reference]
  (B) OURS-wiring: the processor's OWN pixel_values -> our vision/connector -> LM      [tests our WIRING]
  (C) OURS-fast  : uint8 -> GPU resize+normalize -> vision/connector -> LM             [tests our PREPROCESS]
(B) isolates whether the bypass wiring is right; (C) additionally tests our resize/normalize.
Also prints the numeric difference between the processor's pixel_values and ours.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/src")
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DT = torch.float32
Q = "What do you see in this image? Describe the scene."
print(f"device={DEV}", flush=True)


def get_frame():
    try:
        from haf_torch.data import fractal as fx
        for tr in fx.stream_transitions(max_ep=1, n_t=3, image_size=224, mask_ratio=0.0):
            return np.asarray(tr["image"], np.uint8), tr["instruction"]
    except Exception as e:
        print(f"(fractal unavailable: {e}; using synthetic)", flush=True)
    a = np.full((224, 224, 3), 240, np.uint8); a[60:170, 60:170] = [200, 30, 30]
    return a, "pick red block"


img_np, instr = get_frame()
print(f"frame {img_np.shape} | dataset instruction: '{instr}'", flush=True)

model = AutoModelForImageTextToText.from_pretrained(MID, torch_dtype=DT, attn_implementation="sdpa").to(DEV).eval()
proc = AutoProcessor.from_pretrained(MID)
tok = AutoTokenizer.from_pretrained(MID)
inner = model.model

# ---------------- (A) official ----------------
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
inp = proc(text=prompt, images=[Image.fromarray(img_np)], return_tensors="pt").to(DEV)
with torch.no_grad():
    out = model.generate(**inp, max_new_tokens=50, do_sample=False)
print("\n=== (A) OFFICIAL processor + generate ===", flush=True)
print(proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(), flush=True)

pv = inp["pixel_values"].reshape(-1, 3, inp["pixel_values"].shape[-2], inp["pixel_values"].shape[-1])
print(f"\nprocessor pixel_values {tuple(pv.shape)} range[{pv.min():.2f},{pv.max():.2f}]", flush=True)


def greedy_from_pixels(pixels, label):
    with torch.no_grad():
        vis = inner.connector(inner.vision_model(pixel_values=pixels).last_hidden_state)
        vis = vis.reshape(1, -1, vis.shape[-1])
        ids = tok(Q + "\n", return_tensors="pt").to(DEV)["input_ids"]
        h = torch.cat([vis, inner.text_model.embed_tokens(ids)], 1)
        mask = torch.ones(h.shape[:2], dtype=torch.long, device=DEV)
        gen = []
        for _ in range(50):
            hs = inner.text_model(inputs_embeds=h, attention_mask=mask).last_hidden_state[:, -1]
            nid = model.lm_head(hs).argmax(-1, keepdim=True)
            if nid.item() == tok.eos_token_id:
                break
            gen.append(nid.item())
            h = torch.cat([h, inner.text_model.embed_tokens(nid)], 1)
            mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=DEV)], 1)
    print(f"\n=== ({label}) ===", flush=True)
    print(f"visual tokens {tuple(vis.shape)}", flush=True)
    print(tok.decode(gen, skip_special_tokens=True).strip() or "(empty)", flush=True)


greedy_from_pixels(pv, "B) OURS wiring + processor pixels")

x = torch.from_numpy(img_np).to(DEV)[None].permute(0, 3, 1, 2).float() / 255.0
S = pv.shape[-1]
r = min(S / x.shape[-2], S / x.shape[-1])
nh, nw = int(round(x.shape[-2] * r)), int(round(x.shape[-1] * r))
xi = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
o = x.new_zeros((1, 3, S, S)); o[:, :, S - nh:, S - nw:] = xi
ours = (o * 2.0 - 1.0)
print(f"\nours pixel_values {tuple(ours.shape)} range[{ours.min():.2f},{ours.max():.2f}]", flush=True)
if ours.shape[-2:] == pv.shape[-2:]:
    d = (ours - pv[:1]).abs()
    print(f"DIFF vs processor tile0: max {d.max():.4f} mean {d.mean():.4f}", flush=True)
greedy_from_pixels(ours, "C) OURS fast path (GPU resize+norm)")

print("\nCHAT_TEST_OK", flush=True)
