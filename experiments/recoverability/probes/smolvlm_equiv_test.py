"""STRICT equivalence: does our fast path reproduce the OFFICIAL processor path token-for-token?

(A) official : processor(no-split) -> model.generate
(B) ours     : processor's text ids + OUR pixel pipeline -> visual tokens scattered into <image> slots -> generate
Compares generated token ids exactly, plus hidden-state cosine. Also reports pixel diff (our resize vs processor's).
"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/src")
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from haf_torch.models.smolvlm_fast import resize_for_vlm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MID = "HuggingFaceTB/SmolVLM-256M-Instruct"; DT = torch.float32
Q = "What do you see in this image? Describe the scene."

def frame():
    try:
        from haf_torch.data import fractal as fx
        for tr in fx.stream_transitions(max_ep=1, n_t=3):
            return np.asarray(tr["image"], np.uint8)
    except Exception as e:
        print("(synthetic)", e, flush=True)
    a = np.full((256, 320, 3), 240, np.uint8); a[60:170, 60:170] = [200, 30, 30]; return a

img = frame(); print(f"frame {img.shape}", flush=True)
m = AutoModelForImageTextToText.from_pretrained(MID, torch_dtype=DT, attn_implementation="sdpa").to(DEV).eval()
proc = AutoProcessor.from_pretrained(MID); proc.image_processor.do_image_splitting = False   # 64 tokens, 1 tile
tk = proc.tokenizer; inner = m.model
IMG_ID = tk.convert_tokens_to_ids("<image>")

tpl = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}],
                               add_generation_prompt=True)
b = proc(text=tpl, images=[Image.fromarray(img)], return_tensors="pt").to(DEV)
ids, am = b["input_ids"], b["attention_mask"]
print(f"seq len {ids.shape[1]}, <image> slots {(ids==IMG_ID).sum().item()}, pixels {tuple(b['pixel_values'].shape)}", flush=True)

# ---- (A) official ----
with torch.no_grad():
    outA = m.generate(**b, max_new_tokens=40, do_sample=False)
genA = outA[0, ids.shape[1]:]
print("\n=== (A) OFFICIAL ===\n" + tk.decode(genA, skip_special_tokens=True).strip(), flush=True)

# ---- our pixel pipeline ----
# our production path: workers do PIL LANCZOS stretch to 512, GPU only normalizes
from PIL import Image as _I
_lz = np.asarray(_I.fromarray(img).resize((512, 512), _I.LANCZOS), dtype=np.uint8)
ours_px = (torch.from_numpy(_lz).to(DEV)[None].permute(0, 3, 1, 2).float() / 255.0) * 2.0 - 1.0
proc_px = b["pixel_values"].reshape(-1, 3, 512, 512)
print(f"\npixel diff ours vs processor: max {float((ours_px-proc_px).abs().max()):.4f} "
      f"mean {float((ours_px-proc_px).abs().mean()):.4f}", flush=True)

def gen_ours(pixels, label, n=40):
    with torch.no_grad():
        vis = inner.connector(inner.vision_model(pixel_values=pixels).last_hidden_state)   # [1,64,D]
        emb = inner.text_model.embed_tokens(ids).clone()
        emb[ids == IMG_ID] = vis.reshape(-1, vis.shape[-1])
        cur_emb, cur_mask, out_ids = emb, am, []
        for _ in range(n):
            hs = inner.text_model(inputs_embeds=cur_emb, attention_mask=cur_mask).last_hidden_state[:, -1]
            nid = m.lm_head(hs).argmax(-1, keepdim=True)
            out_ids.append(nid.item())
            if nid.item() == tk.eos_token_id: break
            cur_emb = torch.cat([cur_emb, inner.text_model.embed_tokens(nid)], 1)
            cur_mask = torch.cat([cur_mask, torch.ones(1, 1, dtype=cur_mask.dtype, device=DEV)], 1)
    print(f"\n=== ({label}) ===\n" + tk.decode(out_ids, skip_special_tokens=True).strip(), flush=True)
    return out_ids

idsB = gen_ours(proc_px, "B) OURS wiring + PROCESSOR pixels")
idsC = gen_ours(ours_px, "C) OURS wiring + OUR pixels (fast path)")

gA = genA.tolist()
def cmp(x, name):
    k = min(len(gA), len(x)); same = sum(int(a == b) for a, b in zip(gA[:k], x[:k]))
    first = next((i for i, (a, b) in enumerate(zip(gA[:k], x[:k])) if a != b), k)
    print(f"{name}: {same}/{k} tokens identical to (A), first divergence at token {first}", flush=True)
print()
cmp(idsB, "B vs A"); cmp(idsC, "C vs A")
print("\nEQUIV_TEST_OK", flush=True)
