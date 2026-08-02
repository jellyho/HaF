"""RT-1-style lightweight VLA backbone in JAX / Flax (linen) — matches src/haf's backbone style (siglip_gemma3).

Same recipe as the torch version (rt1_vla.py): conv image encoder -> FiLM(language) -> TokenLearner (K tokens) ->
Transformer -> backbone token sequence [B, K, d]. Head-agnostic so action/aux heads plug in (bin256 / FAST-NTP / VQ
/ latent-aux). Self-contained lightweight ConvStem here; swap in src/haf's SigLIP encoder for a pretrained ViT
(then it becomes the pi/openpi-lineage backbone for M2). NHWC tensors (Flax convention).

Smoke on CPU:  JAX_PLATFORMS=cpu python rt1_vla_jax.py
"""
import jax, jax.numpy as jnp, flax.linen as nn


class ConvStem(nn.Module):
    chans: tuple = (32, 64, 128, 256, 512)
    @nn.compact
    def __call__(self, x):                                          # x [B,224,224,3] -> [B,7,7,512]
        for c in self.chans:
            x = nn.Conv(c, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.GroupNorm(num_groups=min(32, c))(x)
            x = nn.gelu(x)
        return x


class FiLM(nn.Module):
    @nn.compact
    def __call__(self, feat, lang):                                # feat [B,H,W,C], lang [B,Ld]
        C = feat.shape[-1]
        g, b = jnp.split(nn.Dense(2 * C)(lang), 2, -1)
        return feat * (1 + g[:, None, None, :]) + b[:, None, None, :]


class TokenLearner(nn.Module):
    k: int = 8
    @nn.compact
    def __call__(self, feat):                                      # [B,H,W,C] -> [B,K,C]
        C = feat.shape[-1]
        a = nn.Conv(C, (1, 1))(feat); a = nn.gelu(a); a = nn.Conv(self.k, (1, 1))(a)   # [B,H,W,K]
        B, H, W, K = a.shape
        a = jax.nn.softmax(a.reshape(B, H * W, K), axis=1)         # attention over spatial
        f = feat.reshape(B, H * W, C)
        return jnp.einsum("bnk,bnc->bkc", a, f)


class TFBlock(nn.Module):
    d: int; nhead: int
    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        x = x + nn.MultiHeadDotProductAttention(num_heads=self.nhead)(h, h)
        h = nn.LayerNorm()(x)
        x = x + nn.Dense(self.d)(nn.gelu(nn.Dense(4 * self.d)(h)))
        return x


class RT1BackboneJAX(nn.Module):
    d_model: int = 512; k_tokens: int = 8; depth: int = 4; nhead: int = 8
    @nn.compact
    def __call__(self, image, lang):                               # -> [B,K,d]
        f = ConvStem()(image)
        f = FiLM()(f, lang)
        t = nn.Dense(self.d_model)(TokenLearner(self.k_tokens)(f))
        pos = self.param("pos", nn.initializers.normal(0.02), (self.k_tokens, self.d_model))
        t = t + pos[None]
        for _ in range(self.depth):
            t = TFBlock(self.d_model, self.nhead)(t)
        return t


class BinActionHeadJAX(nn.Module):
    """RT-1/OpenVLA-style per-dim 256-bin action tokens. R = 1 - CE_val/CE_marg."""
    n_dims: int; bins: int = 256
    @nn.compact
    def __call__(self, tokens):                                    # -> [B, n_dims, bins]
        return nn.Dense(self.n_dims * self.bins)(tokens.mean(1)).reshape(-1, self.n_dims, self.bins)


def ce_loss(logits, ids):
    lp = jax.nn.log_softmax(logits, -1)
    return -jnp.take_along_axis(lp, ids[..., None], -1).mean()


if __name__ == "__main__":
    import math
    rng = jax.random.PRNGKey(0)
    B = 3; img = jax.random.normal(rng, (B, 224, 224, 3)); lang = jax.random.normal(rng, (B, 384))
    bb = RT1BackboneJAX(d_model=512, k_tokens=8, depth=2)
    vars_bb = bb.init(rng, img, lang); tok = bb.apply(vars_bb, img, lang)
    nparam = sum(x.size for x in jax.tree_util.tree_leaves(vars_bb))
    print("backbone tokens:", tok.shape, "| params:", nparam // 1000, "K")
    ah = BinActionHeadJAX(n_dims=7, bins=256)
    vars_ah = ah.init(rng, tok); logits = ah.apply(vars_ah, tok)
    ids = jax.random.randint(rng, (B, 7), 0, 256)
    print("bin256 action CE:", float(ce_loss(logits, ids)), "(random ~ ln256=%.2f)" % math.log(256))
    # grad through backbone (grounding) works
    def loss_fn(pb, pa): return ce_loss(ah.apply(pa, bb.apply(pb, img, lang)), ids)
    g = jax.grad(loss_fn)(vars_bb, vars_ah)
    gn = float(jnp.sqrt(sum((x ** 2).sum() for x in jax.tree_util.tree_leaves(g))))
    print("grad norm through RT-1 backbone:", round(gn, 4), "(>0 => trains)")
    print("SMOKE OK")
