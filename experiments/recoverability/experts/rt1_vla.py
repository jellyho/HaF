"""RT-1-style lightweight VLA backbone (~a few M params on top of EfficientNet-B0).

Faithful to the RT-1 recipe's defining pieces:
  EfficientNet image encoder  ->  FiLM conditioning on the language embedding  ->  TokenLearner (learned spatial
  pooling to K tokens)  ->  Transformer over the tokens  ->  backbone token sequence [B, K, d].

The backbone is head-agnostic so our experts plug straight in as the action / aux heads (mirroring real VLAs):
  * BinActionHead        : RT-1/OpenVLA-style per-dim 256-bin action tokens (cross-entropy). default action head.
  * TokenNTPExpert       : official FAST action tokens, autoregressive NTP.
  * LatentPredictionExpert: the AHA low-recoverability aux (future-frame latent; flow / mse / vq heads).
Native to RT-1/fractal + SimplerEnv (RT-1 is a SimplerEnv baseline). Run in .venv (torch+torchvision+transformers).
"""
import torch, torch.nn as nn


class FiLM(nn.Module):
    """Per-channel feature-wise linear modulation of a conv feature map, conditioned on language."""
    def __init__(self, lang_dim, channels):
        super().__init__(); self.to_gb = nn.Linear(lang_dim, 2 * channels); self.c = channels
    def forward(self, feat, lang):                                   # feat [B,C,H,W], lang [B,Ld]
        g, b = self.to_gb(lang).chunk(2, -1)
        return feat * (1 + g[..., None, None]) + b[..., None, None]


class TokenLearner(nn.Module):
    """Learn K spatial attention maps and weighted-pool the feature map into K tokens."""
    def __init__(self, channels, k=8):
        super().__init__()
        self.att = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.GELU(), nn.Conv2d(channels, k, 1))
        self.k = k
    def forward(self, feat):                                         # [B,C,H,W] -> [B,K,C]
        B, C, H, W = feat.shape
        a = self.att(feat).flatten(2).softmax(-1)                    # [B,K,HW]
        f = feat.flatten(2)                                          # [B,C,HW]
        return torch.einsum("bkn,bcn->bkc", a, f)


class RT1Backbone(nn.Module):
    def __init__(self, lang_dim=384, d_model=512, k_tokens=8, depth=4, nhead=8, pretrained=True):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = efficientnet_b0(weights=w).features         # -> [B,1280,H/32,W/32]
        C = 1280
        self.film = FiLM(lang_dim, C)
        self.tl = TokenLearner(C, k_tokens)
        self.proj = nn.Linear(C, d_model)
        self.pos = nn.Parameter(torch.randn(k_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, batch_first=True, dropout=0.0)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.d_model = d_model; self.k = k_tokens
    def tokens(self, image, lang):                                  # image [B,3,224,224] normalized, lang [B,Ld]
        f = self.features(image)
        f = self.film(f, lang)
        t = self.proj(self.tl(f)) + self.pos.unsqueeze(0)
        return self.tf(t)                                           # [B,K,d]


class BinActionHead(nn.Module):
    """RT-1/OpenVLA-style: discretize each action dim into `bins` uniform bins, predict per-dim (cross-entropy).
    Recoverability of the action, real-VLA way: R = 1 - CE_val/CE_marg."""
    def __init__(self, d_model, n_dims, bins=256):
        super().__init__(); self.n, self.bins = n_dims, bins
        self.head = nn.Linear(d_model, n_dims * bins)
    def logits(self, tokens):                                       # pool tokens -> [B, n_dims, bins]
        return self.head(tokens.mean(1)).view(-1, self.n, self.bins)
    def loss(self, tokens, dim_ids):                                # dim_ids [B, n_dims] long in [0,bins)
        return nn.functional.cross_entropy(self.logits(tokens).reshape(-1, self.bins), dim_ids.reshape(-1))
    @torch.no_grad()
    def predict_bins(self, tokens):
        return self.logits(tokens).argmax(-1)                       # [B, n_dims]


if __name__ == "__main__":
    torch.manual_seed(0); B = 3
    img = torch.randn(B, 3, 224, 224); lang = torch.randn(B, 384)
    bb = RT1Backbone(pretrained=False, d_model=512, k_tokens=8, depth=2)
    tok = bb.tokens(img, lang)
    ntop = sum(p.numel() for p in bb.parameters()) - sum(p.numel() for p in bb.features.parameters())
    print("backbone tokens:", tuple(tok.shape), "| effnet params:",
          sum(p.numel() for p in bb.features.parameters()) // 1_000_000, "M | RT-1 head params:", ntop // 1000, "K")
    # 256-bin action head (7 dims)
    ah = BinActionHead(512, n_dims=7, bins=256)
    ids = torch.randint(0, 256, (B, 7))
    import math; print("bin256 action CE:", float(ah.loss(tok, ids)), "(random ~ ln256=%.2f)" % math.log(256))
    # experts plug in as alternative heads
    import sys; sys.path.insert(0, "/data5/jellyho/Hindsight/HaF/experiments/recoverability/experts")
    from token_ntp_expert import TokenNTPExpert
    from latent_expert import LatentPredictionExpert
    fast = TokenNTPExpert(bb.d_model, vocab=2048, d_model=256, depth=2, nhead=4)
    ft = torch.randint(0, 2048, (B, 30)); fp = torch.zeros(B, 30, dtype=torch.bool)
    print("FAST-NTP action CE (on RT-1 tokens):", float(fast.ntp_loss(tok, ft, fp)))
    lat = LatentPredictionExpert(bb.d_model, latent_dim=768, d_model=256, depth=2, num_queries=1, head="flow")
    print("latent-aux flow loss (on RT-1 tokens):", float(lat.flow_loss(tok, torch.randn(B, 768))))
    print("SMOKE OK")
