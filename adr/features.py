"""Frozen dual encoders: the deployed pooled cosine and the dense patch features ADR reads.

Pooling-native rule. CLS models (CLIP and its fine-tunes): the value path of the last attention block applied to
every token, then ln_post and the projection. MAP models (SigLIP, SigLIP2): all blocks in full, then the attention-
pooling head applied to each final token on its own. Images are resized to a short side of MINSIDE pixels, rounded
to whole patches; captions are embedded word by word and as a whole.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from .lexicon import grid, parse

MINSIDE = 336
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# tag -> (open_clip architecture, pretrained tag or checkpoint env var, patch, family); the fine-tunes are loaded
# from author checkpoints named by the environment variable
MODELS = {
    "clip_b16":     ("ViT-B-16-quickgelu", "openai", 16, "cls"),
    "clip_b32":     ("ViT-B-32-quickgelu", "openai", 32, "cls"),
    "clip_l14":     ("ViT-L-14-quickgelu", "openai", 14, "cls"),
    "laion_b16":    ("ViT-B-16", "laion2b_s34b_b88k", 16, "cls"),
    "dfn_h14":      ("ViT-H-14-378-quickgelu", "dfn5b", 14, "cls"),
    "metaclip_h14": ("ViT-H-14-quickgelu", "metaclip_fullcc", 14, "cls"),
    "negclip":      ("ViT-B-32-quickgelu", "env:ADR_CKPT_NEGCLIP", 32, "cls"),
    "clove":        ("ViT-B-32-quickgelu", "env:ADR_CKPT_CLOVE", 32, "cls"),
    "ceclip":       ("ViT-B-32-quickgelu", "env:ADR_CKPT_CECLIP", 32, "cls"),
    "siglip_b16":   ("ViT-B-16-SigLIP", "webli", 16, "map"),
    "siglip_so400m": ("ViT-SO400M-14-SigLIP-384", "webli", 14, "map"),
    "siglip2_so400m": ("ViT-SO400M-14-SigLIP2-378", "webli", 14, "map"),
}


class Encoder:
    def __init__(self, tag="clip_b16", device=None, cache_dir=None):
        import open_clip
        assert tag in MODELS, "unknown model tag %r, choose from %s" % (tag, sorted(MODELS))
        self.tag = tag
        name, pre, self.patch, self.family = MODELS[tag]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cache_dir = cache_dir or os.environ.get("ADR_CACHE_DIR")
        if pre.startswith("env:"):
            ckpt = os.environ.get(pre[4:])
            assert ckpt and os.path.exists(ckpt), "set %s to the downloaded %s checkpoint" % (pre[4:], tag)
            if tag == "clove":
                import copy
                import open_clip.factory as _F
                cfg = copy.deepcopy(_F.get_model_config(name))
                cfg["text_cfg"]["context_length"] = 64        # CLoVe was fine-tuned with context length 64
                name = name + "-ctx64"
                _F._MODEL_CONFIGS[name] = cfg
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=None, cache_dir=cache_dir)
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ck.get("state_dict", ck)
            sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
            missing, _ = model.load_state_dict(sd, strict=False)
            assert not missing, "missing keys in %s: %s" % (tag, missing[:5])
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pre, cache_dir=cache_dir)
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess
        self.tok = open_clip.get_tokenizer(name)
        if self.family == "map":
            nrm = preprocess.transforms[-1]
            self.norm = T.Normalize(mean=nrm.mean, std=nrm.std)
            trunk = self.model.visual.trunk
            assert trunk.cls_token is None and trunk.attn_pool is not None and trunk.attn_pool.pos_embed is None
        else:
            self.norm = T.Normalize(CLIP_MEAN, CLIP_STD)

    @torch.no_grad()
    def caption(self, cap, lexicon=None):
        """Parse the caption and embed its content words (W) and the whole caption (ce)."""
        seq, edges = parse(cap, lexicon)
        W = F.normalize(self.model.encode_text(self.tok(seq).to(self.device)), dim=-1)
        ce = F.normalize(self.model.encode_text(self.tok([cap]).to(self.device)), dim=-1)[0]
        return {"seq": seq, "edges": edges, "W": W, "ce": ce}

    @torch.no_grad()
    def image(self, img):
        """Dense unit-norm patch features Fp (N x d), the pooled embedding g and the patch grid P (N x 2)."""
        W0, H0 = img.size
        scl = MINSIDE / min(W0, H0)
        p = self.patch
        W1 = max(p, int(round(W0 * scl / p)) * p)
        H1 = max(p, int(round(H0 * scl / p)) * p)
        px = self.norm(T.functional.to_tensor(img.resize((W1, H1), Image.BICUBIC))).unsqueeze(0).to(self.device)
        Fp, gh, gw = self._dense_cls(px) if self.family == "cls" else self._dense_map(px)
        g = F.normalize(self.model.encode_image(self.preprocess(img).unsqueeze(0).to(self.device)), dim=-1)[0]
        return Fp, g, grid(gh, gw)

    def _dense_cls(self, px):
        visual = self.model.visual
        x = visual.conv1(px)
        gh, gw = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + self._interp_pos_cls(visual, gh, gw).to(x.dtype)
        x = visual.ln_pre(x)
        blocks = visual.transformer.resblocks
        for blk in blocks[:-1]:
            x = blk(x)
        last = blocks[-1]
        y = last.ln_1(x)
        d = y.shape[-1]
        w_v = last.attn.in_proj_weight.split(d, 0)[2]
        b_v = last.attn.in_proj_bias.split(d, 0)[2]
        v = F.linear(y, w_v, b_v)
        f = visual.ln_post(last.attn.out_proj(v))[:, 1:, :]
        if visual.proj is not None:
            f = f @ visual.proj
        return F.normalize(f[0], dim=-1), gh, gw

    @staticmethod
    def _interp_pos_cls(visual, gh, gw):
        pe = visual.positional_embedding
        n0 = pe.shape[0] - 1
        s = int(round(n0 ** 0.5))
        cls_pe, grid_pe = pe[:1], pe[1:]
        grid_pe = grid_pe.reshape(1, s, s, -1).permute(0, 3, 1, 2)
        grid_pe = F.interpolate(grid_pe, size=(gh, gw), mode="bicubic", align_corners=False)
        return torch.cat([cls_pe, grid_pe.permute(0, 2, 3, 1).reshape(gh * gw, -1)], 0)

    def _dense_map(self, px):
        trunk = self.model.visual.trunk
        x = trunk.patch_embed.proj(px)
        gh, gw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = trunk.patch_embed.norm(x)
        pos = trunk.pos_embed
        n, c = pos.shape[-2:]
        s = int(round(n ** 0.5))
        pos = F.interpolate(pos.reshape(1, s, s, c).permute(0, 3, 1, 2), size=(gh, gw), mode="bicubic", align_corners=False)
        x = x + pos.permute(0, 2, 3, 1).reshape(1, gh * gw, c).to(x.dtype)
        x = trunk.norm_pre(x)
        for blk in trunk.blocks:
            x = blk(x)
        x = trunk.norm(x)
        ap = trunk.attn_pool                           # on one token the attention over a single key is the identity
        B, N, C = x.shape
        kv = ap.kv(x).reshape(B, N, 2, ap.num_heads, ap.head_dim)
        v = kv[:, :, 1].reshape(B, N, C)
        y = ap.proj(v)
        f = trunk.fc_norm(y + ap.mlp(ap.norm(y)))
        return F.normalize(f[0], dim=-1), gh, gw

    def similarity(self, Fp, pack):
        return (Fp @ pack["W"].T).cpu().numpy().astype(np.float64)

    @staticmethod
    def deployed(g, pack):
        return float(g @ pack["ce"])
