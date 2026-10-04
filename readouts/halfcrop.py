"""The half-crop input intervention: embed the left, right, top and bottom halves with the pooled encoder and score
a caption by cos(half toward +e, subject word) + cos(half toward -e, object word). A single statement is decided by
the sign of the score minus the subject-object swapped score."""
import torch
import torch.nn.functional as F

from adr import Encoder, parse

from .base import ScoringReadout
from .dual_encoder import LEXICONS


class HalfCrop(ScoringReadout):
    kind = "swap"

    def __init__(self, model="clip_b16", encoder=None, device=None):
        self.enc = encoder or Encoder(model, device=device)
        self.lexicon = LEXICONS["main"]
        self._cache = (None, None)

    def use_lexicon(self, name):
        self.lexicon = LEXICONS[name]

    def covers(self, caption):
        return bool(parse(caption, self.lexicon)[1])

    @torch.no_grad()
    def halves(self, img):
        if self._cache[0] is img:
            return self._cache[1]
        W, H = img.size
        crops = {"L": img.crop((0, 0, max(W // 2, 8), H)), "R": img.crop((min(W // 2, W - 8), 0, W, H)),
                 "T": img.crop((0, 0, W, max(H // 2, 8))), "B": img.crop((0, min(H // 2, H - 8), W, H))}
        enc = self.enc
        out = {k: F.normalize(enc.model.encode_image(enc.preprocess(c).unsqueeze(0).to(enc.device)), dim=-1)[0] for k, c in crops.items()}
        self._cache = (img, out)
        return out

    def _score(self, halves, pk, swap=False):
        D = 0.0
        for s_ix, o_ix, e, _ in pk["edges"]:
            if swap:
                s_ix, o_ix = o_ix, s_ix
            if abs(e[0]) >= abs(e[1]):
                hs, ho = ("R", "L") if e[0] > 0 else ("L", "R")
            else:
                hs, ho = ("B", "T") if e[1] > 0 else ("T", "B")
            D += float(halves[hs] @ pk["W"][s_ix]) + float(halves[ho] @ pk["W"][o_ix])
        return D

    def scores(self, img, captions):
        h = self.halves(img)
        return [self._score(h, pk) if pk["edges"] else 0.0 for pk in (self.enc.caption(c, self.lexicon) for c in captions)]

    def swap_score(self, img, caption):
        pk = self.enc.caption(caption, self.lexicon)
        return self._score(self.halves(img), pk, swap=True) if pk["edges"] else 0.0
