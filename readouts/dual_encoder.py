"""The deployed pooled cosine and ADR of one frozen dual encoder. Both share one Encoder instance; the pooled
embedding is the encoder's own image forward at native resolution, the dense features a second pass at short
side 336. The lexicon is set per set through `use_lexicon`."""
import numpy as np

from adr import DEPTH, LEX, LEX_VSR, Encoder, column_weights, displacement, parse
from adr.readout import all_scores

from .base import ScoringReadout

LEXICONS = {"main": LEX, "depth": dict(LEX, **DEPTH), "vsr": LEX_VSR}


class DualEncoderBase(ScoringReadout):
    def __init__(self, model="clip_b16", encoder=None, device=None):
        self.enc = encoder or Encoder(model, device=device)
        self.lexicon = LEX
        self._img_cache = (None, None)

    def use_lexicon(self, name):
        self.lexicon = LEXICONS[name]

    def covers(self, caption):
        return bool(parse(caption, self.lexicon)[1])

    def _image(self, img):
        if self._img_cache[0] is img:
            return self._img_cache[1]
        feats = self.enc.image(img)
        self._img_cache = (img, feats)
        return feats


class Deployed(DualEncoderBase):
    kind = "score"

    def scores(self, img, captions):
        _, g, _ = self._image(img)
        return [self.enc.deployed(g, self.enc.caption(c, self.lexicon)) for c in captions]


class ADR(DualEncoderBase):
    kind = "signed"

    def __init__(self, model="clip_b16", variant="excess", encoder=None, device=None):
        super().__init__(model, encoder, device)
        self.variant = variant

    def scores(self, img, captions):
        Fp, _, P = self._image(img)
        out = []
        for c in captions:
            pk = self.enc.caption(c, self.lexicon)
            if not pk["edges"]:
                out.append(0.0)
                continue
            S = self.enc.similarity(Fp, pk)
            out.append(displacement(column_weights(S, self.variant), P, pk["edges"]))
        return out

    def record(self, img, caption):
        """Weights, masses, centroids and D of one caption, for figures and audits."""
        Fp, _, P = self._image(img)
        pk = self.enc.caption(caption, self.lexicon)
        if not pk["edges"]:
            return {"D": 0.0, "covered": False, "seq": pk["seq"], "edges": []}
        S = self.enc.similarity(Fp, pk)
        w = column_weights(S, self.variant)
        mass = w.sum(0)
        cen = (w.T @ P) / np.maximum(mass, 1e-30)[:, None]
        return {"D": displacement(w, P, pk["edges"]), "covered": True, "seq": pk["seq"], "edges": pk["edges"],
                "mass": mass.tolist(), "centroids": cen.tolist(), "S": S, "weights": w, "grid": P}


class ADRAll(DualEncoderBase):
    """All four column weights from one Sinkhorn run; scores() returns the excess variant and keeps the rest in `last`."""
    kind = "signed"

    def __init__(self, model="clip_b16", encoder=None, device=None):
        super().__init__(model, encoder, device)
        self.last = None

    def scores(self, img, captions):
        Fp, _, P = self._image(img)
        self.last, out = [], []
        for c in captions:
            pk = self.enc.caption(c, self.lexicon)
            s = all_scores(self.enc.similarity(Fp, pk), P, pk["edges"]) if pk["edges"] else {v: 0.0 for v in ("excess", "dense", "pi0", "argmax")}
            self.last.append(s)
            out.append(s["excess"])
        return out
