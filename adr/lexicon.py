"""The frozen relation lexicon and the caption parser.

Each phrase maps to a unit axis in normalised image coordinates (x right, y down), oriented so that a true relation
between correctly localised objects gives a positive displacement e . (c_subject - c_object). The parser keeps the
content words of a caption, takes the longest lexicon match, the nearest noun before it as subject and the nearest
noun after it as object. Both were frozen before any benchmark evaluation.
"""
import re

import numpy as np

LEX = {}
for _ph in ["to the right of", "right of"]:
    LEX[_ph] = (1.0, 0.0)
for _ph in ["to the left of", "left of"]:
    LEX[_ph] = (-1.0, 0.0)
for _ph in ["above", "over", "on top of", "on", "onto", "sitting on", "standing on",
            "lying on", "parked on", "resting on", "walking on", "riding"]:
    LEX[_ph] = (0.0, -1.0)
for _ph in ["below", "under", "beneath", "underneath", "lying under"]:
    LEX[_ph] = (0.0, 1.0)
LEX_SORTED = sorted(LEX, key=len, reverse=True)
LEX_RE = {ph: re.compile(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])") for ph in LEX}
LR_RELS = {"to the left of", "to the right of"}

# VSR's own horizontal phrasings, declared before the VSR run and used only there
LEX_VSR = dict(LEX)
LEX_VSR["at the left side of"] = (-1.0, 0.0)
LEX_VSR["at the right side of"] = (1.0, 0.0)

# depth relations are read on the image vertical axis (What'sUp-B front/behind)
DEPTH = {"in front of": (0.0, 1.0), "behind": (0.0, -1.0)}

STOP = set(("a an the of in on at by with and or to is are was were be been being this that these "
            "those there here it its his her their our your my as for from into over under near next "
            "two three four five some many several front back side has have had").split())

_NLP = None


def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def parse(cap, lexicon=None):
    """Return (seq, edges): the content words and the parsed relations (subject_index, object_index, axis, phrase).
    An empty edge list means the caption is not covered and the readout abstains."""
    lex = LEX if lexicon is None else lexicon
    lex_sorted = sorted(lex, key=len, reverse=True)
    lex_re = {ph: re.compile(r"(?<![a-z])" + re.escape(ph) + r"(?![a-z])") for ph in lex}
    doc = nlp()(cap.lower())
    toks = [t for t in doc if t.is_alpha and len(t.text) > 2 and t.text not in STOP][:12]
    seq = [t.text for t in toks]
    if not seq:
        return ["image"], []
    idx = {t.i: k for k, t in enumerate(toks)}
    edges, used, text = [], [], doc.text
    for ph in lex_sorted:
        for mt in lex_re[ph].finditer(text):
            s0, s1 = mt.span()
            if any(not (s1 <= u0 or s0 >= u1) for u0, u1 in used):
                continue
            used.append((s0, s1))
            subj = obj = None
            for t in doc:
                if t.idx + len(t.text) <= s0 and t.pos_ in ("NOUN", "PROPN") and t.i in idx:
                    subj = t
                if t.idx >= s1 and t.pos_ in ("NOUN", "PROPN") and t.i in idx and obj is None:
                    obj = t
            if subj is not None and obj is not None and subj.i != obj.i:
                edges.append((idx[subj.i], idx[obj.i], lex[ph], ph))
    return seq, edges


def grid(gh, gw):
    """Normalised patch centres, row-major, as an N x 2 array of (x, y)."""
    return np.stack([np.tile((np.arange(gw) + .5) / gw, gh), np.repeat((np.arange(gh) + .5) / gh, gw)], 1)
